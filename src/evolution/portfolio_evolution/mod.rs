use crate::evolution::aggregator::AggregatorError;
use crate::evolution::objective::OptimizationObjective;
use crate::portfolio::Portfolio;
use aegis_athena_contracts::common_consts::FLOAT_COMPARISON_EPSILON;
use aegis_athena_contracts::sampling::Sampler;
use std::boxed::Box;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{self, Write};

use crate::consts::{NUMBER_OF_OPTIMIZATION_OBJECTIVES, PERTURBATION};
use itertools::izip;
use num_cpus;
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::warn;

use futures::{future::BoxFuture, FutureExt};
use std::sync::Arc;
use thiserror::Error;

mod memetic;
mod pareto_evolution;

// Strategy Definition
trait EvolutionConfig {}
trait EvolutionStrategy {
    type Config: EvolutionConfig;
    async fn evolve(
        &self,
        cfg: &Self::Config,
        athena_endpoint: Option<String>,
    ) -> Result<EvolutionResult, EvolutionError>;
}

#[derive(Error, Debug)]
pub enum EvolutionError {
    #[error("Invalid population parameters were passed: `{0}`")]
    BadPopulationParameter(String),
    #[error("Need Athena runner endpoint, if SimRunnerStrategy is not local.")]
    MissingAthenaEndpoint,
    #[error("Cannot pass duplicated objective to evolution strategy.")]
    DuplicatedObjective,
    #[error("An error occured during computation of objective: `{0}`")]
    ObjectiveComputationFailed(AggregatorError),
}

fn default_max_concurrency() -> usize {
    num_cpus::get()
}

pub fn initialize_population(
    population_size: usize,
    assets_under_management: usize,
) -> Result<Vec<Vec<f64>>, EvolutionError> {
    if population_size == 0 && assets_under_management == 0 {
        return Err(EvolutionError::BadPopulationParameter("Both population size and assets under management are zero, but none are supposed to be.".into()));
    } else if population_size == 0 {
        return Err(EvolutionError::BadPopulationParameter(
            "Population size cannot be zero".into(),
        ));
    } else if assets_under_management == 0 {
        return Err(EvolutionError::BadPopulationParameter(
            "Assets under management cannot be zero".into(),
        ));
    }

    let rng = thread_rng();
    let uniform = Uniform::new(0., 1.);

    Ok((0..population_size)
        .map(|_| {
            let mut portfolio = rng
                .clone()
                .sample_iter(uniform)
                .take(assets_under_management)
                .collect::<Vec<f64>>();
            let magnitude = portfolio.iter().sum::<f64>();

            portfolio.iter_mut().for_each(|x| *x /= magnitude);
            portfolio
        })
        .collect::<Vec<_>>())
}

fn turn_weights_into_portfolios(population: &[Vec<f64>], stats: &[Vec<f64>]) -> Vec<Portfolio> {
    population
        .par_iter()
        .zip(stats.par_iter())
        .map(|(weights, &stats)| Portfolio::new(weights.to_vec(), stats.to_vec()))
        .collect()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StandardEvolutionConfig {
    pub time_horizon_in_days: usize,
    pub generations: usize,
    pub population_size: usize,
    pub simulations_per_generation: usize,
    pub assets_under_management: usize,
    pub money_to_invest: f64,
    pub risk_free_rate: f64,
    pub elitism_rate: f64,
    pub mutation_rate: f64,
    pub tournament_size: usize,
    pub sampler: Sampler,
    pub generation_check_interval: usize,
    pub objectives: Vec<Box<dyn OptimizationObjective>>, // add a monitoring metric trait and stuff
    #[serde(default)]
    pub global_seed: Option<u64>,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    pub sim_runner: dyn SimRunnerStrategy,
}
impl EvolutionConfig for StandardEvolutionConfig {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemeticParams {
    /// objective to use during the proximal step
    pub local_objective: BuiltInObjective,
    pub proximal_descent_steps: usize,
    pub proximal_descent_step_size: f64,
    pub high_sharpe_threshold: f64,
    pub low_volatility_threshold: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemeticEvolutionConfig {
    #[serde(flatten)]
    pub base: StandardEvolutionConfig,
    pub memetic: MemeticParams,
}

impl EvolutionConfig for MemeticEvolutionConfig {}

//

/// Contains summary statistics for the final population after evolution.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FinalPopulationSummary {
    pub bests: Vec<f64>,
    pub averages: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EvolutionResult {
    pub pareto_fronts: Vec<Vec<Portfolio>>,
    pub averages_matrix: Vec<Vec<f64>>, // this is the per-generation averages
    pub bests_matrix: Vec<Vec<f64>>,    // this is the per-generation maxes
    pub final_summary: FinalPopulationSummary,
}

pub trait SimRunnerStrategy {
    fn evaluate_population(
        &self,
        config: &StandardEvolutionConfig,
        population: &[Vec<f64>],
        athena_endpoint: Option<String>,
    ) -> Result<PopulationEvaluationResult, EvolutionError>;
}
pub struct Local;
impl SimRunnerStrategy for Local {
    fn evaluate_population(
        config: &StandardEvolutionConfig,
        population: &[Vec<f64>],
        athena_endpoint: Option<String>,
    ) -> PopulationEvaluationResult {
        let sampler = config.sampler;
        // preallocate a matrix of population.len() * objectives.len()
        let mut sum_matrix = vec![vec![0.; config.objectives.len()]; population.len()];
        let simulations_per_generation = config.simulations_per_generation;
        let mut scenario = sampler.sample_returns();
        let objectives = config.objectives;

        // Run the simulations
        for sim_i in 0..config.simulations_per_generation {
            // Just progressively modify the matrix, adding and then dividing (do everything
            // in-place)
            for (i, &objective) in objectives.iter().enumerate() {
                for (j, &weight) in population.iter().enumerate() {
                    let objective_value = objective
                        .compute(weight, scenario)
                        .unwrap_or_else(|e| EvolutionError::ObjectiveComputationFailed(e))?;
                    sum_matrix[j][i] += objective_value;
                }
            }

            scenario = sampler.sampler_returns();
        }

        // then do the averaging
        let inv_sims = 1.0 / simulations_per_generation;
        let average_matrix: Vec<Vec<f64>> = sum_matrix
            .into_iter()
            .map(|row| row.into_iter().map(|x| x * inv_sims).collect())
            .collect();
        let bests: Vec<f64> = (0..objectives.len())
            .map(|objective_idx| average_matrix.iter().map(|row| row[objective_idx]).max())
            .collect();

        // return
        PopulationEvaluationResult {
            average_matrix,
            bests,
            last_scenario_returns: scenario,
        }
    }
}

fn generate_offsprings(
    population: &[&Portfolio],
    offspring_count: usize,
    mutation_rate: f64,
    k: usize,
) -> Vec<Vec<f64>> {
    let mut offsprings = Vec::new();

    while offsprings.len() < offspring_count {
        let (parent_1, parent_2) = select_parents(population, k);

        let mut child_weights = crossover(&parent_1, &parent_2);

        // Toss a coin and stochastically mutate weights based on rate
        mutate(&mut child_weights, mutation_rate);

        offsprings.push(child_weights);
    }
    offsprings
}

fn select_parents(population: &[&Portfolio], k: usize) -> (Vec<f64>, Vec<f64>) {
    let parent_1 = tournament_selection(population, 2);
    let parent_2 = tournament_selection(population, 2);

    (parent_1.to_owned(), parent_2.to_owned())
}

fn tournament_selection(population_refs: &[&Portfolio], k: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    // 1. Choose k references randomly from the population slice.
    // choose_multiple returns an iterator yielding references to the elements of the slice.
    // Since the slice contains `&Portfolio`, the iterator yields `&&Portfolio`.
    let contestants: Vec<&&Portfolio> = population_refs.choose_multiple(&mut rng, k).collect();

    // Handle the case where k is 0 or larger than population size (choose_multiple might return fewer)
    if contestants.is_empty() {
        // This shouldn't happen if k > 0 and population_refs is not empty,
        // but handle defensively. Return uniform weights? Panic?
        // Let's panic as it indicates a configuration issue or unexpected state.
        panic!(
            "Tournament selection failed: No contestants selected (k={}, population_size={})",
            k,
            population_refs.len()
        );
    }

    // 2. Find the best contestant among the selected references.
    // We iterate through the contestants (&&Portfolio) and find the minimum
    // based on the Portfolio's comparison logic (rank then crowding distance).
    // We need to dereference twice (&&Portfolio -> &Portfolio) to use partial_cmp.
    let winner: &&Portfolio = contestants
        .into_iter()
        .min_by(|&&a, &&b| {
            // Compare the &Portfolio references using the implemented PartialOrd
            a.partial_cmp(b).unwrap_or(Ordering::Equal) // Treat non-comparable as equal for sorting
        })
        .expect("Failed to find a winner in tournament selection"); // Should not happen if contestants is not empty

    // 3. Clone the weights of the winning portfolio.
    // Dereference the winner (&&Portfolio -> &Portfolio) and clone its weights.
    winner.weights.clone()
}

fn crossover(parent_1: &Vec<f64>, parent_2: &Vec<f64>) -> Vec<f64> {
    let mut rng = thread_rng();

    parent_1
        .iter()
        .zip(parent_2.iter())
        .map(|(&weight_1, &weight_2)| {
            let alpha: f64 = rng.gen_range(0.0..1.0);
            alpha * weight_1 + (1.0 - alpha) * weight_2
        })
        .collect()
}

fn mutate(weights: &mut Vec<f64>, mutation_rate: f64) {
    let mut rng = thread_rng();

    // Mutate
    for weight in weights.iter_mut() {
        if rng.gen_bool(mutation_rate) {
            let change: f64 = rng.gen_range(-PERTURBATION..PERTURBATION);
            *weight += change;
        }
    }

    // Ensure non-negative weights
    for weight in weights.iter_mut() {
        if *weight < 0.0 {
            *weight = 0.0;
        }
    }

    // Normalize again
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= total);
}

/// Holds the results of evaluating a population over multiple simulations,
/// including both per-portfolio averages and population-wide summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationEvaluationResult {
    // Per-Portfolio Averages
    pub averages_matrix: Vec<Vec<f64>>,
    pub bests: Vec<f64>,
    // Last Simulation Data
    /// The return scenarios sampled during the *last* simulation run (needed for memetic search).
    pub last_scenario_returns: Vec<Vec<f64>>,
}
