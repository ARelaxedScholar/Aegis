use crate::k8s_job::evaluate_generation_in_k8s_job;
use aegis_athena_contracts::common_consts::FLOAT_COMPARISON_EPSILON;
use aegis_athena_contracts::common_portfolio_evolution_ds::compute_portfolio_performance;
use aegis_athena_contracts::common_portfolio_evolution_ds::PortfolioPerformance;
use aegis_athena_contracts::{portfolio::Portfolio, sampling::Sampler};
use std::cmp::Ordering;
use std::io::{self, Write};

use self::gradients::compute_portfolio_gradient;
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

mod gradients;
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
    #[error("Invalid population parameters were passed.")]
    BadPopulationParameter(String),
    #[error("Need Athena runner endpoint, if SimRunnerStrategy is not local.")]
    MissingAthenaEndpoint,
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

fn turn_weights_into_portfolios(
    population: &[Vec<f64>],
    simulation_average_returns: &[f64],
    simulation_average_volatilities: &[f64],
    simulation_average_sharpe_ratios: &[f64],
) -> Vec<Portfolio> {
    let portfolio_simulation_averages: Vec<(&Vec<f64>, &f64, &f64, &f64)> = izip!(
        population,
        simulation_average_returns,
        simulation_average_volatilities,
        simulation_average_sharpe_ratios
    )
    .collect();

    portfolio_simulation_averages
        .par_iter()
        .map(|(portfolio, &ave_ret, &ave_vol, &ave_sharpe)| {
            Portfolio::new(portfolio.to_vec(), ave_ret, ave_vol, ave_sharpe)
        })
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
    #[serde(default)]
    pub global_seed: Option<u64>,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    pub sim_runner: SimRunnerStrategy,
}
impl EvolutionConfig for StandardEvolutionConfig {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemeticParams {
    /// objective to use during the proximal step
    pub local_objective: Objective,
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
    /// Best (highest) average return found in the final population.
    pub best_return: f64,
    /// Average of the average returns across the final population.
    pub population_average_return: f64,
    /// Best (lowest) average volatility found in the final population.
    pub best_volatility: f64,
    /// Average of the average volatilities across the final population.
    pub population_average_volatility: f64,
    /// Best (highest) average Sharpe ratio found in the final population.
    pub best_sharpe: f64,
    /// Average of the average Sharpe ratios across the final population.
    pub population_average_sharpe: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EvolutionResult {
    pub pareto_fronts: Vec<Vec<Portfolio>>,
    pub best_average_return_per_generation: Vec<f64>,
    pub average_return_per_generation: Vec<f64>,
    pub best_average_volatility_per_generation: Vec<f64>,
    pub average_volatility_per_generation: Vec<f64>,
    pub best_average_sharpe_ratio_per_generation: Vec<f64>,
    pub average_sharpe_ratio_per_generation: Vec<f64>,
    pub final_summary: FinalPopulationSummary,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SimRunnerStrategy {
    Local,
    AthenaK8sJob,
    AthenaGrpc,
}

pub fn make_evaluator(
    sim_runner: SimRunnerStrategy,
    config: StandardEvolutionConfig,
    athena_endpoint: Option<String>,
) -> impl Fn(&[Vec<f64>]) -> BoxFuture<'static, anyhow::Result<PopulationEvaluationResult>> + Clone
{
    // wrap in Arcs so we can cheaply clone into each branch
    let cfg = Arc::new(config);
    let ep = Arc::new(athena_endpoint.unwrap());
    let runner = Arc::new(sim_runner);

    move |population: &[Vec<f64>]| {
        // First clone everything we need *owned* into this particular call:
        let cfg = cfg.clone();
        let ep = ep.clone();
        let runner = runner.clone();
        // clone the population slice into an owned Vec<Vec<f64>>
        let pop_owned = population.to_owned();

        // dispatch on runner
        match &*runner {
            SimRunnerStrategy::Local => {
                // synchronous path
                async move {
                    let res = evaluate_population_performance_local(&cfg, &pop_owned);
                    Ok(res)
                }
                .boxed()
            }
            _ => {
                unimplemented!("Yeah just crash here, this will be phased out in due time");
            }
        }
    }
}

// Helper function needs adjustment for threshold parameters
fn find_dominant_objective(
    performance_report: &PortfolioPerformance,
    high_sharpe_threshold: f64,
    low_volatility_threshold: f64,
) -> Objective {
    if performance_report.sharpe_ratio >= high_sharpe_threshold {
        Objective::SharpeRatio
    } else if performance_report.percent_annualized_volatility <= low_volatility_threshold {
        Objective::Volatility
    } else {
        Objective::AnnualizedReturns
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
                                                        // Or Ordering::Less/Greater if a default is needed
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

// For Local Search
// The idea is to push a solution in the solution it already excels in
// This should ideally lead to a more diverse Pareto Front, and more interesting solutions
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize, Copy)]
pub enum Objective {
    AnnualizedReturns,
    SharpeRatio,
    Volatility,
    MaximizeStrength,
}

/// Holds the results of evaluating a population over multiple simulations,
/// including both per-portfolio averages and population-wide summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationEvaluationResult {
    // Per-Portfolio Averages
    /// Average annualized return for each portfolio in the evaluated population.
    pub average_returns: Vec<f64>,
    /// Average annualized volatility (as a percentage) for each portfolio.
    pub average_volatilities: Vec<f64>,
    /// Average Sharpe ratio for each portfolio.
    pub average_sharpe_ratios: Vec<f64>,

    // Last Simulation Data
    /// The return scenarios sampled during the *last* simulation run (needed for memetic search).
    pub last_scenario_returns: Vec<Vec<f64>>,

    // Population-Wide Summary Statistics (calculated from the per-portfolio averages)
    /// Best (highest) average return found in the population for this evaluation.
    pub best_return: f64,
    /// Average of the average returns across the entire population.
    pub population_average_return: f64,
    /// Best (lowest) average volatility found in the population.
    pub best_volatility: f64,
    /// Average of the average volatilities across the entire population.
    pub population_average_volatility: f64,
    /// Best (highest) average Sharpe ratio found in the population.
    pub best_sharpe: f64,
    /// Average of the average Sharpe ratios across the entire population.
    pub population_average_sharpe: f64,
}

/// Evaluates the performance of a given population of portfolios over multiple simulations.
///
/// Calculates per-portfolio average metrics and population-wide summary statistics.
///
/// # Arguments
/// * `population`: A slice of weight vectors representing the portfolios to evaluate.
/// * `config`: The base configuration containing simulation parameters and the sampler.
///
/// # Returns
/// A `PopulationEvaluationResult` struct. Returns default/empty values if the input population is empty.
fn evaluate_population_performance_local(
    config: &StandardEvolutionConfig,
    population: &[Vec<f64>],
) -> PopulationEvaluationResult {
    let population_size = population.len();
    if population_size == 0 {
        return PopulationEvaluationResult {
            average_returns: vec![],
            average_volatilities: vec![],
            average_sharpe_ratios: vec![],
            last_scenario_returns: vec![],
            best_return: f64::NEG_INFINITY,
            population_average_return: 0.0,
            best_volatility: f64::INFINITY,
            population_average_volatility: 0.0,
            best_sharpe: f64::NEG_INFINITY,
            population_average_sharpe: 0.0,
        };
    }

    let simulations_per_generation = config.simulations_per_generation;

    // Initialize accumulators
    let mut accumulated_returns = vec![0.0; population_size];
    let mut accumulated_volatilities = vec![0.0; population_size];
    let mut accumulated_sharpe_ratios = vec![0.0; population_size];
    let mut last_scenario_returns: Vec<Vec<f64>> = vec![];
    let mut cfg_clone = config.clone();

    // --- Simulation Loop ---
    for i in 0..simulations_per_generation {
        let scenario_returns = cfg_clone.sampler.sample_returns();
        if i == simulations_per_generation - 1 {
            last_scenario_returns = scenario_returns.clone();
        }

        let performance_metrics_in_scenario: Vec<(f64, f64, f64)> = population
            .par_iter()
            .map(|portfolio_weights| {
                let performance = compute_portfolio_performance(
                    &scenario_returns,
                    portfolio_weights,
                    config.money_to_invest,
                    config.risk_free_rate,
                    config.time_horizon_in_days as f64,
                );
                (
                    performance.annualized_return,
                    performance.percent_annualized_volatility,
                    performance.sharpe_ratio,
                )
            })
            .collect();

        for (idx, (ret, vol, sharpe)) in performance_metrics_in_scenario.iter().enumerate() {
            accumulated_returns[idx] += ret;
            accumulated_volatilities[idx] += vol;
            accumulated_sharpe_ratios[idx] += sharpe;
        }
    } // --- End of Simulation Loop ---

    // --- Calculate Per-Portfolio Averages ---
    let sim_count_f64 = simulations_per_generation as f64;
    // These vectors hold the average performance for each *individual* portfolio
    let average_returns: Vec<f64> = accumulated_returns
        .par_iter()
        .map(|&sum| sum / sim_count_f64)
        .collect();
    let average_volatilities: Vec<f64> = accumulated_volatilities
        .par_iter()
        .map(|&sum| sum / sim_count_f64)
        .collect();
    let average_sharpe_ratios: Vec<f64> = accumulated_sharpe_ratios
        .par_iter()
        .map(|&sum| sum / sim_count_f64)
        .collect();

    // --- Calculate Population-Wide Summary Statistics ---
    let population_size_f64 = population_size as f64;

    // Use the calculated per-portfolio averages to get population stats
    let best_return = average_returns
        .par_iter()
        .fold(|| f64::NEG_INFINITY, |a, &b| a.max(b))
        .reduce(|| f64::NEG_INFINITY, |a, b| a.max(b));
    let population_average_return = average_returns.par_iter().sum::<f64>() / population_size_f64;

    let best_volatility = average_volatilities
        .par_iter()
        .fold(|| f64::INFINITY, |a, &b| a.min(b))
        .reduce(|| f64::INFINITY, |a, b| a.min(b));
    let population_average_volatility =
        average_volatilities.par_iter().sum::<f64>() / population_size_f64;

    let best_sharpe = average_sharpe_ratios
        .par_iter()
        .fold(|| f64::NEG_INFINITY, |a, &b| a.max(b))
        .reduce(|| f64::NEG_INFINITY, |a, b| a.max(b));
    let population_average_sharpe =
        average_sharpe_ratios.par_iter().sum::<f64>() / population_size_f64;

    // --- Return Results ---
    PopulationEvaluationResult {
        average_returns, // Per-portfolio averages
        average_volatilities,
        average_sharpe_ratios,
        last_scenario_returns, // Data from last sim run
        best_return,           // Population summary stats
        population_average_return,
        best_volatility,
        population_average_volatility,
        best_sharpe,
        population_average_sharpe,
    }
}
