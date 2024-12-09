use core::f64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

use itertools::{izip, Itertools};
use nalgebra::DVector;
use rand::distributions::Uniform;
use rand::prelude::*;
use statrs::distribution::MultivariateNormal;
use statrs::statistics::MeanN;
use std::sync::atomic::AtomicUsize;
use std::sync::{atomic::Ordering, LazyLock};
use tch::{CModule, Kind, TchError, Tensor};

static SUPERVISOR: LazyLock<Result<CModule, TchError>> =
    LazyLock::new(|| CModule::load("../model_weights/supervisor.pt"));
//static SUPERVISOR_INPUT_DIM = 4;//Trained at that dimension so I can't take more or less than that.
const NUMBER_OF_OPTIMIZATION_OBJECTIVES: usize = 3;
const PERTURBATION: f64 = 0.01;

const LOG_RETURNS_MEANS: (f64, f64, f64, f64) = (
    5.32273662e-04,
    6.63425548e-05,
    8.77944050e-05,
    6.45186507e-05,
);
const LOG_RETURNS_COV: [f64; 4 * 4] = [
    1.21196105e-04,
    1.13364388e-06,
    -2.25083039e-05,
    1.18820847e-04,
    1.13364388e-06,
    2.24497315e-06,
    5.60201102e-06,
    1.27454505e-06,
    -2.25083039e-05,
    5.60201102e-06,
    6.78149858e-05,
    -2.89021031e-05,
    1.18820847e-04,
    1.27454505e-06,
    -2.89021031e-05,
    2.58077479e-04,
];

const UNSCALED_MEANS: (f64, f64, f64, f64) = (121.25030248, 61.86589531, 73.73191426, 36.73631967);
const UNSCALED_COV: [f64; 4 * 4] = [
    4.66704616e+01,
    8.73746363e-01,
    1.03911088e+00,
    6.21559427e+00,
    8.73746363e-01,
    1.57423688e-01,
    5.28491535e-01,
    4.24114881e-02,
    1.03911088e+00,
    5.28491535e-01,
    6.72981173e+00,
    -1.23283122e+00,
    6.21559427e+00,
    4.24114881e-02,
    -1.23283122e+00,
    9.15228952e+00,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Portfolio {
    id: usize,
    rank: Option<usize>,
    crowding_distance: Option<f64>,
    weights: Vec<f64>,
    average_returns: f64,
    volatility: f64,
    sharpe_ratio: f64,
}

impl PartialEq for Portfolio {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl PartialOrd for Portfolio {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // if ID is the same return equal
        if self.id == other.id {
            return Some(std::cmp::Ordering::Equal);
        }

        // Compare based on rank
        if self.rank != other.rank {
            return self.rank.partial_cmp(&other.rank);
        }
        // If Rank is the same compare based on Crowding_Distance
        match (self.crowding_distance, other.crowding_distance) {
            (Some(self_distance), Some(other_distance)) => {
                self_distance.partial_cmp(&other_distance)
            }
            _ => panic!("Crowding distance is None for one or both portfolios."), //shouldn't happen after processing
        }
    }
}
impl Eq for Portfolio {}
static PORTFOLIO_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Portfolio {
    fn new(weights: Vec<f64>, average_returns: f64, volatility: f64, sharpe_ratio: f64) -> Self {
        let id = PORTFOLIO_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        let rank = None;
        let crowding_distance = None;
        Portfolio {
            id,
            rank,
            crowding_distance,
            weights,
            average_returns,
            volatility,
            sharpe_ratio,
        }
    }

    fn to_metrics_vector(&self) -> Vec<f64> {
        // We negate the self.volatility to make maximization the global goal
        vec![self.average_returns, -self.volatility, self.sharpe_ratio]
    }

    fn is_dominated_by(&self, other: &Portfolio) -> bool {
        let self_metrics = self.to_metrics_vector();
        let other_metrics = other.to_metrics_vector();
        // Dominating implies being better in one area, while not being worse in any other compared
        // to other. Logically, if we are better than other in at least one we can't be dominated.
        let better_in_at_least_one = self_metrics
            .iter()
            .zip(other_metrics.iter())
            .any(|(&self_metric, &other_metric)| self_metric > other_metric);

        // We check that other_metric is dominating (better than) in at least one category, if that is the case, self cannot
        // be in the Pareto front and is therefore dominated
        let dominated = self_metrics
            .iter()
            .zip(other_metrics.iter())
            .any(|(&self_metric, &other_metric)| self_metric < other_metric);

        // If better_in_at_least_one is true, we complement it since we want to return whether we're dominated
        // If we know we are worse in one metric, we know we are dominated
        // Both must be true for self to be part of the Pareto Front
        !better_in_at_least_one && dominated
    }
}
// Sampler Code
#[derive(Debug)]
enum Sampler {
    Normal {
        normal_distribution: MultivariateNormal,
        periods_to_sample: usize,
    },
    SupervisedNormal {
        normal_distribution: MultivariateNormal,
        periods_to_sample: usize,
        look_ahead: usize,
    }, //uses the supervisor to supervise a normal so that hopefully it has the required temporal characteristics
    SeriesGAN(usize),
}

impl Sampler {
    /// Honestly, only the superverised normal needs the price scenarios
    /// and for theoretical reasons this is gibberish, so it will be revamped later.
    /// But for now we leave it as this.
    fn sample_price_scenario(&self) -> Vec<DVector<f64>> {
        let mut rng = thread_rng();
        match self {
            Sampler::Normal {
                normal_distribution,
                periods_to_sample,
            } => vec![DVector::from_vec(vec![1.])], // SHOULDN'T BE CALLED TRUTHFULLY,
            Sampler::SupervisedNormal {
                normal_distribution,
                periods_to_sample,
                look_ahead,
            } => {
                // ARGUABLY USELESS, THIS ENTIRE SECTION NEEDS REVAMPING.
                let raw_normal_sequence = normal_distribution
                    .sample_iter(&mut rng)
                    .take(*periods_to_sample + look_ahead)
                    .collect::<Vec<_>>();
                Sampler::supervise_sequence(raw_normal_sequence, *look_ahead) //returns supervised_sequence
            }
            Sampler::SeriesGAN(periods_to_sample) => {
                // NOT IMPLEMENTED (WILL WRITE IT SO THAT THERE"S A MODEL THAT WAS TRAINED TO GENERATE THESE, NOT A PRIORITY)
                vec![DVector::from_vec(vec![1.])]
            }
        }
    }
    /// sample_returns
    /// Takes method of Sampler object
    ///
    /// Returns a vector of vectors of f64.
    ///
    /// The goal is to sample returns according to different modalities.
    fn sample_returns(&self) -> Vec<Vec<f64>> {
        let mut rng = thread_rng();
        match self {
            Sampler::Normal {
                normal_distribution,
                periods_to_sample,
            } => normal_distribution
                .sample_iter(&mut rng)
                .take(*periods_to_sample)
                .map(|row| row.iter().cloned().collect())
                .collect::<Vec<_>>(),
            Sampler::SupervisedNormal {
                normal_distribution,
                periods_to_sample,
                look_ahead,
            } => {
                let scenario = self.sample_price_scenario();

                // Compute Returns
                (0..scenario.len() - 1)
                    .into_par_iter()
                    .map(|t| {
                        let current_row = &scenario[t];
                        let next_row = &scenario[t + 1];

                        current_row
                            .iter()
                            .zip(next_row.iter())
                            .map(|(current, next)| (next - current) / current)
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            }
            Sampler::SeriesGAN(usize) => {
                // THE MOST IMPORTANT ONE, WHEN I AM DONE IMPLEMENTING THIS HOPEFULLY GENERATING GOOD PORTFOLIOS WILL BE EASIER
                vec![vec![1.]] //FOR NOW RETURNS UNIT
            }
        }
    }
    // SUPERVISOR FUNCTIONS
    fn find_min_max(raw_sequence: &[DVector<f64>]) -> Result<(Vec<(f64, f64)>, usize), String> {
        if raw_sequence.is_empty() {
            return Err("Passed an empty sequence to supervisor".to_string());
        }
        let dimension = raw_sequence[0].len(); // access first row and then check the number of elements
                                               // must be in this order so that any value is less than INFINITY, and any value is bigger than NEG_INFINITY
        let mut min_max = vec![(f64::INFINITY, f64::NEG_INFINITY); dimension];

        for row in raw_sequence.iter() {
            for (col_idx, &value) in row.iter().enumerate() {
                let (min, max) = &mut min_max[col_idx];
                *min = (*min).min(value);
                *max = (*max).max(value);
            }
        }
        Ok((min_max, dimension))
    }
    fn supervisor_pass(
        preprocessed_sequence: Vec<DVector<f64>>,
        look_ahead: usize,
        number_of_assets: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        if preprocessed_sequence.len() < look_ahead {
            return Err(
                "A sequence less than LOOK_BACK was passed to the supervisor pass.".to_string(),
            );
        }
        // Prepare Slices for Training (reversed to match GRUs expectation)
        let (inputs, input_lengths): (Vec<&[DVector<f64>]>, Vec<usize>) = (1
            ..(preprocessed_sequence.len() + 1 - look_ahead))
            .map(|i| (&preprocessed_sequence[..i], i))
            .rev()
            .unzip();

        let extract_from_vec_storage =
            |vec_storage: nalgebra::DVector<f64>| vec_storage.data.as_vec().clone();

        let vectorized_inputs = inputs
            .iter()
            .map(|input_sequence| {
                input_sequence
                    .iter()
                    .map(|stuff_inside| extract_from_vec_storage(stuff_inside.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let max_length = input_lengths[0];

        let tensor_vector = vectorized_inputs
            .iter()
            .zip(input_lengths.iter())
            .map(|(input_sequence, &length)| {
                Tensor::from_slice(
                    &input_sequence
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<f64>>(),
                )
                .reshape([length as i64, number_of_assets as i64])
                .pad([0, 0, 0, (max_length - length) as i64], "constant", 0.0)
            })
            .collect::<Vec<_>>();

        let stacked_tensor = Tensor::stack(&tensor_vector, 0);
        let lengths_tensor = Tensor::from_slice(
            &input_lengths
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>(),
        );
        let supervisor_input = vec![
            stacked_tensor.to_kind(Kind::Float),
            lengths_tensor.to_kind(Kind::Int64),
        ];
        // Note to me of tomorrow: Right now wrote the code for stacking and padding logic, review to make sure it makes sense.
        // Then pass the thing to supervisor, stack the outputs into one vector and return that.
        let outputs = SUPERVISOR
            .as_ref()
            .expect("The supervisor model to be used")
            .forward_ts(&supervisor_input);

        Ok(Vec::<Vec<f64>>::try_from(outputs.expect("Sequence"))
            .expect("The OK variant of my Converted Supervised Sequence: "))
    }
    fn supervise_sequence(raw_sequence: Vec<DVector<f64>>, look_ahead: usize) -> Vec<DVector<f64>> {
        let (min_max_columns, number_of_assets) = Sampler::find_min_max(&raw_sequence)
            .expect("A list of tuples containing the Min-Max of each column");

        // // Define helper functions
        let min_max_scaling = |value: f64, min: f64, max: f64| (value - min) / (max - min);
        let undo_min_max_scaling =
            |scaled_value: f64, min: f64, max: f64| scaled_value * (max - min) + min;

        // Scaling is necessary since supervisor was trained on scaled sequences
        let min_max_scaled_sequence: Vec<_> = raw_sequence
            .iter()
            .map(|row| {
                // min-max scale all the entries
                let scaled_row = row
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        // presumes min_max_columns contains tuples s.t (min, max)
                        min_max_scaling(value, min_max_columns[i].0, min_max_columns[i].1)
                    })
                    .collect();
                DVector::from_vec(scaled_row)
            })
            .collect();

        // Do Supervisor Pass to Supervise (predict 2 bits ahead until done)
        let supervised_sequence =
            Sampler::supervisor_pass(min_max_scaled_sequence, look_ahead, number_of_assets)
                .expect("The supervised sequence.");

        // Undo Scaling
        let supervised_restored_sequence = supervised_sequence
            .iter()
            .map(|scaled_row| {
                let unscaled_row = scaled_row
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        undo_min_max_scaling(value, min_max_columns[i].0, min_max_columns[i].1)
                    })
                    .collect();
                DVector::from_vec(unscaled_row)
            })
            .collect();

        supervised_restored_sequence
    }
}

fn find_pareto_front(portfolios: &[Portfolio]) -> Vec<Portfolio> {
    // Find all the dominated portfolios within batch
    portfolios
        .par_iter()
        .enumerate()
        .filter(|(i, portfolio_a)| {
            // A portfolio is non-dominated iff no other portfolio dominates it
            !portfolios
                .iter()
                .enumerate()
                .any(|(j, portfolio_b)| *i != j && portfolio_a.is_dominated_by(portfolio_b))
        })
        .map(|(_, portfolio)| portfolio.clone())
        .collect()
}

fn calculate_and_update_crowding_distance(pareto_front: &mut Vec<Portfolio>) {
    // Helper function to help compute the crowding distance
    let get_objective_value = |portfolio: &Portfolio, objective_idx: usize| match objective_idx {
        0 => portfolio.average_returns,
        1 => portfolio.volatility,
        2 => portfolio.sharpe_ratio,
        _ => panic!(
            "You changed the number of objective without 
            \nupdating the matching logic. Lel.
            Reached a part of the code that shouldn't be reached"
        ),
    };

    // Initialize the distances
    pareto_front
        .iter_mut()
        .for_each(|portfolio| portfolio.crowding_distance = Some(0.0));

    for objective_idx in 0..NUMBER_OF_OPTIMIZATION_OBJECTIVES {
        // sort with respect to current objective
        pareto_front.sort_by(|portfolio_a, portfolio_b| {
            let portfolio_a_objective_value = get_objective_value(portfolio_a, objective_idx);
            let portfolio_b_objective_value = get_objective_value(portfolio_b, objective_idx);

            portfolio_a_objective_value
                .partial_cmp(&portfolio_b_objective_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep the best and worst (boundaries) by assigning infinite crowding distance
        let last_index = pareto_front.len() - 1;
        pareto_front[0].crowding_distance = Some(f64::INFINITY);
        pareto_front[last_index].crowding_distance = Some(f64::INFINITY);

        let min_value = get_objective_value(&pareto_front[0], objective_idx);
        let max_value = get_objective_value(&pareto_front[last_index], objective_idx);
        let range = max_value - min_value;

        if range > 0.0 {
            for i in 1..pareto_front.len() - 1 {
                let previous = get_objective_value(&pareto_front[i - 1], objective_idx);
                let next = get_objective_value(&pareto_front[i + 1], objective_idx);
                let current_distance = pareto_front[i].crowding_distance.unwrap_or(0.0);
                pareto_front[i].crowding_distance =
                    Some(current_distance + (next - previous) / range);
            }
        }
    }
}

fn non_dominated_sort(portfolios: &mut [Portfolio]) -> Vec<Vec<Portfolio>> {
    let mut fronts: Vec<Vec<Portfolio>> = Vec::new();
    let mut remaining_portfolios = portfolios.to_vec();

    let mut current_front = 1;
    while !remaining_portfolios.is_empty() {
        // Find the Pareto front & Update Portfolio
        let mut pareto_front = find_pareto_front(&remaining_portfolios);
        pareto_front.iter_mut().for_each(|portfolio| {
            portfolio.rank = Some(current_front);
        });

        // Modifies the crowding distances in-place
        calculate_and_update_crowding_distance(&mut pareto_front);

        // Then add it to the fronts list
        fronts.push(pareto_front.clone());

        // Remove the portfolios in the current front
        let pareto_ids: std::collections::HashSet<_> =
            pareto_front.iter().map(|portfolio| portfolio.id).collect();

        // Remaining Portfolios
        remaining_portfolios.retain(|portfolio| !pareto_ids.contains(&portfolio.id));
        current_front += 1;
    }

    fronts
}
struct EvolutionConfig {
    time_horizon_in_days: usize,
    generations: usize,
    population_size: usize,
    simulations_per_generation: usize,
    assets_under_management: usize,
    money_to_invest: f64,
    risk_free_rate: f64,
    elitism_rate: f64,
    mutation_rate: f64,
    sampler: Sampler,
    generation_check_interval: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct EvolutionResult {
    pareto_fronts: Vec<Vec<Portfolio>>,
    best_average_return_per_generation: Vec<f64>,
    average_return_per_generation: Vec<f64>,
    best_average_volatility_per_generation: Vec<f64>,
    average_volatility_per_generation: Vec<f64>,
    best_average_sharpe_ratio_per_generation: Vec<f64>,
    average_sharpe_ratio_per_generation: Vec<f64>,
}

// Algo Logic
// 1. Generate a bunch of portfolio for the first batch based on a parameter passed (initial training size)
// 2. Sample based on the provided length using whichever method was specified by the user
// 3. Run the simulation computing every day the daily returns, and whatnot. <- Target for multithreading if possible
// 4. At the end use a Pareto front optimization strategy to select from a sample of non-dominated solutions and let them explore
// 4.i To an extent we'd also want the ability to explore so we'd have a randomly chosen percentage between 0 and 15 percent of dominated
// 4.ii solutions that would be allowed to reproduce using crossover.
// 4.iii We'd also mutate the allocation stochastically in such away tha randomly an increment is done to one component (and equivalent decrement is done to another)
// 5. Repeat until golden brown. lel
//
// This will be made into a python endpoint to write the code in Python simpler.
fn evolve_portfolios(config: EvolutionConfig) -> EvolutionResult {
    // Initialization Phase
    //
    // Common Enough to Alias
    let population_size = config.population_size;
    let generations = config.generations;
    let simulations_per_generation = config.simulations_per_generation;

    let rng = thread_rng();
    let uniform = Uniform::new(0., 1.);
    let elite_population_size = ((population_size as f64) * config.elitism_rate) as usize;
    let offspring_count = population_size - elite_population_size;

    // For each portfolio we sample from a Uniform and then normalize
    let mut population: Vec<Vec<f64>> = (0..config.population_size)
        .map(|_| {
            let mut portfolio = rng
                .clone()
                .sample_iter(uniform)
                .take(config.assets_under_management)
                .collect::<Vec<f64>>();
            let magnitude = portfolio.iter().sum::<f64>();

            portfolio.iter_mut().for_each(|x| *x /= magnitude);
            portfolio
        })
        .collect::<Vec<_>>();

    // Put here so that we can pass it to the final step
    let mut simulation_average_returns: Vec<f64> = vec![0.; population_size];
    let mut simulation_average_volatilities: Vec<f64> = vec![0.; population_size];
    let mut simulation_average_sharpe_ratios: Vec<f64> = vec![0.; population_size];
    // Metrics Vectors
    let mut best_average_return_per_generation: Vec<f64> = vec![0.; generations];
    let mut average_return_per_generation: Vec<f64> = vec![0.; generations];

    let mut best_average_volatility_per_generation: Vec<f64> = vec![0.; generations];
    let mut average_volatility_per_generation: Vec<f64> = vec![0.; generations];

    let mut best_average_sharpe_ratio_per_generation: Vec<f64> = vec![0.; generations];
    let mut average_sharpe_ratio_per_generation: Vec<f64> = vec![0.; generations];

    let turn_weights_into_portfolios =
        |population: Vec<Vec<f64>>,
         simulation_average_returns: Vec<f64>,
         simulation_average_volatilities: Vec<f64>,
         simulation_average_sharpe_ratios: Vec<f64>| {
            let portfolio_simulation_averages: Vec<(Vec<f64>, f64, f64, f64)> = izip!(
                population,
                simulation_average_returns,
                simulation_average_volatilities,
                simulation_average_sharpe_ratios
            )
            .collect();
            portfolio_simulation_averages
                .par_iter()
                .map(|(portfolio, ave_ret, ave_vol, ave_sharpe)| {
                    Portfolio::new(portfolio.to_vec(), *ave_ret, *ave_vol, *ave_sharpe)
                })
                .collect()
        };

    // EVOLUTION BABY!!!
    for generation in 0..generations {
        // Reset Arrays for Generation
        simulation_average_returns.fill(0.0);
        simulation_average_volatilities.fill(0.0);
        simulation_average_sharpe_ratios.fill(0.0);

        // Run the simulations and collect the results
        for _ in 0..simulations_per_generation {
            // Sample the data for this simulation
            let scenario_returns = config.sampler.sample_returns();
            let portfolios_performance_metrics = population
                .par_iter()
                .map(|portfolio| {
                    compute_portfolio_performance(
                        scenario_returns.clone(),
                        portfolio.clone(),
                        config.money_to_invest,
                        config.risk_free_rate,
                        config.time_horizon_in_days as f64,
                    )
                })
                .collect::<Vec<(Vec<f64>, f64, f64, f64)>>();

            // Run the Cumulative sum of all the metrics over the number of simulations
            portfolios_performance_metrics.iter().enumerate().for_each(
                |(i, &(_, average_return, volatility, sharpe_ratio))| {
                    simulation_average_returns[i] += average_return;
                    simulation_average_volatilities[i] += volatility;
                    simulation_average_sharpe_ratios[i] += sharpe_ratio;
                },
            );
        }

        // Divide everything by simulation number to get the average
        simulation_average_returns
            .par_iter_mut()
            .for_each(|ret| *ret /= simulations_per_generation as f64);
        simulation_average_volatilities
            .par_iter_mut()
            .for_each(|volatility| *volatility /= simulations_per_generation as f64);
        simulation_average_sharpe_ratios
            .par_iter_mut()
            .for_each(|sharpe_ratio| *sharpe_ratio /= simulations_per_generation as f64);

        // PREPARING & REPORTING METRICS FOR GENERATION
        // Bests
        best_average_return_per_generation[generation] = simulation_average_returns // Higher is better
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        best_average_volatility_per_generation[generation] = simulation_average_volatilities // Lower is better
            .iter()
            .fold(f64::INFINITY, |a, b| a.min(*b));
        best_average_sharpe_ratio_per_generation[generation] = simulation_average_sharpe_ratios // Higher is better
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b));

        // Averages
        average_return_per_generation[generation] =
            simulation_average_returns.iter().sum::<f64>() / (population_size as f64);
        average_volatility_per_generation[generation] =
            simulation_average_volatilities.iter().sum::<f64>() / (population_size as f64);
        average_sharpe_ratio_per_generation[generation] =
            simulation_average_sharpe_ratios.iter().sum::<f64>() / (population_size as f64);

        if (generation % config.generation_check_interval == 0) || (generation == generations - 1) {
            println!(
    "Generation {}: Best Return: {:.4}, Avg Return: {:.4}, Best Sharpe: {:.4}, Avg Sharpe: {:.4}, Best Volatility: {:.4}, Avg Volatility: {:.4}",
    generation,
    best_average_return_per_generation[generation],
    average_return_per_generation[generation],
    best_average_sharpe_ratio_per_generation[generation],
    average_sharpe_ratio_per_generation[generation],
    best_average_volatility_per_generation[generation],
    average_volatility_per_generation[generation],
);
        }
        // NEXT GENERATION CREATION LOGIC
        // Initialize the Structs and then find Pareto Front

        let mut portfolio_structs: Vec<Portfolio> = turn_weights_into_portfolios(
            population.clone(),
            simulation_average_returns.clone(),
            simulation_average_volatilities.clone(),
            simulation_average_sharpe_ratios.clone(),
        );

        let mut fronts = non_dominated_sort(&mut portfolio_structs);

        // Prepare the population vector for next generation
        population.clear(); // reset the vector

        // Adding Elites (Exploitation)
        for front in fronts.iter_mut() {
            // Sort array before proceeding!
            front.sort_by(|portfolio_a, portfolio_b| {
                portfolio_a
                    .partial_cmp(portfolio_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for portfolio in front.iter() {
                if population.len() >= elite_population_size {
                    break;
                }
                population.push(portfolio.weights.clone());
            }

            if population.len() >= elite_population_size {
                break;
            }
        }

        let offsprings = generate_offsprings(&population, offspring_count, config.mutation_rate);
        population.extend(offsprings);
    }
    //
    EvolutionResult {
        pareto_fronts: non_dominated_sort(&mut turn_weights_into_portfolios(
            population,
            simulation_average_returns,
            simulation_average_volatilities,
            simulation_average_sharpe_ratios,
        )), // Includes both best and their offsprings
        best_average_return_per_generation,
        average_return_per_generation,
        best_average_volatility_per_generation,
        average_volatility_per_generation,
        best_average_sharpe_ratio_per_generation,
        average_sharpe_ratio_per_generation,
    }
}

fn generate_offsprings(
    population: &[Vec<f64>],
    offspring_count: usize,
    mutation_rate: f64,
) -> Vec<Vec<f64>> {
    let mut offsprings = Vec::new();

    while offsprings.len() < offspring_count {
        let (parent_1, parent_2) = select_parents(population);

        let mut child_weights = crossover(&parent_1, &parent_2);

        // Toss a coin and stochastically mutate weights based on rate
        mutate(&mut child_weights, mutation_rate);

        offsprings.push(child_weights);
    }
    offsprings
}

fn select_parents(population: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();
    let parent_1 = &population[rng.gen_range(0..population.len())];
    let parent_2 = &population[rng.gen_range(0..population.len())];

    (parent_1.to_owned(), parent_2.to_owned())
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

fn compute_portfolio_performance(
    returns: Vec<Vec<f64>>,
    weights: Vec<f64>,
    money_to_invest: f64,
    risk_free_rate: f64,
    time_horizon_in_days: f64, // these are days
) -> (Vec<f64>, f64, f64, f64) {
    // Returns per row
    let portfolio_returns = returns
        .par_iter()
        .map(|row| {
            row.into_par_iter()
                .zip(weights.par_iter())
                .map(|(log_return, weight)| ((log_return.exp() - 1.0) * weight) * money_to_invest)
                .sum::<f64>()
        })
        .collect::<Vec<f64>>();

    let average_return = portfolio_returns.iter().sum::<f64>() / (portfolio_returns.len() as f64);
    let volatility = (portfolio_returns
        .iter()
        .map(|ret| (ret - average_return).powi(2))
        .sum::<f64>()
        / (portfolio_returns.len() as f64 - 1.0))
        .sqrt();

    // Annualizing!
    let number_of_periods = portfolio_returns.len() as f64;
    let time_horizon_in_years = time_horizon_in_days / 365.0;
    let periods_per_year = number_of_periods / time_horizon_in_years;

    let annualized_return = average_return * periods_per_year;
    let annualized_volatility = volatility * periods_per_year.sqrt();
    let percent_annualized_volatility = annualized_volatility / money_to_invest;

    // Adjust risk-free rate to the provided time horizon
    let risk_free_return = money_to_invest * risk_free_rate;

    let sharpe_ratio = if volatility != 0.0 {
        (annualized_return - risk_free_return) / annualized_volatility
    } else if annualized_return > risk_free_return {
        // Portfolio offers a guaranteed return above the risk-free rate
        f64::MAX // Or a large constant like 1e6
    } else if annualized_return < risk_free_return {
        // Portfolio offers a guaranteed return below the risk-free rate
        f64::MIN // Or a large negative constant like -1e6
    } else {
        // Portfolio return equals the risk-free rate
        0.0
    };
    (
        portfolio_returns,
        annualized_return,
        percent_annualized_volatility,
        sharpe_ratio,
    )
}

// Algo Code
fn main() {
    let monthly_scaler = 21;
    let mvn = MultivariateNormal::new(
        vec![
            (monthly_scaler as f64) * LOG_RETURNS_MEANS.0,
            (monthly_scaler as f64) * LOG_RETURNS_MEANS.1,
            (monthly_scaler as f64) * LOG_RETURNS_MEANS.2,
            (monthly_scaler as f64) * LOG_RETURNS_MEANS.3,
        ],
        LOG_RETURNS_COV
            .into_iter()
            .map(|cov_component| (monthly_scaler as f64) * cov_component)
            .collect(),
    )
    .expect("Wanted a multivariate normal");

    let repeats = 10;
    let time_horizons = vec![100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300];
    for time in time_horizons {
        let mean = (0..repeats)
            .map(|_| {
                let normal_sampler = Sampler::Normal {
                    normal_distribution: mvn.clone(),
                    periods_to_sample: 15,
                };
                let start = std::time::Instant::now();
                evolve_portfolios(EvolutionConfig {
                    time_horizon_in_days: time,
                    generations: 100,
                    population_size: 100,
                    simulations_per_generation: 10_000,
                    assets_under_management: 4,
                    money_to_invest: 1_000_000.,
                    risk_free_rate: 0.02,
                    elitism_rate: 0.05,
                    mutation_rate: 0.1,
                    sampler: normal_sampler,
                    generation_check_interval: 10,
                });
                start.elapsed()
            })
            .collect::<Vec<_>>()
            .iter()
            .sum::<std::time::Duration>()
            / repeats as u32;
        println!("Mean Time taken: {:?}", mean);
        println!("For one generation for a population size of 100 and doing 10,000 simulations per portfolio.\n When sampling for {time} steps\n
        on a AMD Ryzen 5 5500U with Radeon Graphics");
    }
}
