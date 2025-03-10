// IMPORTS
use core::f64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::portfolio::portoflio::Portfolio;
use crate::sampling::sampling::Sampler;
use itertools::izip;
use rand::distributions::Uniform;
use rand::prelude::*;
use statrs::distribution::MultivariateNormal;

// Modules
mod portfolio;
mod python_bindings;
mod sampling;

// Actual Code
static SUPERVISOR_INPUT_DIM: usize = 4; //Trained at that dimension so I can't take more or less than that.
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
    tournament_size: usize,
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

        let mut portfolio_structs: Vec<Portfolio> = turn_weights_into_portfolios(
            population.clone(),
            simulation_average_returns.clone(),
            simulation_average_volatilities.clone(),
            simulation_average_sharpe_ratios.clone(),
        );

        let mut fronts = non_dominated_sort(&mut portfolio_structs);
        let mut next_generation: Vec<Vec<f64>> = Vec::new();

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
                next_generation.push(portfolio.weights.clone());
            }

            if population.len() >= elite_population_size {
                break;
            }
        }

        let offsprings = generate_offsprings(
            &fronts.into_iter().flatten().collect::<Vec<Portfolio>>(),
            offspring_count,
            config.mutation_rate,
            config.tournament_size,
        );
        next_generation.extend(offsprings);
        population = next_generation;
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
    population: &[Portfolio],
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

fn select_parents(population: &[Portfolio], k: usize) -> (Vec<f64>, Vec<f64>) {
    let parent_1 = tournament_selection(population, 2);
    let parent_2 = tournament_selection(population, 2);

    (parent_1.to_owned(), parent_2.to_owned())
}

fn tournament_selection(population: &[Portfolio], k: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let mut k_portfolios: Vec<Portfolio> =
        population.choose_multiple(&mut rng, k).cloned().collect();
    k_portfolios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
    k_portfolios[0].weights.clone()
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
