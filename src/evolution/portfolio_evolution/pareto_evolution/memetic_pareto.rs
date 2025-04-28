use std::io;

use aegis_athena_contracts::{
    common_portfolio_evolution_ds::compute_portfolio_performance, portfolio::Portfolio,
};
use rayon::prelude::*;
use tracing::warn;

use crate::evolution::portfolio_evolution::memetic::lamarckian_proximal_descent;
use crate::evolution::portfolio_evolution::pareto_evolution::build_pareto_fronts;
use crate::evolution::portfolio_evolution::{
    find_dominant_objective, generate_offsprings, initialize_population, make_evaluator,
    tournament_selection, turn_weights_into_portfolios, EvolutionError, EvolutionResult,
    EvolutionStrategy, FinalPopulationSummary, MemeticEvolutionConfig, Objective,
    PortfolioPerformance,
};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

pub async fn memetic_evolve_portfolios(
    config: MemeticEvolutionConfig,
    athena_endpoint: Option<String>,
) -> Result<EvolutionResult, EvolutionError> {
    let population_size = config.base.population_size;
    let generations = config.base.generations;
    let elite_population_size =
        ((population_size as f64) * config.base.elitism_rate).round() as usize; // Use round for clarity
    let offspring_count = population_size - elite_population_size;
    let simulations_per_generation = config.base.simulations_per_generation;

    // Ensure elite size is reasonable
    if elite_population_size == 0 && config.base.elitism_rate > 0.0 {
        warn!(
            "Warning: Elite population size rounded to 0, check population size and elitism rate."
        );
    }
    if elite_population_size >= population_size {
        panic!("Elite population size cannot be >= total population size.");
    }

    let mut population: Vec<Vec<f64>> =
        initialize_population(population_size, config.base.assets_under_management)?;
    let runner = config.base.sim_runner.clone();
    let cfg = config.base.clone();
    let population_evaluator = make_evaluator(runner, cfg, athena_endpoint.clone());

    // Add new config params needed for memetic part
    let proximal_steps = config.memetic.proximal_descent_steps;
    let step_size = config.memetic.proximal_descent_step_size;
    let high_sharpe = config.memetic.high_sharpe_threshold;
    let low_vol = config.memetic.low_volatility_threshold;

    // Put here so that we can pass it to the final step
    let simulation_average_returns: Vec<f64> = vec![0.; population_size];
    let simulation_average_volatilities: Vec<f64> = vec![0.; population_size];
    let simulation_average_sharpe_ratios: Vec<f64> = vec![0.; population_size];
    // Metrics Vectors
    let mut best_average_return_per_generation: Vec<f64> = vec![0.; generations];
    let mut average_return_per_generation: Vec<f64> = vec![0.; generations];

    let mut best_average_volatility_per_generation: Vec<f64> = vec![0.; generations];
    let mut average_volatility_per_generation: Vec<f64> = vec![0.; generations];

    let mut best_average_sharpe_ratio_per_generation: Vec<f64> = vec![0.; generations];
    let mut average_sharpe_ratio_per_generation: Vec<f64> = vec![0.; generations];

    // --- Main Evolution Loop ---
    for generation in 0..generations {
        // Evaluate the current population
        let eval_result = population_evaluator(&population)
            .await
            .expect("Failed to evaluate population");

        // --- Extract results ---
        let simulation_average_returns = eval_result.average_returns; // Per-portfolio
        let simulation_average_volatilities = eval_result.average_volatilities;
        let simulation_average_sharpe_ratios = eval_result.average_sharpe_ratios;
        let last_scenario_returns = eval_result.last_scenario_returns; // For memetic step

        // --- Store generation metrics directly from eval_result ---
        best_average_return_per_generation[generation] = eval_result.best_return;
        average_return_per_generation[generation] = eval_result.population_average_return;
        best_average_volatility_per_generation[generation] = eval_result.best_volatility;
        average_volatility_per_generation[generation] = eval_result.population_average_volatility;
        best_average_sharpe_ratio_per_generation[generation] = eval_result.best_sharpe;
        average_sharpe_ratio_per_generation[generation] = eval_result.population_average_sharpe;

        // --- Create Portfolio Structs (uses per-portfolio averages) ---
        let portfolio_structs: Vec<Portfolio> = turn_weights_into_portfolios(
            &population,
            &simulation_average_returns,
            &simulation_average_volatilities,
            &simulation_average_sharpe_ratios,
        );

        // Non-dominated sort modifies ranks and crowding distances in place
        let fronts = build_pareto_fronts(portfolio_structs.as_slice());

        // --- The Memetic Part (Local Search) ---
        let mut next_generation_elites: Vec<Vec<f64>> = Vec::with_capacity(elite_population_size);
        let breeding_pool: Vec<&Portfolio> = fronts.iter().flatten().collect(); // Keep original population for breeding pool

        let mut elite_candidates: Vec<Portfolio> = Vec::new();
        for front in fronts.iter() {
            // Add entire front if it fits within elite size budget
            if elite_candidates.len() + front.len() <= elite_population_size {
                elite_candidates.extend(front.iter().cloned());
            } else {
                // Sort front by crowding distance (desc) and take the remaining needed
                let mut sorted_front = front.clone();
                sorted_front.sort_by(|a, b| {
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let remaining_slots = elite_population_size - elite_candidates.len();
                elite_candidates.extend(sorted_front.into_iter().take(remaining_slots));
                break; // Elites are full
            }
            if elite_candidates.len() >= elite_population_size {
                break;
            }
        }

        // Apply local search in parallel to the selected elite candidates
        next_generation_elites = elite_candidates
            .par_iter()
            .map(|elite_portfolio| {
                let mut current_weights = elite_portfolio.weights.clone();
                // Calculate performance based on the *averaged* metrics from this generation's simulations
                // This avoids recalculating from raw returns just for the local search starting point.
                let base_performance = PortfolioPerformance {
                    portfolio_returns: vec![], // Not needed directly for gradient calc logic as implemented
                    annualized_return: elite_portfolio.average_returns,
                    percent_annualized_volatility: elite_portfolio.volatility,
                    sharpe_ratio: elite_portfolio.sharpe_ratio,
                };
                let objective_for_descent = match config.memetic.local_objective {
                    Objective::MaximizeStrength => {
                        find_dominant_objective(&base_performance, high_sharpe, low_vol)
                    }
                    other => other,
                };
                for _ in 0..proximal_steps {
                    // Recalculate performance needed *inside* gradient computation
                    let current_performance_for_gradient = compute_portfolio_performance(
                        &last_scenario_returns, // Need the returns for gradient calculation
                        &current_weights,
                        config.base.money_to_invest,
                        config.base.risk_free_rate,
                        config.base.time_horizon_in_days as f64,
                    );

                    current_weights = lamarckian_proximal_descent(
                        &last_scenario_returns, // Pass the sampled returns
                        &current_weights,
                        current_performance_for_gradient, // Pass performance for gradient calculation
                        config.base.money_to_invest,
                        config.base.risk_free_rate,
                        config.base.time_horizon_in_days as f64,
                        step_size,
                        objective_for_descent,
                        config.memetic.high_sharpe_threshold,
                        config.memetic.low_volatility_threshold,
                    );
                }
                current_weights // Return the improved weights
            })
            .collect();

        // --- Generate Offspring (using original population before elite improvement) ---
        let offspring_count = population_size - next_generation_elites.len();
        let offspring_weights = generate_offsprings(
            &breeding_pool, // Use the full sorted population as potential parents
            offspring_count,
            config.base.mutation_rate,
            config.base.tournament_size,
        );

        // --- Combine Improved Elites and New Offspring ---
        let mut next_generation = next_generation_elites; // Start with improved elites
        next_generation.extend(offspring_weights);

        // Final check for population size consistency
        if next_generation.len() != population_size {
            warn!("Warning: Population size mismatch ({}) at end of generation {}. Should be {}. Resizing.", next_generation.len(), generation, population_size);
            // Simple resize/fill strategy, might need refinement
            next_generation.resize_with(population_size, || {
                let mut rng = thread_rng();
                let uniform = Uniform::new(0.0, 1.0);
                let mut weights = (&mut rng)
                    .sample_iter(uniform)
                    .take(config.base.assets_under_management)
                    .collect::<Vec<f64>>();
                let sum: f64 = weights.iter().sum();
                if sum > 1e-9 {
                    weights.iter_mut().for_each(|w| *w /= sum);
                } else {
                    weights.fill(1.0 / config.base.assets_under_management as f64);
                }
                weights
            });
        }

        population = next_generation; // Update population for next iteration
    } // End of generation loop

    // --- Final Evaluation After the Loop ---
    let final_eval_result = population_evaluator(&population)
        .await
        .expect("Failed to evaluate final population");

    // Create final portfolio structs using the final weights and *final* evaluation results
    let final_portfolio_structs = turn_weights_into_portfolios(
        &population,
        &final_eval_result.average_returns,
        &final_eval_result.average_volatilities,
        &final_eval_result.average_sharpe_ratios,
    );

    // Create the final summary object
    let final_summary = FinalPopulationSummary {
        best_return: final_eval_result.best_return,
        population_average_return: final_eval_result.population_average_return,
        best_volatility: final_eval_result.best_volatility,
        population_average_volatility: final_eval_result.population_average_volatility,
        best_sharpe: final_eval_result.best_sharpe,
        population_average_sharpe: final_eval_result.population_average_sharpe,
    };

    // --- Prepare and return the EvolutionResult ---
    Ok(EvolutionResult {
        pareto_fronts: build_pareto_fronts(final_portfolio_structs.as_slice()),
        // Generation history vectors were filled during the loop
        best_average_return_per_generation,
        average_return_per_generation,
        best_average_volatility_per_generation,
        average_volatility_per_generation,
        best_average_sharpe_ratio_per_generation,
        average_sharpe_ratio_per_generation,
        // Include the final summary
        final_summary,
    })
}
