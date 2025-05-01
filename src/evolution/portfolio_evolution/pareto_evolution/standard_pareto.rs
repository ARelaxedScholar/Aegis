use tracing::warn;

use crate::evolution::portfolio_evolution::{
    generate_offsprings, initialize_population, make_evaluator,
    pareto_evolution::build_pareto_fronts, turn_weights_into_portfolios, EvolutionError,
    EvolutionResult, EvolutionStrategy, FinalPopulationSummary, SimRunnerStrategy,
    StandardEvolutionConfig,
};
use aegis_athena_contracts::portfolio::Portfolio;

struct StandardParetoEvolution {}

impl EvolutionStrategy for StandardParetoEvolution {
    type Config = StandardEvolutionConfig;
    async fn evolve(
        &self,
        config: &StandardEvolutionConfig,
        athena_endpoint: Option<String>,
    ) -> Result<EvolutionResult, EvolutionError> {
        // Initialization Phase
        // Check that we received the athena endpoint if we're gonna need it.
        if !(matches!(config.sim_runner, SimRunnerStrategy::Local)) && athena_endpoint.is_none() {
            return Err(EvolutionError::MissingAthenaEndpoint);
        }
        // Check that optimization objectives are non-duplicated
        let objectives = config.objectives;
        let mut seen = Vec::with_capacity(objectives.len());

        for objective in objectives.iter() {
            let tid = objective.as_any().type_id();
            if seen.contains(tid) {
                return Err(DuplicateObjective);
            }
            seen.push(tid);
        }
        drop(seen); // we won't use this anymore, so free memory.

        // aliasing common variables
        let population_size = config.population_size;
        let generations = config.generations;
        let simulations_per_generation = config.simulations_per_generation;

        let elite_population_size = ((population_size as f64) * config.elitism_rate) as usize;
        let mut population =
            initialize_population(population_size, config.assets_under_management)?;

        // Ensure elite size is reasonable
        if elite_population_size == 0 && config.elitism_rate > 0.0 {
            warn!(
            "Warning: Elite population size rounded to 0, check population size and elitism rate."
        );
        }
        if elite_population_size >= population_size {
            return Err(EvolutionError::BadPopulationParameter(
                "Elite population size cannot be >= total population size.".into(),
            ));
        }
        let offspring_count = population_size - elite_population_size;

        // For each portfolio we sample from a Uniform and then normalize
        let mut population: Vec<Vec<f64>> = population;

        // Put here so that we can pass it to the final step
        let mut averages_matrix = Vec::<Vec<f64>>::with_capacity(generations);
        let mut bests_matrix = Vec::<Vec<f64>>::with_capacity(generations);
        // EVOLUTION BABY!!!
        for generation in 0..generations {
            eprintln!("Generation {} starting.", generation + 1);
            let eval_result: Vec<Vec<f64>> =
                config
                    .sim_runner
                    .evaluate_population(&config, &population, athena_endpoint)?;

            // --- Store generation metrics directly from eval_result ---
            let mean_stats = (0..objectives.len())
                .iter()
                .map(|(objective_idx)| {
                    eval_result
                        .averages_matrix
                        .iter()
                        .map(|portfolio_stats| portfolio_stats[objective_idx])
                        .sum::<f64>()
                        / population.len()
                })
                .collect::<Vec<f64>>();

            averages_matrix.push(mean_stats());
            bests_matrix.push(eval_result.bests);

            // --- Create Portfolio Structs (uses per-portfolio averages) ---
            let portfolio_structs: Vec<Portfolio> =
                turn_weights_into_portfolios(&population, &eval_result.averages_matrix);
            let fronts = build_pareto_fronts(portfolio_structs.as_slice());
            let breeding_pool: Vec<&Portfolio> = fronts.iter().flatten().collect();
            let mut next_generation: Vec<Vec<f64>> = Vec::new();

            // Adding Elites (Exploitation)
            for front in fronts.iter() {
                // Iterate through fronts (already ordered by rank: F1, F2, ...)
                if next_generation.len() >= elite_population_size {
                    // Stop if we already have enough elites
                    break;
                }
                if next_generation.len() + front.len() <= elite_population_size {
                    // If the entire current front fits within the remaining elite slots, add them all.
                    for portfolio in front.iter() {
                        next_generation.push(portfolio.weights.clone());
                    }
                } else {
                    // If the entire front doesn't fit, we need to take the best ones based on crowding distance.
                    let needed = elite_population_size - next_generation.len();
                    // Sort *this specific front* by crowding distance (descending).
                    // Cloning the front is necessary to sort it without affecting the original `fronts` structure.
                    let mut sorted_partial_front = front.clone();
                    sorted_partial_front.sort_by(|a, b| {
                        // Crowding distance should be Some(_) after calculate_and_update_crowding_distance runs.
                        // Use unwrap_or just in case, assigning a low value if None (shouldn't happen).
                        // Sort descending: b compared to a. Handle Inf correctly (comes first).
                        let dist_a = a.crowding_distance.unwrap_or(f64::NEG_INFINITY);
                        let dist_b = b.crowding_distance.unwrap_or(f64::NEG_INFINITY);
                        dist_b.total_cmp(&dist_a) // total_cmp is safer for floats
                    });
                    // Take the top 'needed' portfolios from the sorted front.
                    for portfolio in sorted_partial_front.iter().take(needed) {
                        next_generation.push(portfolio.weights.clone());
                    }
                    // Elites are now full, break the outer loop over fronts.
                    break;
                }
            }

            let offsprings = generate_offsprings(
                &breeding_pool,
                offspring_count,
                config.mutation_rate,
                config.tournament_size,
            );
            next_generation.extend(offsprings);
            population = next_generation;
        }

        // --- Final Evaluation After the Loop ---
        let final_eval_result = population_evaluator(&population)
            .await
            .expect("Failed to evaluate final population");

        // Create final portfolio structs using the final weights and *final* evaluation results
        let final_portfolio_structs =
            turn_weights_into_portfolios(&population, &final_eval_result.averages_matrix);

        let mean_stats = (0..objectives.len())
            .iter()
            .map(|(objective_idx)| {
                final_eval_result
                    .averages_matrix
                    .iter()
                    .map(|portfolio_stats| portfolio_stats[objective_idx])
                    .sum::<f64>()
                    / population.len()
            })
            .collect::<Vec<f64>>();

        // Create the final summary object
        let final_summary = FinalPopulationSummary {
            averages: mean_stats,
            bests: final_eval_result.bests,
        };

        // --- Prepare and return the EvolutionResult ---
        Ok(EvolutionResult {
            pareto_fronts: build_pareto_fronts(final_portfolio_structs.as_slice()),
            // Generation history vectors were filled during the loop
            averages_matrix,
            bests_matrix,
            // Include the final summary
            final_summary,
        })
        //
    }
}
