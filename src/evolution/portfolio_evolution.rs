pub mod portfolio_evolution {
    use std::cmp::Ordering;
    use std::f64::EPSILON;

    use crate::{Portfolio, Sampler, FLOAT_COMPARISON_EPSILON};
    use crate::{NUMBER_OF_OPTIMIZATION_OBJECTIVES, PERTURBATION};
    use athena_client::evaluate_population_distributed;
    use itertools::izip;
    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rayon::prelude::*;
    use serde::{Deserialize, Serialize};

<<<<<<< HEAD
=======
    fn default_max_concurrency() -> usize {
        num_cpus::get()
    }

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
    /// Finds the indices of non-dominated portfolios within a slice.
    ///
    /// A portfolio is non-dominated if no other portfolio in the slice
    /// strictly dominates it across all objectives (using the `is_dominated_by` method).
    ///
    /// # Arguments
    /// * `portfolios`: A slice of `Portfolio` structs to analyze.
    ///
    /// # Returns
    /// A `Vec<usize>` containing the indices of the non-dominated portfolios
    /// relative to the input `portfolios` slice.
    ///
    /// # Complexity
    /// Currently O(N^2) due to the nested comparison, but parallelized.
    /// Consider more efficient algorithms for large N.
    fn find_non_dominated_indices(portfolios: &[Portfolio]) -> Vec<usize> {
        // Handle empty input immediately
        if portfolios.is_empty() {
            return vec![];
        }

        portfolios
            .par_iter() // Iterate over portfolios in parallel
            .enumerate() // Get (index `i`, portfolio `&portfolio_a`)
            .filter(|(i, portfolio_a)| {
                // Check if *any* other portfolio_b dominates portfolio_a
                // The inner loop checks all other portfolios (j != i)
                let is_dominated = portfolios
                    .iter() // We use sequential iterator here since outer loop is already parallelized
                    .enumerate()
                    .any(|(j, portfolio_b)| {
                        // Ensure we don't compare a portfolio to itself
                        // and check for domination using the Portfolio method
                        *i != j && portfolio_a.is_dominated_by(portfolio_b)
                    });
                // The `filter` keeps items where `is_dominated` is false
                !is_dominated
            })
            .map(|(i, _portfolio_a)| i) // We only need the index `i` of the non-dominated portfolio
            .collect() // Collect the indices into a Vec<usize>
    }

    fn calculate_and_update_crowding_distance(pareto_front: &mut Vec<Portfolio>) {
        // Helper function to help compute the crowding distance
        let get_objective_value = |portfolio: &Portfolio, objective_idx: usize| match objective_idx
        {
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

    fn build_pareto_fronts(portfolios: &[Portfolio]) -> Vec<Vec<Portfolio>> {
        let mut fronts: Vec<Vec<Portfolio>> = Vec::new();
        // Clone the input initially to work with an owned Vec we can modify
        let mut remaining_portfolios: Vec<Portfolio> = portfolios.to_vec(); // Initial clone needed to allow `retain`

        let mut current_front_rank = 1;
        while !remaining_portfolios.is_empty() {
            // Find the *indices* of the non-dominated portfolios in the current remaining set
            let non_dominated_indices = find_non_dominated_indices(&remaining_portfolios);

            if non_dominated_indices.is_empty() {
                // Should not happen if there are remaining portfolios unless there's an issue
                // or potentially if all remaining portfolios are identical and dominate each other somehow?
                // like even for the final front, non_dominated_indices would just contain all the portfolios since they are all mutually non-dominated.
<<<<<<< HEAD
                eprintln!("Warning: Found no non-dominated portfolios among remaining {} portfolios. Breaking sort.", remaining_portfolios.len());
=======
                warn!("Warning: Found no non-dominated portfolios among remaining {} portfolios. Breaking sort.", remaining_portfolios.len());
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
                // We treat those weird portfolios as a single final front
                if !remaining_portfolios.is_empty() {
                    fronts.push(remaining_portfolios.clone()); // Add remaining as a last front
                }
                break;
            }

            // Create the Pareto front Vec<Portfolio> by cloning *only* the non-dominated ones
            let mut pareto_front: Vec<Portfolio> = non_dominated_indices
                .iter()
                .map(|&idx| remaining_portfolios[idx].clone())
                .collect();

            // Update rank for portfolios in this front
            pareto_front.iter_mut().for_each(|portfolio| {
                portfolio.rank = Some(current_front_rank);
            });

            // Calculate crowding distance for this front
            calculate_and_update_crowding_distance(&mut pareto_front);

            // Get the IDs of the portfolios added to this front
            // We need IDs because indices change after `retain`
            let pareto_ids: std::collections::HashSet<_> =
                pareto_front.iter().map(|portfolio| portfolio.id).collect();

            // Add the calculated front to the list of fronts
            fronts.push(pareto_front); // `pareto_front` is already Vec<Portfolio>

            // Remove the portfolios that were just added to the front from the remaining list
            remaining_portfolios.retain(|portfolio| !pareto_ids.contains(&portfolio.id));

            current_front_rank += 1;
        }

        fronts // Return the calculated fronts
    }

    fn initialize_population(
        population_size: usize,
        assets_under_management: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        if population_size == 0 || assets_under_management == 0 {
            return Err("Please pass postive values".into());
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
        #[serde(default = "default_max_concurrency")]
        pub max_concurrency: usize,
    }

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

    // Algo Logic
    // 1. Generate a bunch of portfolio for the first batch based on a parameter passed (initial training size)
    // 2. Sample based on the provided length using whichever method was specified by the user
    // 3. Run the simulation computing every day the daily returns, and whatnot. <- Target for multithreading if possible
    // 4. At the end use a Pareto front optimization strategy to select from a sample of non-dominated solutions and let them explore
    // 4.i To an extent we'd also want the ability to explore so we'd have a randomly chosen percentage between 0 and 15 percent of dominated
    // 4.ii solutions that would be allowed to reproduce using crossover.
    // 4.iii We'd also mutate the allocation stochastically in such away tha randomly an increment is done to one component (and equivalent decrement is done to another)
    // 5. Repeat until golden brown. lel
<<<<<<< HEAD
    pub fn standard_evolve_portfolios(config: StandardEvolutionConfig) -> EvolutionResult {
=======
    pub async fn standard_evolve_portfolios(config: StandardEvolutionConfig, athena_endpoint: String, population : Vec<Vec<f64>>) -> EvolutionResult {
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        // Initialization Phase
        //
        // Common Enough to Alias
        let population_size = config.population_size;
        let generations = config.generations;
        let simulations_per_generation = config.simulations_per_generation;

        let elite_population_size = ((population_size as f64) * config.elitism_rate) as usize;
        // Ensure elite size is reasonable
        if elite_population_size == 0 && config.elitism_rate > 0.0 {
<<<<<<< HEAD
            eprintln!("Warning: Elite population size rounded to 0, check population size and elitism rate.");
=======
            warn!("Warning: Elite population size rounded to 0, check population size and elitism rate.");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        }
        if elite_population_size >= population_size {
            panic!("Elite population size cannot be >= total population size.");
        }
        let offspring_count = population_size - elite_population_size;

        // For each portfolio we sample from a Uniform and then normalize
<<<<<<< HEAD
        let mut population: Vec<Vec<f64>> =
            initialize_population(population_size, config.assets_under_management).unwrap();
=======
        let mut population: Vec<Vec<f64>> = population;
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)

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

        // EVOLUTION BABY!!!
        for generation in 0..generations {
<<<<<<< HEAD
            let eval_result = evaluate_population_performance(&population, &config);
=======
            let eval_result =
                evaluate_population_distributed(&population, &config, &athena_endpoint)
                    .await
                    .expect("Failed to evaluate population");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)

            // --- Extract results ---
            let simulation_average_returns = eval_result.average_returns; // Per-portfolio
            let simulation_average_volatilities = eval_result.average_volatilities;
            let simulation_average_sharpe_ratios = eval_result.average_sharpe_ratios;
            let last_scenario_returns = eval_result.last_scenario_returns; // For memetic

            // --- Store generation metrics directly from eval_result ---
            best_average_return_per_generation[generation] = eval_result.best_return;
            average_return_per_generation[generation] = eval_result.population_average_return;
            best_average_volatility_per_generation[generation] = eval_result.best_volatility;
            average_volatility_per_generation[generation] =
                eval_result.population_average_volatility;
            best_average_sharpe_ratio_per_generation[generation] = eval_result.best_sharpe;
            average_sharpe_ratio_per_generation[generation] = eval_result.population_average_sharpe;

            // --- Create Portfolio Structs (uses per-portfolio averages) ---
            let mut portfolio_structs: Vec<Portfolio> = turn_weights_into_portfolios(
                &population,
                &simulation_average_returns,
                &simulation_average_volatilities,
                &simulation_average_sharpe_ratios,
            );
<<<<<<< HEAD
            
=======

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
            let mut fronts = build_pareto_fronts(&portfolio_structs);
            let breeding_pool: Vec<&Portfolio> = fronts.iter().flatten().collect();
            let mut next_generation: Vec<Vec<f64>> = Vec::new();

            // Adding Elites (Exploitation)
<<<<<<< HEAD
            for front in fronts.iter() { // Iterate through fronts (already ordered by rank: F1, F2, ...)
=======
            for front in fronts.iter() {
                // Iterate through fronts (already ordered by rank: F1, F2, ...)
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
                if next_generation.len() >= elite_population_size {
                    // Stop if we already have enough elites
                    break;
                }
<<<<<<< HEAD
        
=======

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
                if next_generation.len() + front.len() <= elite_population_size {
                    // If the entire current front fits within the remaining elite slots, add them all.
                    for portfolio in front.iter() {
                        next_generation.push(portfolio.weights.clone());
                    }
                } else {
                    // If the entire front doesn't fit, we need to take the best ones based on crowding distance.
                    let needed = elite_population_size - next_generation.len();
<<<<<<< HEAD
        
=======

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
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
<<<<<<< HEAD
        
=======

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
                    // Take the top 'needed' portfolios from the sorted front.
                    for portfolio in sorted_partial_front.iter().take(needed) {
                        next_generation.push(portfolio.weights.clone());
                    }
<<<<<<< HEAD
        
=======

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
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
<<<<<<< HEAD
        let final_eval_result = evaluate_population_performance(&population, &config);
=======
        let final_eval_result =
            evaluate_population_performance_distributed(&population, &config, &athena_endpoint)
                .await
                .expect("Failed to evaluate final population");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)

        // Create final portfolio structs using the final weights and *final* evaluation results
        let mut final_portfolio_structs = turn_weights_into_portfolios(
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
        EvolutionResult {
            pareto_fronts: build_pareto_fronts(&mut final_portfolio_structs),
            // Generation history vectors were filled during the loop
            best_average_return_per_generation,
            average_return_per_generation,
            best_average_volatility_per_generation,
            average_volatility_per_generation,
            best_average_sharpe_ratio_per_generation,
            average_sharpe_ratio_per_generation,
            // Include the final summary
            final_summary,
        }
        //
    }

<<<<<<< HEAD
    pub fn memetic_evolve_portfolios(config: MemeticEvolutionConfig) -> EvolutionResult {
=======
    pub async fn memetic_evolve_portfolios(config: MemeticEvolutionConfig, athena_endpoint: String, population: Vec<Vec<f64>>) -> EvolutionResult {
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        let population_size = config.base.population_size;
        let generations = config.base.generations;
        let elite_population_size =
            ((population_size as f64) * config.base.elitism_rate).round() as usize; // Use round for clarity
        let offspring_count = population_size - elite_population_size;
        let simulations_per_generation = config.base.simulations_per_generation;

        // Ensure elite size is reasonable
        if elite_population_size == 0 && config.base.elitism_rate > 0.0 {
<<<<<<< HEAD
            eprintln!("Warning: Elite population size rounded to 0, check population size and elitism rate.");
=======
            warn!("Warning: Elite population size rounded to 0, check population size and elitism rate.");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        }
        if elite_population_size >= population_size {
            panic!("Elite population size cannot be >= total population size.");
        }

        let mut population: Vec<Vec<f64>> =
<<<<<<< HEAD
            initialize_population(population_size, config.base.assets_under_management).unwrap();
=======
            population;
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)

        // Add new config params needed for memetic part
        let proximal_steps = config.memetic.proximal_descent_steps;
        let step_size = config.memetic.proximal_descent_step_size;
        let high_sharpe = config.memetic.high_sharpe_threshold;
        let low_vol = config.memetic.low_volatility_threshold;

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

        // --- Main Evolution Loop ---
        for generation in 0..generations {
            // Evaluate the current population
<<<<<<< HEAD
            let eval_result = evaluate_population_performance(&population, &config.base);
=======
            let eval_result = evaluate_population_performance_distributed(
                &population,
                &config.base,
                &athena_endpoint,
            )
            .await
            .expect("Failed to evaluate population");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)

            // --- Extract results ---
            let simulation_average_returns = eval_result.average_returns; // Per-portfolio
            let simulation_average_volatilities = eval_result.average_volatilities;
            let simulation_average_sharpe_ratios = eval_result.average_sharpe_ratios;
            let last_scenario_returns = eval_result.last_scenario_returns; // For memetic

            // --- Store generation metrics directly from eval_result ---
            best_average_return_per_generation[generation] = eval_result.best_return;
            average_return_per_generation[generation] = eval_result.population_average_return;
            best_average_volatility_per_generation[generation] = eval_result.best_volatility;
            average_volatility_per_generation[generation] =
                eval_result.population_average_volatility;
            best_average_sharpe_ratio_per_generation[generation] = eval_result.best_sharpe;
            average_sharpe_ratio_per_generation[generation] = eval_result.population_average_sharpe;

            // --- Create Portfolio Structs (uses per-portfolio averages) ---
            let mut portfolio_structs: Vec<Portfolio> = turn_weights_into_portfolios(
                &population,
                &simulation_average_returns,
                &simulation_average_volatilities,
                &simulation_average_sharpe_ratios,
            );

            // Non-dominated sort modifies ranks and crowding distances in place
            let fronts = build_pareto_fronts(&portfolio_structs);

            // --- The Memetic Part (Local Search) ---
            let mut next_generation_elites: Vec<Vec<f64>> =
                Vec::with_capacity(elite_population_size);
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
<<<<<<< HEAD
                eprintln!("Warning: Population size mismatch ({}) at end of generation {}. Should be {}. Resizing.", next_generation.len(), generation, population_size);
=======
                warn!("Warning: Population size mismatch ({}) at end of generation {}. Should be {}. Resizing.", next_generation.len(), generation, population_size);
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
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
<<<<<<< HEAD
        let final_eval_result = evaluate_population_performance(&population, &config.base);
=======
        let final_eval_result = evaluate_population_performance_distributed(
            &population,
            &config.base,
            &athena_endpoint,
        )
        .await
        .expect("Failed to evaluate final population");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)

        // Create final portfolio structs using the final weights and *final* evaluation results
        let mut final_portfolio_structs = turn_weights_into_portfolios(
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
        EvolutionResult {
            pareto_fronts: build_pareto_fronts(&mut final_portfolio_structs),
            // Generation history vectors were filled during the loop
            best_average_return_per_generation,
            average_return_per_generation,
            best_average_volatility_per_generation,
            average_volatility_per_generation,
            best_average_sharpe_ratio_per_generation,
            average_sharpe_ratio_per_generation,
            // Include the final summary
            final_summary,
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
<<<<<<< HEAD
    
=======

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        // 1. Choose k references randomly from the population slice.
        // choose_multiple returns an iterator yielding references to the elements of the slice.
        // Since the slice contains `&Portfolio`, the iterator yields `&&Portfolio`.
        let contestants: Vec<&&Portfolio> = population_refs.choose_multiple(&mut rng, k).collect();
<<<<<<< HEAD
    
        // Handle the case where k is 0 or larger than population size (choose_multiple might return fewer)
        if contestants.is_empty() {
            // This shouldn't happen if k > 0 and population is not empty,
            panic!("Tournament selection failed: No contestants selected (k={}, population_size={})", k, population_refs.len());
        }
    
=======

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

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        // 2. Find the best contestant among the selected references.
        // We iterate through the contestants (&&Portfolio) and find the minimum
        // based on the Portfolio's comparison logic (rank then crowding distance).
        // We need to dereference twice (&&Portfolio -> &Portfolio) to use partial_cmp.
<<<<<<< HEAD
        let winner: &&Portfolio = contestants.into_iter().min_by(|&&a, &&b| {
            // Compare the &Portfolio references using the implemented PartialOrd
            a.partial_cmp(b)
             .unwrap_or(Ordering::Equal) // Treat non-comparable as equal for sorting
                                        // Or Ordering::Less/Greater if a default is needed
        }).expect("Failed to find a winner in tournament selection"); // Should not happen if contestants is not empty
    
=======
        let winner: &&Portfolio = contestants
            .into_iter()
            .min_by(|&&a, &&b| {
                // Compare the &Portfolio references using the implemented PartialOrd
                a.partial_cmp(b).unwrap_or(Ordering::Equal) // Treat non-comparable as equal for sorting
                                                            // Or Ordering::Less/Greater if a default is needed
            })
            .expect("Failed to find a winner in tournament selection"); // Should not happen if contestants is not empty

>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
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
    enum Objective {
        AnnualizedReturns,
        SharpeRatio,
        Volatility,
        MaximizeStrength,
    }
    // For conversion to memetic algorithm
    // Takes a single step in the direction indicated by the gradient (proximal operator)
    fn lamarckian_proximal_descent(
        returns: &[Vec<f64>],
        weights: &[f64],
        performance_report: PortfolioPerformance,
        money_to_invest: f64,
        risk_free_rate: f64,
        time_horizon_in_days: f64,
        step_size: f64,
        objective: Objective,
        high_sharpe_threshold: f64,
        low_volatility_threshold: f64,
    ) -> Vec<f64> {
        let mut objective_for_descent = objective;
        if objective_for_descent == Objective::MaximizeStrength {
            objective_for_descent = find_dominant_objective(
                &performance_report,
                high_sharpe_threshold,
                low_volatility_threshold,
            );
        }

        let portfolio_gradient = compute_portfolio_gradient(
            &returns,
            &weights,
            performance_report,
            money_to_invest,
            risk_free_rate,
            time_horizon_in_days,
            objective_for_descent,
        );

        // gradient step
        let tentative_new_portfolio = weights
            .iter()
            .zip(portfolio_gradient)
            .map(|(w, g)| w - step_size * g)
            .collect::<Vec<f64>>();

        // proximal step returns the new portfolio to use
        proximal_step(&tentative_new_portfolio)
    }

    //
    fn proximal_step(weights: &Vec<f64>) -> Vec<f64> {
        // In this context, the proximal step reduces to projecting
        // the weight vector on the simplex defined by w_i >= 0, and sum(w_i) = 1.
        // Which is equivalent to solving the QP problem 1/2||w-x||^2 subject to the constraints

        // The following code adapts this idea without the overhead of an actual
        // QP solver
        let n = weights.len();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Find all the k non-zero values
        let mut sum = 0.0;
        let mut k = n;

        for i in 0..n {
            sum += sorted_weights[i];
            let theta = (sum - 1.0) / (i + 1) as f64;
            if sorted_weights[i] - theta <= 0.0 {
                k = i; // 1, 2, 3,... kth value, ... nth value
                break;
            }
        }

        // Compute the threshold value
        let sum_topk = sorted_weights[..k].iter().sum::<f64>();
        // Distributes the excess across the non-zero values
        let theta = (sum_topk - 1.0) / k as f64;

        // Finally project
        let projected_weights = weights
            .iter()
            .map(|w| (w - theta).max(0.))
            .collect::<Vec<f64>>();

        // Checks if all weights were squashed to 0.
        let projected_sum = projected_weights.iter().sum::<f64>();
        if projected_sum.abs() < 1e-10 * n as f64 {
            // return the uniform distribution if weights were already all negative or smth
            vec![1. / n as f64; n]
        } else {
            // Final normalization for stability
            projected_weights
                .into_iter()
                .map(|w| w / projected_sum)
                .collect::<Vec<f64>>()
        }
    }

    /// Holds the results of evaluating a population over multiple simulations,
    /// including both per-portfolio averages and population-wide summary statistics.
    #[derive(Debug, Clone)]
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
    fn evaluate_population_performance(
        population: &[Vec<f64>],
        config: &StandardEvolutionConfig,
    ) -> PopulationEvaluationResult {
        let population_size = population.len();
        if population_size == 0 {
<<<<<<< HEAD
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
=======
            panic!("I am pretty sure I got checks for that already, but we shouldn't be here with a population of 0.");
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        }

        let simulations_per_generation = config.simulations_per_generation;

        // Initialize accumulators
        let mut accumulated_returns = vec![0.0; population_size];
        let mut accumulated_volatilities = vec![0.0; population_size];
        let mut accumulated_sharpe_ratios = vec![0.0; population_size];
        let mut last_scenario_returns: Vec<Vec<f64>> = vec![];

        // --- Simulation Loop ---
        for i in 0..simulations_per_generation {
            let scenario_returns = config.sampler.sample_returns();
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
        let population_average_return =
            average_returns.par_iter().sum::<f64>() / population_size_f64;

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

    fn compute_portfolio_gradient(
        returns: &[Vec<f64>],
        weights: &[f64],
        base_performance: PortfolioPerformance,
        money_to_invest: f64,
        risk_free_rate: f64,
        time_horizon_in_days: f64,
        objective: Objective,
    ) -> Vec<f64> {
        // this is how little we perturb the solution
        let epsilon = 1e-6;
        let mut gradient = Vec::with_capacity(weights.len());

        // Compute all the partial derivatives to get the gradient vector
        for i in 0..weights.len() {
            // Perturb gand renormalize
            let mut perturbed_weights = weights.to_vec();
            perturbed_weights[i] += epsilon;
            let total = 1.0 + epsilon; // the weights are assumed to be valid

            perturbed_weights = perturbed_weights
                .iter()
                .map(|w| *w / total)
                .collect::<Vec<f64>>();

            // Compute the performances for the perturbed vector
            let perturbed_performance = compute_portfolio_performance(
                returns,
                &perturbed_weights,
                money_to_invest,
                risk_free_rate,
                time_horizon_in_days,
            );

            // Then compute the partial gradient based on objective!
            let partial_gradient = match objective {
                Objective::AnnualizedReturns => {
                    (perturbed_performance.annualized_return - base_performance.annualized_return)
                        / epsilon
                }
                Objective::Volatility => {
                    (perturbed_performance.percent_annualized_volatility
                        - base_performance.percent_annualized_volatility)
                        / epsilon
                }
                Objective::SharpeRatio => {
                    (perturbed_performance.sharpe_ratio - base_performance.sharpe_ratio) / epsilon
                }
                Objective::MaximizeStrength => {
                    unreachable!(
                        "Objective::MaximizeStrength should never be called in this context"
                    );
                }
            };
            if partial_gradient.is_nan() {
                panic!(
                    "NaN encountered in gradient calculation! \
                     Index: {}, Objective: {:?}, Epsilon: {}, \
                     Base Perf: {:?}, Perturbed Perf: {:?}, Weights: {:?}",
                    i,
                    objective,
                    epsilon,
                    base_performance,
                    perturbed_performance,
                    weights // Log relevant context
                );
            } else if !partial_gradient.is_finite() {
                // For robustness (we still log this)
<<<<<<< HEAD
                eprintln!(
=======
                warn!(
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
                    "Warning: Non-finite gradient ({}) encountered. \
                     Index: {}, Objective: {:?}, Epsilon: {}, \
                     Base Perf: {:?}, Perturbed Perf: {:?}, Weights: {:?}. \
                     Substituting gradient component with 0.0.",
                    partial_gradient,
                    i,
                    objective,
                    epsilon,
                    base_performance,
                    perturbed_performance,
                    weights
                );
                gradient.push(0.0); // Substitute 0.0 for Inf/-Inf
            } else {
                // Finite gradient is ideal (? I want more money)
                gradient.push(partial_gradient);
            }
        }
        gradient
    }

    #[derive(Debug, Clone)]
    struct PortfolioPerformance {
        portfolio_returns: Vec<f64>,
        annualized_return: f64,
        percent_annualized_volatility: f64,
        sharpe_ratio: f64,
    }

    fn compute_portfolio_performance(
        returns: &[Vec<f64>],
        weights: &[f64],
        money_to_invest: f64,
        risk_free_rate: f64,
        time_horizon_in_days: f64,
    ) -> PortfolioPerformance {
        // --- Edge Case Checks ---
        // Check 1: Invalid Configuration for Time/Money (Panic)
        if time_horizon_in_days.abs() < FLOAT_COMPARISON_EPSILON {
            panic!("Configuration Error: time_horizon_in_days cannot be zero.");
        }
        if money_to_invest.abs() < FLOAT_COMPARISON_EPSILON {
            panic!("Configuration Error: money_to_invest cannot be zero.");
        }

        let number_of_periods = returns.len() as f64;

        // Check 2: Insufficient Return Periods for Volatility/Sharpe (Panic)
        if number_of_periods < 2.0 {
            panic!(
            "Configuration Error: Cannot compute volatility or Sharpe ratio with fewer than 2 return periods (found {}). \
             Check 'periods_to_sample' in Sampler configuration.",
             returns.len()
         );
        }

        // --- Main Calculation (Now guaranteed N >= 2) ---
        let portfolio_returns = returns
            .par_iter()
            .map(|row| {
                row.par_iter()
                    .zip(weights.par_iter())
                    .map(|(log_return, weight)| {
                        ((log_return.exp() - 1.0) * *weight) * money_to_invest
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();

        let average_return = portfolio_returns.iter().sum::<f64>() / number_of_periods;

        // Calculate variance (N-1 in denominator is now safe)
        let variance = portfolio_returns
            .iter()
            .map(|ret| (ret - average_return).powi(2))
            .sum::<f64>()
            / (number_of_periods - 1.0);
        let volatility = variance.sqrt(); // Standard deviation (dollar terms)

        // Annualizing!
        let time_horizon_in_years = time_horizon_in_days / 365.0;
        let periods_per_year = number_of_periods / time_horizon_in_years;

        let annualized_return = average_return * periods_per_year;
        let annualized_volatility = volatility * periods_per_year.sqrt();
        let percent_annualized_volatility = annualized_volatility / money_to_invest;

        // Adjust risk-free rate
        let risk_free_return = money_to_invest * risk_free_rate; // Annual dollar risk-free

        // Calculate Sharpe
        let sharpe_ratio = if annualized_volatility.abs() >= FLOAT_COMPARISON_EPSILON {
            // CASE 1: Volatility is significantly NON-ZERO
            (annualized_return - risk_free_return) / annualized_volatility
        } else {
            // CASE 2: Volatility IS effectively ZERO
            // Throwaway cause that's a useless portfolio (just cap it at 0. sharpe tadum-tsh)
<<<<<<< HEAD
                0.0
            
=======
            0.0
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
        };

        PortfolioPerformance {
            portfolio_returns,
            annualized_return,
            percent_annualized_volatility,
            sharpe_ratio,
        }
    }

    // TESTS

    #[cfg(test)]
    mod tests {
        use statrs::distribution::MultivariateNormal;

        use super::*;
        use std::f64::EPSILON;

        // Helper: create a dummy Portfolio.
        // Adjust this function if your Portfolio::new signature is different.
        fn create_portfolio(average_returns: f64, volatility: f64, sharpe_ratio: f64) -> Portfolio {
            // Assume Portfolio::new takes (weights, average_returns, volatility, sharpe_ratio)
            Portfolio::new(
                vec![0.25, 0.25, 0.25, 0.25],
                average_returns,
                volatility,
                sharpe_ratio,
            )
        }

        // Test initialize_population: ensure it errors on zero and produces normalized vectors.
        #[test]
        fn test_initialize_population_errors_and_normalization() {
            assert!(
                initialize_population(0, 4).is_err(),
                "Population size of 0 should error"
            );
            assert!(
                initialize_population(10, 0).is_err(),
                "Assets under management of 0 should error"
            );

            let population = initialize_population(5, 4).unwrap();
            for weights in population.iter() {
                let sum: f64 = weights.iter().sum();
                assert!(
                    (sum - 1.0).abs() < FLOAT_COMPARISON_EPSILON,
                    "Weights should sum to 1. Got {}",
                    sum
                );
            }
        }

        // Test find_non_dominated_indices for empty input and single portfolio.
        #[test]
        fn test_find_non_dominated_indices() {
            let empty: Vec<Portfolio> = vec![];
            let indices = find_non_dominated_indices(&empty);
            assert!(
                indices.is_empty(),
                "Empty input should return empty indices"
            );

            let p1 = create_portfolio(0.10, 0.05, 2.0);
            let indices = find_non_dominated_indices(&[p1.clone()]);
            assert_eq!(indices, vec![0], "Single portfolio should be non-dominated");

            // Test with two portfolios where one dominates the other.
            let p2 = create_portfolio(0.05, 0.10, 1.0); // Clearly dominated by p1
            let indices = find_non_dominated_indices(&[p1.clone(), p2.clone()]);
            assert_eq!(
                indices,
                vec![0],
                "Only the dominating portfolio should remain"
<<<<<<< HEAD
            );
        }

        // Test build_pareto_fronts: Check that the front is built correctly.
        #[test]
        fn test_build_pareto_fronts() {
            let p1 = create_portfolio(0.10, 0.05, 2.0); // Non-dominated
            let p2 = create_portfolio(0.05, 0.10, 1.0); // Dominated by p1
            let p3 = create_portfolio(0.12, 0.06, 2.5); // Non-dominated
            let portfolios = vec![p1.clone(), p2.clone(), p3.clone()];
            let fronts = build_pareto_fronts(&portfolios);
            // Expect front 1 to contain p1 and p3 (order may vary)
            assert_eq!(
                fronts[0].len(),
                2,
                "Pareto front should have two portfolios"
            );
            // Check that the dominated one (p2) is not in the first front.
            for portfolio in fronts[0].iter() {
                assert_ne!(
                    portfolio.average_returns, p2.average_returns,
                    "Dominated portfolio should be removed"
                );
            }
        }

        // Test proximal_step: ensure projection onto the simplex.
        #[test]
        fn test_proximal_step_projection() {
            let unsorted = vec![0.5, 0.2, 0.3, -0.1];
            let projected = proximal_step(&unsorted);
            for &w in projected.iter() {
                assert!(w >= 0.0, "All weights must be non-negative");
            }
            let sum: f64 = projected.iter().sum();
            assert!(
                (sum - 1.0).abs() < FLOAT_COMPARISON_EPSILON,
                "Projected weights must sum to 1, got {}",
                sum
            );
        }

        // Test compute_portfolio_performance with zero returns.
        #[test]
        fn test_compute_portfolio_performance_zero_returns() {
            // 10 periods, 4 assets, all returns zero.
            let returns = vec![vec![0.0; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let performance =
                compute_portfolio_performance(&returns, &weights, 10000.0, 0.02, 365.0);

            assert!(
                (performance.annualized_return).abs() < FLOAT_COMPARISON_EPSILON,
                "Annualized return should be 0, but got {}",
                performance.annualized_return
            );
            assert!(
                (performance.percent_annualized_volatility).abs() < FLOAT_COMPARISON_EPSILON,
                "Volatility should be 0, but got {}",
                performance.percent_annualized_volatility
            );
            assert_eq!(
                performance.sharpe_ratio, 0.0,
                "Sharpe ratio should be 0 when returns equal risk-free, but got {}",
                performance.sharpe_ratio
            );
        }

        // Test compute_portfolio_performance panics with zero time horizon.
        #[test]
        #[should_panic(expected = "time_horizon_in_days cannot be zero")]
        fn test_compute_portfolio_performance_zero_time() {
            let returns = vec![vec![0.0; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let _ = compute_portfolio_performance(&returns, &weights, 10000.0, 0.02, 0.0);
        }

        // Test compute_portfolio_performance panics with zero money to invest.
        #[test]
        #[should_panic(expected = "money_to_invest cannot be zero")]
        fn test_compute_portfolio_performance_zero_money() {
            let returns = vec![vec![0.0; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let _ = compute_portfolio_performance(&returns, &weights, 0.0, 0.02, 365.0);
        }

        // Test compute_portfolio_gradient: Ensure no NaNs or Infs.
        #[test]
        fn test_compute_portfolio_gradient_no_nan() {
            let returns = vec![vec![0.01; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let base_perf = compute_portfolio_performance(&returns, &weights, 10000.0, 0.02, 365.0);
            let grad = compute_portfolio_gradient(
                &returns,
                &weights,
                base_perf,
                10000.0,
                0.02,
                365.0,
                Objective::AnnualizedReturns,
            );
            for g in grad.iter() {
                assert!(g.is_finite(), "Gradient component must be finite");
                assert!(!g.is_nan(), "Gradient component must not be NaN");
            }
        }

        // Integration test for standard_evolve_portfolios.
        // This is more of a smoke test than a strict functional test.
        #[test]
        fn test_standard_evolve_portfolios_smoke() {
            // Use a dummy Sampler with constant returns. We use the Normal variant.
            let dummy_normal = MultivariateNormal::new(
                vec![0.01, 0.01, 0.01, 0.01],
                vec![
                    0.001, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0,
                    0.0, 0.001,
                ],
            )
            .unwrap();
            let config = StandardEvolutionConfig {
                time_horizon_in_days: 365,
                generations: 3,
                population_size: 10,
                simulations_per_generation: 3,
                assets_under_management: 4,
                money_to_invest: 10000.0,
                risk_free_rate: 0.02,
                elitism_rate: 0.2,
                mutation_rate: 0.05,
                tournament_size: 2,
                sampler: Sampler::Normal {
                    normal_distribution: dummy_normal,
                    periods_to_sample: 10,
                },
                generation_check_interval: 1,
            };
            let result = standard_evolve_portfolios(config);
            // Check that we have a final Pareto front.
            assert!(
                !result.pareto_fronts.is_empty(),
                "There should be at least one Pareto front"
            );
            // Check that the final summary has finite values.
            assert!(
                result.final_summary.best_return.is_finite(),
                "Best return should be finite"
=======
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
            );
        }

        // Test build_pareto_fronts: Check that the front is built correctly.
        #[test]
<<<<<<< HEAD
    fn test_memetic_evolve_portfolios_smoke() {
        // Create a dummy Normal sampler with constant returns.
        let dummy_normal = MultivariateNormal::new(
            vec![0.01, 0.01, 0.01, 0.01],
            vec![
                0.001, 0.0,   0.0,   0.0,
                0.0,   0.001, 0.0,   0.0,
                0.0,   0.0,   0.001, 0.0,
                0.0,   0.0,   0.0,   0.001,
            ]
        ).unwrap();

        // Build a base evolution configuration.
        let base_config = StandardEvolutionConfig {
            time_horizon_in_days: 365,
            generations: 3,
            population_size: 10,
            simulations_per_generation: 3,
            assets_under_management: 4,
            money_to_invest: 10000.0,
            risk_free_rate: 0.02,
            elitism_rate: 0.2,
            mutation_rate: 0.05,
            tournament_size: 2,
            sampler: Sampler::Normal {
                normal_distribution: dummy_normal,
                periods_to_sample: 10,
            },
            generation_check_interval: 1,
        };

        // Define memetic parameters.
        let memetic_params = MemeticParams {
            local_objective: Objective::SharpeRatio,
            proximal_descent_steps: 3,
            proximal_descent_step_size: 0.001,
            high_sharpe_threshold: 2.0,
            low_volatility_threshold: 0.01,
        };

        // Combine into a memetic evolution configuration.
        let config = MemeticEvolutionConfig {
            base: base_config,
            memetic: memetic_params,
        };

        // Run the memetic evolution.
        let result = memetic_evolve_portfolios(config);

        // Check that we have at least one Pareto front.
        assert!(!result.pareto_fronts.is_empty(), "Expected at least one Pareto front");

        // Check that final summary values are finite.
        assert!(result.final_summary.best_return.is_finite(), "Final best return should be finite");
        assert!(result.final_summary.population_average_return.is_finite(), "Population average return should be finite");
        assert!(result.final_summary.best_volatility.is_finite(), "Final best volatility should be finite");
        assert!(result.final_summary.population_average_volatility.is_finite(), "Population average volatility should be finite");
        assert!(result.final_summary.best_sharpe.is_finite(), "Final best Sharpe ratio should be finite");
        assert!(result.final_summary.population_average_sharpe.is_finite(), "Population average Sharpe ratio should be finite");

        
        println!("Final Population Summary: {:?}", result.final_summary);
    }
=======
        fn test_build_pareto_fronts() {
            let p1 = create_portfolio(0.10, 0.05, 2.0); // Non-dominated
            let p2 = create_portfolio(0.05, 0.10, 1.0); // Dominated by p1
            let p3 = create_portfolio(0.12, 0.06, 2.5); // Non-dominated
            let portfolios = vec![p1.clone(), p2.clone(), p3.clone()];
            let fronts = build_pareto_fronts(&portfolios);
            // Expect front 1 to contain p1 and p3 (order may vary)
            assert_eq!(
                fronts[0].len(),
                2,
                "Pareto front should have two portfolios"
            );
            // Check that the dominated one (p2) is not in the first front.
            for portfolio in fronts[0].iter() {
                assert_ne!(
                    portfolio.average_returns, p2.average_returns,
                    "Dominated portfolio should be removed"
                );
            }
        }

        // Test proximal_step: ensure projection onto the simplex.
        #[test]
        fn test_proximal_step_projection() {
            let unsorted = vec![0.5, 0.2, 0.3, -0.1];
            let projected = proximal_step(&unsorted);
            for &w in projected.iter() {
                assert!(w >= 0.0, "All weights must be non-negative");
            }
            let sum: f64 = projected.iter().sum();
            assert!(
                (sum - 1.0).abs() < FLOAT_COMPARISON_EPSILON,
                "Projected weights must sum to 1, got {}",
                sum
            );
        }

        // Test compute_portfolio_performance with zero returns.
        #[test]
        fn test_compute_portfolio_performance_zero_returns() {
            // 10 periods, 4 assets, all returns zero.
            let returns = vec![vec![0.0; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let performance =
                compute_portfolio_performance(&returns, &weights, 10000.0, 0.02, 365.0);

            assert!(
                (performance.annualized_return).abs() < FLOAT_COMPARISON_EPSILON,
                "Annualized return should be 0, but got {}",
                performance.annualized_return
            );
            assert!(
                (performance.percent_annualized_volatility).abs() < FLOAT_COMPARISON_EPSILON,
                "Volatility should be 0, but got {}",
                performance.percent_annualized_volatility
            );
            assert_eq!(
                performance.sharpe_ratio, 0.0,
                "Sharpe ratio should be 0 when returns equal risk-free, but got {}",
                performance.sharpe_ratio
            );
        }

        // Test compute_portfolio_performance panics with zero time horizon.
        #[test]
        #[should_panic(expected = "time_horizon_in_days cannot be zero")]
        fn test_compute_portfolio_performance_zero_time() {
            let returns = vec![vec![0.0; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let _ = compute_portfolio_performance(&returns, &weights, 10000.0, 0.02, 0.0);
        }

        // Test compute_portfolio_performance panics with zero money to invest.
        #[test]
        #[should_panic(expected = "money_to_invest cannot be zero")]
        fn test_compute_portfolio_performance_zero_money() {
            let returns = vec![vec![0.0; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let _ = compute_portfolio_performance(&returns, &weights, 0.0, 0.02, 365.0);
        }

        // Test compute_portfolio_gradient: Ensure no NaNs or Infs.
        #[test]
        fn test_compute_portfolio_gradient_no_nan() {
            let returns = vec![vec![0.01; 4]; 10];
            let weights = vec![0.25, 0.25, 0.25, 0.25];
            let base_perf = compute_portfolio_performance(&returns, &weights, 10000.0, 0.02, 365.0);
            let grad = compute_portfolio_gradient(
                &returns,
                &weights,
                base_perf,
                10000.0,
                0.02,
                365.0,
                Objective::AnnualizedReturns,
            );
            for g in grad.iter() {
                assert!(g.is_finite(), "Gradient component must be finite");
                assert!(!g.is_nan(), "Gradient component must not be NaN");
            }
        }
        
>>>>>>> 59a3cdd (feat(portfolio_evolution.rs): It is now distributed, yay.)
    }
}
