pub mod portfolio_evolution {
    use crate::{Portfolio, Sampler};
    use crate::{NUMBER_OF_OPTIMIZATION_OBJECTIVES, PERTURBATION};
    use itertools::izip;
    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rayon::prelude::*;
    use serde::{Deserialize, Serialize};

    fn find_pareto_front(portfolios: &[Portfolio]) -> Vec<Portfolio> {
        // Find all the non-dominated portfolios within batch
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
        population: Vec<Vec<f64>>,
        simulation_average_returns: Vec<f64>,
        simulation_average_volatilities: Vec<f64>,
        simulation_average_sharpe_ratios: Vec<f64>,
    ) -> Vec<Portfolio> {
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
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct MemeticParams {
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
    pub fn standard_evolve_portfolios(config: StandardEvolutionConfig) -> EvolutionResult {
        // Initialization Phase
        //
        // Common Enough to Alias
        let population_size = config.population_size;
        let generations = config.generations;
        let simulations_per_generation = config.simulations_per_generation;

        let elite_population_size = ((population_size as f64) * config.elitism_rate) as usize;
        // Ensure elite size is reasonable
        if elite_population_size == 0 && config.elitism_rate > 0.0 {
            eprintln!("Warning: Elite population size rounded to 0, check population size and elitism rate.");
        }
        if elite_population_size >= population_size {
            panic!("Elite population size cannot be >= total population size.");
        }
        let offspring_count = population_size - elite_population_size;

        // For each portfolio we sample from a Uniform and then normalize
        let mut population: Vec<Vec<f64>> =
            initialize_population(population_size, config.assets_under_management).unwrap();

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
            let eval_result = evaluate_population_performance(&population, &config);

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
                population,
                simulation_average_returns,
                simulation_average_volatilities,
                simulation_average_sharpe_ratios,
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
                    if next_generation.len() >= elite_population_size {
                        break;
                    }
                    next_generation.push(portfolio.weights.clone());
                }

                if next_generation.len() >= elite_population_size {
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

        // --- Final Evaluation After the Loop ---
        let final_eval_result = evaluate_population_performance(&population, &config);

        // Create final portfolio structs using the final weights and *final* evaluation results
        let mut final_portfolio_structs = turn_weights_into_portfolios(
            population,
            final_eval_result.average_returns,
            final_eval_result.average_volatilities,
            final_eval_result.average_sharpe_ratios,
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
            pareto_fronts: non_dominated_sort(&mut final_portfolio_structs),
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

    pub fn memetic_evolve_portfolios(config: MemeticEvolutionConfig) -> EvolutionResult {
        let population_size = config.base.population_size;
        let generations = config.base.generations;
        let elite_population_size =
            ((population_size as f64) * config.base.elitism_rate).round() as usize; // Use round for clarity
        let offspring_count = population_size - elite_population_size;
        let simulations_per_generation = config.base.simulations_per_generation;

        // Ensure elite size is reasonable
        if elite_population_size == 0 && config.base.elitism_rate > 0.0 {
            eprintln!("Warning: Elite population size rounded to 0, check population size and elitism rate.");
        }
        if elite_population_size >= population_size {
            panic!("Elite population size cannot be >= total population size.");
        }

        let mut population: Vec<Vec<f64>> =
            initialize_population(population_size, config.base.assets_under_management).unwrap();

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
            let eval_result = evaluate_population_performance(&population, &config.base);

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
                population,
                simulation_average_returns,
                simulation_average_volatilities,
                simulation_average_sharpe_ratios,
            );

            // Non-dominated sort modifies ranks and crowding distances in place
            let fronts = non_dominated_sort(&mut portfolio_structs);

            // --- The Memetic Part (Local Search) ---
            let mut next_generation_elites: Vec<Vec<f64>> =
                Vec::with_capacity(elite_population_size);
            let mut individuals_for_breeding = portfolio_structs.clone(); // Keep original population for breeding pool

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

                    for _ in 0..proximal_steps {
                        // Determine objective: Use find_dominant_objective or fix it (e.g., Sharpe)
                        // Let's use the heuristic here as an example
                        let objective_for_descent =
                            find_dominant_objective(&base_performance, high_sharpe, low_vol);
                        // OR: let objective_for_descent = Objective::SharpeRatio;

                        // Recalculate performance needed *inside* gradient computation
                        let current_performance_for_gradient = compute_portfolio_performance(
                            last_scenario_returns.clone(), // Need the returns for gradient calculation
                            current_weights.clone(),
                            config.base.money_to_invest,
                            config.base.risk_free_rate,
                            config.base.time_horizon_in_days as f64,
                        );

                        current_weights = lamarckian_proximal_descent(
                            last_scenario_returns.clone(), // Pass the sampled returns
                            current_weights,
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
                &individuals_for_breeding, // Use the full sorted population as potential parents
                offspring_count,
                config.base.mutation_rate,
                config.base.tournament_size,
            );

            // --- Combine Improved Elites and New Offspring ---
            let mut next_generation = next_generation_elites; // Start with improved elites
            next_generation.extend(offspring_weights);

            // Final check for population size consistency
            if next_generation.len() != population_size {
                eprintln!("Warning: Population size mismatch ({}) at end of generation {}. Should be {}. Resizing.", next_generation.len(), generation, population_size);
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
        let final_eval_result = evaluate_population_performance(&population, &config.base);

        // Create final portfolio structs using the final weights and *final* evaluation results
        let mut final_portfolio_structs = turn_weights_into_portfolios(
            population,
            final_eval_result.average_returns,
            final_eval_result.average_volatilities,
            final_eval_result.average_sharpe_ratios,
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
            pareto_fronts: non_dominated_sort(&mut final_portfolio_structs),
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

    // For Local Search
    // The idea is to push a solution in the solution it already excels in
    // This should ideally lead to a more diverse Pareto Front, and more interesting solutions
    #[derive(PartialEq)]
    enum Objective {
        AnnualizedReturns,
        SharpeRatio,
        Volatility,
        MaximizeStrength,
    }
    // For conversion to memetic algorithm
    // Takes a single step in the direction indicated by the gradient (proximal operator)
    fn lamarckian_proximal_descent(
        returns: Vec<Vec<f64>>,
        weights: Vec<f64>,
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
            returns,
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
            return PopulationEvaluationResult {
                average_returns: vec![],
                average_volatilities: vec![],
                average_sharpe_ratios: vec![],
                last_scenario_returns: vec![],
                best_return: f64::NEG_INFINITY,
                population_average_return: 0.0, // Or f64::NAN? Or handle upstream?
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
                        scenario_returns.clone(),
                        portfolio_weights.clone(),
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
        returns: Vec<Vec<f64>>,
        weights: &Vec<f64>,
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
            // Perturb and renormalize
            let mut perturbed_weights = weights.clone();
            perturbed_weights[i] += epsilon;
            let total = perturbed_weights.iter().sum::<f64>();
            perturbed_weights = perturbed_weights
                .into_iter()
                .map(|w| w / total)
                .collect::<Vec<f64>>();

            // Compute the performances for the perturbed vector
            let perturbed_performance = compute_portfolio_performance(
                returns.clone(),
                perturbed_weights,
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

            gradient.push(partial_gradient);
        }
        gradient
    }

    struct PortfolioPerformance {
        portfolio_returns: Vec<f64>,
        annualized_return: f64,
        percent_annualized_volatility: f64,
        sharpe_ratio: f64,
    }
    fn compute_portfolio_performance(
        returns: Vec<Vec<f64>>,
        weights: Vec<f64>,
        money_to_invest: f64,
        risk_free_rate: f64,
        time_horizon_in_days: f64, // these are days
    ) -> PortfolioPerformance {
        // Returns per row
        let portfolio_returns = returns
            .par_iter()
            .map(|row| {
                row.into_par_iter()
                    .zip(weights.par_iter())
                    .map(|(log_return, weight)| {
                        ((log_return.exp() - 1.0) * weight) * money_to_invest
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();

        let average_return =
            portfolio_returns.iter().sum::<f64>() / (portfolio_returns.len() as f64);
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
        use crate::evolution::portfolio_evolution::portfolio_evolution::find_pareto_front;
        use crate::portfolio::portoflio::Portfolio;

        // Helper function to create portfolios with specified return, volatility, and sharpe ratio.
        fn create_portfolio(average_returns: f64, volatility: f64, sharpe_ratio: f64) -> Portfolio {
            Portfolio::new(
                vec![0.25, 0.25, 0.25, 0.25], // Example weights
                average_returns,
                volatility,
                sharpe_ratio,
            )
        }

        #[test]
        fn test_empty_portfolio_list() {
            let portfolios: Vec<Portfolio> = vec![];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(
                pareto_front.len(),
                0,
                "Pareto front of an empty list should be empty"
            );
        }

        #[test]
        fn test_single_portfolio() {
            let portfolios = vec![create_portfolio(0.10, 0.05, 2.0)];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(
                pareto_front.len(),
                1,
                "Pareto front of a single portfolio should contain that portfolio"
            );
            assert_eq!(
                pareto_front[0].average_returns, 0.10,
                "The single portfolio's return should be preserved"
            );
        }

        #[test]
        fn test_clearly_dominated_portfolio() {
            let portfolios = vec![
                create_portfolio(0.10, 0.05, 2.0),
                create_portfolio(0.05, 0.10, 0.5), // Dominated
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(
                pareto_front.len(),
                1,
                "Pareto front should only contain the non-dominated portfolio"
            );
            assert_eq!(
                pareto_front[0].average_returns, 0.10,
                "The non-dominated portfolio's return should be preserved"
            );
        }

        #[test]
        fn test_no_domination() {
            // These portfolios are non-dominated because one has higher return but also higher volatility.
            let portfolios = vec![
                create_portfolio(0.10, 0.05, 2.0),
                create_portfolio(0.12, 0.06, 2.0),
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(
                pareto_front.len(),
                2,
                "Pareto front should contain both portfolios when neither dominates"
            );
        }

        #[test]
        fn test_identical_portfolios() {
            let portfolios = vec![
                create_portfolio(0.10, 0.05, 2.0),
                create_portfolio(0.10, 0.05, 2.0),
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(pareto_front.len(), 2, "Pareto front should contain identical portfolios (to preserve diversity or alternative weights)");
        }

        #[test]
        fn test_multiple_dominated() {
            let portfolios = vec![
                create_portfolio(0.10, 0.05, 2.0),  // Non-dominated
                create_portfolio(0.05, 0.10, 0.5),  // Dominated
                create_portfolio(0.08, 0.06, 1.0),  // Dominated
                create_portfolio(0.12, 0.07, 1.71), // Non-dominated
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(
                pareto_front.len(),
                2,
                "Pareto front should contain only the two non-dominated portfolios"
            );
        }

        #[test]
        fn test_large_number_of_portfolios() {
            let mut portfolios = Vec::new();
            for i in 0..100 {
                portfolios.push(create_portfolio(
                    0.10 + (i as f64 * 0.001),
                    0.05 + (i as f64 * 0.0001),
                    2.0,
                ));
            }
            let pareto_front = find_pareto_front(&portfolios);
            //In this case, none of the portfolios dominate each other since both return and volatility increase together
            assert_eq!(
                pareto_front.len(),
                100,
                "Pareto front should contain all portfolios when none dominate"
            );
        }

        #[test]
        fn test_negative_returns() {
            let portfolios = vec![
                create_portfolio(-0.05, 0.1, -0.5), // Dominated
                create_portfolio(-0.10, 0.2, -0.5), // Dominated
                create_portfolio(0.02, 0.02, 0.9),  // Non Dominated
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(
                pareto_front.len(),
                1,
                "Pareto front should only contain the two non-dominated portfolios"
            );
        }

        #[test]
        fn test_zero_volatility() {
            let portfolios = vec![
                create_portfolio(0.10, 0.00, 2.0), // Higher Sharpe, non-dominated
                create_portfolio(0.05, 0.00, 0.5), // Lower Sharpe, dominated by the first portfolio.
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(pareto_front.len(), 1, "Pareto front should contain only the first portfolio because has the higher Sharpe with same vol");
        }
        #[test]
        fn test_equal_sharpe_ratio() {
            let portfolios = vec![
                create_portfolio(0.10, 0.02, 2.0), // Equal Sharpe
                create_portfolio(0.06, 0.03, 2.0), // Equal Sharpe, dominated by the first portfolio due to higher return and lower vol
            ];
            let pareto_front = find_pareto_front(&portfolios);
            assert_eq!(pareto_front.len(), 1, "Pareto front should contain only the undominated portfolio when Sharpe ratios are equal.");
        }
    }
}
