use crate::consts::NUMBER_OF_OPTIMIZATION_OBJECTIVES;
use crate::evolution::portfolio_evolution::{
    generate_offsprings, initialize_population, make_evaluator, turn_weights_into_portfolios,
    EvolutionError, EvolutionResult, EvolutionStrategy, FinalPopulationSummary, SimRunnerStrategy,
    StandardEvolutionConfig,
};
use aegis_athena_contracts::portfolio::Portfolio;
use rayon::prelude::*;
use tracing::warn;

mod memetic_pareto;
mod standard_pareto;

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
            warn!("Warning: Found no non-dominated portfolios among remaining {} portfolios. Breaking sort.", remaining_portfolios.len());
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
