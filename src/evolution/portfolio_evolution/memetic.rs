use crate::evolution::portfolio_evolution::find_dominant_objective;
use crate::evolution::portfolio_evolution::Objective;
use aegis_athena_contracts::common_portfolio_evolution_ds::PortfolioPerformance;

use super::gradients::compute_portfolio_gradient;
// For conversion to memetic algorithm
// Takes a single step in the direction indicated by the gradient (proximal operator)
pub fn lamarckian_proximal_descent(
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
        .map(|(w, g)| w + step_size * g)
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
