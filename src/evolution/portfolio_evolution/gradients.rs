use aegis_athena_contracts::common_portfolio_evolution_ds::compute_portfolio_performance;
use tracing::warn;

use crate::evolution::portfolio_evolution::{BuiltInObjective, PortfolioPerformance};

pub fn compute_portfolio_gradient(
    returns: &[Vec<f64>],
    weights: &[f64],
    base_performance: PortfolioPerformance,
    money_to_invest: f64,
    risk_free_rate: f64,
    time_horizon_in_days: f64,
    objective: BuiltInObjective,
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
            BuiltInObjective::AnnualizedReturns => {
                (perturbed_performance.annualized_return - base_performance.annualized_return)
                    / epsilon
            }
            BuiltInObjective::Volatility => {
                (perturbed_performance.percent_annualized_volatility
                    - base_performance.percent_annualized_volatility)
                    / epsilon
            }
            BuiltInObjective::SharpeRatio => {
                (perturbed_performance.sharpe_ratio - base_performance.sharpe_ratio) / epsilon
            }
            BuiltInObjective::MaximizeStrength => {
                unreachable!("BuiltInObjective::MaximizeStrength should never be called in this context");
            }
        };
        if partial_gradient.is_nan() {
            panic!(
                "NaN encountered in gradient calculation! \
                     Index: {}, BuiltInObjective: {:?}, Epsilon: {}, \
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
            warn!(
                "Warning: Non-finite gradient ({}) encountered. \
                     Index: {}, BuiltInObjective: {:?}, Epsilon: {}, \
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
