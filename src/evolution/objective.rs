use crate::evolution::aggregator::{
    Aggregator, AggregatorError, ArithmeticMean, StandardDeviation,
};
use aegis_athena_contracts::common_consts::FLOAT_COMPARISON_EPSILON;
use dyn_clone::DynClone;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::any::Any;
// This is a trait for any (I might reuse this, if not the code works)
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
// Actual objective and optimization stuff
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub enum OptimizationDirection {
    Maximize,
    Minimize,
}

#[typetag::serde(tag = "type")]
pub trait OptimizationObjective: DynClone + std::fmt::Debug + AsAny + Send + Sync {
    fn compute(&self, weights: &[f64], scenario: &[Vec<f64>]) -> Result<f64, AggregatorError>;
    /// A default (numerical) implementation of gradients for any OptimizationObjective
    /// any scalar objective can thus be differentiated from the get-go.
    fn gradient(
        &self,
        weights: &[f64],
        scenario: &[Vec<f64>],
        direction: Option<OptimizationDirection>,
    ) -> Result<Vec<f64>, AggregatorError> {
        let epsilon = 1e-6;
        let direction = direction.unwrap_or(self.default_direction());
        let mut gradient = Vec::with_capacity(weights.len());
        let objective_untouched = self.compute(weights, scenario);

        // central numerical gradient
        for (i, w) in weights.iter().enumerate() {
            // perturb and normalize
            let mut objective_plus = weights.to_vec();
            let mut objective_minus = weights.to_vec();

            // modify the vectors
            objective_plus[i] += epsilon;
            objective_minus[i] -= epsilon;
            // compute objective with respect to new vector
            let f_plus = self.compute(&objective_plus, scenario).unwrap();
            let f_minus = self.compute(&objective_minus, scenario).unwrap();
            // compute central gradient
            let mut partial_grad = (f_plus - f_minus) / (2. * epsilon);
            // adjust based on objective
            partial_grad = match direction {
                //Then gradient ascent (desired form is x_{t+1} = x_t + grad
                OptimizationDirection::Maximize => -partial_grad, // x_t+1 = x_t - (-grad)
                //Standard gradient descent
                OptimizationDirection::Minimize => partial_grad, // keep it as is.
            };
            gradient.push(partial_grad);
        }

        Ok(gradient)
    }
    fn default_direction(&self) -> OptimizationDirection;

    fn direction(&self) -> OptimizationDirection;
}
dyn_clone::clone_trait_object!(OptimizationObjective);
// The Built-In Objectives
#[derive(Serialize, Deserialize, Debug)]
pub struct Returns<A: Aggregator> {
    aggregator: A,
    direction: Option<OptimizationDirection>,
}

#[typetag::serde]
impl<A: Aggregator + 'static> OptimizationObjective for Returns<A> {
    fn compute(&self, weights: &[f64], scenario: &[Vec<f64>]) -> Result<f64, AggregatorError> {
        // compute returns
        let returns = scenario
            .par_iter()
            .map(|row| {
                row.par_iter()
                    .zip(weights.par_iter())
                    .map(|(log_return, weight)| ((log_return.exp() - 1.0) * *weight))
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        // aggregated returns
        self.aggregator.value(&returns)
    }
    fn gradient(
        &self,
        weights: &[f64],
        scenario: &[Vec<f64>],
        direction: Option<OptimizationDirection>,
    ) -> Result<Vec<f64>, AggregatorError> {
        // early fallback
        let returns_vec = scenario
            .iter()
            .map(|row| {
                row.iter()
                    .zip(weights.iter())
                    .map(|(log_return, wi)| (log_return.exp() - 1.) * wi)
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        let gradient_wrt_returns = self.aggregator.gradient_wrt_series(&returns_vec);
        if gradient_wrt_returns.is_err() {
            return OptimizationObjective::gradient(self, weights, scenario, direction);
        }

        // if the gradient_wrt_series is provided, we can be cleverer.
        let gradient_wrt_returns = gradient_wrt_returns.unwrap(); // we already checked.
        let asset_jacobian = scenario
            .iter()
            .map(|row| row.iter().map(|r_ti| r_ti.exp() - 1.).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>(); // T x asset_number

        // compute the gradient
        Ok((0..weights.len())
            .map(|i| {
                asset_jacobian // sum across rows for a given asset
                    .iter()
                    .zip(gradient_wrt_returns.iter())
                    .map(|(jacobian_row, g_wi)| jacobian_row[i] * g_wi)
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>())
    }
    fn default_direction(&self) -> OptimizationDirection {
        OptimizationDirection::Maximize
    }
    fn direction(&self) -> OptimizationDirection {
        self.direction.unwrap_or(self.default_direction())
    }
}

pub type MeanReturns = Returns<ArithmeticMean>;

#[derive(Serialize, Deserialize, Debug)]
pub struct Volatility(pub Returns<StandardDeviation>);
impl OptimizationObjective for Volatility {
    fn compute(&self, weights: &[f64], scenario: &[Vec<f64>]) -> Result<f64, AggregatorError> {
        // we simply delegate
        self.0.compute(weights, scenario)
    }
    fn default_direction(&self) -> OptimizationDirection {
        OptimizationDirection::Minimize
    }
    fn direction(&self) -> OptimizationDirection {
        self.direction.unwrap_or(self.default_direction())
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SharpeRatio {
    mean_returns: Returns<ArithmeticMean>,
    volatility: Volatility,
    time_horizon_in_days: usize,
    risk_free_rate: f64,
    direction: Option<OptimizationDirection>,
}

#[typetag::serde]
impl OptimizationObjective for SharpeRatio {
    fn compute(&self, weights: &[f64], scenario: &[Vec<f64>]) -> Result<f64, AggregatorError> {
        let time_horizon_in_years = (self.time_horizon_in_days as f64) / 365.;
        let periods_per_years = (scenario.len() as f64) / time_horizon_in_years;
        // the arithmetic mean should never fail, hence we use expect on it.
        let annualized_returns = self
            .mean_returns
            .compute(weights, scenario)
            .expect("Failed to get mean returns")
            * periods_per_years;
        let annualized_volatility =
            self.volatility.compute(weights, scenario)? * periods_per_years.sqrt();

        // Compute and return sharpe ratio
        if annualized_volatility.abs() >= FLOAT_COMPARISON_EPSILON {
            // vol is significant enough, so use it.
            Ok((annualized_returns - self.risk_free_rate) / annualized_volatility)
        } else {
            // effectively useless portfolio, so penalize it (vol == 0 is impossible IRL)
            Ok(0.)
        }
    }

    fn default_direction(&self) -> OptimizationDirection {
        OptimizationDirection::Maximize
    }
    fn direction(&self) -> OptimizationDirection {
        self.direction.unwrap_or(self.default_direction())
    }
}
