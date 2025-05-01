use crate::evolution::objective::AsAny;
use thiserror::Error;
/// Aggregator trait which allows to reduce a series to a single f64 number.
/// among other things allows the implementation of Volatility and Sharpe ratio
/// as returns + aggregators.
pub trait Aggregator: Sync + Send + AsAny {
    fn value(&self, series: &[f64]) -> Result<f64, AggregatorError>;
    fn gradient_wrt_series(&self, series: &[f64]) -> Result<Vec<f64>, AggregatorError> {
        Err(AggregatorError::GradientUnimplemented)
    }
}

#[derive(Error, Debug)]
pub enum AggregatorError {
    #[error("Gradient is unimplemented for this aggregator")]
    GradientUnimplemented,
    #[error("Number of periods is invalid for aggregator: `{0}`")]
    InvalidNumberOfPeriods(String),
}
pub struct ArithmeticMean;
impl Aggregator for ArithmeticMean {
    fn value(&self, series: &[f64]) -> Result<f64, AggregatorError> {
        Ok(series.iter().sum::<f64>() / (series.len() as f64))
    }
    fn gradient_wrt_series(&self, series: &[f64]) -> Result<Vec<f64>, AggregatorError> {
        let inv_t = 1. / (series.len() as f64);
        Ok(vec![inv_t; series.len()])
    }
}
pub struct StandardDeviation;
impl Aggregator for StandardDeviation {
    fn value(&self, series: &[f64]) -> Result<f64, AggregatorError> {
        if series.len() <= 1 {
            return Err(AggregatorError::InvalidNumberOfPeriods(
                "Standard deviation cannot be computed for series with less than 2 elements."
                    .into(),
            ));
        }
        let number_of_periods = series.len() as f64;
        let mean = series.iter().sum::<f64>() / (series.len() as f64);
        let variance =
            series.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (number_of_periods - 1.);
        Ok(variance.sqrt())
    }
}
