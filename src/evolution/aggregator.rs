/// Aggregator trait which allows to reduce a series to a single f64 number.
/// among other things allows the implementation of Volatility and Sharpe ratio
/// as returns + aggregators.
pub trait Aggregator {
    fn value(&self, series: &[f64]) -> Result<f64, &'static str>;
    fn gradient_wrt_series(&self, series: &[f64]) -> Option<Vec<f64>> {
        None
    }
}

pub struct ArithmeticMean;
impl Aggregator for ArithmeticMean {
    fn value(&self, series: &[f64]) -> Result<f64, &'static str> {
        Ok(series.iter().sum::<f64>() / (series.len() as f64))
    }
    fn gradient_wrt_series(&self, series: &[f64]) -> Option<Vec<f64>> {
        let inv_t = 1. / (series.len() as f64);
        Some(vec![inv_t; series.len()])
    }
}
pub struct StandardDeviation;
impl Aggregator for StandardDeviation {
    fn value(&self, series: &[f64]) -> Result<f64, &'static str> {
        if series.len() <= 1 {
            return Err(
                "Standard deviation cannot be computed for series with less than 2 elements.",
            );
        }
        let number_of_periods = series.len() as f64;
        let mean_series = series.iter().sum::<f64>() / (series.len() as f64);
        let variance =
            series.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (number_of_periods - 1.);
        variance.sqrt()
    }
}
