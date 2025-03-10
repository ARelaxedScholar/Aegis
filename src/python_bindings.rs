use crate::{
    evolve_portfolios as native_evolve_portfolios, EvolutionConfig, EvolutionResult, Portfolio,
    Sampler,
};
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PyEvolutionConfig {
    time_horizon_in_days: usize,
    generations: usize,
    population_size: usize,
    simulations_per_generation: usize,
    assets_under_management: usize,
    money_to_invest: f64,
    risk_free_rate: f64,
    elitism_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    sampler: Sampler,
    generation_check_interval: usize,
}
#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PyEvolutionResult {
    pareto_fronts: Vec<Vec<Portfolio>>,
    best_average_return_per_generation: Vec<f64>,
    average_return_per_generation: Vec<f64>,
    best_average_volatility_per_generation: Vec<f64>,
    average_volatility_per_generation: Vec<f64>,
    best_average_sharpe_ratio_per_generation: Vec<f64>,
    average_sharpe_ratio_per_generation: Vec<f64>,
}

#[pyfunction]
fn evolve_portfolios(config: PyEvolutionConfig) -> PyEvolutionResult {
    let config: EvolutionConfig = config.into();
    // call function
    let evolution_result = native_evolve_portfolios(config);
    //
    let result: PyEvolutionResult = evolution_result.into();
    result
}

#[pymodule]
fn rusty_evolution(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEvolutionConfig>()?;
    m.add_class::<PyEvolutionResult>()?;
    m.add_function(wrap_pyfunction!(evolve_portfolios, m)?)?;
    Ok(())
}

//  Conversion Functions
impl From<PyEvolutionConfig> for EvolutionConfig {
    fn from(py_config: PyEvolutionConfig) -> Self {
        EvolutionConfig {
            time_horizon_in_days: py_config.time_horizon_in_days,
            generations: py_config.generations,
            population_size: py_config.population_size,
            simulations_per_generation: py_config.simulations_per_generation,
            assets_under_management: py_config.assets_under_management,
            money_to_invest: py_config.money_to_invest,
            risk_free_rate: py_config.risk_free_rate,
            elitism_rate: py_config.elitism_rate,
            mutation_rate: py_config.mutation_rate,
            tournament_size: py_config.tournament_size,
            generation_check_interval: py_config.generation_check_interval,
            sampler: py_config.sampler, // Ensure Sampler implements needed traits.
        }
    }
}

impl From<EvolutionResult> for PyEvolutionResult {
    fn from(result: EvolutionResult) -> Self {
        PyEvolutionResult {
            pareto_fronts: result.pareto_fronts,
            best_average_return_per_generation: result.best_average_return_per_generation,
            average_return_per_generation: result.average_return_per_generation,
            best_average_volatility_per_generation: result.best_average_volatility_per_generation,
            average_volatility_per_generation: result.average_volatility_per_generation,
            best_average_sharpe_ratio_per_generation: result
                .best_average_sharpe_ratio_per_generation,
            average_sharpe_ratio_per_generation: result.average_sharpe_ratio_per_generation,
        }
    }
}
