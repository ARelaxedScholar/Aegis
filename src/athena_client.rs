use crate::evolution::portfolio_evolution::{PopulationEvaluationResult, StandardEvolutionConfig};
use aegis_athena_contracts::sampling::Sampler;
use aegis_athena_contracts::simulation::simulation_service_client::SimulationServiceClient;
use aegis_athena_contracts::simulation::SimulationBatchRequest;
use aegis_athena_contracts::simulation::{self, SimulationScenario, WeightVector};
use anyhow::anyhow;
use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use std::time::SystemTime;
use tonic::transport::Channel;

pub async fn evaluate_population_performance_distributed(
    population: &[Vec<f64>],
    config: &StandardEvolutionConfig,
    athena_endpoint: &str,
) -> anyhow::Result<PopulationEvaluationResult> {
    // 0) Pick a reproducible or fresh seed per generation
    let global_seed: u64 = config.global_seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Clock went backwards")
            .as_secs()
    });

    // 1) Total sims and concurrency
    let total_sims = config.simulations_per_generation;
    let max_conc = config.max_concurrency;

    // 2) Decide batch size & number of batches
    let batch_size = (total_sims + max_conc - 1) / max_conc;
    let num_batches = (total_sims + batch_size - 1) / batch_size;

    // 3) Prepare a single gRPC client to the Athena Service
    let channel = Channel::from_shared(athena_endpoint.to_string())?
        .connect()
        .await?;
    let client = SimulationServiceClient::new(channel);

    let population = Arc::new(population);
    let config = Arc::new(config.clone());

    // 4) Build all the batch requests
    let requests: Vec<_> = (0..num_batches)
        .map(|i| {
            let cfg = Arc::clone(&config);

            // Split the total number of simulations across independent batches
            // (e.g. 1000 sims, 4 batches = 250 sims each)
            // The last batch may be smaller if total_sims is not divisible by max_conc
            // (e.g. 1000 sims, 3 batches = 334, 333, 333)
            let sims = if i + 1 == num_batches {
                total_sims - batch_size * i
            } else {
                batch_size
            };
            SimulationBatchRequest {
                sampler: Some(cfg.sampler.clone().into()),
                population: population
                    .iter()
                    .map(|p| WeightVector {
                        weights: p.to_vec(),
                    })
                    .collect(),
                config: Some((*cfg).clone().into()),
                iterations: sims as i32,
                seed: global_seed.wrapping_add(i as u64), // per‐batch seed
            }
        })
        .collect();

    // 5) Run them with up to max_concurrency in flight
    let partials = stream::iter(requests)
        .map(|req| {
            let mut c = client.clone();
            async move {
                c.run_batch(req)
                    .await
                    .map(|r| r.into_inner())
                    .map_err(|e| anyhow::anyhow!("{}", e))
            }
        })
        .buffer_unordered(max_conc)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    // 6) Aggregate partial sums
    let n = population.len();
    let mut sum_r = vec![0.0; n];
    let mut sum_v = vec![0.0; n];
    let mut sum_s = vec![0.0; n];
    let mut last_scenario = SimulationScenario::default();

    for p in partials {
        for i in 0..n {
            sum_r[i] += p.sum_returns[i];
            sum_v[i] += p.sum_volatilities[i];
            sum_s[i] += p.sum_sharpes[i];
        }
        last_scenario = p
            .last_scenario
            .to_owned()
            .ok_or_else(|| anyhow!("missing last_scenario"))?;
    }

    // 7) Build the final PopulationEvaluationResult
    let total = total_sims as f64;
    let avg_r: Vec<f64> = sum_r.iter().map(|&x| x / total).collect();
    let avg_v: Vec<f64> = sum_v.iter().map(|&x| x / total).collect();
    let avg_s: Vec<f64> = sum_s.iter().map(|&x| x / total).collect();

    let best_r = avg_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pop_r = avg_r.iter().sum::<f64>() / (n as f64);
    let best_v = avg_v.iter().cloned().fold(f64::INFINITY, f64::min);
    let pop_v = avg_v.iter().sum::<f64>() / (n as f64);
    let best_s = avg_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pop_s = avg_s.iter().sum::<f64>() / (n as f64);

    let unflatten_assets_returns = |flat: Vec<f64>, n: usize| -> Vec<Vec<f64>> {
        assert_eq!(flat.len() % n, 0); // check that the n that is passed is correct
        flat.chunks(n) // iterator over &[f64] slices of length n
            .map(|row| row.to_vec())
            .collect() // Vec<Vec<f64>>
    };

    Ok(PopulationEvaluationResult {
        average_returns: avg_r,
        average_volatilities: avg_v,
        average_sharpe_ratios: avg_s,
        last_scenario_returns: unflatten_assets_returns(
            last_scenario.returns,
            config.assets_under_management,
        ),
        best_return: best_r,
        population_average_return: pop_r,
        best_volatility: best_v,
        population_average_volatility: pop_v,
        best_sharpe: best_s,
        population_average_sharpe: pop_s,
    })
}

impl From<StandardEvolutionConfig> for simulation::EvolutionConfig {
    fn from(cfg: StandardEvolutionConfig) -> Self {
        let periods_to_sample = match cfg.sampler {
            Sampler::FactorModel {
                periods_to_sample, ..
            } => periods_to_sample,
            Sampler::Normal {
                periods_to_sample, ..
            } => periods_to_sample,
            _ => panic!("Unsupported sampler type"),
        };
        simulation::EvolutionConfig {
            time_horizon_in_days: cfg.time_horizon_in_days as i32,
            periods_to_sample: periods_to_sample as u32,
            money_to_invest: cfg.money_to_invest,
            risk_free_rate: cfg.risk_free_rate,
        }
    }
}
