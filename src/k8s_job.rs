use anyhow::{Error, Result};
use futures::{StreamExt, TryStreamExt};
use k8s_openapi::{
    api::{
        batch::v1::{Job, JobSpec},
        core::v1::{Container, EnvVar, PodSpec, PodTemplateSpec},
    },
    apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta},
};
use kube::api::ListParams;
use kube::{
    api::{Api, DeleteParams, PostParams},
    runtime::{watcher, watcher::Config, WatchStreamExt},
    Client,
};
//use kube_runtime::watcher::{self, watcher, Event};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, env};
use uuid::Uuid;

use crate::evolution::portfolio_evolution::{PopulationEvaluationResult, StandardEvolutionConfig};
use aegis_athena_contracts::sampling::SerdeSampler;

#[derive(Deserialize)]
struct SimulationBatchResultDTO {
    // Aggregated sum of returns for each portfolio.
    sum_returns: Vec<f64>,
    // Aggregated sum of volatilities for each portfolio.
    sum_volatilities: Vec<f64>,
    // Aggregated sum of Sharpe ratios for each portfolio.
    sum_sharpes: Vec<f64>,
    // The market scenario (i.e. series of returns) from the last simulation.
    last_scenario: ScenarioDTO,
}

#[derive(Deserialize)]
struct ScenarioDTO {
    returns: Vec<f64>,
}

pub async fn evaluate_generation_in_k8s_job(
    config: &StandardEvolutionConfig,
    population: &[Vec<f64>],
) -> Result<PopulationEvaluationResult> {
    // Turn your internal Sampler into the Serde‐friendly DTO
    let serde_sampler = SerdeSampler::from(config.sampler.clone());

    // Build the JSON payload the runner expects
    let payload = json!({
        "sampler": serde_sampler,
        "time_horizon_in_days": config.time_horizon_in_days as i32,
        "periods_to_sample": config.simulations_per_generation as u32,
        "money_to_invest": config.money_to_invest,
        "risk_free_rate": config.risk_free_rate,
        "population": population,
        "iterations": config.simulations_per_generation as i32,
        "seed": config.global_seed.unwrap_or(0),
    })
    .to_string();

    // Which image should we launch?
    let image = env::var("ATHENA_RUNNER_IMAGE")
        .map_err(|_| Error::msg("ATHENA_RUNNER_IMAGE must be set"))?;

    // Unique name + labels for the Job & its Pods
    let job_name = format!("evolve-{}", Uuid::new_v4());
    let mut labels = BTreeMap::new();
    labels.insert("job-name".to_string(), job_name.clone());

    // Construct the Kubernetes Job
    let job = Job {
        metadata: ObjectMeta {
            name: Some(job_name.clone()),
            labels: Some(labels.clone()),
            ..Default::default()
        },
        spec: Some(JobSpec {
            parallelism: Some(config.max_concurrency as i32),
            completions: Some(config.max_concurrency as i32),
            completion_mode: Some("Indexed".to_string()),
            selector: Some(LabelSelector {
                match_labels: Some(labels.clone()),
                ..Default::default()
            }),
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels.clone()),
                    ..Default::default()
                }),
                spec: Some(PodSpec {
                    restart_policy: Some("Never".into()),
                    containers: vec![ Container {
                                            name:  "athena-runner".into(),
                                            image: Some(image.clone()),
                                            env: Some(vec![
                                                // your existing PAYLOAD_JSON
                                                EnvVar {
                                                    name:  "PAYLOAD_JSON".into(),
                                                    value: Some(payload.clone()),
                                                    ..Default::default()
                                                },
                                                EnvVar {
                                                    name: "COMPLETIONS".into(),
                                                    value: Some(config.max_concurrency.to_string()), // matches spec.completions
                                                    ..Default::default()
                                                  },
                                                // Inject the pod’s completion index:
                                                EnvVar {
                                                    name: "JOB_COMPLETION_INDEX".into(),
                                                    value_from: Some(k8s_openapi::api::core::v1::EnvVarSource {
                                                        field_ref: Some(k8s_openapi::api::core::v1::ObjectFieldSelector {
                                                            api_version: None,
                                                            field_path: "metadata.annotations['batch.kubernetes.io/job-completion-index']".into(),
                                                        }),
                                                        ..Default::default()
                                                    }),
                                                    ..Default::default()
                                                },
                                            ]),
                                            ..Default::default()
                                        }],
                    ..Default::default()
                }),
            },
            ..Default::default()
        }),
        ..Default::default()
    };

    // Submit it
    let client = Client::try_default().await?;
    let jobs: Api<Job> = Api::namespaced(client.clone(), "default");
    jobs.create(&PostParams::default(), &job).await?;

    // Wait for all Pods to succeed
    // Create a watcher stream for that one Jo

    // As soon as we see the Job’s succeeded count hit our target, we break
    // create the watcher on *just* our Job by name
    let mut stream = watcher(
        Api::<Job>::namespaced(client.clone(), "default"),
        Config {
            field_selector: Some(format!("metadata.name={}", job_name)),
            ..Default::default()
        },
    )
    .applied_objects()
    .boxed(); // only “Applied” events, so we get the Job once on LIST, then again on each UPDATE

    // Pull events one by one
    while let Some(job) = stream.try_next().await? {
        if let Some(succeeded) = job.status.as_ref().and_then(|s| s.succeeded) {
            if succeeded >= config.max_concurrency as i32 {
                break; // we’ve seen all Pods finish
            }
        }
    }

    // Find any one of the Pods, fetch its logs (stdout JSON), parse
    let lp = ListParams {
        label_selector: Some(format!("job-name={}", job_name)),
        ..ListParams::default()
    };
    let pods: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(client.clone(), "default");
    let pod_list = pods.list(&lp).await?;

    let mut partials = Vec::new();
    for pod in pod_list.items {
        if let Some(name) = pod.metadata.name {
            let logs = pods.logs(&name, &Default::default()).await?;
            partials.push(serde_json::from_str::<SimulationBatchResultDTO>(&logs)?);
        }
    }

    let n = population.len();
    let mut sum_r = vec![0.0; n];
    let mut sum_v = vec![0.0; n];
    let mut sum_s = vec![0.0; n];

    // Dummy intialization
    let mut last_scenario = ScenarioDTO {
        returns: vec![0.; n],
    };

    // Each partial has its own sums; we just add them up
    for p in partials {
        for i in 0..n {
            sum_r[i] += p.sum_returns[i];
            sum_v[i] += p.sum_volatilities[i];
            sum_s[i] += p.sum_sharpes[i];
        }
        last_scenario = p.last_scenario;
    }

    // Do the aggregation
    let total = config.simulations_per_generation as f64;
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

    // Clean-up and submit
    jobs.delete(&job_name, &DeleteParams::background()).await?;
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
