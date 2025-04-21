use aegis::evolution::portfolio_evolution::{
    initialize_population, memetic_evolve_portfolios, standard_evolve_portfolios,
    MemeticEvolutionConfig, MemeticParams, Objective, SimRunner, StandardEvolutionConfig,
};
use aegis_athena_contracts::sampling::Sampler;
use futures::stream::{self, StreamExt};
use serde_json;
use std::{fs::File, io::Write, time::Instant};
use tokio::task;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let time_horizon_in_days = 252;
    let assets_under_management = 50;
    let number_of_factors = 5; // chosen for no specific reason, for the vibes.
    let seed = 42; // a classic
                   // Build your base config; max_concurrency is unused for Local.
    let base_config = StandardEvolutionConfig {
        time_horizon_in_days,
        generations: 100,
        population_size: 100,
        simulations_per_generation: 4600, // 1000 * log()
        assets_under_management,
        money_to_invest: 1_000_000.0,
        risk_free_rate: 0.01,
        elitism_rate: 0.1,
        mutation_rate: 0.1,
        tournament_size: 3,
        sampler: Sampler::factor_model_synthetic(
            assets_under_management,
            number_of_factors,
            time_horizon_in_days,
            Some(seed),
        )
        .expect("Failed to get sampler"),
        generation_check_interval: 10,
        global_seed: Some(seed),
        max_concurrency: 0, // not used by Local
        sim_runner: SimRunner::Local,
    };

    let fake_athena_endpoint = "wedontusethisanyways.com"; // required cause the functions can also use kubernetes/gRPC runners I configured, but not used here.
    let population = initialize_population(
        base_config.population_size,
        base_config.assets_under_management,
    )
    .expect("Failed to get stuff");

    let fake_ep = fake_athena_endpoint.to_string();
    let pop = population.clone();
    let base = base_config.clone();

    println!("Starting experiment: ");
    let start = Instant::now();

    stream::iter(0..=5)
        .map(|prox_steps| {
            println!("Run {prox_steps} was launched");
            // capture clones for each future
            let cfg = base.clone();
            let ep = fake_ep.clone();
            let pop = pop.clone();

            async move {
                // attach memetic if needed
                let memetic_params = MemeticParams {
                    local_objective: Objective::SharpeRatio,
                    proximal_descent_steps: prox_steps,
                    proximal_descent_step_size: 0.001,
                    high_sharpe_threshold: 1.8,
                    low_volatility_threshold: 0.10,
                };

                // run the correct async evolve
                let result = task::spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("failed to build mini-runtime");

                    rt.block_on(async {
                        if prox_steps == 0 {
                            standard_evolve_portfolios(cfg.clone(), ep.clone(), pop.clone()).await
                        } else {
                            let mem_cfg = MemeticEvolutionConfig {
                                base: cfg.clone(),
                                memetic: memetic_params,
                            };
                            memetic_evolve_portfolios(mem_cfg.clone(), ep.clone(), pop.clone())
                                .await
                        }
                    })
                })
                .await
                .expect("blocking task panicked");

                // offload blocking file I/O
                let filename = format!("ga_result_{}steps.json", prox_steps);
                let json = serde_json::to_string_pretty(&result)?;
                task::spawn_blocking(move || {
                    let mut f = File::create(&filename)?;
                    f.write_all(json.as_bytes())?;
                    Ok::<_, std::io::Error>(())
                })
                .await??;

                anyhow::Ok((prox_steps, result))
            }
        })
        // only 6 of these futures will run at once:
        .buffer_unordered(6)
        // print each result as it comes in
        .for_each(|res| async {
            match res {
                Ok((steps, r)) => {
                    println!(
                        "→ run({} steps): best_sharpe = {:.4}, pop_avg_sharpe = {:.4}",
                        steps,
                        r.final_summary.best_sharpe,
                        r.final_summary.population_average_sharpe
                    );
                }
                Err(e) => eprintln!("experiment failed: {e:#}"),
            }
        })
        .await;

    let elapsed = start.elapsed();
    println!("Total experiment time: {:.2?}", elapsed);

    Ok(())
}
