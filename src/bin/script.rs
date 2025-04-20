use std::{
    fs::File,
    io::Write,
    sync::mpsc::channel,
    thread,
    time::Instant,   
};
use threadpool::ThreadPool;
use serde_json;
use aegis::evolution::portfolio_evolution::{Objective, StandardEvolutionConfig, MemeticEvolutionConfig, MemeticParams, initialize_population, SimRunner, standard_evolve_portfolios, memetic_evolve_portfolios};
use aegis_athena_contracts::sampling::Sampler;
use futures::stream::{self, StreamExt};
use tokio::task;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build your base config; max_concurrency is unused for Local.
    let base_config = StandardEvolutionConfig {
        time_horizon_in_days: 252,
        generations: 10,
        population_size: 2,
        simulations_per_generation: 1_00, // small for testing
        assets_under_management: 2,
        money_to_invest: 1_000_000.0,
        risk_free_rate: 0.01,
        elitism_rate: 0.05,
        mutation_rate: 0.05,
        tournament_size: 3,
        sampler: Sampler::factor_model_synthetic(2, 5, 252, Some(42)).expect("Failed to get sampler"),
        generation_check_interval: 10,
        global_seed: Some(42),
        max_concurrency: 0,     // not used by Local
        sim_runner: SimRunner::Local,
    };
    if let Sampler::FactorModel{ref mu_assets, ..} = base_config.sampler {
    println!("This is the mu_assets: {:?}", mu_assets);
    } else {
    	panic!("We'll never get here");
    }
    

    // Launcher: 5 threads max, channel to collect back steps + results.
    let pool = ThreadPool::new(5);
    let fake_athena_endpoint = "wedontusethisanyways.com";
    let population =
        initialize_population(base_config.population_size, base_config.assets_under_management).expect("Failed to get stuff");

    let fake_ep = fake_athena_endpoint.to_string();
    let pop = population.clone();
    let base = base_config.clone();

    println!("Starting experiment: ");
    let start = Instant::now(); 
    
    stream::iter(0..=5)
        .map(|prox_steps| {
            println!("Run {prox_steps} was launched");
            // capture clones for each future
            let mut cfg = base.clone();
            let ep = fake_ep.clone();
            let pop = pop.clone();

            async move {
                // attach memetic if needed
                let memetic_params  =  MemeticParams {
                        local_objective: Objective::AnnualizedReturns,
                        proximal_descent_steps: prox_steps,
                        proximal_descent_step_size: 0.001,
                        high_sharpe_threshold: 1.8,
                        low_volatility_threshold: 0.10,
                    };

                // run the correct async evolve
                let result = if prox_steps == 0 {
                    standard_evolve_portfolios(cfg.clone(), ep.clone(), pop.clone()).await
                } else {
                    let mem_cfg = MemeticEvolutionConfig {
                        base: cfg.clone(),
                        memetic: memetic_params,
                    };
                    memetic_evolve_portfolios(mem_cfg.clone(), ep.clone(), pop.clone()).await
                };

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
        // only 5 of these futures will run at once:
        .buffer_unordered(5)
        // print each result as it comes in
        .for_each(|res| async {
            match res {
                Ok((steps, r)) => {
                    println!(
                        "→ run({} steps): best_sharpe = {:.4}, pop_avg_sharpe = {:.4}",
                        steps, r.final_summary.best_sharpe, r.final_summary.population_average_sharpe
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

