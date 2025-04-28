use dotenv::dotenv;

use crate::web_app::build_app;
use std::{env, net::SocketAddr};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod athena_client;
mod consts;
mod evolution;
mod k8s_job;
mod web_app;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    // App host/port
    let host = env::var("APP_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = env::var("APP_PORT")
        .unwrap_or_else(|_| "3000".into())
        .parse::<u16>()
        .expect("APP_PORT must be a number");
    let addr = SocketAddr::from((host.parse::<std::net::IpAddr>().unwrap(), port));
    println!("Server starting on {}", addr);
    tracing::info!("Server starting on {}", addr);

    // Router
    let app = build_app();

    // Run server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
