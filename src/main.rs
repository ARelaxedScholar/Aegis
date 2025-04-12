use axum::{routing::post, Router};
use dotenv::dotenv;
use handlers::{handle_memetic_evolve, handle_standard_evolve};
use http::header::HeaderName;
use std::{env, net::SocketAddr};
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, request_id::MakeRequestUuid,
    request_id::SetRequestIdLayer, trace::TraceLayer,
};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod athena_client;
mod consts;
mod evolution;
mod handlers;

#[tokio::main]
async fn main() {
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
    let x_request_id = HeaderName::from_static("x-request-id");
    // Router
    let app = Router::new()
        .route("/evolve/standard", post(handle_standard_evolve))
        .route("/evolve/memetic", post(handle_memetic_evolve))
        .layer(
            tower::ServiceBuilder::new()
                .layer(SetRequestIdLayer::new(
                    x_request_id.clone(),
                    MakeRequestUuid,
                ))
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .into_inner(),
        );

    // Run server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}
