use crate::docs::ApiDoc;
use axum::middleware::from_fn;
use axum::{http::StatusCode, routing::post, Router};
use dotenv::dotenv;
use handlers::{handle_memetic_evolve, handle_standard_evolve};
use std::{env, net::SocketAddr, sync::Arc};
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, request_id::MakeRequestUuid,
    request_id::SetRequestIdLayer, trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa_swagger_ui::{SwaggerUi, Url};

#[tokio::main]
async fn main() {
    dotenv().ok();

    // Logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    // App host/port
    let host = env::var("APP_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = env::var("APP_PORT")
        .unwrap_or_else(|_| "3000".into())
        .parse::<u16>()
        .expect("APP_PORT must be a number");
    let addr = SocketAddr::from((host.parse::<std::net::IpAddr>().unwrap(), port));

    tracing::info!("Server starting on {}", addr);

    // Router
    let app_routes = Router::new()
        .route("/evolve/standard", post(evolve_standard))
        .route("/evolve/memetic", post(evolve_memetic))
        .layer(
            tower::ServiceBuilder::new()
                .layer(SetRequestIdLayer::new(MakeRequestUuid))
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .into_inner(),
        );

    // Build the OpenAPI object
    let openapi = ApiDoc::openapi();

    // Mount your swagger UI at `/docs`.
    // The .url(...) argument sets the path to serve the raw JSON at.
    let swagger_ui = SwaggerUi::new("/docs").url("/api-docs/openapi.json", openapi);

    let app = app_routes.merge(swagger_ui);

    // Graceful shutdown
    let shutdown_signal = async {
        use tokio::signal;
        let ctrl_c = signal::ctrl_c();
        #[cfg(unix)]
        let terminate = {
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate()).unwrap();
            sigterm.recv()
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        tracing::warn!("Shutdown signal received");
    };

    // Run server
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal)
        .await
        .unwrap();
}
