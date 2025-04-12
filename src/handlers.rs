use crate::evolution::portfolio_evolution::{
    initialize_population, memetic_evolve_portfolios, standard_evolve_portfolios, EvolutionResult,
    MemeticEvolutionConfig, StandardEvolutionConfig,
};

use axum::{extract::Json, http::StatusCode};
use serde::{Deserialize, Serialize};

use crate::consts::ATHENA_ENDPOINT;

#[derive(Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub async fn handle_standard_evolve(
    Json(payload): Json<StandardEvolutionConfig>,
) -> Result<(StatusCode, Json<EvolutionResult>), (StatusCode, Json<ErrorResponse>)> {
    // now `?` works, because our return type is `Result<…, (StatusCode, Json<ErrorResponse>)>`
    let population =
        initialize_population(payload.population_size, payload.assets_under_management)
            .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    // call async evolution
    let evolution_result =
        standard_evolve_portfolios(payload, ATHENA_ENDPOINT.to_string(), population).await;

    // wrap in Ok for the successful path
    Ok((StatusCode::OK, Json(evolution_result)))
}

pub async fn handle_memetic_evolve(
    Json(payload): Json<MemeticEvolutionConfig>,
) -> Result<(StatusCode, Json<EvolutionResult>), (StatusCode, Json<ErrorResponse>)> {
    // Now `?` works, returning Err(...) on failure
    let population = initialize_population(
        payload.base.population_size,
        payload.base.assets_under_management,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    // Run async evolution logic
    let evolution_result =
        memetic_evolve_portfolios(payload, ATHENA_ENDPOINT.to_string(), population).await;

    // Wrap the successful result in Ok(...)
    Ok((StatusCode::OK, Json(evolution_result)))
}
