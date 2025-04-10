use crate::{
    evolution::portfolio_evolution::portfolio_evolution::MemeticEvolutionConfig,
    portfolio_evolution::{EvolutionResult, StandardEvolutionConfig},
};
use axum::{extract::Json, http::StatusCode, response::IntoResponse};
use serde::Deserialize;
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

#[utoipa::path(
    post,
    path = "/evolve/standard",
    tag = "Evolution",
    request_body = StandardEvolutionConfig,
    responses(
        (status = 200, description = "Evolution completed successfully", body = EvolutionResult),
        (status = 400, description = "Bad request", body = ErrorResponse)
    )
)]
async fn handle_standard_evolve(Json(payload): Json<StandardEvolutionConfig>) -> impl IntoResponse {
    // Initialize or load population
    let population =
        initialize_population(payload.population_size, payload.assets_under_management)
            .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    // Call the evolution
    match standard_evolve_portfolios(payload, payload.athena_endpoint.clone(), population).await {
        result => {
            let res: EvolutionResult = result;
            (StatusCode::OK, Json(res))
        }
    }
}

#[utoipa::path(
    post,
    path = "/evolve/memetic",
    tag = "Evolution",
    request_body = MemeticEvolutionConfig,
    responses(
        (status = 200, description = "Evolution completed successfully", body = EvolutionResult),
        (status = 400, description = "Bad request", body = ErrorResponse)
    )
)]
async fn handle_memetic_evolve(Json(payload): Json<MemeticEvolutionConfig>) -> impl IntoResponse {
    let population = initialize_population(
        payload.base.population_size,
        payload.base.assets_under_management,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    match memetic_evolve_portfolios(payload, payload.base.athena_endpoint.clone(), population).await
    {
        result => {
            let res: EvolutionResult = result;
            (StatusCode::OK, Json(res))
        }
    }
}
