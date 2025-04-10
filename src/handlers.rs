use crate::portfolio_evolution::{EvolutionResult, StandardEvolutionConfig};
use axum::{extract::Json, http::StatusCode, response::IntoResponse};

async fn handle_standard_evolve(Json(payload): Json<StandardEvolveRequest>) -> impl IntoResponse {
    // Initialize or load population
    let population = if let Some(pop) = payload.population {
        pop
    } else {
        initialize_population(
            payload.config.population_size,
            payload.config.assets_under_management,
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?
    };

    // Call the evolution
    match standard_evolve_portfolios(
        payload.config,
        payload.config.athena_endpoint.clone(),
        population,
    )
    .await
    {
        result => {
            let res: EvolutionResult = result;
            (StatusCode::OK, Json(res))
        }
    }
}

async fn handle_memetic_evolve(Json(payload): Json<MemeticEvolveRequest>) -> impl IntoResponse {
    let population = if let Some(pop) = payload.population {
        pop
    } else {
        initialize_population(
            payload.config.base.population_size,
            payload.config.base.assets_under_management,
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?
    };

    match memetic_evolve_portfolios(
        payload.config,
        payload.config.base.athena_endpoint.clone(),
        population,
    )
    .await
    {
        result => {
            let res: EvolutionResult = result;
            (StatusCode::OK, Json(res))
        }
    }
}
