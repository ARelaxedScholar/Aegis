use utoipa::OpenApi;
use crate::handlers::handle_standard_evolve;
use crate::portfolio_evolution::{StandardEvolutionConfig, EvolutionResult};

#[derive(OpenApi)]
#[openapi(
    paths(
        handle_standard_evolve
    ),
    components(
        schemas(
            StandardEvolutionConfig,
            MemeticEvolutionConfig
            EvolutionResult,
            ErrorResponse
        )
    ),
    tags(
        (name = "Evolution", description = "Endpoints for evolving stuff")
    )
)]
pub struct ApiDoc;
