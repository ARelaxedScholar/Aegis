use axum::{routing::post, Router};
use http::header::HeaderName;
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, request_id::MakeRequestUuid,
    request_id::SetRequestIdLayer, trace::TraceLayer,
};

pub fn build_app() -> Router {
    let x_request_id = HeaderName::from_static("x-request-id");
    Router::new().layer(
        tower::ServiceBuilder::new()
            .layer(SetRequestIdLayer::new(
                x_request_id.clone(),
                MakeRequestUuid,
            ))
            .layer(TraceLayer::new_for_http())
            .layer(CompressionLayer::new())
            .layer(CorsLayer::permissive())
            .into_inner(),
    )
}
