use once_cell::sync::Lazy;
use std::env;

pub const NUMBER_OF_OPTIMIZATION_OBJECTIVES: usize = 3;
pub const PERTURBATION: f64 = 0.01;
pub const LOG_RETURNS_MEANS: (f64, f64, f64, f64) = (
    5.32273662e-04,
    6.63425548e-05,
    8.77944050e-05,
    6.45186507e-05,
);
pub const LOG_RETURNS_COV: [f64; 4 * 4] = [
    1.21196105e-04,
    1.13364388e-06,
    -2.25083039e-05,
    1.18820847e-04,
    1.13364388e-06,
    2.24497315e-06,
    5.60201102e-06,
    1.27454505e-06,
    -2.25083039e-05,
    5.60201102e-06,
    6.78149858e-05,
    -2.89021031e-05,
    1.18820847e-04,
    1.27454505e-06,
    -2.89021031e-05,
    2.58077479e-04,
];
pub static ATHENA_ENDPOINT: Lazy<String> =
    Lazy::new(|| env::var("ATHENA_ENDPOINT").expect("ATHENA_ENDPOINT must be set"));
