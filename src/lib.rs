// IMPORTS
use core::f64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::portfolio::portoflio::Portfolio;
use crate::sampling::sampling::Sampler;
use itertools::izip;
use rand::distributions::Uniform;
use rand::prelude::*;
use statrs::distribution::MultivariateNormal;
use crate::consts::*;

// Modules
mod consts;
mod portfolio;
mod python_bindings;
mod evolution;
mod sampling;

// Actual Code

