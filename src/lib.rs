#![allow(warnings)]
// IMPORTS
use core::f64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::consts::*;
use crate::portfolio::portoflio::Portfolio;
use crate::sampling::sampling::Sampler;
use itertools::izip;
use rand::distributions::Uniform;
use rand::prelude::*;
use statrs::distribution::MultivariateNormal;

// Modules
mod athena_client;
mod consts;
mod evolution;
mod handlers;
mod portfolio;
mod sampling;

// Actual Code
