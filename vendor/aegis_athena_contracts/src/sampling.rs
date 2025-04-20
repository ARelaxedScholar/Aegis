use pyo3::prelude::*;

use crate::sampling::Sampler as RustSampler;
use crate::simulation::sampler_config::Kind;
use crate::simulation::SamplerConfig;
use crate::simulation::{FactorModelConfig, NormalConfig, SeriesGanConfig};
use nalgebra::Dyn;
use rand::RngCore;
use rand::{distributions::Distribution, rngs::OsRng, Rng, SeedableRng};
use rand_chacha::rand_core::block;
use rand_chacha::ChaCha20Rng;
use rand_distr::Uniform;
use serde::de::{self, Error as DeError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::DeserializeFromStr;
use statrs::distribution::MultivariateNormal;
use statrs::statistics::{MeanN, VarianceN};
use std::convert::{TryFrom, TryInto};
// ---- Quick setup for the Serialization logic
const CURRENT_SERIALIZER_VERSION: u32 = 1;
fn default_version() -> u32 {
    CURRENT_SERIALIZER_VERSION
}

// ---- This is the code for the PySampler (This is what will be used in the PyO3 bindings when I reimplement them.)
// ---- We can deserialize PySampler for free.
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct PySampler {
    #[serde(flatten)]
    sampler: Sampler,
}

#[pymethods]
impl PySampler {
    /// Construct a new sampler: mode = "factor" or "normal", seed = Option<u64>
    #[new]
    #[pyo3(signature = (mode, assets, factors, periods, seed = None))]
    fn new(
        mode: &str,
        assets: usize,
        factors: usize,
        periods: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let sampler = match mode {
            "factor" => Sampler::factor_model_synthetic(assets, factors, periods, seed)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?,
            "normal" => {
                Sampler::normal(
                    &vec![0.0; assets],
                    &vec![1.0; assets * assets],
                    periods,
                    seed,
                )
            }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "mode must be \"factor\" or \"normal\"",
                ));
            }
        };
        Ok(PySampler { sampler })
    }

    /// Sample returns; advances internal RNG
    fn sample_returns(&mut self) -> Vec<Vec<f64>> {
        self.sampler.sample_returns()
    }

    /// Reseed the internal RNG mid-flight
    fn reseed(&mut self, seed: u64) {
        self.sampler.reseed(seed);
    }
}

// ---- Impl of Sampler (this is methods)
#[derive(Debug, Clone)]
pub enum Sampler {
    FactorModel {
        assets_under_management: usize,
        periods_to_sample: usize,
        number_of_factors: usize,

        mu_factors: Vec<f64>,
        covariance_factors: Vec<Vec<f64>>,
        loadings: Vec<Vec<f64>>,
        idiosyncratic_variances: Vec<f64>,

        mu_assets: Vec<f64>,
        covariance_assets: Vec<Vec<f64>>,

        normal_distribution: MultivariateNormal<Dyn>,
        rng: ChaCha20Rng,
        seed: u64,
    },

    Normal {
        periods_to_sample: usize,
        normal_distribution: MultivariateNormal<Dyn>,
        rng: ChaCha20Rng,
        seed: u64,
    },

    SeriesGAN(usize),
}

// ---- Impl of Sampler (this is static functions.)
impl Sampler {
    /// If seed is None, we generate one from OsRng so we can record it.
    pub fn factor_model_synthetic(
        assets_under_management: usize,
        number_of_factors: usize,
        periods_to_sample: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if assets_under_management == 0 || number_of_factors == 0 {
            return Err("Assets and Factors should be positives".into());
        }

        // determine seed
        let seed = seed.unwrap_or_else(|| OsRng.next_u64());
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // build factor model...
        let small_returns = 0.001;
        let mu_factors = vec![small_returns; number_of_factors];
        let covariance_factors = Self::generate_covariance_matrix(number_of_factors)?;

        let throwaway = MultivariateNormal::new(
            mu_factors.clone(),
            covariance_factors.clone().into_iter().flatten().collect(),
        )
        .map_err(|e| format!("MVN init failed: {}", e))?;
        let factor_returns = throwaway.sample(&mut rng);

        let uniform = rand::distributions::Uniform::new(0.5, 1.5);
        let mut loadings = vec![vec![0.0; number_of_factors]; assets_under_management];
        for i in 0..assets_under_management {
            for j in 0..number_of_factors {
                loadings[i][j] = rng.sample(uniform);
            }
        }

        let idiosyncratic_variances = vec![0.01; assets_under_management];
        let mut mu_assets = vec![0.0; assets_under_management];
        for i in 0..assets_under_management {
            mu_assets[i] = loadings[i]
                .iter()
                .zip(factor_returns.iter())
                .map(|(l, f)| l * f)
                .sum();
        }

        let mut covariance_assets = Self::compute_asset_covariance(
            &loadings,
            &covariance_factors,
            &idiosyncratic_variances,
        )?;

        // Enforce exact symmetry up to machine‐precision:
        let n = covariance_assets.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (covariance_assets[i][j] + covariance_assets[j][i]);
                covariance_assets[i][j] = avg;
                covariance_assets[j][i] = avg;
            }
        }

        // Add a tiny “jitter” to the diagonal so it’s numerically positive‑definite:
        for i in 0..n {
            covariance_assets[i][i] += 1e-8;
        }

        // now flatten in row‑major order:
        let flat_cov: Vec<f64> = covariance_assets.iter().flat_map(|row| row.iter().cloned()).collect();

        // Pass to distrib
        let normal_distribution = MultivariateNormal::new(
            mu_assets.clone(),
            covariance_assets.clone().into_iter().flatten().collect(),
        )
        .map_err(|e| format!("Failed to create MVN: {}", e))?;

        Ok(Sampler::FactorModel {
            assets_under_management,
            periods_to_sample,
            number_of_factors,
            mu_factors,
            covariance_factors,
            loadings,
            idiosyncratic_variances,
            mu_assets,
            covariance_assets,
            normal_distribution,
            rng,
            seed,
        })
    }

    pub fn run_batch_factor_model_maker(
        mu_assets: &[f64],
        covariance_assets: &[f64],
        periods_to_sample: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let normal_sampler = Self::normal(mu_assets, &covariance_assets, periods_to_sample, seed)?;
        let (periods_to_sample, normal_distribution, rng, seed) = {
            if let Self::Normal {
                periods_to_sample,
                normal_distribution,
                rng,
                seed,
            } = normal_sampler
            {
                (periods_to_sample, normal_distribution, rng, seed)
            } else {
                panic!(
                    "Schema of Normal Distribution sampler was changed without updating related function."
                )
            }
        };

        let unflatten_square = |flat: Vec<f64>, n: usize| -> Vec<Vec<f64>> {
            assert_eq!(flat.len(), n * n, "flat.len() must be n*n");
            flat.chunks(n) // iterator over &[f64] slices of length n
                .map(|row| row.to_vec())
                .collect() // Vec<Vec<f64>>
        };

        Ok(Sampler::FactorModel {
            // we don't care since the important info is passed by the asset_means implicitly
            // or is only useful for logging/construction
            assets_under_management: mu_assets.len(),
            number_of_factors: 0,
            mu_factors: vec![0.],
            covariance_factors: vec![vec![0.]],
            loadings: vec![vec![0.]],
            idiosyncratic_variances: vec![0.],

            // Given but yet again we don't care?
            mu_assets: mu_assets.to_vec(),
            covariance_assets: unflatten_square(covariance_assets.to_vec(), mu_assets.len()),

            // This is what is actually used for sampling
            periods_to_sample,
            normal_distribution,
            rng,
            seed,
        })
    }

    pub fn normal(
        means: &[f64],
        cov: &[f64],
        periods_to_sample: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let seed = seed.unwrap_or_else(|| OsRng.next_u64());
        let rng = ChaCha20Rng::seed_from_u64(seed);

        let normal_distribution = MultivariateNormal::new(means.to_vec(), cov.to_vec())
            .map_err(|e| format!("Failed to create distribution for normal sampler: {}", e))?;

        Ok(Sampler::Normal {
            periods_to_sample,
            normal_distribution,
            rng,
            seed,
        })
    }

    fn generate_covariance_matrix(number_of_factors: usize) -> Result<Vec<Vec<f64>>, String> {
        if number_of_factors == 0 {
            return Err("Number of factors must be greater than zero".to_string());
        }
        let mut rng = ChaCha20Rng::from_entropy();

        let uniform = Uniform::new(0.01, 0.2);

        let mut m = vec![vec![0.0; number_of_factors]; number_of_factors];
        for i in 0..number_of_factors {
            m[i][i] = rng.sample(uniform);
        }
        Ok(m)
    }

    fn compute_asset_covariance(
        loadings: &Vec<Vec<f64>>,
        covariance_factors: &Vec<Vec<f64>>,
        idiosyncratic_variances: &Vec<f64>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let assets = loadings.len();
        let factors = covariance_factors.len();
        if loadings[0].len() != factors {
            return Err("Incompatible dimensions".to_string());
        }
        if idiosyncratic_variances.len() != assets {
            return Err("Incompatible dimensions".to_string());
        }

        let mut bsf = vec![vec![0.0; factors]; assets];
        for i in 0..assets {
            for j in 0..factors {
                for k in 0..factors {
                    bsf[i][j] += loadings[i][k] * covariance_factors[k][j];
                }
            }
        }

        let mut cov = vec![vec![0.0; assets]; assets];
        for i in 0..assets {
            for j in 0..assets {
                for k in 0..factors {
                    cov[i][j] += bsf[i][k] * loadings[j][k];
                }
            }
        }

        for i in 0..assets {
            cov[i][i] += idiosyncratic_variances[i];
        }
        Ok(cov)
    }

    pub fn sample_returns(&mut self) -> Vec<Vec<f64>> {
        match self {
            Sampler::FactorModel {
                normal_distribution,
                periods_to_sample,
                rng,
                ..
            } => normal_distribution
                .clone()
                .sample_iter(rng)
                .take(*periods_to_sample)
                .map(|row| row.as_slice().to_vec())
                .collect(),

            Sampler::Normal {
                normal_distribution,
                periods_to_sample,
                rng,
                ..
            } => normal_distribution
                .clone()
                .sample_iter(rng)
                .take(*periods_to_sample)
                .map(|row| row.as_slice().to_vec())
                .collect(),

            Sampler::SeriesGAN(periods_to_sample) => vec![vec![1.0]; *periods_to_sample],
        }
    }

    pub fn reseed(&mut self, new_seed: u64) {
        match self {
            Sampler::FactorModel { rng, seed, .. } | Sampler::Normal { rng, seed, .. } => {
                *rng = ChaCha20Rng::seed_from_u64(new_seed);
                *seed = new_seed;
            }
            _ => {}
        }
    }

    pub fn find_min_max(raw_sequence: &[Vec<f64>]) -> Result<(Vec<(f64, f64)>, usize), String> {
        if raw_sequence.is_empty() {
            return Err("Passed an empty sequence".to_string());
        }
        let dim = raw_sequence[0].len();
        let mut mm = vec![(f64::INFINITY, f64::NEG_INFINITY); dim];
        for row in raw_sequence {
            for (i, &v) in row.iter().enumerate() {
                let (min, max) = &mut mm[i];
                *min = (*min).min(v);
                *max = (*max).max(v);
            }
        }
        Ok((mm, dim))
    }
}

// ---- This is the converting function for SamplerConfig (Protobufs)
impl From<RustSampler> for SamplerConfig {
    fn from(s: RustSampler) -> Self {
        match s {
            RustSampler::FactorModel {
                mu_assets,
                covariance_assets,
                ..
            } => {
                // flatten the Vec<Vec<f64>> into Vec<f64> (row‑major)
                let flat_cov = covariance_assets
                    .into_iter()
                    .flatten()
                    .collect::<Vec<f64>>();

                SamplerConfig {
                    kind: Some(Kind::FactorModel(FactorModelConfig {
                        mu_assets,
                        covariance_assets: flat_cov,
                    })),
                }
            }
            RustSampler::Normal {
                normal_distribution,
                ..
            } => {
                // we should be passed a well form sampler.
                let means: Vec<f64> = normal_distribution.mean().unwrap().as_slice().to_vec();
                let cov = normal_distribution.variance().unwrap().as_slice().to_vec();
                SamplerConfig {
                    kind: Some(Kind::Normal(NormalConfig { means, cov })),
                }
            }
            RustSampler::SeriesGAN(periods) => SamplerConfig {
                kind: Some(Kind::SeriesGan(SeriesGanConfig {
                    periods: periods as u32,
                })),
            },
        }
    }
}

mod u128_string {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(val: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // write the u128 as a decimal string
        serializer.serialize_str(&val.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        // first deserialize a string, then parse
        let s = String::deserialize(deserializer)?;
        s.parse::<u128>()
            .map_err(|e| serde::de::Error::custom(format!("invalid u128: {}", e)))
    }
}

// ---- This is the the serializable Sampler to pass around
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SerdeSampler {
    FactorModel {
        #[serde(default = "default_version")]
        version: u32,

        assets_under_management: usize,
        periods_to_sample: usize,
        number_of_factors: usize,

        mu_factors: Vec<f64>,
        covariance_factors: Vec<Vec<f64>>,
        loadings: Vec<Vec<f64>>,
        idiosyncratic_variances: Vec<f64>,

        mu_assets: Vec<f64>,
        covariance_assets: Vec<Vec<f64>>,

        /// Original seed used to initialize the RNG
        seed: u64,
        /// ChaCha20 “stream” / nonce
        stream: u64,
        /// ChaCha20 word‐position
        #[serde(with = "u128_string")]
        word_pos: u128,
    },

    Normal {
        #[serde(default = "default_version")]
        version: u32,

        periods_to_sample: usize,
        means: Vec<f64>,
        covariance: Vec<Vec<f64>>,

        seed: u64,
        stream: u64,
        #[serde(with = "u128_string")]
        word_pos: u128,
    },

    SeriesGan {
        #[serde(default = "default_version")]
        version: u32,
        periods_to_sample: usize,
    },
}

// This converts from SerdeSampler to Sampler
impl From<Sampler> for SerdeSampler {
    fn from(s: Sampler) -> Self {
        match s {
            Sampler::FactorModel {
                assets_under_management,
                periods_to_sample,
                number_of_factors,
                mu_factors,
                covariance_factors,
                loadings,
                idiosyncratic_variances,
                mu_assets,
                covariance_assets,
                rng,
                seed,
                ..
            } => {
                let stream = rng.get_stream();
                let word_pos = rng.get_word_pos();
                SerdeSampler::FactorModel {
                    version: CURRENT_SERIALIZER_VERSION,
                    assets_under_management,
                    periods_to_sample,
                    number_of_factors,
                    mu_factors,
                    covariance_factors,
                    loadings,
                    idiosyncratic_variances,
                    mu_assets,
                    covariance_assets,
                    seed,
                    stream,
                    word_pos,
                }
            }

            Sampler::Normal {
                periods_to_sample,
                normal_distribution,
                rng,
                seed,
            } => {
                let stream = rng.get_stream();
                let word_pos = rng.get_word_pos();
                let means: Vec<f64> = normal_distribution.mean().unwrap().as_slice().to_vec();
                let var: Vec<f64> = normal_distribution.variance().unwrap().as_slice().to_vec();
                let dim = means.len();
                let covariance = var.chunks(dim).map(|r| r.to_vec()).collect::<Vec<_>>();

                SerdeSampler::Normal {
                    version: CURRENT_SERIALIZER_VERSION,
                    periods_to_sample,
                    means,
                    covariance,
                    seed,
                    stream,
                    word_pos,
                }
            }

            Sampler::SeriesGAN(periods) => SerdeSampler::SeriesGan {
                version: CURRENT_SERIALIZER_VERSION,
                periods_to_sample: periods,
            },
        }
    }
}

impl TryFrom<SerdeSampler> for Sampler {
    type Error = String;

    fn try_from(ss: SerdeSampler) -> Result<Self, Self::Error> {
        match ss {
            SerdeSampler::FactorModel {
                version,
                assets_under_management,
                periods_to_sample,
                number_of_factors,
                mu_factors,
                covariance_factors,
                loadings,
                idiosyncratic_variances,
                mu_assets,
                covariance_assets,
                seed,
                stream,
                word_pos,
            } => {
                if version != CURRENT_SERIALIZER_VERSION {
                    eprintln!(
                        "Warning: deserializing sampler version {} but code expects {}",
                        version, CURRENT_SERIALIZER_VERSION
                    );
                }
                let mut rng = ChaCha20Rng::seed_from_u64(seed);
                rng.set_stream(stream);
                rng.set_word_pos(word_pos);

                let flat_cov = covariance_assets.clone().into_iter().flatten().collect();
                let normal_distribution = MultivariateNormal::new(mu_assets.clone(), flat_cov)
                    .map_err(|e| format!("Rebuild MVN failed: {}", e))?;

                Ok(Sampler::FactorModel {
                    assets_under_management,
                    periods_to_sample,
                    number_of_factors,
                    mu_factors,
                    covariance_factors,
                    loadings,
                    idiosyncratic_variances,
                    mu_assets,
                    covariance_assets,
                    normal_distribution,
                    rng,
                    seed,
                })
            }

            SerdeSampler::Normal {
                version,
                periods_to_sample,
                means,
                covariance,
                seed,
                stream,
                word_pos,
            } => {
                if version != CURRENT_SERIALIZER_VERSION {
                    eprintln!(
                        "Warning: deserializing sampler version {} but code expects {}",
                        version, CURRENT_SERIALIZER_VERSION
                    );
                }
                let mut rng = ChaCha20Rng::seed_from_u64(seed);
                rng.set_stream(stream);
                rng.set_word_pos(word_pos);

                let flat_cov = covariance.into_iter().flatten().collect::<Vec<_>>();
                let normal_distribution = MultivariateNormal::new(means.clone(), flat_cov)
                    .map_err(|e| format!("Rebuild Normal MVN failed: {}", e))?;

                Ok(Sampler::Normal {
                    periods_to_sample,
                    normal_distribution,
                    rng,
                    seed,
                })
            }

            SerdeSampler::SeriesGan {
                version,
                periods_to_sample,
            } => {
                if version != CURRENT_SERIALIZER_VERSION {
                    eprintln!(
                        "Warning: deserializing sampler version {} but code expects {}",
                        version, CURRENT_SERIALIZER_VERSION
                    );
                }
                Ok(Sampler::SeriesGAN(periods_to_sample))
            }
        }
    }
}

// ---- Actual Serialize/Deserialize implementation using Serde sampler
impl Serialize for Sampler {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let helper: SerdeSampler = self.clone().into();
        helper.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Sampler {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = SerdeSampler::deserialize(deserializer)?;
        helper.try_into().map_err(DeError::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_covariance_matrix_valid_size() {
        let number_of_factors = 3;
        let result = Sampler::generate_covariance_matrix(number_of_factors);
        assert!(result.is_ok());
        let cov = result.unwrap();
        assert_eq!(cov.len(), number_of_factors);
        for row in cov {
            assert_eq!(row.len(), number_of_factors);
        }
    }

    #[test]
    fn test_generate_covariance_matrix_zero() {
        assert!(Sampler::generate_covariance_matrix(0).is_err());
    }

    fn test_synthetica_factor_model_function() {
        let number_of_factors = 5;
        let result = Sampler::factor_model_synthetic();
    }
}
