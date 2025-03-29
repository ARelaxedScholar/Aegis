pub mod sampling {
    use pyo3::pyclass;
    use rand::prelude::*; use rand::distributions::Uniform;
    use statrs::distribution::MultivariateNormal;
    use serde::{Serialize, Serializer, Deserialize, Deserializer};
use serde::de::{self, Visitor, SeqAccess, MapAccess};
use std::fmt;
    
    #[pyclass]
    pub struct PySampler {
        sampler: Sampler,
    }

    #[derive(Debug, Clone)]
    pub enum Sampler {
        FactorModel {
            assets_under_management: usize,
            periods_to_sample: usize,
            number_of_factors: usize,
        
            // Parameters of the factor model (with potential for Bayesian priors)
            mu_factors: Vec<f64>,
            covariance_factors: Vec<Vec<f64>>,
            loadings: Vec<Vec<f64>>,
            idiosyncratic_variances: Vec<f64>, // Added
        
            // Derived quantities (can be computed from the parameters)
            mu_assets: Vec<f64>,               
            covariance_assets: Vec<Vec<f64>>,   
        
            // The multivariate normal distribution (for sampling)
            normal_distribution: MultivariateNormal,
        },
        
        Normal {
            periods_to_sample: usize,
            normal_distribution: MultivariateNormal,
   
        },
        SeriesGAN(usize), // Might never be implemented
    }

    impl Sampler {
        fn factor_model_synthetic(
            assets_under_management: usize,
            number_of_factors: usize,
            periods_to_sample: usize,
        ) -> Result<Self, String> {
            if assets_under_management == 0 || number_of_factors == 0 {
                return Err("Assets and Factors should be positives".into())
            }


            let mut rng = rand::thread_rng();
            let small_returns = 0.001;
            let mu_factors: Vec<f64> = vec![small_returns; number_of_factors];
            let covariance_factors: Vec<Vec<f64>> = Self::generate_covariance_matrix(number_of_factors).unwrap();

            let throwaway = MultivariateNormal::new(mu_factors.clone(), covariance_factors.clone().into_iter().flatten().collect()).unwrap();
            let factor_returns = throwaway.sample(&mut rng);
            drop(throwaway); // we no longer need it

            let mut loadings: Vec<Vec<f64>> =
                vec![vec![0.0; number_of_factors]; assets_under_management];
    
            let uniform = Uniform::new(0.5, 1.5);
            for i in 0..assets_under_management {
                for j in 0..number_of_factors {
                    loadings[i][j] = rng.sample(uniform);
                }
            }

            // Random variance not explained by factors
            let mut idiosyncratic_variances: Vec<f64> = vec![0.01; assets_under_management];

            let mut mu_assets: Vec<f64> = vec![0.0; assets_under_management];
            for i in 0..assets_under_management {
                mu_assets[i] = loadings[i]
                    .iter()
                    .zip(factor_returns.iter())
                    .map(|(loading, factor)| loading * factor)
                    .sum::<f64>();
            }

            let covariance_assets = Self::compute_asset_covariance(&loadings, &covariance_factors, &idiosyncratic_variances)?;
                let normal_distribution = MultivariateNormal::new(mu_assets.clone(), covariance_assets.clone().into_iter().flatten().collect())
                .map_err(|e| format!("Failed to create MultivariateNormal: {}", e))?;

            Ok(Self::FactorModel {
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
            })
        }

        fn generate_covariance_matrix(number_of_factors: usize) -> Result<Vec<Vec<f64>>, String> {
            if number_of_factors == 0 {
                return Err("Number of factors must be greater than zero".to_string());
            }
        
            let mut rng = thread_rng();
            let uniform = Uniform::new(0.01, 0.2); // Chosen arbitrarily tbh
        
            let mut covariance_matrix: Vec<Vec<f64>> =
                vec![vec![0.0; number_of_factors]; number_of_factors];
        
            for i in 0..number_of_factors {
                covariance_matrix[i][i] = rng.sample(uniform); // Sample variances from uniform distribution
            }
        
            Ok(covariance_matrix)
        }

        fn compute_asset_covariance(
            loadings: &Vec<Vec<f64>>,
            covariance_factors: &Vec<Vec<f64>>,
            idiosyncratic_variances: &Vec<f64>,
        ) -> Result<Vec<Vec<f64>>, String> {
            let assets_under_management = loadings.len();
            let number_of_factors = covariance_factors.len();
        
            // Check dimensions
            if loadings[0].len() != number_of_factors {
                return Err("Incompatible dimensions: Number of columns in loadings must equal number of factors".to_string());
            }
        
            if idiosyncratic_variances.len() != assets_under_management {
                return Err("Incompatible dimensions: Number of idiosyncratic variances must equal number of assets".to_string());
            }
        
            // 1. Calculate B * Sigma_f (loadings * covariance_factors)
            let mut b_sigma_f: Vec<Vec<f64>> = vec![vec![0.0; number_of_factors]; assets_under_management];
            for i in 0..assets_under_management {
                for j in 0..number_of_factors {
                    for k in 0..number_of_factors {
                        b_sigma_f[i][j] += loadings[i][k] * covariance_factors[k][j];
                    }
                }
            }
        
            // 2. Calculate (B * Sigma_f) * B.transpose()
            let mut b_sigma_f_bt: Vec<Vec<f64>> = vec![vec![0.0; assets_under_management]; assets_under_management];
            for i in 0..assets_under_management {
                for j in 0..assets_under_management {
                    for k in 0..number_of_factors {
                        b_sigma_f_bt[i][j] += b_sigma_f[i][k] * loadings[j][k];
                    }
                }
            }
        
            // 3. Add the idiosyncratic variances to the diagonal
            for i in 0..assets_under_management {
                b_sigma_f_bt[i][i] += idiosyncratic_variances[i];
            }
        
            Ok(b_sigma_f_bt)
        }

        fn normal(means: Vec<f64>, cov: Vec<f64>, periods_to_sample: usize) -> Self {
            let normal_distribution = MultivariateNormal::new(means, cov).unwrap();

            Self::Normal {
                normal_distribution,
                periods_to_sample,
            }
        }
    }

    impl Sampler {
        /// Honestly, only the superverised normal needs the price scenarios
        /// and for theoretical reasons this is gibberish, so it will be revamped later.
        /// But for now we leave it as this.
        fn sample_price_scenario(&self) -> Vec<Vec<f64>> {
            let rng = thread_rng();
            match self {
                Sampler::FactorModel{
                    number_of_factors,
                    periods_to_sample,
                    assets_under_management, normal_distribution, mu_factors, covariance_factors, mu_assets, covariance_assets, loadings, idiosyncratic_variances
                }=> {
                    vec![vec![1.]]
                }
                Sampler::Normal {
                    normal_distribution,
                    periods_to_sample,
                } => vec![vec![1.]], // SHOULDN'T BE CALLED TRUTHFULLY,

                Sampler::SeriesGAN(periods_to_sample) => {
                    // NOT IMPLEMENTED (WILL WRITE IT SO THAT THERE"S A MODEL THAT WAS TRAINED TO GENERATE THESE, NOT A PRIORITY)
                    vec![vec![1.]]
                }
            }
        }
        /// sample_returns
        /// Takes method of Sampler object
        ///
        /// Returns a vector of vectors of f64.
        ///
        /// The goal is to sample returns according to different modalities.
        pub fn sample_returns(&self) -> Vec<Vec<f64>> {
            let mut rng = thread_rng();
            match self {
                Sampler::FactorModel{
                    assets_under_management,
                    number_of_factors,
                    normal_distribution,
                    loadings,
                    mu_factors,
                    covariance_factors,
                    mu_assets,
                    covariance_assets,
                    periods_to_sample,
                    idiosyncratic_variances,
                } => normal_distribution
                    .sample_iter(&mut rng)
                    .take(*periods_to_sample)
                    .map(|row| row.iter().cloned().collect())
                    .collect::<Vec<_>>(),
                Sampler::Normal {
                    normal_distribution,
                    periods_to_sample,
                } => normal_distribution
                    .sample_iter(&mut rng)
                    .take(*periods_to_sample)
                    .map(|row| row.iter().cloned().collect())
                    .collect::<Vec<_>>(),
                Sampler::SeriesGAN(usize) => {
                    // THE MOST IMPORTANT ONE, WHEN I AM DONE IMPLEMENTING THIS HOPEFULLY GENERATING GOOD PORTFOLIOS WILL BE EASIER
                    vec![vec![1.]] //FOR NOW RETURNS UNIT
                }
            }
        }

        pub fn find_min_max(raw_sequence: &[Vec<f64>]) -> Result<(Vec<(f64, f64)>, usize), String> {
            if raw_sequence.is_empty() {
                return Err("Passed an empty sequence to supervisor".to_string());
            }
            let dimension = raw_sequence[0].len(); // access first row and then check the number of elements
                                                   // must be in this order so that any value is less than INFINITY, and any value is bigger than NEG_INFINITY
            let mut min_max = vec![(f64::INFINITY, f64::NEG_INFINITY); dimension];

            for row in raw_sequence.iter() {
                for (col_idx, &value) in row.iter().enumerate() {
                    let (min, max) = &mut min_max[col_idx];
                    *min = (*min).min(value);
                    *max = (*max).max(value);
                }
            }
            Ok((min_max, dimension))
        }
    }

impl Serialize for Sampler {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Sampler::FactorModel {
                assets_under_management,
                periods_to_sample,
                number_of_factors,
                normal_distribution: _, // Don't serialize this field
                mu_factors,
                covariance_factors,
                mu_assets,
                covariance_assets,
                loadings,
                idiosyncratic_variances, 
            } => {
                // Serialize the FactorModel variant, excluding the normal_distribution
                // You'll need to define a custom struct or tuple to represent the serialized form
                // and then serialize that.

                #[derive(Serialize)]
                struct FactorModelData {
                    assets_under_management: usize,
                    number_of_factors: usize,
                    mu_factors: Vec<f64>,
                    covariance_factors: Vec<f64>,
                    mu_assets: Vec<f64>,
                    covariance_assets: Vec<f64>,
                    loadings: Vec<Vec<f64>>,
                }
                let data = FactorModelData {
                    assets_under_management: *assets_under_management,
                    number_of_factors: *number_of_factors,
                    mu_factors: mu_factors.clone(),
                    covariance_factors: covariance_factors.clone().into_iter().flatten().collect(), // Flattening for serialization, adjust as needed
                    mu_assets: mu_assets.clone(),
                    covariance_assets: covariance_assets.clone().into_iter().flatten().collect(), // Flattening for serialization, adjust as needed
                    loadings: loadings.clone(),
                };
                data.serialize(serializer)
            }
            Sampler::Normal { periods_to_sample, .. } => {
                // Handle the Normal variant
                // Serialize the Normal variant, excluding the normal_distribution
                serializer.serialize_i32(*periods_to_sample as i32)
            }
            Sampler::SeriesGAN(usize) => {
              todo!()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Sampler {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
    {
        struct SamplerVisitor;

        impl<'de> Visitor<'de> for SamplerVisitor {
            type Value = Sampler;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Sampler enum")
            }

            fn visit_map<M>(self, access: M) -> Result<Self::Value, M::Error>
                where
                    M: MapAccess<'de>,
            {
                todo!()
            }
        }
        deserializer.deserialize_enum("Sampler", &["FactorModel", "Normal", "SeriesGAN"], SamplerVisitor)
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

        let covariance_matrix = result.unwrap();
        assert_eq!(
            covariance_matrix.len(),
            number_of_factors,
            "Covariance matrix should have the correct number of rows"
        );
        for row in &covariance_matrix {
            assert_eq!(
                row.len(),
                number_of_factors,
                "Covariance matrix should be square"
            );
        }
    }

    #[test]
    fn test_generate_covariance_matrix_positive_diagonal() {
        let number_of_factors = 3;
        let result = Sampler::generate_covariance_matrix(number_of_factors);

        assert!(result.is_ok());

        let covariance_matrix = result.unwrap();
        for i in 0..number_of_factors {
            assert!(
                covariance_matrix[i][i] > 0.0,
                "Diagonal elements should be positive"
            );
        }
    }

    #[test]
    fn test_generate_covariance_matrix_zero_factors() {
        let number_of_factors = 0;
        let result = Sampler::generate_covariance_matrix(number_of_factors);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Number of factors must be greater than zero".to_string()
        );
    }
}
}


