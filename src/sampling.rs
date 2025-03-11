pub mod sampling {
    use pyo3::pyclass;
    use rand::prelude::*;
    use serde::{Deserialize, Serialize};
    use statrs::distribution::MultivariateNormal;

    #[derive(Debug, Clone, Deserialize, Serialize)]
    #[pyclass]
    pub enum Sampler {
        FactorModel {
            // Calibrate on this (parameters and whatnot)
            assets_under_management: usize,
            number_of_factors: usize,
            normal_distribution: MultivariateNormal,
            // Kept for diagnostic purposes
            mu_factors: Vec<f64>,
            covariance_factors: Vec<f64>,
            mu_assets: Vec<f64>,
            covariance_assets: Vec<f64>,
            loadings: Vec<Vec<f64>>,
        },
        #[serde(skip)]
        Normal {
            // Evolve portfolio on this once you are confident with performance
            normal_distribution: MultivariateNormal,
            periods_to_sample: usize,
        },
        SeriesGAN(usize), // Might never be implemented
    }

    impl Sampler {
        fn factor_model(
            assets_under_management: usize,
            number_of_factors: usize,
            periods_to_sample: usize,
        ) {
            let small_returns = 0.001;
            let mu_factors: Vec<f64> = vec![small_returns; number_of_factors];
            let covariance_factors: Vec<Vec<f64>> = generate_covariance_matrix(number_of_factors);

            let throwaway = MultivariateNormal::new(mu_factors, covariance_factors).unwrap();
            let factor_returns: Vec<f64> = throwaway.sample();
            drop(throwaway); // we no longer need it

            let mut loadings: Vec<Vec<f64>> =
                vec![vec![0.0; number_of_factors]; assets_under_management];
            let mut rng = rand::rng();
            let uniform = Uniform::new(0.5, 1.5);
            for i in 0..assets_under_management {
                for j in 0..number_of_factors {
                    loadings[i][j] = rng.sample(uniform);
                }
            }

            // Random variance not explained by factors
            let mut idiosyncratic_variances: Vec<f64> = vec![0.0; assets_under_management];

            let mu_assets: Vec<f64> = vec![0.0; assets_under_management];
            for i in 0..assets_under_management {
                mu_assets[i] = loadings[i]
                    .iter()
                    .zip(factor_returns.iter())
                    .map(|(loading, factor)| loading * factor)
                    .sum::<f64>();
            }

            let covariance_assets: Vec<f64> =
                compute_asset_covariance(&loadings, &covariance_factors, &idiosyncratic_variances);
            let normal_distribution = MultivariateNormal::new(mu_assets, covariance_assets);

            FactorModel {
                assets_under_management,
                number_of_factors,
                mu_factors,
                normal_distribution,
                loadings,
                mu_factors,
                covariance_factors,
                mu_assets,
                covariance_assets,
            }
        }

        fn generate_covariance_matrix(number_of_factors: usize) {
            unimplemented!()
        }

        fn compute_asset_covariance(loadings : &Vec<f64>, covariance_factors : &Vec<f64>, &Vec<f64>) {
            unimplemented!()
        }

        fn normal(means: Vec<f64>, cov: Vec<f64>, periods_to_sample: usize) {
            let normal_distribution = MultivariateNormal::new(means, cov).unwrap();

            Normal {
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
                Sampler::FactorModel(
                    number_of_factors,
                    periods_to_sample,
                    assets_under_management,
                ) => {
                    let mu_factors = vec![.001; number_of_factors];
                    let sigma_factors = generate_covariance_matrix(number_of_factors);
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
}
