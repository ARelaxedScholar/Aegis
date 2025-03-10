pub mod sampling {
    use rand::prelude::*;
    use serde::{Deserialize, Serialize};
    use statrs::distribution::MultivariateNormal;

    #[pyclass]
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub enum Sampler {
        #[serde(skip)]
        Normal {
            normal_distribution: MultivariateNormal,
            periods_to_sample: usize,
        },
        SeriesGAN(usize),
    }

    impl Sampler {
        /// Honestly, only the superverised normal needs the price scenarios
        /// and for theoretical reasons this is gibberish, so it will be revamped later.
        /// But for now we leave it as this.
        fn sample_price_scenario(&self) -> Vec<Vec<f64>> {
            let rng = thread_rng();
            match self {
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
