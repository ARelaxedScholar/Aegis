use crate::Sampler::{Normal, SeriesGAN};
use nalgebra::base::dimension::{Const, Dyn};
use nalgebra::{DMatrix, DVector, Matrix, VecStorage};
use rand::prelude::*;
use statrs::distribution::{Continuous, MultivariateNormal};
use statrs::statistics::{MeanN, VarianceN};

// Sampler Code
enum Sampler {
    Normal(MultivariateNormal, usize),
    SeriesGAN(usize),
}

impl Sampler {
    fn sample(&self) -> Vec<DVector<f64>> {
        let mut rng = thread_rng();
        match self {
            Sampler::Normal(multivariate_normal_distribution, number_of_steps) => {
                multivariate_normal_distribution
                    .sample_iter(&mut rng)
                    .take(*number_of_steps)
                    .collect::<Vec<_>>()
            }
            Sampler::SeriesGAN(number_of_steps) => {
                //
                let mut rng = thread_rng();
                let dist = MultivariateNormal::new(vec![0.0, 1.0], vec![1., 0., 0., 1.])
                    .expect("Multivariate normal");
                dist.sample_iter(&mut rng)
                    .take(*number_of_steps)
                    .collect::<Vec<_>>()
            }
        }
    }
}

// Algo Code

fn main() {
    let mvn = MultivariateNormal::new(vec![0., 0.], vec![1., -0.5, -0.5, 1.])
        .expect("Wanted a multivariate normal");
    let normal_sampler = Sampler::Normal(mvn, 30);

    normal_sampler
        .sample()
        .iter()
        .for_each(|element| println!("{}", element));
}
