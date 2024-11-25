use crate::Sampler::{Normal, SeriesGAN};
use nalgebra::base::dimension::{Const, Dyn};
use nalgebra::{DMatrix, DVector, Matrix, VecStorage};
use rand::distributions::Uniform;
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

// Algo Logic
// 1. Generate a bunch of portfolio for the first batch based on a parameter passed (initial training size)
// 2. Sample based on the provided length using whichever method was specified by the user
// 3. Run the simulation computing every day the daily returns, and whatnot. <- Target for multithreading if possible
// 4. At the end use a Pareto front optimization strategy to select from a sample of non-dominated solutions and let them explore
// 4.i To an extent we'd also want the ability to explore so we'd have a randomly chosen percentage between 0 and 15 percent of dominated
// 4.ii solutions that would be allowed to reproduce using crossover.
// 4.iii We'd also mutate the allocation stochastically in such away tha randomly an increment is done to one component (and equivalent decrement is done to another)
// 5. Repeat until golden brown. lel
//
// This will be made into a python endpoint to write the code in Python simpler.
fn evolve_portfolios(
    time_horizon: usize,
    generations: usize,
    population_size: usize,
    assets_under_management: usize,
    sampler: Sampler,
) {
    // Initialization Phase
    let rng = thread_rng();
    let uniform = Uniform::new(0., 1.);
    let initial_population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| {
            rng.clone()
                .sample_iter(uniform)
                .take(assets_under_management)
                .collect::<Vec<f64>>()
        })
        .collect();

    dbg!(initial_population);
}
// Algo Code

fn main() {
    let mvn = MultivariateNormal::new(vec![0., 0.], vec![1., -0.5, -0.5, 1.])
        .expect("Wanted a multivariate normal");
    let normal_sampler = Sampler::Normal(mvn, 30);
    evolve_portfolios(1, 1, 10, 1, normal_sampler);
}
