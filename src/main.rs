use std::f64::MAX_EXP;

use crate::Sampler::{Normal, SupervisedNormal, _SeriesGAN};
use nalgebra::base::dimension::{Const, Dyn};
use nalgebra::{DMatrix, DVector, Matrix, VecStorage};
use rand::distributions::Uniform;
use rand::prelude::*;
use statrs::distribution::{Continuous, MultivariateNormal};
use statrs::statistics::{MeanN, VarianceN};
use std::sync::LazyLock;
use tch::{CModule, TchError};

static SUPERVISOR: LazyLock<Result<CModule, TchError>> =
    LazyLock::new(|| CModule::load("/model_weights/supervisor.pt"));

// // Sampler Code
enum Sampler {
    Normal(MultivariateNormal, usize),
    SupervisedNormal(MultivariateNormal, usize, usize), //uses the supervisor to supervise a normal so that hopefully it has the required temporal characteristics
    _SeriesGAN(usize),
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
            Sampler::SupervisedNormal(
                multivariate_normal_distribution,
                number_of_steps,
                look_ahead,
            ) => {
                let raw_normal_sequence = multivariate_normal_distribution
                    .sample_iter(&mut rng)
                    .take(*number_of_steps + look_ahead)
                    .collect::<Vec<_>>();
                let supervised_sequence =
                    Sampler::supervise_sequence(raw_normal_sequence, *look_ahead);
                supervised_sequence
            }
            Sampler::_SeriesGAN(number_of_steps) => {
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
    // SUPERVISOR FUNCTIONS
    fn find_min_max(raw_sequence: &Vec<DVector<f64>>) -> Vec<(f64, f64)> {
        vec![(1., 1.)]
    }
    fn supervisor_pass(
        preprocessed_sequence: Vec<DVector<f64>>,
        look_ahead: usize,
    ) -> Vec<Vec<f64>> {
        vec![vec![1., 1., 1.]]
    }
    fn supervise_sequence(raw_sequence: Vec<DVector<f64>>, look_ahead: usize) -> Vec<DVector<f64>> {
        let min_max_columns = Sampler::find_min_max(&raw_sequence);

        // // Define helper functions
        let min_max_scaling = |value: f64, min: f64, max: f64| (value - min) / (max - min);
        let undo_min_max_scaling =
            |scaled_value: f64, min: f64, max: f64| scaled_value * (max - min) + min;

        // Scaling is necessary since supervisor was trained on scaled sequences
        let min_max_scaled_sequence = raw_sequence
            .iter()
            .map(|row| {
                // min-max scale all the entries
                let scaled_row = row
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        // presumes min_max_columns contains tuples s.t (min, max)
                        min_max_scaling(value, min_max_columns[i].0, min_max_columns[i].1)
                    })
                    .collect();
                DVector::from_vec(scaled_row)
            })
            .collect();

        // Do Supervisor Pass to Supervise (predict 2 bits ahead until done)
        let supervised_sequence = Sampler::supervisor_pass(min_max_scaled_sequence, look_ahead);

        // Undo Scaling
        let supervised_restored_sequence = supervised_sequence
            .iter()
            .map(|scaled_row| {
                let unscaled_row = scaled_row
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        undo_min_max_scaling(value, min_max_columns[i].0, min_max_columns[i].1)
                    })
                    .collect();
                DVector::from_vec(unscaled_row)
            })
            .collect();

        supervised_restored_sequence
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
    let population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| {
            let mut portfolio = rng
                .clone()
                .sample_iter(uniform)
                .take(assets_under_management)
                .collect::<Vec<f64>>();
            let magnitude = portfolio.iter().sum::<f64>();

            portfolio.iter_mut().map(|x| *x / magnitude).collect()
        })
        .collect::<Vec<_>>();

    for generation in 0..generations {
        // Sample the data for this generation
        let sample_days = sampler.sample();

        // Compute the metrics for each element in the portfolio
        // based on this data.
    }
}
// Algo Code

fn main() {
    let mvn = MultivariateNormal::new(vec![0., 0.], vec![1., -0.5, -0.5, 1.])
        .expect("Wanted a multivariate normal");
    let normal_sampler = Sampler::Normal(mvn, 30);
    evolve_portfolios(1, 10, 100, 4, normal_sampler);

    let test = |x: f64| {
        println!("{}", x);
    };
    test(5.);
}
