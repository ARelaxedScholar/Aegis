use core::f64;

use crate::Sampler::{Normal, SupervisedNormal, _SeriesGAN};
use nalgebra::base::dimension::{Const, Dyn};
use nalgebra::{DMatrix, DVector, Matrix, VecStorage};
use rand::distributions::Uniform;
use rand::{prelude::*, Error};
use statrs::distribution::{Continuous, MultivariateNormal};
use statrs::statistics::{MeanN, VarianceN};
use std::sync::LazyLock;
use tch::{nn, CModule, Kind, TchError, Tensor};

static SUPERVISOR: LazyLock<Result<CModule, TchError>> =
    LazyLock::new(|| CModule::load("../model_weights/supervisor.pt"));
static SUPERVISOR_INPUT_DIM = 4;//Trained at that dimension so I can't take more or less than that.

// // Sampler Code
enum Sampler {
    Normal {
        normal_distribution: MultivariateNormal,
        days_to_sample: usize,
    },
    SupervisedNormal {
        normal_distribution: MultivariateNormal,
        days_to_sample: usize,
        look_ahead: usize,
    }, //uses the supervisor to supervise a normal so that hopefully it has the required temporal characteristics
    _SeriesGAN(usize),
}

impl Sampler {
    fn sample(&self) -> Vec<DVector<f64>> {
        let mut rng = thread_rng();
        match self {
            Sampler::Normal {
                normal_distribution,
                days_to_sample,
            } => normal_distribution
                .sample_iter(&mut rng)
                .take(*days_to_sample)
                .collect::<Vec<_>>(),
            Sampler::SupervisedNormal {
                normal_distribution,
                days_to_sample,
                look_ahead,
            } => {
                let raw_normal_sequence = normal_distribution
                    .sample_iter(&mut rng)
                    .take(*days_to_sample + look_ahead)
                    .collect::<Vec<_>>();
                Sampler::supervise_sequence(raw_normal_sequence, *look_ahead) //returns supervised_sequence
            }
            Sampler::_SeriesGAN(days_to_sample) => {
                //
                let mut rng = thread_rng();
                let dist = MultivariateNormal::new(vec![0.0, 1.0], vec![1., 0., 0., 1.])
                    .expect("Multivariate normal");
                dist.sample_iter(&mut rng)
                    .take(*days_to_sample)
                    .collect::<Vec<_>>()
            }
        }
    }
    // SUPERVISOR FUNCTIONS
    fn find_min_max(raw_sequence: &[DVector<f64>]) -> Result<(Vec<(f64, f64)>, usize), String> {
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
    fn supervisor_pass(
        preprocessed_sequence: Vec<DVector<f64>>,
        look_ahead: usize,
        number_of_assets: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        if preprocessed_sequence.len() < look_ahead {
            return Err(
                "A sequence less than LOOK_BACK was passed to the supervisor pass.".to_string(),
            );
        }
        // Prepare Slices for Training (reversed to match GRUs expectation)
        let (inputs, input_lengths): (Vec<&[DVector<f64>]>, Vec<usize>) = (1
            ..(preprocessed_sequence.len() + 1 - look_ahead))
            .map(|i| (&preprocessed_sequence[..i], i))
            .rev()
            .unzip();

        let extract_from_vec_storage =
            |vec_storage: nalgebra::DVector<f64>| vec_storage.data.as_vec().clone();

        let vectorized_inputs = inputs
            .iter()
            .map(|input_sequence| {
                input_sequence
                    .iter()
                    .map(|stuff_inside| extract_from_vec_storage(stuff_inside.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let max_length = input_lengths[0];

        let tensor_vector = vectorized_inputs
            .iter()
            .zip(input_lengths.iter())
            .map(|(input_sequence, &length)| {
                Tensor::from_slice(
                    &input_sequence
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<f64>>(),
                )
                .reshape([length as i64, number_of_assets as i64])
                .pad([0, 0, 0, (max_length - length) as i64], "constant", 0.0)
            })
            .collect::<Vec<_>>();

        let stacked_tensor = Tensor::stack(&tensor_vector, 0);
        let lengths_tensor = Tensor::from_slice(
            &input_lengths
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>(),
        );
        let supervisor_input = vec![
            stacked_tensor.to_kind(Kind::Float),
            lengths_tensor.to_kind(Kind::Int64),
        ];
        // Note to me of tomorrow: Right now wrote the code for stacking and padding logic, review to make sure it makes sense.
        // Then pass the thing to supervisor, stack the outputs into one vector and return that.
        let outputs = SUPERVISOR
            .as_ref()
            .expect("The supervisor model to be used")
            .forward_ts(&supervisor_input);

        Ok(Vec::<Vec<f64>>::try_from(outputs.expect("Sequence"))
            .expect("The OK variant of my Converted Supervised Sequence: "))
    }
    fn supervise_sequence(raw_sequence: Vec<DVector<f64>>, look_ahead: usize) -> Vec<DVector<f64>> {
        let (min_max_columns, number_of_assets) = Sampler::find_min_max(&raw_sequence)
            .expect("A list of tuples containing the Min-Max of each column");

        // // Define helper functions
        let min_max_scaling = |value: f64, min: f64, max: f64| (value - min) / (max - min);
        let undo_min_max_scaling =
            |scaled_value: f64, min: f64, max: f64| scaled_value * (max - min) + min;

        // Scaling is necessary since supervisor was trained on scaled sequences
        let min_max_scaled_sequence: Vec<_> = raw_sequence
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
        let supervised_sequence =
            Sampler::supervisor_pass(min_max_scaled_sequence, look_ahead, number_of_assets)
                .expect("The supervised sequence.");

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

    // For each portfolio we sample from a Uniform and then normalize
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

    // We'd want to compute for each day
    let generation_data = Vec::new();
    for generation in 0..generations {
        // Sample the data for this generation
        let sample_days = sampler.sample();

        // Compute the metrics for each element in the portfolio
        // based on this data.
    }
}
// Algo Code

fn main() {
    let mvn = MultivariateNormal::new(
        vec![0., 0., 0., 0.],
        vec![
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ],
    )
    .expect("Wanted a multivariate normal");
    let mvn_dimension = mvn.mean().expect("VecStorage of means").len());
    let normal_sampler = Sampler::SupervisedNormal {
        normal_distribution: mvn,
        days_to_sample: 30,
        look_ahead: 2,
    };
    evolve_portfolios(1, 1, 100, 4, normal_sampler);

    let test = |x: f64| {
        println!("{}", x);
    };
    test(5.);
}
