use crate::evolution::objective::{OptimizationDirection, OptimizationObjective};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub id: usize,
    pub rank: Option<usize>,
    pub crowding_distance: Option<f64>,
    pub weights: Vec<f64>,
    pub objectives: Vec<Box<dyn OptimizationObjective>>,
    pub stats: Vec<f64>,
}

impl PartialEq for Portfolio {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl PartialOrd for Portfolio {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // if ID is the same return equal
        if self.id == other.id {
            return Some(std::cmp::Ordering::Equal);
        }

        // Compare based on rank
        if self.rank != other.rank {
            return self.rank.partial_cmp(&other.rank);
        }
        // If Rank is the same compare based on Crowding_Distance
        match (self.crowding_distance, other.crowding_distance) {
            (Some(self_distance), Some(other_distance)) => {
                self_distance.partial_cmp(&other_distance)
            }
            _ => panic!("Crowding distance is None for one or both portfolios."), //shouldn't happen after processing
        }
    }
}
impl Eq for Portfolio {}
static PORTFOLIO_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Portfolio {
    pub fn new(
        weights: Vec<f64>,
        objective: Vec<dyn OptimizationObjective>,
        stats: Vec<f64>,
    ) -> Self {
        let id = PORTFOLIO_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        let rank = None;
        let crowding_distance = None;
        Portfolio {
            id,
            rank,
            crowding_distance,
            weights,
            objectives,
            stats,
        }
    }

    pub fn turn_objectives_to_score(&self) -> Vec<f64> {
        let mut scores = vec![0.; self.objectives.len()];

        for (i, Box::<objective>) in self.objectives.iter().enumerate() {
            let to_push = match self.objective.direction() {
                OptimizationDirection::Maximize => self.stats[i],
                OptimizationDirection::Minimize => -self.stats[i],
            };
            scores.push(to_push);
        }
        scores
    }

    pub fn is_dominated_by(&self, other: &Portfolio) -> bool {
        let self_metrics = self.turn_objectives_to_score();
        let other_metrics = other.turn_objectives_to_score();

        // Check if 'other' is at least as good as 'self' in all objectives
        let other_is_at_least_as_good_in_all = self_metrics
            .iter()
            .zip(other_metrics.iter())
            .all(|(&self_metric, &other_metric)| other_metric >= self_metric);

        // Check if 'other' is strictly better than 'self' in at least one objective
        let other_is_strictly_better_in_one = self_metrics
            .iter()
            .zip(other_metrics.iter())
            .any(|(&self_metric, &other_metric)| other_metric > self_metric);

        // 'self' is dominated by 'other' if 'other' is at least as good in all
        // objectives AND strictly better in at least one objective
        other_is_at_least_as_good_in_all && other_is_strictly_better_in_one
    }
}
