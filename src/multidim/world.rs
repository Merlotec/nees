use std::cmp::Ordering;
use rand::distributions::{Distribution, Open01, Uniform};
use rand::distributions::uniform::SampleUniform;
use rand_distr::{Beta, LogNormal, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use crate::distribution::DistributionParams;
use crate::world::{House, Household, School, World};
use super::{Item, Agent};

/// Cobb-Douglas agent with D dimensions and heterogeneous parameters.
#[derive(Debug, Clone)]
pub struct CDAgent<const D: usize, F: num::Float> {
    pub id: usize,
    pub income: F,
    pub pref_params: [F; D],
}

impl<const D: usize, F: num::Float> CDAgent<D, F> {
    /// Debug information about the household.
    pub fn new(id: usize, income: F, pref_params: [F; D]) -> Self {
        Self {
            id,
            income,
            pref_params,
        }
    }
}

impl<const D: usize, F: num::Float> Agent<D> for CDAgent<D, F> {
    type FloatType = F;
    /// Returns the item ID.
    fn agent_id(&self) -> usize {
        self.id
    }

    /// Returns the income of the agent.
    fn income(&self) -> F {
        self.income
    }

    /// Utility function used for this agent.
    fn utility(&self, price: F, quality: [F; D]) -> F {
        // Additively separable utility function.
        let mut param_sum = F::zero();
        for p in self.pref_params {
            param_sum = param_sum + p;
        }
        let mut acc = ((self.income - price) / F::from(1000.0).unwrap()).powf(F::one() - param_sum);

        for (q, p) in quality.iter().zip(self.pref_params.iter()) {
            acc = acc * q.powf(*p);
        }

        acc
    }


}

#[derive(Debug, Clone)]
pub struct DimItem<const D: usize, F: num::Float> {
    /// Store quality here to improve cache locality and avoid having to look up quality from school.
    pub quality: [F; D],
}

impl<const D: usize, F: num::Float> DimItem<D, F> {
    pub fn new(quality: [F; D]) -> Self {
        Self {
            quality,
        }
    }
}

impl<const D: usize, F: num::Float> Item<D> for DimItem<D, F> {
    type FloatType = F;

    /// Returns the quality of the school.
    fn quality(&self) -> [F; D] {
        self.quality
    }
}



#[derive(Debug, Clone)]
pub struct DimWorld<const D: usize, F: num::Float, I: Item<D, FloatType = F>, A: Agent<D, FloatType = F>> {
    pub items: Vec<I>,
    pub agents: Vec<A>,
}


pub fn create_cb_world<const D: usize, F: num::Float + SampleUniform, Q: Distribution<F>, P: Distribution<F>, Y: Distribution<F>>(n: usize, quality_distributions: [Q; D], pref_distributions: [P; D], income_distribution: Y) -> DimWorld<D, F, DimItem<D, F>, CDAgent<D, F>>
where StandardNormal: Distribution<F> {

    let mut items: Vec<DimItem<D, F>> = Vec::with_capacity(n);
    let mut agents: Vec<CDAgent<D, F>> = Vec::with_capacity(n);

    let mut rng = rand::thread_rng();

    for i in 0..n {
        let quality_params: [F; D] = quality_distributions.iter().map(|x| x.sample(&mut rng).max(F::from(0.05).unwrap())).collect::<Vec<F>>().try_into().unwrap_or_else(|_| panic!("Failed to generate random distribution"));
        items.push(DimItem::new(quality_params));

        let mut pref_params: [F; D]  = pref_distributions.iter().map(|x| x.sample(&mut rng).max(F::from(0.05).unwrap())).collect::<Vec<F>>().try_into().unwrap_or_else(|_| panic!("Failed to generate random distribution"));
        loop {
            let mut sum = F::zero();
            for p in pref_params {
                sum = sum + p;
            }

            if sum < F::from(0.95).unwrap() {
                break;
            } else {
                pref_params = pref_distributions.iter().map(|x| x.sample(&mut rng).clamp(F::from(0.1).unwrap(), F::from(0.9).unwrap())).collect::<Vec<F>>().try_into().unwrap_or_else(|_| panic!("Failed to generate random distribution"));
            }
        }
        let income = income_distribution.sample(&mut rng).max(F::from(5.0).unwrap());
        agents.push(CDAgent::new(i, income, pref_params));
    }

    DimWorld { items, agents }
}

pub fn test_multidim<const D: usize, F: num::Float + SampleUniform>(n: usize)
where StandardNormal: Distribution<F> {
    let am = F::one() / F::from(D + 1).unwrap();
    let world: DimWorld<D, F, _, _> = create_cb_world(n, std::array::from_fn(|_| Normal::new(F::from(0.5).unwrap(), F::from(0.1).unwrap()).unwrap()), std::array::from_fn(|_| Normal::new(am, am * F::from(0.1).unwrap()).unwrap()), Normal::new(F::from(100.0).unwrap(), F::from(50.0).unwrap()).unwrap());

    let settings = super::allocate::FractalSettings { epsilon: F::from(1e-8).unwrap(), max_iter: 400, constraint_price: F::zero() };

    let allocations = super::allocate::root(world.agents, world.items, settings).unwrap();

    if super::verify_solution(&allocations, &settings) {
        println!("MULTIDIM VERIFICATION SUCCESSFUL!!! (n={})", n);
    } else {
        println!("MULTIDIM VERIFICATION FAILED :(");
    }
}