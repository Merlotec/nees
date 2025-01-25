use std::cmp::Ordering;
use std::fs;
use rand::distributions::{Distribution, Open01, Uniform};
use rand::distributions::uniform::SampleUniform;
use rand_distr::{Beta, LogNormal, Normal, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::DeserializeOwned;
use crate::distribution::DistributionParams;
use crate::world::{House, Household, School, World};
use super::{Item, Agent};
use serde_big_array::BigArray;
use serde_with::serde_as;
use crate::distribution;

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

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimItemRecord<const D: usize, F: num::Float + Serialize + DeserializeOwned> {
    /// Store quality here to improve cache locality and avoid having to look up quality from school.
    #[serde_as(as = "[_; D]")]
    pub quality: [F; D],
}

impl<const D: usize, F: num::Float + Serialize + DeserializeOwned> From<DimItem<D, F>> for DimItemRecord<D, F> {
    fn from(value: DimItem<D, F>) -> Self {
        Self {quality: value.quality}
    }
}

impl<const D: usize, F: num::Float + Serialize + DeserializeOwned> From<DimItemRecord<D, F>> for DimItem<D, F> {
    fn from(value: DimItemRecord<D, F>) -> Self {
        Self {quality: value.quality}
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CDAgentRecord<const D: usize, F: num::Float + Serialize> {
    /// Store quality here to improve cache locality and avoid having to look up quality from school.
    pub id: usize,
    pub income: F,
    #[serde_as(as = "[_; D]")]
    pub pref_params: [F; D],
}

impl<const D: usize, F: num::Float + Serialize + DeserializeOwned> From<CDAgentRecord<D, F>> for CDAgent<D, F> {
    fn from(value: CDAgentRecord<D, F>) -> Self {
        Self {id: value.id, income: value.income, pref_params: value.pref_params}
    }
}

impl<const D: usize, F: num::Float + Serialize + DeserializeOwned> From<CDAgent<D, F>> for CDAgentRecord<D, F> {
    fn from(value: CDAgent<D, F>) -> Self {
        Self {id: value.id, income: value.income, pref_params: value.pref_params}
    }
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



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimWorld<I, A> {
    pub items: Vec<I>,
    pub agents: Vec<A>,
}

impl<I, A> DimWorld<I, A> {
    fn from_similar<I0, A0>(value: DimWorld<I0, A0>) -> Self
    where A: From<A0>, I: From<I0> {
        Self {items: value.items.into_iter().map(|x| I::from(x)).collect(), agents: value.agents.into_iter().map(|x| A::from(x)).collect()}
    }
}


pub fn create_cb_world<const D: usize, F: num::Float + SampleUniform, Q: Distribution<F>, P: Distribution<F>, Y: Distribution<F>>(n: usize, quality_distributions: [Q; D], pref_distributions: [P; D], income_distribution: Y) -> DimWorld<DimItem<D, F>, CDAgent<D, F>>
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
                pref_params = pref_distributions.iter().map(|x| x.sample(&mut rng).clamp(F::from(0.3 / D as f64).unwrap(), F::from(0.9).unwrap())).collect::<Vec<F>>().try_into().unwrap_or_else(|_| panic!("Failed to generate random distribution"));
            }
        }
        let income = income_distribution.sample(&mut rng).max(F::from(5.0).unwrap());
        agents.push(CDAgent::new(i, income, pref_params));
    }

    DimWorld { items, agents }
}

pub fn test_multidim<const D: usize, F: num::Float + SampleUniform + Serialize + DeserializeOwned>(n: usize)
where StandardNormal: Distribution<F> {
    let am = F::one() / F::from(D + 1).unwrap();
    let mut world: DimWorld<DimItem<D, F>, CDAgent<D, F>>;

    let settings = super::allocate::FractalSettings { epsilon: F::from(1e-8).unwrap(), max_iter: 400, constraint_price: F::zero() };

    if let Ok(s) = fs::read_to_string("config_multidim.json") {
        let ws: DimWorld<DimItemRecord<D, F>, CDAgentRecord<D, F>> = serde_json::from_str(s.as_str()).unwrap();
        world = DimWorld::from_similar(ws);
    } else {
        world = create_cb_world(n, std::array::from_fn(|_| Normal::new(F::from(0.5).unwrap(), F::from(0.1).unwrap()).unwrap()), std::array::from_fn(|_| Normal::new(am, am * F::from(0.1).unwrap()).unwrap()), Normal::new(F::from(100.0).unwrap(), F::from(50.0).unwrap()).unwrap());
        let ws: DimWorld<DimItemRecord<D, F>, CDAgentRecord<D, F>> = DimWorld::from_similar(world.clone());
        fs::write("config_multidim.json", serde_json::to_string_pretty(&ws).unwrap()).unwrap();
    }

    println!("Running algorithm (D={}, n={})", D, n);
    let mut allocations = super::allocate::root(world.agents, world.items, settings).unwrap();

    // Test to ensure verification works - should fail with this code.
    //allocations[0].set_price(F::from(1.0).unwrap());

    if super::verify_non_envy_configuration(&allocations, &settings) {
        println!("MULTIDIM NON-ENVY VERIFICATION SUCCESSFUL!!! (D={}, n={})", D, allocations.len());
    } else {
        println!("MULTIDIM NON-ENVY VERIFICATION FAILED :( (D={}, n={})", D, allocations.len());
    }
}