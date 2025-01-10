use bevy::scene::ron::value::Float;
use utility::indifferent_price;
use crate::multidim::allocate::FractalSettings;

pub mod allocate;
pub mod utility;
pub mod world;
mod graph;

pub trait Agent<const D: usize> {
    type FloatType: num::Float;

    fn agent_id(&self) -> usize;
    fn income(&self) -> Self::FloatType;
    fn utility(&self, price: Self::FloatType, quality: [Self::FloatType; D]) -> Self::FloatType;
}

impl<A> Agent<1> for A
where
    A: crate::solver::Agent,
{
    type FloatType = <A as crate::solver::Agent>::FloatType;
    fn agent_id(&self) -> usize {
        <A as crate::solver::Agent>::agent_id(&self)
    }

    fn income(&self) -> Self::FloatType {
        <A as crate::solver::Agent>::income(&self)
    }
    fn utility(&self, price: Self::FloatType, quality: [Self::FloatType; 1]) -> Self::FloatType {
        <A as crate::solver::Agent>::utility(&self, price, quality[0])
    }
}

pub trait Item<const D: usize> {
    type FloatType: num::Float;

    fn quality(&self) -> [Self::FloatType; D];
}

impl<I> Item<1> for I
where
    I: crate::solver::Item,
{
    type FloatType = <I as crate::solver::Item>::FloatType;
    fn quality(&self) -> [Self::FloatType; 1] {
        [<I as crate::solver::Item>::quality(&self)]
    }
}

#[derive(Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Allocation<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
> {
    agent: A,
    item: I,

    price: F,
    utility: F,
}

impl<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
> std::fmt::Debug for Allocation<D, F, A, I>
where A: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Allocation: a_id: {}, a: {:?} , p: {}", self.agent().agent_id(), self.agent(), self.price().to_f32().unwrap())
    }
}

impl<
        F: num::Float,
        A: Agent<1, FloatType = F> + crate::solver::Agent<FloatType = F>,
        I: Item<1, FloatType = F> + crate::solver::Item<FloatType = F>,
    > Into<crate::solver::Allocation<F, A, I>> for Allocation<1, F, A, I>
{
    fn into(self) -> crate::solver::Allocation<F, A, I> {
        crate::solver::Allocation::new(self.agent, self.item, self.price)
    }
}

#[allow(dead_code)]
impl<const D: usize, F: num::Float, A: Agent<D, FloatType = F>, I: Item<D, FloatType = F>>
    Allocation<D, F, A, I>
{
    pub fn new(agent: A, item: I, price: F) -> Self {
        let utility = agent.utility(price, item.quality());
        Self {
            agent,
            item,

            price,
            utility,
        }
    }

    pub fn decompose(self) -> (A, I) {
        (self.agent, self.item)
    }

    pub fn agent(&self) -> &A {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut A {
        &mut self.agent
    }

    pub fn item(&self) -> &I {
        &self.item
    }

    pub fn quality(&self) -> [F; D] {
        self.item.quality()
    }

    pub fn set_item(&mut self, mut item: I) -> I {
        std::mem::swap(&mut self.item, &mut item);
        self.utility = self.agent.utility(self.price, self.item.quality());
        return item;
    }

    pub fn set_agent(&mut self, mut agent: A) -> A {
        assert!(self.price < agent.income());

        std::mem::swap(&mut self.agent, &mut agent);
        self.utility = self.agent.utility(self.price, self.item.quality());
        return agent;
    }

    pub fn set_agent_and_price(&mut self, mut agent: A, price: F) -> A {
        self.price = price;
        assert!(self.price < agent.income());

        std::mem::swap(&mut self.agent, &mut agent);
        self.utility = self.agent.utility(self.price, self.item.quality());
        return agent;
    }

    fn set_price(&mut self, price: F) {
        assert!(price < self.agent.income());

        self.price = price;
        self.utility = self.agent.utility(self.price, self.item.quality());
    }

    pub fn price(&self) -> F {
        self.price
    }

    pub fn utility(&self) -> F {
        self.utility
    }

    pub fn indifferent_price(&self, quality: [F; D], settings: &FractalSettings<F>) -> Option<F> {
        let (x_min, x_max) = if self.agent().utility(self.price, quality) > self.utility() {
            (self.price, self.agent.income())
        } else {
            assert!(self.price >= settings.constraint_price);
            (settings.constraint_price, self.price)
        };
        indifferent_price(
            self.agent(),
            quality,
            self.utility,
            x_min,
            x_max,
            settings.epsilon,
            settings.max_iter,
        )
    }

    pub fn prefers(&self, other: &Self, epsilon: F) -> bool {
        self.agent().utility(other.price(), other.quality()) > self.utility() + epsilon
    }

    pub fn is_preferred_by(&self, others: &[Self], epsilon: F) -> bool {
        for other in others {
            if other.prefers(self, epsilon) {
                return true;
            }
        }
        false
    }
}

pub fn verify_solution<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    allocations: &[Allocation<D, F, A, I>],
    settings: &FractalSettings<F>,
) -> bool {
    let mut valid = true;

    for (i, allocation_i) in allocations.iter().enumerate() {
        let u = allocation_i
            .agent
            .utility(allocation_i.price, allocation_i.item.quality());
        if (u - allocation_i.utility()).abs() > settings.epsilon {
            println!("Agent {} has a utility mismatch!", i);
            return false;
        }

        for (j, allocation_j) in allocations.iter().enumerate() {
            if i != j {
                if allocation_j.agent.agent_id() == allocation_i.agent.agent_id() {
                    println!(
                        "Agent {} has the same item_id as {}; item_id= {}",
                        i,
                        j,
                        allocation_j.agent.agent_id()
                    );
                    valid = false;
                }

                // Compute the utility agent i would get from allocation j
                let u_alt = allocation_i
                    .agent
                    .utility(allocation_j.price, allocation_j.quality());
                if u_alt > u + settings.epsilon {
                    let p_alt = allocation_i
                        .indifferent_price(allocation_j.quality(), settings)
                        .unwrap();
                    println!(
                        "Agent {} prefers allocation {}, (delta_u = {}, delta_p = {})",
                        i,
                        j,
                        (u_alt - u).to_f32().unwrap(),
                        (p_alt - allocation_j.price()).to_f32().unwrap(),
                    );
                    valid = false;
                }
            }
        }
    }

    valid
}

pub fn favourite<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    agent: &A,
    allocations: &[Allocation<D, F, A, I>],
    epsilon: F,
) -> Option<(usize, F)> {
    let mut u_max = F::zero();
    let mut fav: Option<usize> = None;
    for (l, other) in allocations.iter().enumerate().rev() {
        if agent.income() > other.price() + epsilon {
            let u = agent.utility(other.price(), other.quality());
            if fav.is_none() || u > u_max + epsilon {
                u_max = u;
                fav = Some(l)
            }
        }
    }

    fav.map(|x| (x, u_max))
}

pub fn min_favourite<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    allocations: &[Allocation<D, F, A, I>],
    range: std::ops::RangeInclusive<usize>,
    epsilon: F,
) -> Option<usize> {
    let mut fav_min: Option<usize> = None;
    for l in range {
        if let Some((fav, _)) = favourite(allocations[l].agent(), allocations, epsilon) {
            if let Some(min) = &mut fav_min {
                if &fav < min {
                    *min = fav;
                }
            } else {
                fav_min = Some(fav);
            }
        }
    }

    fav_min
}

pub fn max_favourite<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    allocations: &[Allocation<D, F, A, I>],
    range: std::ops::RangeInclusive<usize>,
    epsilon: F,
) -> Option<usize> {
    let mut fav_max: Option<usize> = None;
    for l in range {
        if let Some((fav, _)) = favourite(allocations[l].agent(), allocations, epsilon) {
            if let Some(min) = &mut fav_max {
                if &fav > min {
                    *min = fav;
                }
            } else {
                fav_max = Some(fav);
            }
        }
    }

    fav_max
}
