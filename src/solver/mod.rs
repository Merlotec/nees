use utility::indifferent_price;

pub mod utility;
pub mod switchbranch;

pub trait Agent {
    type FloatType: num::Float;

    fn item_id(&self) -> usize;
    fn income(&self) -> Self::FloatType;
    fn utility(&self, price: Self::FloatType, quality: Self::FloatType) -> Self::FloatType;
}

pub trait Item {
    type FloatType: num::Float;

    fn quality(&self) -> Self::FloatType;
}
pub struct Allocation<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> {
    agent: A,
    item: I,

    price: F,
    utility: F,
}

impl<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> Allocation<F, A, I> {
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

    pub fn item(&self) -> &I {
        &self.item
    }

    pub fn quality(&self) -> F {
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

    pub fn indifferent_price(&self, quality: F, epsilon: F, max_iter: usize) -> Option<F> {
        let (x_min, x_max) = if quality > self.quality() {
            (self.price, self.agent.income())
        } else {
            (F::zero(), self.price)
        };
        indifferent_price(self.agent(), self.quality(), self.utility, x_min, x_max, epsilon, max_iter)
    }
}

pub struct Solver<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> {
    agents: Vec<A>,
    items: Vec<I>,

    allocations: Vec<Allocation<F, A, I>>,
}