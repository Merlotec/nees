use super::*;
use crate::render::RenderAllocation;
use crate::solver::switchbranch::{boundary_point, steepest_alignment, SBError, SBResult};
use crate::solver::{Agent, Allocation, Item};
use std::alloc::alloc;
use std::mem;
use std::sync::{Arc, Mutex};
#[derive(Debug, Copy, Clone)]
pub enum FractalError {
    NoIndifference,
    NoCandidate,
    IncomeExceeded,
    NoSupport,
    NoDoublecross,
    InvalidInsertion,
    NoIntermediateAgent,
}

pub type FractalResult<T> = Result<T, FractalError>;

#[derive(Debug, Clone)]
pub enum AgentHolder<A: Agent> {
    Empty,
    Agent(A),
}

impl<A: Agent> AgentHolder<A> {
    pub fn has_agent(&self) -> bool {
        match self {
            AgentHolder::Empty => false,
            AgentHolder::Agent(_) => true,
        }
    }
    pub fn take(&mut self) -> Self {
        let mut other = AgentHolder::Empty;
        mem::swap(self, &mut other);
        other
    }

    pub fn to_option(self) -> Option<A> {
        match self {
            AgentHolder::Agent(a) => Some(a),
            AgentHolder::Empty => None,
        }
    }

    pub fn agent(&self) -> &A {
        match &self {
            AgentHolder::Agent(a) => a,
            AgentHolder::Empty => panic!("AgentHolder::Empty"),
        }
    }
}

impl<A: Agent> Agent for AgentHolder<A> {
    type FloatType = A::FloatType;

    fn agent_id(&self) -> usize {
        self.agent().agent_id()
    }

    fn income(&self) -> Self::FloatType {
        self.agent().income()
    }

    fn utility(&self, price: Self::FloatType, quality: Self::FloatType) -> Self::FloatType {
        self.agent().utility(price, quality)
    }
}

// q0 < q1
pub fn next_agent_up<F: num::Float, A: Agent<FloatType = F>>(
    agents: &[A],
    q0: F,
    p0: F,
    q1: F,
    epsilon: F,
    max_iter: usize,
) -> FractalResult<(usize, F)> {
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;
    for (a, agent) in agents.iter().enumerate() {
        if agent.income() <= p0 {
            return Err(FractalError::IncomeExceeded); // Should not happenn.
        }
        let p_indif = indifferent_price(
            agent,
            q1,
            agent.utility(p0, q0),
            p0,
            agent.income(),
            epsilon,
            max_iter,
        )
        .ok_or(FractalError::NoIndifference)
        .unwrap();

        if to_allocate.is_none() || p_indif < p_min {
            p_min = p_indif;
            to_allocate = Some(a);
        }
    }
    to_allocate
        .map(|x| (x, p_min))
        .ok_or(FractalError::NoCandidate)
}

// q0 > q1
// TODO: deal with situation where we may have asymptotic effects as we approach income.
pub fn next_agent_down<F: num::Float, A: Agent<FloatType = F>>(
    agents: &[A],
    q0: F,
    p0: F,
    q1: F,
    epsilon: F,
    max_iter: usize,
) -> FractalResult<(usize, F)> {
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;
    for (a, agent) in agents.iter().enumerate() {
        if agent.income() <= p0 {
            continue; // Should not happen.
        }
        let p_indif = indifferent_price(
            agent,
            q1,
            agent.utility(p0, q0),
            F::zero(),
            p0, // since p0 > p1
            epsilon,
            max_iter,
        )
        .ok_or(FractalError::NoIndifference)
        .unwrap();

        if to_allocate.is_none() || p_indif < p_min {
            p_min = p_indif;
            to_allocate = Some(a);
        }
    }
    to_allocate
        .map(|x| (x, p_min))
        .ok_or(FractalError::NoCandidate)
}

pub fn envelope_boundary<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, AgentHolder<A>, I>],
    quality: F,
    epsilon: F,
    max_iter: usize,
) -> Option<(usize, F)> {
    let mut p_max: F = num::zero();
    let mut i_max: Option<usize> = None;
    for (i, alloc) in allocations.iter().enumerate().rev() {
        if alloc.agent().has_agent() {
            if i_max.is_none() {
                p_max = alloc.indifferent_price(quality, epsilon, max_iter)?;
                i_max = Some(i);
            } else {
                let u_other = alloc.agent().utility(p_max, quality);
                if u_other > alloc.utility() {
                    p_max = alloc.indifferent_price(quality, epsilon, max_iter)?;
                    i_max = Some(i);
                }
            }
        }
    }

    i_max.map(|i| (i, p_max))
}

pub fn boundary<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, A, I>],
    quality: F,
    epsilon: F,
    max_iter: usize,
) -> Option<(usize, F)> {
    let mut p_max: F = num::zero();
    let mut i_max: Option<usize> = None;
    for (i, alloc) in allocations.iter().enumerate().rev() {
        if i_max.is_none() {
            p_max = alloc.indifferent_price(quality, epsilon, max_iter)?;
            i_max = Some(i);
        } else {
            let u_other = alloc.agent().utility(p_max, quality);
            if u_other > alloc.utility() {
                p_max = alloc.indifferent_price(quality, epsilon, max_iter)?;
                i_max = Some(i);
            }
        }
    }

    i_max.map(|i| (i, p_max))
}
/// Inclusive start and end.
fn align_down<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    start: usize,
    end: usize,
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> FractalResult<()> {
    assert!(end <= start);
    let agent_holders: Vec<AgentHolder<A>> = allocations[end..=start]
        .iter_mut()
        .map(|alloc| alloc.agent.take())
        .collect();
    let mut agents = Vec::with_capacity(agent_holders.len());
    for ah in agent_holders {
        agents.push(ah.to_option().ok_or(FractalError::NoIntermediateAgent)?);
    }

    let mut it = (end..=start).rev();
    for l in &mut it {
        let q0 = allocations[l].quality();
        // Start with a.
        // Since we are first allocation, we can take the current price of the allocation as given.
        let (inner_b, p0) = if l == start {
            (None, F::zero())
        } else if let Some((inner_b, p)) =
            envelope_boundary(&allocations, q0, epsilon, max_iter)
        {
            // TODO: If the inner_b < a then it is an old agent, but we know that it does not interfere due to it previously being a valid allocation.
            // This is because for this to be a problem, the allocations must have been farther out in the previous valid state which is a contradiction.

            // TODO: Sln: if we hit this boundary, we can simply use old allocations after this as we know they are valid because our movement above.
            // will not disrupt these allocations.
            (if inner_b >= start { Some(inner_b) } else { break; }, p)
        } else {
            return Err(FractalError::NoSupport);
        };


        // May need to promote since we have pushed all the agents out (so may cross over the DC agent).
        // Find the steepest boundary.
        let agent = if l > end {
            let q1 = allocations[l - 1].quality();
            let (a, p1) = next_agent_down(&agents, q0, p0, q1, epsilon, max_iter)
                .ok_or(FractalError::NoCandidate)?;
            let agent = agents.remove(a);
        } else {
            // Last agent to allocate, can allocate wherever.
            agents.pop().ok_or(FractalError::NoCandidate)?
        };
        allocations[l].set_agent(AgentHolder::Agent(agent), p0);
        // RECURSION!!!
        allocate(
            allocations,
            start,
            inner_b,
            epsilon,
            max_iter,
            render_pipe.clone(),
        )?;
    }

    // If there are still agents to allocate we allocate them up from the bottom (since this means that we break due to a recrossing).
    // Reverse again to go up from the s.
    for l in it.rev() {
        let p = allocations[l].price();
        allocations[l].set_agent(AgentHolder::Agent(agents.remove(0)), p);
    }
}

/// Inclusive start and end.
fn align_up<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    start: usize,
    end: usize,
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> FractalResult<()> {
    assert!(end >= start);
    let agent_holders: Vec<AgentHolder<A>> = allocations[start..=end]
        .iter_mut()
        .map(|alloc| alloc.agent.take())
        .collect();
    let mut agents = Vec::with_capacity(agent_holders.len());
    for ah in agent_holders {
        agents.push(ah.to_option().ok_or(FractalError::NoIntermediateAgent));
    }

    let mut it = end..=start;
    for l in &mut it {
        let q0 = allocations[l].quality();
        // Start with a.
        // Since we are first allocation, we can take the current price of the allocation as given.
        let (inner_b, p0) = if l == start {
            (None, F::zero())
        } else if let Some((inner_b, p)) =
            envelope_boundary(&allocations, q0, epsilon, max_iter)
        {
            // TODO: If the inner_b < a then it is an old agent, but we know that it does not interfere due to it previously being a valid allocation.
            // This is because for this to be a problem, the allocations must have been farther out in the previous valid state which is a contradiction.
            (if inner_b <= start { Some(inner_b) } else { break; }, p)
        } else {
            return Err(FractalError::NoSupport);
        };

        // Find the steepest boundary.
        let agent = if l > end {
            let q1 = allocations[l - 1].quality();
            let (a, p1) = next_agent_up(&agents, q0, p0, q1, epsilon, max_iter)
                .ok_or(FractalError::NoCandidate)?;
            let agent = agents.remove(a);
        } else {
            // Last agent to allocate, can allocate wherever.
            agents.pop().ok_or(FractalError::NoCandidate)?
        };
        allocations[l].set_agent(AgentHolder::Agent(agent), p0);
        // RECURSION!!!
        allocate(
            allocations,
            start,
            inner_b,
            epsilon,
            max_iter,
            render_pipe.clone(),
        )?;
    }

    for l in it.rev() {
        let p = allocations[l].price();
        allocations[l].set_agent(AgentHolder::Agent(agents.pop().unwrap()), p);
    }
}


// Recursive allocation function.
// Allocations from [b, a) must be non empty.
// Allocation a must be empty or not exist.
pub fn allocate<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    i: usize,
    boundary: Some(usize),
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> FractalResult<()> {
    if let Some(b) = boundary {
        if i < allocations.len() {
            if let AgentHolder::Empty = allocations[i].agent() {
                // Already agent there - invalid insertion.
                return Err(FractalError::InvalidInsertion);
            }
        } else {
            return Err(FractalError::InvalidInsertion);
        }

        if i.abs_diff(b) > 1 {
            // We have a double-crossing.
            // Must dismantle all items from (b, a) and attempt to reallocate.
            // We allocate depending on the direction.
            if b < i {
                let s = b + 1;
                align_down(allocations, i, s, epsilon, max_iter, render_pipe.clone())?;

                // Check if the final agent prefers a lower agent - if so we must shift up!!
                // TODO: could it be possible that final agent doublecrosses and prefers a lower allocation? What do we do then?
                if allocations[s]
                    .agent()
                    .utility(allocations[b].quality(), allocations[b].price())
                    > allocations[s].utility() + epsilon
                {
                    // We must promote agent at b because we cannot get under it.
                    let to_promote = allocations[b].agent_mut().take().unwrap();
                    // Take all the agents up to a.
                    align_up(allocations, i - 1, b, epsilon, max_iter, render_pipe.clone())?;
                    let (b_promoted, p_promoted) = envelope_boundary(allocations, allocations[i].quality(), epsilon, max_iter);
                    // Add the promoted agent. The algorithm relies on this being a valid allocation.
                    allocations[i].set_agent(to_promote, p_promoted);
                    allocate(allocations, i, b_promoted, epsilon, max_iter, render_pipe.clone())?;
                }
            } else if i < b {
                let s = b - 1;
                align_down(allocations, i, s, epsilon, max_iter, render_pipe.clone())?;

                // Check if the final agent prefers a higher agent - if so we must shift up!!
                // TODO: could it be possible that final agent doublecrosses and prefers a lower allocation? What do we do then?
                if allocations[s]
                    .agent()
                    .utility(allocations[b].quality(), allocations[b].price())
                    > allocations[s].utility() + epsilon
                {
                    // We must demote agent at b because we cannot get under it.
                    let to_demote = allocations[b].agent_mut().take().unwrap();
                    // Take all the agents up to a.
                    align_up(allocations, i + 1, b, epsilon, max_iter, render_pipe.clone())?;
                    let (b_demoted, p_demoted) = envelope_boundary(allocations, allocations[i].quality(), epsilon, max_iter);
                    // Add the promoted agent. The algorithm relies on this being a valid allocation.
                    allocations[i].set_agent(to_demote, p_demoted);
                    allocate(allocations, i, b_demoted, epsilon, max_iter, render_pipe.clone())?;
                }
            }
        }
    }

    Ok(())
}

// This is the 'outer most' layer of the process, and the only one that can add new agents into the system.
// The inner layers all deal with already allocated agents.
pub fn root<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut agents: Vec<A>,
    mut items: Vec<I>,
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> FractalResult<Vec<Allocation<F, A, I>>> {
    assert_eq!(agents.len(), items.len());

    let mut allocations: Vec<Allocation<F, A, I>> = Vec::new();

    // The allocation loop.
    while !items.is_empty() {
        // Get the quality of the next allocation.
        let q0 = items.first().unwrap().quality();
        let (b, p0): (Option<usize>, F) = if allocations.is_empty() {
            (None, F::zero())
        } else {
            // Should have a boundary point if there are existing allocations.
            let (b, p1) = envelope_boundary(&allocations, q0, epsilon, max_iter)
                .ok_or(FractalError::NoSupport)?;
            (Some(b), p1)
        };

        let a = if items.len() > 1 {
            // Calculate the agent to allocate.
            let (a, _) = next_agent_up(&agents, q0, p0, items[1].quality(), epsilon, max_iter)?;
            a
        } else {
            0 // There should be only one agent left.
        };

        let new_allocation = Allocation::new(AgentHolder::Agent(agents.remove(a)), items.remove(0), p0);
        let i = allocations.len();
        allocations.push(new_allocation);

        // By calling this recursive function we ensure that the newly added agent is dealt with such that
        // the resulting allocation state is valid.
        allocate(&mut allocations, i, b, epsilon, max_iter, render_pipe.clone())?;
    }

    Ok(allocations)
}
