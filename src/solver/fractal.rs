use super::*;
use crate::render::RenderAllocation;
use crate::solver::{Agent, Allocation, Item};
use std::mem;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Copy, Clone)]
pub enum FractalError {
    NoIndifference,
    NoCandidate,
    IncomeExceeded,
    NoBoundary,
    NoDoublecross,
    InvalidInsertion,
    NoIntermediateAgent,
    EmptyAllocation,
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

pub fn displace<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    i: usize,
    to: usize,
) {
    if i > to {
        let mut buffer = allocations[to].agent_mut().take();
        let agent = allocations[i].agent_mut().take();
        allocations[to].set_agent(agent);
        for j in (to + 1..=i) {
            let hold = allocations[j].agent_mut().take();
            allocations[j].set_agent(buffer.take());
            buffer = hold;
        }

        allocations[i].set_agent(buffer);
    } else if to > i {
        let mut buffer = allocations[to].agent_mut().take();
        let agent = allocations[i].agent_mut().take();
        allocations[to].set_agent(agent);
        for j in (i..to).rev() {
            let hold = allocations[j].agent_mut().take();
            allocations[j].set_agent(buffer.take());
            buffer = hold;
        }
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
    assert!(q1 >= q0);
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
    assert!(q1 <= q0);
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;
    for (a, agent) in agents.iter().enumerate() {
        if agent.income() <= p0 {
            // TODO: what if??
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
    conventional_alignment: bool,
    epsilon: F,
    max_iter: usize,
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

    let mut restore_to: Option<usize> = None;

    for l in (end..=start).rev() {
        let q0 = allocations[l].quality();
        // Start with a.
        // Since we are first allocation, we can take the current price of the allocation as given.
        let (inner_b, p0) = if l == start && !conventional_alignment {
            (None, allocations[l].price())
        } else if let Some((inner_b, p)) = envelope_boundary(
            if conventional_alignment {
                &allocations[start..allocations.len()]
            } else {
                &allocations
            },
            q0,
            epsilon,
            max_iter,
        ) {
            // TODO: If the inner_b < a then it is an old agent, but we know that it does not interfere due to it previously being a valid allocation.
            // This is because for this to be a problem, the allocations must have been farther out in the previous valid state which is a contradiction.

            // TODO: Sln: if we hit this boundary, we can simply use old allocations after this as we know they are valid because our movement above.
            // will not disrupt these allocations.
            (
                if inner_b >= end {
                    Some(inner_b)
                } else {
                    restore_to = Some(l);
                    break;
                },
                p,
            )
        } else {
            return Err(FractalError::NoBoundary);
        };

        if !conventional_alignment && allocations[l].price() > p0 {
            // Old price was greater, meaning we can restore...
            // TODO: might ignore double-crossings lower down???
            restore_to = Some(l);
            break;
        }

        // May need to promote since we have pushed all the agents out (so may cross over the DC agent).
        // Find the steepest boundary.
        let agent = if l > end {
            let q1 = allocations[l - 1].quality();
            let (a, p1) = next_agent_down(&agents, q0, p0, q1, epsilon, max_iter)?;
            agents.remove(a)
        } else {
            // Last agent to allocate, can allocate wherever.
            agents.pop().ok_or(FractalError::NoCandidate)?
        };

        println!("align_down: l={}, id={}, b={:?}, p={}", l, agent.agent_id(), inner_b, p0.to_f64().unwrap());

        allocations[l].set_agent_and_price(AgentHolder::Agent(agent), p0);
        // RECURSION!!!
        allocate(
            allocations,
            l,
            inner_b,
            epsilon,
            max_iter,
        )?;
    }

    // If there are still agents to allocate we allocate them up from the bottom (since this means that we break due to a recrossing).
    // Reverse again to go up from the s.
    // TODO: order swap IS an issue
    if let Some(restore_to) = restore_to {
        assert!(!conventional_alignment);
        for l in end..=restore_to {
            let agent = agents.remove(0);
            println!("align_down, restore: l={}, id={}, p={}", l, agent.agent_id(), allocations[l].price().to_f64().unwrap());
            allocations[l].set_agent(AgentHolder::Agent(agent));
        }
        // In case of an order switch we need to realign.
        align_up(allocations, end, restore_to, true, epsilon, max_iter)?;
    }

    // TODO: we dont always restore agents when the actual allocation point is farther out than the boundary (due to lower pressure).
    // if allocations.len() > 92 {
    //     if allocations[92].prefers(&allocations[91], epsilon) {
    //         println!("BANG: pindiff={}, p_actual={}", allocations[92].indifferent_price(allocations[91].quality(), epsilon, max_iter).unwrap().to_f64().unwrap(), allocations[91].price().to_f64().unwrap());
    //     }
    //     if allocations[91].prefers(&allocations[92], epsilon) {
    //         println!("bingo");
    //     }
    // }

    Ok(())
}

/// Inclusive start and end.
fn align_up<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    start: usize,
    end: usize,
    conventional_alignment: bool,
    epsilon: F,
    max_iter: usize,
) -> FractalResult<()> {
    assert!(end >= start);
    let agent_holders: Vec<AgentHolder<A>> = allocations[start..=end]
        .iter_mut()
        .map(|alloc| alloc.agent.take())
        .collect();
    let mut agents = Vec::with_capacity(agent_holders.len());
    for ah in agent_holders {
        agents.push(ah.to_option().ok_or(FractalError::NoIntermediateAgent)?);
    }
    let mut restore_to: Option<usize> = None;

    for l in start..=end {
        let q0 = allocations[l].quality();
        // Start with a.
        // Since we are first allocation, we can take the current price of the allocation as given.
        let (inner_b, p0) = if l == start && !conventional_alignment {
            (None, allocations[l].price())
        } else if let Some((inner_b, p)) = envelope_boundary(
            if conventional_alignment {
                &allocations[0..end]
            } else {
                &allocations
            },
            q0,
            epsilon,
            max_iter,
        ) {
            // TODO: If the inner_b < a then it is an old agent, but we know that it does not interfere due to it previously being a valid allocation.
            // This is because for this to be a problem, the allocations must have been farther out in the previous valid state which is a contradiction.
            (
                if inner_b <= end {
                    Some(inner_b)
                } else {
                    restore_to = Some(l);
                    break;
                },
                p,
            )
        } else {
            return Err(FractalError::NoBoundary);
        };

        if !conventional_alignment && allocations[l].price() > p0 {
            // Old price was greater, meaning we can restore...
            // TODO: order problems? and dc coming from lower down?
            restore_to = Some(l);
            break;
        }

        // Find the steepest boundary.
        let agent = if l < end {
            let q1 = allocations[l + 1].quality();
            let (a, p1) = next_agent_up(&agents, q0, p0, q1, epsilon, max_iter)?;
            agents.remove(a)
        } else {
            // Last agent to allocate, can allocate wherever.
            agents.pop().ok_or(FractalError::NoCandidate)?
        };

        println!("align_up: l={}, id={}, b={:?}, p={}", l, agent.agent_id(), inner_b, p0.to_f64().unwrap());

        allocations[l].set_agent_and_price(AgentHolder::Agent(agent), p0);
        // RECURSION!!!
        allocate(
            allocations,
            l,
            inner_b,
            epsilon,
            max_iter,
        )?;
    }
    if let Some(restore_to) = restore_to {
        assert!(!conventional_alignment);
        for l in (restore_to..=end).rev() {

            let agent = agents.pop().unwrap();
            println!("align_down, restore: l={}, id={}, p={}", l, agent.agent_id(), allocations[l].price().to_f64().unwrap());
            allocations[l].set_agent(AgentHolder::Agent(agent));
        }
        // In case of an order switch we need to realign.
        align_down(allocations, end, restore_to, true, epsilon, max_iter)?;
    }

    Ok(())
}

// Recursive allocation function.
// Allocations from [b, a) must be non empty.
// Allocation a must be empty or not exist.
pub fn allocate<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    i: usize,
    boundary: Option<usize>,
    epsilon: F,
    max_iter: usize,
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

        if i.abs_diff(b) > 0 {
            // We have a double-crossing.
            // Must dismantle all items from (b, a) and attempt to reallocate.
            // We allocate depending on the direction.
            if b < i {
                let s = b + 1;
                align_down(
                    allocations,
                    i,
                    s,
                    false,
                    epsilon,
                    max_iter,
                )?;

                // Check if the final agent prefers a lower agent - if so we must shift up!!
                // TODO: could it be possible that final agent doublecrosses and prefers a lower allocation? What do we do then?
                if allocations[s].prefers(&allocations[b], epsilon) {
                    // TODO: agent s should be allocated first in some cases??
                    println!("p_star={}, p={}", allocations[b].indifferent_price(allocations[s].quality(), epsilon, max_iter).unwrap().to_f64().unwrap(), allocations[s].price().to_f64().unwrap());
                    // We must promote agent at b because we cannot get under it.
                    // let to_promote = allocations[b].agent_mut().take().to_option().unwrap();
                    displace(allocations, b, i);
                    // Remove agent for now.
                    let to_promote = allocations[i].agent_mut().take();
                    // Take all the agents up to a.
                    // TODO: problem here??
                    align_up(
                        allocations,
                        b,
                        i - 1,
                        true,
                        epsilon,
                        max_iter,
                    )?;

                    let (b_promoted, p_promoted) =
                        envelope_boundary(allocations, allocations[i].quality(), epsilon, max_iter)
                            .ok_or(FractalError::NoBoundary)?;

                    // Otherwise the promotion was invalid!!
                    //assert!(p_promoted <= allocations[i].price());

                    println!(
                        "promoted: b={}, b_prom={}, i={}, prom_id={}",
                        b,
                        b_promoted,
                        i,
                        to_promote.agent().agent_id()
                    );

                    // Add the promoted agent. The algorithm relies on this being a valid allocation.
                    // We have already moved our
                    allocations[i].set_agent_and_price(to_promote, p_promoted);
                    allocate(
                        allocations,
                        i,
                        Some(b_promoted),
                        epsilon,
                        max_iter,
                    )?;

                    if allocations.len() > 93 {
                        if allocations[92].prefers(&allocations[93], epsilon) {
                            println!("PREFERS (92)!!");
                        }
                        if allocations[93].prefers(&allocations[92], epsilon) {
                            println!("PREFERS (93)!!");
                        }
                    }
                }
            } else if i < b {
                let s = b - 1;
                println!("align_up.... i={}, s={}", i, s);
                align_up(
                    allocations,
                    i,
                    s,
                    false,
                    epsilon,
                    max_iter,
                )?;

                // Check if the final agent prefers a higher agent - if so we must shift up!!
                // TODO: could it be possible that final agent doublecrosses and prefers a lower allocation? What do we do then?
                if allocations[s].prefers(&allocations[b], epsilon) {
                    // We must demote agent at b because we cannot get under it.
                    displace(allocations, b, i);
                    // Remove agent for now.
                    let to_demote = allocations[i].agent_mut().take();
                    println!(
                        "demoted: b={}, i={}, prom_id={}",
                        b,
                        i,
                        to_demote.agent().agent_id()
                    );
                    // Take all the agents up to a.
                    align_down(
                        allocations,
                        b,
                        i + 1,
                        true,
                        epsilon,
                        max_iter,
                    )?;
                    let (b_demoted, p_demoted) =
                        envelope_boundary(allocations, allocations[i].quality(), epsilon, max_iter)
                            .ok_or(FractalError::NoBoundary)?;
                    // Add the promoted agent. The algorithm relies on this being a valid allocation.
                    allocations[i].set_agent_and_price(to_demote, p_demoted);
                    allocate(
                        allocations,
                        i,
                        Some(b_demoted),
                        epsilon,
                        max_iter,
                    )?;
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

    let mut allocations: Vec<Allocation<F, AgentHolder<A>, I>> = Vec::new();

    // The allocation loop.
    while !items.is_empty() {
        // Get the quality of the next allocation.
        let q0 = items.first().unwrap().quality();
        let (b, p0): (Option<usize>, F) = if allocations.is_empty() {
            (None, F::zero())
        } else {
            // Should have a boundary point if there are existing allocations.
            let (b, p1) = envelope_boundary(&allocations, q0, epsilon, max_iter)
                .ok_or(FractalError::NoBoundary)?;
            (Some(b), p1)
        };

        let a = if items.len() > 1 {
            // Calculate the agent to allocate.
            let (a, _) = next_agent_up(&agents, q0, p0, items[1].quality(), epsilon, max_iter)?;
            a
        } else {
            0 // There should be only one agent left.
        };

        let new_allocation =
            Allocation::new(AgentHolder::Agent(agents.remove(a)), items.remove(0), p0);
        let i = allocations.len();

        // By calling this recursive function we ensure that the newly added agent is dealt with such that
        // the resulting allocation state is valid.
        println!(
            "Alloc i={}, b={:?}, p0={}, q0={}, id={}",
            i,
            b,
            p0.to_f32().unwrap(),
            q0.to_f32().unwrap(),
            new_allocation.agent().agent_id()
        );

        allocations.push(new_allocation);

        if allocations.len() > 1 {
            *render_pipe.lock().unwrap().deref_mut() = Some(
                allocations
                    .iter()
                    .map(|x| RenderAllocation::from_allocation(&x, F::one(), epsilon, max_iter))
                    .collect(),
            );

            //std::thread::sleep(Duration::from_millis(300));
        }

        // if allocations.len() == 137 {
        //     loop {
        //         sleep(Duration::from_secs(1));
        //     }
        // }

        allocate(
            &mut allocations,
            i,
            b,
            epsilon,
            max_iter,
        )?;
    }

    let mut cleaned = Vec::with_capacity(allocations.len());

    for alloc in allocations {
        let p = alloc.price();
        let (mut agent, item) = alloc.decompose();

        cleaned.push(Allocation::new(
            agent
                .take()
                .to_option()
                .ok_or(FractalError::EmptyAllocation)?,
            item,
            p,
        ));
    }

    Ok(cleaned)
}
