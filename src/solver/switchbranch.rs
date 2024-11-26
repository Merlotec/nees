use core::alloc;
use std::{
    ops::DerefMut,
    sync::{Arc, Mutex}, time::Duration,
};

use crate::render::RenderAllocation;

use super::{utility::indifferent_price, Agent, Allocation, Item};

#[derive(Debug, Copy, Clone)]
pub enum SBError {
    NoIndifference,
    NoCandidate,
    IncomeExceeded,
    NoSupport,
}

pub struct AgentHolder<A: Agent> {
    pub agent: A,
    pub min_alloc: usize,
}

impl<A: Agent> Agent for AgentHolder<A> {
    type FloatType = A::FloatType;

    fn item_id(&self) -> usize {
        self.agent.item_id()
    }

    fn income(&self) -> Self::FloatType {
        self.agent.income()
    }

    fn utility(&self, price: Self::FloatType, quality: Self::FloatType) -> Self::FloatType {
        self.agent.utility(price, quality)
    }
}

pub type SBResult<T> = Result<T, SBError>;

pub fn backtrack<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<AgentHolder<A>>,
    items: &mut Vec<I>,
    allocations: &mut Vec<Allocation<F, AgentHolder<A>, I>>,
    n: usize,
) {

    for _ in 0..n {
        let alloc = allocations.pop().unwrap();
        let (agent, item) = alloc.decompose();
        agents.push(agent);
        items.insert(0, item); // Maintain order of items - very important.
    }
}

pub fn check_dc<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(allocations: &[Allocation<F, A, I>], q1: F, p_min: F, epsilon: F, max_iter: usize) -> Option<(usize, F)> {
    let mut p_dc = F::zero();
    let mut l_dc: Option<usize> = None;
    for (l, alloc) in allocations.iter().enumerate() {
        if alloc.agent().income() > p_min + epsilon {
            if alloc.agent().utility(p_min, q1) > alloc.utility() {
                // We have a double crossing.
                let p_indif = alloc.indifferent_price(
                    q1,
                    epsilon,
                    max_iter,
                )
                .ok_or(SBError::NoIndifference)
                .unwrap();

                if l_dc.is_none() || p_indif > p_dc {
                    p_dc = p_indif;
                    l_dc = Some(l);
                }
            }
        }
    }

    l_dc.map(|x| (x, p_dc))
}

pub fn align_left<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<AgentHolder<A>>,
    items: &mut Vec<I>,
    mut allocations: Vec<Allocation<F, AgentHolder<A>, I>>,
    mut n: usize,
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> SBResult<Vec<Allocation<F, AgentHolder<A>, I>>> {
    assert!(agents.len() <= items.len());
    if items.is_empty() {
        return Ok(allocations);
    }

    let mut q0: F;
    let mut p0: F;

    let (mut q0, mut p0) = if allocations.is_empty() {
        (items.first().unwrap().quality(), F::zero())
    } else {
        let q = items.first().unwrap().quality();
        (
            q,
            allocations
                .last()
                .unwrap()
                .indifferent_price(q, epsilon, max_iter)
                .ok_or(SBError::NoIndifference)
                .unwrap(),
        )
    };

    // Check if this first allocation is in fact valid:
    if let Some(dc) = check_dc(&allocations, q0, p0, epsilon, max_iter) {
        
    }

    while !items.is_empty() {
        if n == 0 {
            return Ok(allocations);
        }
        // Find the steepest agent.
        let mut to_allocate: Option<usize> = None;
        if items.len() == 1 {
            // we are on last item so we just allocate last agent.
            let alloc = Allocation::new(agents.remove(0), items.remove(0), p0);
            allocations.push(alloc);
        } else {
            let q1 = items[1].quality();
            let mut p_min = F::zero();
            for (a, agent) in agents.iter().enumerate() {
                if agent.min_alloc > allocations.len() {
                    continue; // Dont allocate an agent that has a min alloc set above this as its a dc agent and will cause loops.
                }

                if agent.income() <= p0 {
                    return Err(SBError::IncomeExceeded); // Should not happenn.
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
                .ok_or(SBError::NoIndifference)
                .unwrap();

                if to_allocate.is_none() || p_indif < p_min {
                    p_min = p_indif;
                    to_allocate = Some(a);
                }
            }
            // Check if there would be a double crossing...

            let mut dc = false; 

            let mut q_next = q1;
            let mut p_next = p_min;

            // This could cause a further double crossing - we only know that the allocation of the previous agent is valid, but we don't know
            // if the next allocation will be valid, and since right alignment means that allocations are done using p0, q0, without checking
            // validity, we need to check this now.
            println!("tickleft");
            while let Some((l_dc, _)) = check_dc(&allocations, q_next, p_next, epsilon, max_iter) {
                if dc {
                    println!("inner");
                }
                dc = true;
                if n == 0 {
                    break;
                }
                // We do not want to remove the dc agent because we want to try to allocate under it.
                // If this does not work then it will be shifted above, but by the align_right function.

                let aid = allocations[l_dc].agent().item_id();
                let sub_n = allocations.len() - l_dc - 1;
                backtrack(agents, items, &mut allocations, sub_n);
                println!("switchright: {}, {}, {} {}, {}", allocations.len(), agents.len(), items.len(), sub_n, aid);
                // Allocate over.
                allocations = align_right(
                    agents,
                    items,
                    allocations,
                    sub_n + 1, // +1 ensures that we make progress and dont have infinite switching.
                    epsilon,
                    max_iter,
                    render_pipe.clone(),
                )?;

                n -= 1;

                // We now have +1 agent allocated.
                // Check the new agent to see if its indeed valid for left allocaction.
                if let Some(last) = allocations.last() {
                    if let Some(next_item) = items.first() {
                        q_next = next_item.quality();
                        p_next = last
                            .indifferent_price(q_next, epsilon, max_iter)
                            .ok_or(SBError::NoIndifference)
                            .unwrap();

                    }
                } else {
                    q_next = F::zero();
                    p_next = F::zero();
                    // Checking dc should be vacuously true.
                }
                *render_pipe.lock().unwrap().deref_mut() = Some(
                    allocations
                        .iter()
                        .map(|x| RenderAllocation::from_allocation(&x, F::one(), epsilon, max_iter))
                        .collect(),
                );

                std::thread::sleep(Duration::from_millis(300));
            } 

            if !dc {
                // We have a valid allocation.
                println!("{}, {}, {}", allocations.len(), agents.len(), items.len());
                let alloc = Allocation::new(
                    agents.remove(to_allocate.ok_or(SBError::NoCandidate).unwrap()),
                    items.remove(0),
                    p0,
                );
                allocations.push(alloc);
                n -= 1;
            }

            q0 = q_next;
            p0 = p_next;
        }
    }

    Ok(allocations)
}

pub fn align_right<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<AgentHolder<A>>,
    items: &mut Vec<I>,
    mut allocations: Vec<Allocation<F, AgentHolder<A>, I>>,
    mut n: usize,
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> SBResult<Vec<Allocation<F, AgentHolder<A>, I>>> {
    assert!(agents.len() <= items.len());

    let mut q0: F = F::zero();
    let mut p0: F = F::zero();

    if let Some(last) = allocations.last() {
        q0 = last.quality();
        p0 = last.price();
    }

    while !items.is_empty() {
        if n == 0 {
            return Ok(allocations);
        }
        
        // Find the steepest agent.
        let mut to_allocate: Option<usize> = None;

        let q1 = items.first().unwrap().quality();
        let mut p_min = F::zero();
        for (a, agent) in agents.iter().enumerate() {
            if agent.min_alloc > allocations.len() {
                continue; // Dont allocate an agent that has a min alloc set above this as its a dc agent and will cause loops.
            }

            if agent.income() <= p0 {
                return Err(SBError::IncomeExceeded); // Should not happenn.
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
            .ok_or(SBError::NoIndifference)
            .expect(&format!("FAILED: {}, n= {}", allocations.len(), n));

            if to_allocate.is_none() || p_indif < p_min {
                p_min = p_indif;
                to_allocate = Some(a);
            }
        }
        println!("tickright");
        // Check if there would be a double crossing...
        if let Some((l_dc, _)) = check_dc(&allocations, q1, p_min, epsilon, max_iter) {
            // Go back to the dc agent and try to go under, if we cant, allocate this agent above.
            // Return allocated items to the pool.
            let aid = allocations[l_dc].agent().item_id();
            let sub_n = allocations.len() - l_dc;
            backtrack(agents, items, &mut allocations, sub_n - 1);
            let (mut dc_agent, dc_item) = allocations.pop().unwrap().decompose(); // Backtracks to remove the final dc agent.
            assert_eq!(aid, dc_agent.item_id());
            items.insert(0, dc_item);
            println!("switchleft: {}, {}, {}, {}, {}", allocations.len(), agents.len(), items.len(), sub_n, aid);
            // Allocate over.
            allocations = align_left(
                agents,
                items,
                allocations,
                sub_n, // We don't add 1 because we have the 'withheld' agent to push up. This will add one after.
                epsilon,
                max_iter,
                render_pipe.clone(),
            )?;

            println!("after: {}", aid);

            dc_agent.min_alloc = allocations.len();
            agents.push(dc_agent);
            // TODO: next position may be invalid!!! because we have cut the left shift early... 
            allocations = align_left(
                agents,
                items,
                allocations,
                1, // Add the next agent (might be the withheld agent).
                epsilon,
                max_iter,
                render_pipe.clone(),
            )?;

            n -= 1;

            if let Some(last) = allocations.last() {
                q0 = last.quality();
                p0 = last.price();
            } else {
                return Err(SBError::NoSupport);
            }

            *render_pipe.lock().unwrap().deref_mut() = Some(
                allocations
                    .iter()
                    .map(|x| RenderAllocation::from_allocation(&x, F::one(), epsilon, max_iter))
                    .collect(),
            );
            std::thread::sleep(Duration::from_millis(300));

            // Insert after - we know the agent must prefer this point. This should be done automatically by continuing.
        } else {
            // We have a valid allocation.
            let alloc = Allocation::new(
                agents.remove(to_allocate.ok_or(SBError::NoCandidate).unwrap()),
                items.remove(0),
                p_min,
            );
            allocations.push(alloc);

            n -= 1;

            q0 = q1;
            p0 = p_min;
        }
    }

    Ok(allocations)
}

pub fn swichbranch<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: Vec<A>,
    mut items: Vec<I>,
    epsilon: F,
    max_iter: usize,
    render_pipe: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
) -> SBResult<Vec<Allocation<F, AgentHolder<A>, I>>> {
    assert_eq!(agents.len(), items.len());

    let mut agents: Vec<AgentHolder<A>> = agents.into_iter().map(|x| AgentHolder { agent: x, min_alloc: 0 }).collect();

    let n = agents.len();

    align_left(
        &mut agents,
        &mut items,
        Vec::new(),
        n,
        epsilon,
        max_iter,
        render_pipe,
    )
}
