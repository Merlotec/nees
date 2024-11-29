use core::alloc;
use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
    time::Duration,
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

    fn agent_id(&self) -> usize {
        self.agent.agent_id()
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
    decrement_min: bool,
) {
    for _ in 0..n {
        let alloc = allocations.pop().unwrap();
        let (mut agent, item) = alloc.decompose();
        agent.min_alloc = 0; //if agent.min_alloc == 0 { 0 } else { agent.min_alloc - 1 };
        agents.push(agent);
        items.insert(0, item); // Maintain order of items - very important.
    }

    if decrement_min {
        for agent in agents.iter_mut() {
            if agent.min_alloc > 0 {
                agent.min_alloc -= 1;
            }
        }
    }
}

pub fn boundary_point<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, AgentHolder<A>, I>],
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

pub fn steepest_alignment<F: num::Float, A: Agent<FloatType = F>>(
    agents: &[AgentHolder<A>],
    q0: F,
    p0: F,
    q1: F,
    i: usize,
    epsilon: F,
    max_iter: usize,
) -> SBResult<(usize, F)> {
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;
    for (a, agent) in agents.iter().enumerate() {
        if agent.min_alloc > i {
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
    to_allocate.map(|x| (x, p_min)).ok_or(SBError::NoCandidate)
}

pub fn favourite<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agent: &A,
    allocations: &[Allocation<F, A, I>],
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

pub fn steepest_rightshift<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &[AgentHolder<A>],
    allocations: &[Allocation<F, AgentHolder<A>, I>],
    q1: F,
    i: usize,
    epsilon: F,
    max_iter: usize,
) -> SBResult<(usize, F, usize)> {
    // Find the agent that this agent prefers to allocate relative to.
    let mut p_min = F::zero();
    let mut to_allocate: Option<(usize, usize)> = None;
    // We make the agent indifferent to the next favourite entity (so that we get the farthest out allocation).
    for (a, agent) in agents.iter().enumerate() {
        if agent.min_alloc > i {
            continue;
        }

        let (l, u0) = favourite(agent, allocations, epsilon).ok_or(SBError::NoCandidate)?;
        let fav = &allocations[l];
        let p = indifferent_price(
            agent,
            q1,
            u0,
            fav.price(),
            agent.income(),
            epsilon,
            max_iter,
        )
        .ok_or(SBError::NoIndifference)?;

        if to_allocate.is_none() || p < p_min {
            p_min = p;
            to_allocate = Some((a, l));
        }
    }
    to_allocate.map(|(a, l)| (a, p_min, l)).ok_or(SBError::NoCandidate)
}

pub fn check_dc<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, A, I>],
    q1: F,
    p_min: F,
    epsilon: F,
    max_iter: usize,
    check_last: bool,
) -> Vec<(usize, F)> {
    let mut dcs: Vec<(usize, F)> = vec![];
    let len = allocations.len();
    for (l, alloc) in allocations.iter().enumerate() {
        // if l + 1 == len && !check_last {
        //     break;
        // }
        if alloc.agent().income() > p_min + epsilon {
            if alloc.agent().utility(p_min, q1) > alloc.utility() + epsilon {
                // We have a double crossing.
                let p_indif = alloc
                    .indifferent_price(q1, epsilon, max_iter)
                    .ok_or(SBError::NoIndifference)
                    .unwrap();
                //
                // if l_dc.is_none() || p_indif > p_dc {
                //     p_dc = p_indif;
                //     l_dc = Some(l);
                // }
                dcs.push((l, p_indif));
            }
        }
    }

    dcs
}

pub fn check_favourite<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    alloc: &Allocation<F, A, I>,
    allocations: &[Allocation<F, A, I>],
    epsilon: F,
    max_iter: usize,
    check_last: bool,
) -> Option<(usize, F)> {
    let mut u_max = F::zero();
    let mut fav: Option<usize> = None;
    let len = allocations.len();
    for (l, other) in allocations.iter().enumerate() {
        if alloc.agent().income() > other.price() + epsilon {
            let u = alloc.agent().utility(other.price(), other.quality());
            if u > alloc.utility() + epsilon {
                // We have a double crossing.

                if fav.is_none() || u > u_max {
                    u_max = u;
                    fav = Some(l)
                }
            }
        }
    }

    fav.map(|x| (x, u_max))
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

    while !items.is_empty() {
        if n == 0 {
            return Ok(allocations);
        }

        let q0: F = items.first().unwrap().quality();

        let (b, p0): (Option<usize>, F) =
            if let Some((b, p0)) = boundary_point(&allocations, q0, epsilon, max_iter) {
                (Some(b), p0)
            } else {
                (None, F::zero())
            };

        let new_alloc = if items.len() == 1 {
            // Allocate last.
            Allocation::new(agents.remove(0), items.remove(0), p0)
        } else {
            // p0, q0 is the left aligned valid allocation point.
            let q1 = items[1].quality();

            let (a, p1) = steepest_alignment(
                agents.as_slice(),
                q0,
                p0,
                q1,
                allocations.len(),
                epsilon,
                max_iter,
            )?;

            Allocation::new(agents.remove(a), items.remove(0), p0)
        };

        let a_id = new_alloc.agent().agent_id();

        if let Some((_, _)) = check_favourite(&new_alloc, &allocations, epsilon, max_iter, false) {
            // Since we were pushed back by the boundary agent b, this must be our double crossing agent.
            let l_dc = b.unwrap(); // Must exist for a doublecross

            let dc_id = allocations[l_dc].agent().agent_id();

            let sub_n = allocations.len() - l_dc - 1;
            // Deconstruct the new alloc - MUST HAPPEN BEFORE BACKTRACK.
            let (new_agent, new_item) = new_alloc.decompose();
            agents.push(new_agent);
            items.insert(0, new_item);
            // Now we backtrack to the dc agent.
            backtrack(agents, items, &mut allocations, sub_n, false);
            println!(
                "switchright ({}): al={}, id={}, dc_id={}",
                sub_n,
                allocations.len(),
                a_id,
                dc_id
            );
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
        } else {
            println!(
                "addleft: a_id={}, i={}, q={} ({}), p={}",
                new_alloc.agent().agent_id(),
                allocations.len(),
                q0.to_f32().unwrap(),
                new_alloc.item().quality().to_f32().unwrap(),
                p0.to_f32().unwrap()
            );

            // We have a valid allocation.
            allocations.push(new_alloc);
            n -= 1;

            if allocations.len() > 1 {
                *render_pipe.lock().unwrap().deref_mut() = Some(
                    allocations
                        .iter()
                        .map(|x| RenderAllocation::from_allocation(&x, F::one(), epsilon, max_iter))
                        .collect(),
                );

                //std::thread::sleep(Duration::from_millis(300));
            }
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

        let mut q0: F = F::zero();
        let mut p0: F = F::zero();

        if let Some(last) = allocations.last() {
            q0 = last.quality();
            p0 = last.price();
        }
        let q1 = items.first().unwrap().quality();
        // Find the steepest agent.
        // let (a, p1, b) = steepest_rightshift(
        //     agents.as_slice(),
        //     &allocations,
        //     q1,
        //     allocations.len(),
        //     epsilon,
        //     max_iter,
        // )?;

        let (a, p1) = steepest_alignment(
            agents.as_slice(),
            q0,
            p0,
            q1,
            allocations.len(),
            epsilon,
            max_iter,
        )?;

        // if b != allocations.len() - 1 {
        //     println!("NE: b={}, bz={}, p1={}, p1z={}", b, allocations.len() - 1, p1.to_f32().unwrap(), p1z.to_f32().unwrap());
        // }
        // //let p1 = p1z;

        // println!("tickright (al={}, b={}, a_id={}, b_id= {})", allocations.len(), b, a_id, allocations[b].agent().agent_id());

        let mut l_bt = None;
        let mut dcs = check_dc(&allocations, q1, p1, epsilon, max_iter, true);
        if dcs.len() > 1 {
            for dc in &dcs {
                println!("DOUBLEDC: {}", dc.0);
            }
        }
        dcs.sort_by_key(|x| x.0);

        if let Some((l_dc, _)) = dcs.last() {
            l_bt = Some(*l_dc);
        }
        // if b + 1 < allocations.len() {
        //     l_bt = Some(b);
        // }

        let a_id = agents[a].agent_id();
        // Check if there would be a double crossing...
        if let Some(l_bt) = l_bt {
            // Go back to the dc agent and try to go under, if we cant, allocate this agent above.
            // Return allocated items to the pool.

            // todo - maybe issues because dc occurs with immediately previous agent.
            let bt_id = allocations[l_bt].agent().agent_id();
            let sub_n = allocations.len() - l_bt;
            let min_alloc= allocations.len(); //l_dc + 1;
            backtrack(agents, items, &mut allocations, sub_n - 1, false);
            let (mut dc_agent, dc_item) = allocations.pop().unwrap().decompose(); // Backtracks to remove the final dc agent.
            //assert_eq!(dc_id, dc_agent.agent_id());
            items.insert(0, dc_item);
            dc_agent.min_alloc = min_alloc.max(dc_agent.min_alloc + 1);
            println!(
                "switchleft ({}): al={}, id={} -> min_alloc={}, l_bt={}, bt_id={} -> min_alloc={}",
                sub_n,
                allocations.len(),
                a_id,
                agents[a].min_alloc,
                l_bt,
                bt_id,
                dc_agent.min_alloc
            );
            agents.push(dc_agent);

            // As we are shifting all subsequent agents down, we need to also reset their min_alloc, else they may be prevented from moving down properly.
            // if agents[a].min_alloc > 0 {
            //     agents[a].min_alloc -= 1;
            // }

            // Allocate over.
            allocations = align_left(
                agents,
                items,
                allocations,
                sub_n + 1, // We don't add 1 because we have the 'withheld' agent to push up. This will add one after.
                epsilon,
                max_iter,
                render_pipe.clone(),
            )?;

            println!("after: {}", bt_id);

            n -= 1;

            if let Some(last) = allocations.last() {
                q0 = last.quality();
                p0 = last.price();
            } else {
                return Err(SBError::NoSupport);
            }

            // Insert after - we know the agent must prefer this point. This should be done automatically by continuing.
        } else {
            println!(
                "addright: a_id={}, i={}",
                agents[a].agent_id(),
                allocations.len()
            );
            // We have a valid allocation.
            let alloc = Allocation::new(agents.remove(a), items.remove(0), p1);
            allocations.push(alloc);
            n -= 1;
        }
    }

    if (allocations.len() > 1) {
        *render_pipe.lock().unwrap().deref_mut() = Some(
            allocations
                .iter()
                .map(|x| RenderAllocation::from_allocation(&x, F::one(), epsilon, max_iter))
                .collect(),
        );

        //std::thread::sleep(Duration::from_millis(300));
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

    let mut agents: Vec<AgentHolder<A>> = agents
        .into_iter()
        .map(|x| AgentHolder {
            agent: x,
            min_alloc: 0,
        })
        .collect();

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
