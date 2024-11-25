use core::alloc;

use super::{utility::indifferent_price, Agent, Allocation, Item};

#[derive(Debug, Copy, Clone)]
pub enum SBError {
    NoIndifference,
    NoCandidate,
    IncomeExceeded,
    NoSupport,
}

pub type SBResult<T> = Result<T, SBError>;

pub fn backtrack<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<A>,
    items: &mut Vec<I>,
    allocations: &mut Vec<Allocation<F, A, I>>,
    n: usize,
) {
    for _ in 0..n {
        let alloc = allocations.pop().unwrap();
        let (agent, item) = alloc.decompose();
        agents.push(agent);
        items.insert(0, item); // Maintain order of items - very important.
    }
}

pub fn align_left<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<A>,
    items: &mut Vec<I>,
    mut allocations: Vec<Allocation<F, A, I>>,
    mut n: usize,
    epsilon: F,
    max_iter: usize,
) -> SBResult<Vec<Allocation<F, A, I>>> {
    assert_eq!(agents.len(), items.len());
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
                .ok_or(SBError::NoIndifference).unwrap(),
        )
    };

    while !items.is_empty() {
        if n == 0 {
            return Ok(allocations);
        }
        n -= 1;
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
                ).ok_or(SBError::NoIndifference).unwrap();

                if to_allocate.is_none() || p_indif < p_min {
                    p_min = p_indif;
                    to_allocate = Some(a);
                }
            }
            // Check if there would be a double crossing...
            let mut p_dc = F::zero();
            let mut l_dc: Option<usize> = None;
            for (l, alloc) in allocations.iter().enumerate() {
                if alloc.agent().income() > p_min {
                    if alloc.agent().utility(p_min, q1) > alloc.utility() {
                        // We have a double crossing.
                        let p_indif = indifferent_price(
                            alloc.agent(),
                            q1,
                            alloc.utility(),
                            p_min,
                            alloc.agent().income(),
                            epsilon,
                            max_iter,
                        ).ok_or(SBError::NoIndifference).unwrap();

                        if l_dc.is_none() || p_indif > p_dc {
                            p_dc = p_indif;
                            l_dc = Some(l);
                        }
                    }
                }
            }

            if let Some(l_dc) = l_dc {
                // Go back to the dc agent and try to go under, if we cant, allocate this agent above.
                let sub_n = allocations.len() - l_dc;
                backtrack(agents, items, &mut allocations, sub_n);

                // Allocate over.
                allocations =
                    align_right(agents, items, allocations, sub_n - 1, epsilon, max_iter)?;

                if let Some(last) = allocations.last() {
                    q0 = last.quality();
                    p0 = last.price();
                } else {
                    q0 = F::zero();
                    p0 = F::zero();
                }
            } else {
                // We have a valid allocation.
                let alloc = Allocation::new(agents.remove(to_allocate.ok_or(SBError::NoCandidate)?), items.remove(0), p0);
                allocations.push(alloc);

                q0 = q1;
                p0 = p_min;
            }
        }
    }

    Ok(allocations)
}

pub fn align_right<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<A>,
    items: &mut Vec<I>,
    mut allocations: Vec<Allocation<F, A, I>>,
    mut n: usize,
    epsilon: F,
    max_iter: usize,
) -> SBResult<Vec<Allocation<F, A, I>>> {
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
        n -= 1;
        // Find the steepest agent.
        let mut to_allocate: Option<usize> = None;

        let q1 = items[0].quality();
        let mut p_min = F::zero();
        for (a, agent) in agents.iter().enumerate() {
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
            ).ok_or(SBError::NoIndifference).expect(&format!("FAILED: {}, n= {}", allocations.len(), n));

            if to_allocate.is_none() || p_indif < p_min {
                p_min = p_indif;
                to_allocate = Some(a);
            }
        }
        // Check if there would be a double crossing...
        let mut p_dc = F::zero();
        let mut l_dc: Option<usize> = None;
        for (l, alloc) in allocations.iter().enumerate() {
            if alloc.agent().income() > p_min {
                if alloc.agent().utility(p_min, q1) > alloc.utility() {
                    // We have a double crossing.
                    let p_indif = indifferent_price(
                        alloc.agent(),
                        q1,
                        alloc.utility(),
                        p_min,
                        alloc.agent().income(),
                        epsilon,
                        max_iter,
                    ).ok_or(SBError::NoIndifference).unwrap();

                    if l_dc.is_none() || p_indif > p_dc {
                        p_dc = p_indif;
                        l_dc = Some(l);
                    }
                }
            }
        }

        if let Some(l_dc) = l_dc {
            // Go back to the dc agent and try to go under, if we cant, allocate this agent above.
            // Return allocated items to the pool.
            let sub_n = allocations.len() - l_dc;
            backtrack(agents, items, &mut allocations, sub_n);

            // Allocate over.
            allocations = align_left(agents, items, allocations, sub_n - 1, epsilon, max_iter)?;

            if let Some(last) = allocations.last() {
                if let Some(next_item) = items.last() {
                    q0 = next_item.quality();
                    p0 = last.indifferent_price(q0, epsilon, max_iter).ok_or(SBError::NoIndifference).unwrap();
                }
            } else {
                return Err(SBError::NoSupport);
            }

            // Insert after - we know the agent must prefer this point. This should be done automatically by continuing.
            continue;
        } else {
            // We have a valid allocation.
            let alloc = Allocation::new(agents.remove(to_allocate.ok_or(SBError::NoCandidate)?), items.remove(0), p_min);
            allocations.push(alloc);

            q0 = q1;
            p0 = p_min;
        }
    }

    Ok(allocations)
}

pub fn swichbranch<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agents: &mut Vec<A>,
    items: &mut Vec<I>,
    epsilon: F,
    max_iter: usize,
) -> SBResult<Vec<Allocation<F, A, I>>> {
    assert_eq!(agents.len(), items.len());
    align_right(agents, items, Vec::new(), items.len(), epsilon, max_iter)
}
