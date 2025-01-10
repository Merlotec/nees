use bevy::reflect::Enum;
use petgraph::matrix_graph::NodeIndex;
use petgraph::visit::Dfs;
use petgraph::{Direction, Graph};

use super::*;
use super::{Agent, Allocation, Item};
use crate::render::RenderAllocation;
use std::collections::{HashSet, VecDeque};
use std::mem;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use petgraph::dot::{Config, Dot};

use super::graph::*;

/// Represents errors that can occur during the fractal equilibrium computation.
#[derive(Debug, Copy, Clone)]
pub enum FractalError {
    /// No price found that makes an agent indifferent between two reference qualities.
    NoIndifference,
    /// No suitable agent could be identified for allocation at the required step.
    NoCandidate,
    /// An agent's income was exceeded during allocation. The optional `usize` may identify the agent index.
    IncomeExceeded(Option<usize>),
    /// No boundary could be determined for the current set of allocations.
    NoBoundary,
    /// Attempted to insert an agent into an invalid position (e.g., slot already occupied).
    InvalidInsertion,
    /// Required an intermediate agent for realignment, but none was available.
    NoIntermediateAgent,
    /// An allocation was found empty where an agent was expected.
    EmptyAllocation,
    /// An allocation would be allocated to a price below the constraint price.
    UnconstrainedBoundary(Index),
    /// Similar to constraint violation, but hypothetical.
    ConstraintBreach,
}

/// A specialized result type for the fractal module.
pub type FractalResult<T> = Result<T, FractalError>;

/// Configuration parameters for fractal computations, such as tolerance and iteration limits.
#[derive(Debug, Copy, Clone)]
pub struct FractalSettings<F: num::Float> {
    /// Convergence tolerance for iterative computations.
    pub epsilon: F,
    /// Maximum number of iterations for convergence attempts.
    pub max_iter: usize,
    /// The constraint price (lowest price in the configuration).
    pub constraint_price: F,
}

/// A container that can either hold an `Agent` or be empty. This is useful when temporarily
/// displacing agents during the reallocation steps.
#[derive(Clone)]
pub enum AgentHolder<const D: usize, A: Agent<D>> {
    /// No agent currently held.
    Empty,
    /// An agent is stored here.
    Agent(A),
    Deactivated(A),
}

impl<const D: usize, A: Agent<D>> std::fmt::Debug for AgentHolder<D, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentHolder::Empty => write!(f, "Empty"),
            AgentHolder::Agent(a) => write!(f, "Agent"),
            AgentHolder::Deactivated(a) => write!(f, "Deactivated"),
        }
    }
}

impl<const D: usize, A: Agent<D>> AgentHolder<D, A> {
    /// Checks if this holder currently contains an agent.
    pub fn has_agent(&self) -> bool {
        matches!(self, AgentHolder::Agent(_) | AgentHolder::Deactivated(_))
    }

    /// Removes the agent from this holder and returns it as a new `AgentHolder`.
    pub fn take(&mut self) -> Self {
        let mut other = AgentHolder::Empty;
        mem::swap(self, &mut other);
        other
    }

    /// Converts this `AgentHolder` into an `Option<A>`, consuming it.
    pub fn to_option(self) -> Option<A> {
        if let AgentHolder::Agent(a) | AgentHolder::Deactivated(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// Returns a reference to the contained agent, or panics if empty.
    pub fn agent(&self) -> &A {
        match self {
            AgentHolder::Agent(a) | AgentHolder::Deactivated(a) => a,
            AgentHolder::Empty => panic!("Attempted to access an empty AgentHolder."),
        }
    }

    pub fn active(&self) -> bool {
        matches!(self, AgentHolder::Agent(_))
    }

    pub fn deactivate(&mut self) {
        if let AgentHolder::Agent(a) | AgentHolder::Deactivated(a) = self.take() {
            *self = AgentHolder::Deactivated(a);
        }
    }

    pub fn activate(&mut self) {
        if let AgentHolder::Agent(a) | AgentHolder::Deactivated(a) = self.take() {
            *self = AgentHolder::Agent(a);
        }
    }
}

impl<const D: usize, A: Agent<D>> Agent<D> for AgentHolder<D, A> {
    type FloatType = A::FloatType;

    fn agent_id(&self) -> usize {
        self.agent().agent_id()
    }

    fn income(&self) -> Self::FloatType {
        self.agent().income()
    }

    fn utility(&self, price: Self::FloatType, quality: [Self::FloatType; D]) -> Self::FloatType {
        self.agent().utility(price, quality)
    }
}

// pub struct AllocationGraph<
//     const D: usize,
//     F: num::Float,
//     A: Agent<D, FloatType = F>,
//     I: Item<D, FloatType = F>,
// > {
//     pub graph: petgraph::Graph<Allocation<D, F, A, I>, Direction>,
// }

// impl<const D: usize, F: num::Float, A: Agent<D, FloatType = F>, I: Item<D, FloatType = F>>
//     AllocationGraph<D, F, A, I>
// {
//     pub fn insert_to(&mut self) {
//         self.graph.add
//     }
// }
//

/// Finds the agent to allocate next when moving "up" (increasing indices) along the quality axis.
/// This is used when q0 < q1 to determine the next agent that achieves a non-envy state.
/// Choosing q1 is not obvious like in the n=1 case.
pub fn next_agent_up<const D: usize, F: num::Float, A: Agent<D, FloatType = F>>(
    agents: &[A],
    q0: [F; D],
    p0: F,
    q1: [F; D],
    settings: &FractalSettings<F>,
) -> FractalResult<(usize, F)> {
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;

    for (a, agent) in agents.iter().enumerate() {
        if agent.income() <= p0 {
            return Err(FractalError::IncomeExceeded(Some(a)));
        }
        let u = agent.utility(p0, q0);
        if let Some(p_indif) = indifferent_price(
            agent,
            q1,
            u,
            settings.constraint_price,
            agent.income(),
            settings.epsilon,
            settings.max_iter,
        ) {
            if to_allocate.is_none() || p_indif < p_min {
                p_min = p_indif;
                to_allocate = Some(a);
            }
        }
        // } else {
        //     if agent.utility(settings.constraint_price, q1) < u {
        //         return Err(FractalError::ConstraintBreach);
        //     }
        // }
    }

    to_allocate
        .map(|x| (x, p_min))
        .ok_or(FractalError::NoCandidate)
}

fn debug_quality<const D: usize, F: num::Float>(q: &[F; D]) -> String {
    let mut buf = String::new();
    for f in q.iter() {
        buf += &f.to_f32().unwrap().to_string();
        buf += ",";
    }
    buf
}

/// Attempts to determine a boundary point in the allocations for a given quality.
/// A boundary point corresponds to an allocation where an agent is indifferent at some price.
pub fn partial_boundary<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &AllocationGraph<D, F, A, I>,
    quality: [F; D],
    settings: &FractalSettings<F>,
) -> FractalResult<(Option<Index>, F)> {
    let mut p_max: F = num::zero();
    let mut i_max: Option<Index> = None;

    let mut null_ind = Vec::new();

    for i in graph.node_indices().rev() {
        if !graph[i].agent().active() {
            continue;
        }
        let alloc = &graph[i];
        if i_max.is_none() {
            if let Some(p) = alloc
                .indifferent_price(quality, settings) {
                i_max = Some(i);
                p_max = p;
            } else {
                null_ind.push(i);
            }
        } else {
            let u_other = alloc.agent().utility(p_max, quality);
            if u_other > alloc.utility() {
                if let Some(p) = alloc
                    .indifferent_price(quality, settings) {
                    i_max = Some(i);
                    p_max = p;
                } else {
                    null_ind.push(i);
                }
            }
        }
    }

    if let Some(i_max) = i_max {
        for i in null_ind {
            if graph[i].agent().utility(p_max, quality) > graph[i].utility() + settings.epsilon {
                if graph[i].agent().utility(settings.constraint_price, quality) > graph[i].utility() {
                    // We have a constraint violation.
                    return Err(FractalError::NoBoundary);
                }
            }
        }
        Ok((Some(i_max), p_max))
    } else {
        if !null_ind.is_empty() {
            // Check for constraint breach.
            for i in null_ind.iter() {
                if graph[*i].agent().utility(settings.constraint_price, quality) > graph[*i].utility() {
                    // We have a constraint violation.
                    return Err(FractalError::NoBoundary);
                }
            }
        }
        Ok((None, settings.constraint_price))
    }
}

/// The recursive allocation function that ensures no-envy conditions are maintained.
/// It attempts to place an agent at allocation index `i`, potentially resolving double-crosses
/// by rearranging envelopes above or below, depending on the direction.
pub fn restore<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &mut AllocationGraph<D, F, A, I>,
    i: Index,
    settings: &FractalSettings<F>,
) -> FractalResult<()> {
    // Check to see if this agent prefers any other agents.
    for j in graph.node_indices() {
        if !graph[j].agent().active() {
            continue;
        }
        if graph[i].prefers(&graph[j], settings.epsilon) {
            // Check if we have a cycle in the graph, if so, we must promote.
            // Having a cycle is equivalent to an envelope breach in the 1D case.

            if let Some(mut cycle) = find_path_bfs(graph, j, i) {
                println!("cycle {:?}", &cycle);
                // Deal with a cycle.
                // To do this we essentially move each agent to the allocation along its edge, and since we have a cycle, each agent can move.
                // Start with the last item in the path.
                // Must also reallocate anything that has updated.
                // So poison nodes connected to j, then reallocate them all in the opposite order starting from i at allocation j (by cycling around).

                // Rotate along the path.
                rotate_agents_along_path(graph, &cycle);

                // Get the allocations in order of how we want to reallocate them.
                let mut path = multi_dir_bfs(graph, i, j);
                //let mut to_allocate = path;
               // to_allocate.insert(0, j);
                // We want to allocate the allocatino j has moved to last.

                let mut to_allocate = path;

                to_allocate.retain(|x| !cycle.contains(x));

                to_allocate.push(cycle.remove(0));
                cycle.reverse();
                to_allocate.append(&mut cycle);
                // Add the 'rejoin' allocation as the first allocation to reallocate.
                //to_allocate.insert(0, j);

                reallocate_path(graph, &to_allocate, settings)?;
                std::fs::write("cycle.dot", format!("{:?}", Dot::with_config(graph as &_, &[Config::EdgeNoLabel])))
                    .expect("Failed to write DOT file");
            } else {
                //println!("pull: {:?}, {:?}", i, j);
                // Pull back this allocation to the boundary and allocate recursively.
                let (b, p) = partial_boundary(graph, graph[j].quality(), settings)?;
                allocate(graph, j, b, p, settings)?;
            }
        }
    }

    Ok(())
}

/// Reallocates in the order specified.
pub fn reallocate_path<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &mut AllocationGraph<D, F, A, I>,
    to_allocate: &[Index],
    settings: &FractalSettings<F>,
) -> FractalResult<()> {
    let mut to_allocate = to_allocate.to_vec();

    deactivate_path(graph, &to_allocate, settings)?;

    while !to_allocate.is_empty() {
        // Allocate i
        let l = to_allocate.pop().unwrap();
        let (b, p) =
            partial_boundary(graph, graph[l].quality(), settings)?;
        allocate(graph, l, b, p, settings)?;
    }
    Ok(())
}

pub fn deactivate_path<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &mut AllocationGraph<D, F, A, I>,
    to_deactivate: &[Index],
    settings: &FractalSettings<F>,
) -> FractalResult<()> {
    for a in to_deactivate.iter() {
        graph[*a].agent_mut().deactivate();
    }

    graph.retain_edges(|g, e| {
        let (src, target) = g.edge_endpoints(e).unwrap();

        if !to_deactivate.contains(&src) && !to_deactivate.contains(&target) {
            true
        } else {
            assert!(!g[target].agent().active());
            false
        }
    });

    Ok(())
}

pub fn verify_integrity<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &mut AllocationGraph<D, F, A, I>,
    settings: &FractalSettings<F>,
) -> bool {
    for i in graph.node_indices() {
        if !graph[i].agent().active() {
            continue;
        }
        if graph[i].price() > settings.constraint_price {
            if graph.edges_directed(i, Direction::Incoming).count() == 0 {
                println!("Node {:?} has no incoming edges and price {}", i, graph[i].price().to_f32().unwrap());
                return false;
            }
        }
    }
    true
}

pub fn allocate<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &mut AllocationGraph<D, F, A, I>,
    i: Index,
    b: Option<Index>,
    p: F,
    settings: &FractalSettings<F>,
) -> FractalResult<()> {

    if p < graph[i].agent().income() {
        graph[i].set_price(p);

        // Remove old edges.
        graph.retain_edges(|g, e| {
            let (_, target) = g.edge_endpoints(e).unwrap();
            target != i
        });
        if let Some(b) = b {
            assert_ne!(i, b);
            // Add new edge.
            assert!(graph[b].agent().active());
            graph.add_edge(b, i, ());
        }

        graph[i].agent_mut().activate();

        restore(graph, i, settings)?;
    } else {
        println!("income_exceeded: i: {:?}, b: {:?}", i, b);
        // Find the nearest allocation at which we could allocate.
        let income = graph[i].agent().income();
        let b = b.ok_or(FractalError::NoBoundary)?;
        let mut path = traverse_backwards_until(graph, b, |alloc| alloc.price() < income).ok_or(FractalError::IncomeExceeded(None))?;
        path.insert(0, i);
        // cycle the path.
        rotate_agents_along_path(graph, &path);
        let last = *path.last().unwrap();
        let mut to_allocate = bfs_forward(graph, last);
        //to_allocate.insert(0, last);
        let mut invalid_path = bfs_forward(graph, i);
        to_allocate.insert(0, invalid_path.remove(0));
        to_allocate.append(&mut invalid_path);
        reallocate_path(graph, &to_allocate, settings)?;
    }
    //assert!(verify_integrity(graph, settings));
    Ok(())
}

/// The entry point for computing a non-envy equilibrium allocation of agents to items.
/// - `agents`: a vector of agents (with heterogeneous utilities and incomes).
/// - `items`: a vector of items (with heterogeneous qualities).
/// - `settings`: fractal computation settings (tolerance, iterations).
/// - `render_pipe`: optional pipe for rendering allocations as they evolve.
///
/// Returns a vector of `Allocation` representing a final non-envy equilibrium if successful.
pub fn root<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    mut agents: Vec<A>,
    mut items: Vec<I>,
    settings: FractalSettings<F>,
) -> FractalResult<Vec<Allocation<D, F, A, I>>> {
    assert_eq!(agents.len(), items.len());

    // We first want to order the itmes using some reasonable mechanism.
    // The better the estimate, the fewer cycles we will have to fix in our graph.
    items.sort_by(|a, b| {
        square_of_sum_sqrts(a.quality())
            .partial_cmp(&square_of_sum_sqrts(b.quality()))
            .unwrap()
    });

    let mut graph: AllocationGraph<D, F, A, I> = AllocationGraph::new();

    // Main allocation loop: each iteration picks an item and finds the appropriate agent to allocate.
    while !items.is_empty() {

        let q0 = items.first().unwrap().quality();
        let (b, p0) = partial_boundary(&graph, q0, &settings)?;

        let a = if items.len() > 1 {
            // Find the agent that ensures a non-envy equilibrium with the next item's quality.
            match next_agent_up(&agents, q0, p0, items[1].quality(), &settings) {
                Ok((a, _)) => a,
                Err(FractalError::IncomeExceeded(Some(a))) => a,
                Err(e) => return Err(e),
            }
        } else {
            // Only one agent and one item left, must be allocated directly.
            0
        };

        let new_allocation =
            Allocation::new(AgentHolder::Agent(agents.remove(a)), items.remove(0), p0);

        let i = graph.add_node(new_allocation);
        println!("add: {:?}", i);
        // Optionally store a snapshot of the current allocations for rendering or analysis.
        // if allocations.len() > 1 {
        //     if let Some(render_pipe) = &render_pipe {
        //         *render_pipe.lock().unwrap().deref_mut() = Some(
        //             allocations
        //                 .iter()
        //                 .map(|x| RenderAllocation::from_allocation(x, F::from(100.0).unwrap(), settings.epsilon, settings.max_iter))
        //                 .collect(),
        //         );
        //     }
        // }

        // Recursively ensure no-envy conditions by aligning envelopes if needed.
        allocate(&mut graph, i, b, p0, &settings)?;
    }

    std::fs::write("sln.dot", format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel])))
        .expect("Failed to write DOT file");

    // Final clean-up: extract the actual agents and items from `AgentHolder` and return the final solution.
    let mut cleaned = Vec::with_capacity(graph.node_count());
    for alloc in take_all_nodes(&mut graph) {
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

    println!("cleaned: {}", cleaned.len());
    Ok(cleaned)
}

/// Remove all nodes from the graph and return their data in a `Vec<T>`.
fn take_all_nodes<T, E>(graph: &mut Graph<T, E>) -> Vec<T> {
    // Collect existing node indices first
    let indices: Vec<Index> = graph.node_indices().collect();

    // Reserve capacity to store all nodes
    let mut extracted = Vec::with_capacity(indices.len());

    // Remove each node by index, retrieving its data (weight)
    for ix in indices.into_iter().rev() {
        if let Some(weight) = graph.remove_node(ix) {
            extracted.push(weight);
        }
    }

    extracted.reverse();
    extracted
}

fn square_of_sum_sqrts<const D: usize, F: num::Float>(arr: [F; D]) -> F {
    // 1. Sum the square roots
    let sum_of_sqrts_iter = arr.iter().map(|&x| x.sqrt());
    let mut sum = F::zero();
    for s in sum_of_sqrts_iter {
        sum = sum + s;
    }
    // 2. Square that sum
    sum.powi(2)
}
