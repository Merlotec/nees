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

pub type AllocationGraph<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
> = petgraph::Graph<Allocation<D, F, AgentHolder<D, A>, I>, ()>;

pub type Index = NodeIndex<u32>;

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
#[derive(Debug, Clone)]
pub enum AgentHolder<const D: usize, A: Agent<D>> {
    /// No agent currently held.
    Empty,
    /// An agent is stored here.
    Agent(A),
    Deactivated(A),
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

/// Find a single path from `start` to `goal` using BFS.
/// Returns `None` if no path is found.
fn find_path_bfs<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &AllocationGraph<D, F, A, I>,
    start: Index,
    goal: Index,
) -> Option<Vec<Index>> {
    // Queue for BFS, storing (current_node, predecessor)
    let mut queue = VecDeque::new();
    // Keep track of visited nodes
    let mut visited = vec![false; graph.node_count()];
    // Keep track of how we reached each node
    let mut parent = vec![None; graph.node_count()];

    // Initialize BFS
    visited[start.index()] = true;
    queue.push_back(start);

    // BFS loop
    while let Some(current) = queue.pop_front() {
        if current == goal {
            // We found the goal, reconstruct the path
            let mut path = vec![current];
            // Walk backwards from `goal` to `start` using `parent`
            let mut p = parent[current.index()];
            while let Some(prev) = p {
                path.push(prev);
                p = parent[prev.index()];
            }
            path.reverse();
            return Some(path);
        }

        // Traverse neighbors
        for neighbor in graph.neighbors_directed(current, petgraph::Direction::Outgoing) {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                parent[neighbor.index()] = Some(current);
                queue.push_back(neighbor);
            }
        }
    }

    // No path found
    None
}

/// Perform a reverse BFS starting from `start_index`, traversing
/// *incoming* edges. We stop exploring neighbors if we find a node
/// whose `agent` matches `end_agent`.
///
/// Returns a `Vec<NodeIndex>` in the order they were visited by BFS.
fn reverse_bfs_with_termination<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &AllocationGraph<D, F, A, I>,
    start_index: Index,
    end_index: Index,
) -> Vec<Index> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut order = Vec::new();

    // Initialize the BFS queue
    visited.insert(start_index);
    queue.push_back(start_index);

    while let Some(current) = queue.pop_front() {
        // Record the visitation order
        order.push(current);

        // Look at predecessors in the graph (incoming edges)
        for neighbor in graph.neighbors_directed(current, Direction::Incoming) {
            // Skip if already visited
            if visited.contains(&neighbor) {
                continue;
            }

            // If this neighbor has the end agent, skip enqueueing,
            // effectively terminating this branch.
            if neighbor == end_index {
                continue;
            }

            // Mark visited and enqueue
            visited.insert(neighbor);
            queue.push_back(neighbor);
        }
    }

    order
}
/// Traverse from `start` in both directions (incoming and outgoing edges),
/// but handle `end` specially:
///  - Don't add `end` to the visited order.
///  - Don't explore incoming edges of `end`.
///  - Do explore outgoing edges of `end`.
fn multi_dir_bfs<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &AllocationGraph<D, F, A, I>,  // or Graph<MyNode, E>, etc.
    start: Index,
    end: Index,
) -> Vec<Index> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut order = Vec::new();

    // Begin BFS from the `start` node
    visited.insert(start);
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == end {
            // Skip adding `end` to the order
            // BUT do explore forward (outgoing) edges from `end`.
            for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        } else {
            // Regular node (not `end`):
            // 1) Add `current` to the BFS visitation order
            order.push(current);

            // 2) Explore incoming edges (backwards)
            for neighbor in graph.neighbors_directed(current, Direction::Incoming) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }

            // 3) Explore outgoing edges (forwards)
            for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    order
}

/// Rotate the agents along the given `path` in the direction of edges,
/// moving the last agent in the path to the first node.
fn rotate_agents_along_path<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &mut AllocationGraph<D, F, A, I>,
    path: &[Index],
) {
    if path.len() < 2 {
        return; // Nothing to rotate if path has 0 or 1 node
    }

    // 1. Temporarily take the agent from the last node in the path
    let last_index = path[path.len() - 1];
    let last_agent = graph[last_index].agent.take();

    // 2. Move agents forward along the path in reverse order
    //    (so we don't overwrite an agent before moving it).
    //    For path [0, 1, 2, 3], we do:
    //        Node[3] <- Node[2]
    //        Node[2] <- Node[1]
    //        Node[1] <- Node[0]
    //        Node[0] <- last_agent
    //    This effectively rotates them one step forward in edge direction.
    for i in (1..path.len()).rev() {
        let to = path[i];
        let from = path[i - 1];
        // Move the agent from `from` to `to`
        graph[to].agent = graph[from].agent.take();
    }

    // 3. Place the previously-last agent into the first node
    let first_index = path[0];
    graph[first_index].agent = last_agent;
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

            if let Some(cycle) = find_path_bfs(graph, j, i) {
                println!("cycle({}): {:?}, {:?}", cycle.len(), i, j);
                // Deal with a cycle.
                // To do this we essentially move each agent to the allocation along its edge, and since we have a cycle, each agent can move.
                // Start with the last item in the path.
                // Must also reallocate anything that has updated.
                // So poison nodes connected to j, then reallocate them all in the opposite order starting from i at allocation j (by cycling around).

                // Rotate along the path.
                rotate_agents_along_path(graph, &cycle);

                // Get the allocations in order of how we want to reallocate them.
                let mut to_allocate = multi_dir_bfs(graph, i, j);

                // Start with agent j
                graph[j].agent_mut().deactivate();
                // Deactivate intermediate agents
                for a in to_allocate.iter() {
                    graph[*a].agent_mut().deactivate();
                }
                println!("to_allocate: {:?}", to_allocate);

                std::fs::write("graph_cycle.dot", format!("{:?}", Dot::with_config(graph as &_, &[Config::NodeIndexLabel, Config::EdgeNoLabel])))
                    .expect("Failed to write DOT file");

                // Allocate j first.
                {
                    let (b, p) =
                        partial_boundary(graph, graph[j].quality(), settings)?;
                    graph[j].set_price(p);
                    allocate(graph, j, b, settings)?;
                }

                while !to_allocate.is_empty() {
                    // Allocate i
                    let l = *to_allocate.last().unwrap();
                    let (b, p) =
                        partial_boundary(graph, graph[l].quality(), settings)?;
                    graph[l].set_price(p);
                    allocate(graph, l, b, settings)?;
                    to_allocate.pop();
                }
            } else {
                println!("pull: {:?}, {:?}", i, j);
                // Pull back this allocation to the boundary and allocate recursively.
                let (b, p) = partial_boundary(graph, graph[j].quality(), settings)?;
                if p > graph[j].agent().income() {
                    std::fs::write("graph.dot", format!("{:?}", Dot::with_config(graph as &_, &[Config::NodeIndexLabel, Config::EdgeNoLabel])))
                        .expect("Failed to write DOT file");
                    panic!("boom!");
                }
                graph[j].set_price(p);
                allocate(graph, j, b, settings)?;
            }
        }
    }

    Ok(())
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
    settings: &FractalSettings<F>,
) -> FractalResult<()> {
    // Remove old edges.
    graph.retain_edges(|g, e| {
        let (_, target) = g.edge_endpoints(e).unwrap();
        target != i
    });
    if let Some(b) = b {
        // Add new edge.
        assert!(graph[b].agent().active());
        graph.add_edge(b, i, ());
    }

    graph[i].agent_mut().activate();

    restore(graph, i, settings)?;
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
        allocate(&mut graph, i, b, &settings)?;
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
