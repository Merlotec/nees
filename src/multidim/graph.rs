use std::collections::{HashSet, VecDeque};
use petgraph::{Direction, Graph};
use petgraph::graph::NodeIndex;
use crate::multidim::{Agent, Allocation, Item};
use crate::multidim::allocate::{AgentHolder};


pub type AllocationGraph<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
> = petgraph::Graph<Allocation<D, F, AgentHolder<D, A>, I>, ()>;

pub type Index = NodeIndex<u32>;

/// Find a single path from `start` to `goal` using BFS.
/// Returns `None` if no path is found.
pub fn find_path_bfs<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D,  FloatType= F>,
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
pub fn reverse_bfs_with_termination<
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
pub fn multi_dir_bfs<
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
pub fn rotate_agents_along_path<
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


/// Reconstruct the path from `start` to `end` using the `parent` array.
/// Returns a Vec of node indices: [start, ..., end].
fn reconstruct_path(
    start: Index,
    end: Index,
    parent: &[Option<Index>],
) -> Vec<Index> {
    let mut path = vec![end];
    let mut current = end;
    // Walk backwards until we reach `start`
    while current != start {
        let p = parent[current.index()]
            .expect("No parent found while reconstructing path. This shouldn't happen if BFS is correct.");
        path.push(p);
        current = p;
    }
    path.reverse();
    path
}

/// Traverse backwards from `start` by following incoming edges.
/// Stop if we find a node that satisfies `predicate`.
/// Return the path [start -> ... -> found_node] if successful, or `None` if no match was found.
pub fn traverse_backwards_until<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
    G
>(
    graph: &AllocationGraph<D, F, A, I>,
    start: Index,
    mut predicate: G,
) -> Option<Vec<Index>>
where
// The predicate reads node data (N) and returns a bool
    G: FnMut(&Allocation<D, F, AgentHolder<D, A>, I>) -> bool,
{
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    // Store each node's parent to reconstruct path
    // `parent[i] = Some(j)` means we reached i from j.
    let mut parent = vec![None; graph.node_count()];

    // Initialize BFS
    visited.insert(start);
    queue.push_back(start);

    // If the start node itself satisfies the predicate, we can return immediately.
    if predicate(&graph[start]) {
        return Some(vec![start]);
    }

    // Standard BFS loop, but only along incoming edges
    while let Some(current) = queue.pop_front() {
        // For each predecessor of `current`
        for neighbor in graph.neighbors_directed(current, Direction::Incoming) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                parent[neighbor.index()] = Some(current);

                // Check the predicate
                if predicate(&graph[neighbor]) {
                    // Found a satisfying node, reconstruct the path and return
                    return Some(reconstruct_path(start, neighbor, &parent));
                }

                queue.push_back(neighbor);
            }
        }
    }

    // No match found
    None
}

/// Perform a breadth-first search (BFS) starting from `start`,
/// exploring only *outgoing* edges, and return a vector of
/// node indices in the order they were visited.
pub fn bfs_forward<
    const D: usize,
    F: num::Float,
    A: Agent<D, FloatType = F>,
    I: Item<D, FloatType = F>,
>(
    graph: &AllocationGraph<D, F, A, I>, // Or `Graph<MyNode, E>` if you store custom data
    start: Index,
) -> Vec<Index> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut order = Vec::new();

    // Initialize BFS with the start node
    visited.insert(start);
    queue.push_back(start);

    // Standard BFS loop
    while let Some(current) = queue.pop_front() {
        // Record the current node in visitation order
        order.push(current);

        // Explore all outgoing edges from `current`
        for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
            // If the neighbor hasn't been visited, mark and enqueue it
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    order
}