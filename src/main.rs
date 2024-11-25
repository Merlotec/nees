use std::sync::{Arc, Mutex};

use render::RenderAllocation;
use solver::{switchbranch, Agent, Allocation};
use world::{House, Household};

mod render;
mod world;
mod solver;
mod distribution;

fn main() {

    // let allocations = vec![
    //     Allocation::new(Household::new(0, 100f64, 0.5f64, 0.5f64), House::new(0f64, 0f64, None, 0.6f64), 50f64),
    //     Allocation::new(Household::new(0, 120f64, 0.6f64, 0.5f64), House::new(0f64, 0f64, None, 0.8f64), 68f64)
    // ];

    let mut world = distribution::create_world::<f64>(100, 100);

    let allocations = switchbranch::swichbranch(&mut world.households, &mut world.houses, 1e-5, 200).unwrap();

    let render_allocs = allocations.iter().map(|x| RenderAllocation::from_allocation(&x, 1f64, 1e-5, 200)).collect();

    let to_allocate: Arc<Mutex<Option<Vec<RenderAllocation>>>> = Arc::new(Mutex::new(Some(render_allocs)));

    render::render_test(to_allocate);
}
