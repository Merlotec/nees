use std::{fs, ops::DerefMut, sync::{Arc, Mutex}, time::Duration};

use render::RenderAllocation;
use solver::{switchbranch::{self, AgentHolder}, verify_solution, Agent, Allocation};
use world::{House, Household};
use crate::solver::fractal;
use crate::world::World;

mod distribution;
mod render;
mod solver;
mod world;

fn main() {
    // let allocations = vec![
    //     Allocation::new(Household::new(0, 100f64, 0.5f64, 0.5f64), House::new(0f64, 0f64, None, 0.6f64), 50f64),
    //     Allocation::new(Household::new(0, 120f64, 0.6f64, 0.5f64), House::new(0f64, 0f64, None, 0.8f64), 68f64)
    // ];

    let epsilon = 1e-7;
    let max_iter = 500;
    let n = 200;
    //let mut world = distribution::create_world::<f64>(100, 100);

    let mut world: World<f64>;
    if let Ok(s) = fs::read_to_string("config.json") {
        world = serde_json::from_str(s.as_str()).unwrap();
    } else {
        world = distribution::create_world::<f64>(n, n);
        while !world.validate() {
            world = distribution::create_world::<f64>(n, n);
        }
        fs::write("config.json", serde_json::to_string_pretty(&world).unwrap()).unwrap();
    }

    //fs::write("config.json", serde_json::to_string_pretty(&world).unwrap()).unwrap();

    let to_allocate: Arc<Mutex<Option<Vec<RenderAllocation>>>> = Arc::new(Mutex::new(None));
    let pipe = to_allocate.clone();

    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(5));
        let allocations: Vec<Allocation<f64, Household<f64>, House<f64>>> =
            fractal::root(
                world.households,
                world.houses,
                epsilon,
                max_iter,
                pipe.clone(),
            )
            .unwrap();
        *pipe.lock().unwrap().deref_mut() = Some(
            allocations
                .iter()
                .map(|x| RenderAllocation::from_allocation(&x, 1.0, epsilon, max_iter))
                .collect(),
        );
        if verify_solution(&allocations, epsilon, max_iter) {
            println!("VERIFICATION SUCCESSFUL");
        } else {
            println!("VERIFICATION FAILED");
        }
    });

    render::render_test(to_allocate);
}
