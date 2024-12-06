use std::{fs, ops::DerefMut, sync::{Arc, Mutex}, time::Duration};

use serde::{de::DeserializeOwned, Serialize};
use rand_distr::{uniform::SampleUniform, Distribution, Open01, StandardNormal};
use render::RenderAllocation;
use solver::verify_solution;
use crate::solver::fractal;
use crate::solver::fractal::FractalSettings;
use crate::world::World;

mod distribution;
mod render;
mod solver;
mod world;
mod export;

fn main()  {
    run::<f64>();
}

fn run<F: 'static + Send + num::Float + Serialize + DeserializeOwned>()
where F: SampleUniform, StandardNormal: Distribution<F>, Open01: Distribution<F> {

    let epsilon = F::from(1e-8).unwrap();
    let max_iter = 400;
    let n = 200;
    //let mut world = distribution::create_world::<f64>(100, 100);

    let mut actual_n = n;

    let mut world: World<F>;
    if let Ok(s) = fs::read_to_string("config.json") {
        world = serde_json::from_str(s.as_str()).unwrap();
        actual_n = world.houses.len();
    } else {
        world = distribution::create_world::<F>(n, n);
        while !world.validate() {
            world = distribution::create_world::<F>(n, n);
        }
        fs::write("config.json", serde_json::to_string_pretty(&world).unwrap()).unwrap();
    }

    let to_allocate: Arc<Mutex<Option<Vec<RenderAllocation>>>> = Arc::new(Mutex::new(None));
    let pipe = to_allocate.clone();

    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(5));
        let allocations =
            fractal::root(
                world.households,
                world.houses,
                FractalSettings {
                    epsilon,
                    max_iter,
                },
                Some(pipe.clone()),
            )
            .unwrap();
        *pipe.lock().unwrap().deref_mut() = Some(
            allocations
                .iter()
                .map(|x| RenderAllocation::from_allocation(&x, F::from(100.0).unwrap(), epsilon, max_iter))
                .collect(),
        );
        if verify_solution(&allocations, epsilon, max_iter) {
            println!("VERIFICATION SUCCESSFUL (n={}, epsilon={})", actual_n, epsilon.to_f64().unwrap());
            if let Err(e) = export::serialize_allocations_to_csv(allocations, "sln.csv") {
                println!("Failed to write solution to output sln.csv: {}", e);
            } else {
                println!("Written solution to sln.csv");
            }
        } else {
            println!("VERIFICATION FAILED (n={}, epsilon={})", actual_n, epsilon.to_f64().unwrap());
        }
    });

    render::render_test(to_allocate);
}
