use std::{
    fs,
    ops::DerefMut,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::distribution::DistributionParams;
use crate::solver::fractal;
use crate::solver::fractal::FractalSettings;
use crate::world::World;
use rand_distr::{uniform::SampleUniform, Distribution, Open01, StandardNormal};
use render::RenderAllocation;
use serde::{de::DeserializeOwned, Serialize};
use solver::verify_solution;

mod cstats;
mod distribution;
mod export;
mod multidim;
mod render;
mod solver;
mod vectorrender;
mod world;

fn main() {
    //run_cstat::<f64>();
    //run_config::<f64>()
    multidim::world::test_multidim::<7, f64>(300);
}

fn run_cstat<F: 'static + Send + num::Float + Serialize + DeserializeOwned>()
where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let epsilon = F::from(1e-8).unwrap();
    let max_iter = 400;
    let n = 500;

    cstats::run_all_cstat(n, n, &FractalSettings { epsilon, max_iter })
}

fn run_config<F: 'static + Send + num::Float + Serialize + DeserializeOwned>()
where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let epsilon = F::from(1e-8).unwrap();
    let max_iter = 400;
    let n = 500;

    let params = DistributionParams {
        inc_mean: F::from(100.0).unwrap(),
        asp_std: F::from(0.15).unwrap(),
        ..Default::default()
    };

    let mut actual_n = n;

    let mut world: World<F>;
    if let Ok(s) = fs::read_to_string("config.json") {
        world = serde_json::from_str(s.as_str()).unwrap();
        actual_n = world.houses.len();
    } else {
        world = distribution::create_world::<F>(n, n, &params);
        while !world.validate() {
            world = distribution::create_world::<F>(n, n, &params);
        }
        fs::write("config.json", serde_json::to_string_pretty(&world).unwrap()).unwrap();
    }

    let to_allocate: Arc<Mutex<Option<Vec<RenderAllocation>>>> = Arc::new(Mutex::new(None));
    let pipe = to_allocate.clone();

    std::thread::spawn(move || {
    //     std::thread::sleep(Duration::from_secs(5));
    //     let allocations = fractal::root(
    //         world.households,
    //         world.houses,
    //         F::zero(),
    //         FractalSettings { epsilon, max_iter },
    //         Some(pipe.clone()),
    //     )
    //     .unwrap();

        let allocations = multidim::allocate::root(
            world.households,
            world.houses,
            multidim::allocate::FractalSettings {
                epsilon,
                max_iter,
                constraint_price: F::zero(),
            },
        )
        .unwrap();

        let allocations: Vec<solver::Allocation<_, _, _>> =
            allocations.into_iter().map(|x| x.into()).collect();

        let ra: Vec<RenderAllocation> = allocations
            .iter()
            .map(|x| {
                RenderAllocation::from_allocation(&x, F::from(1.0).unwrap(), epsilon, max_iter)
            })
            .collect();

        let svg = vectorrender::render_allocations_to_svg(&ra, None, 1000, 1000, false);
        std::fs::write("render.svg", svg).unwrap();

        println!("Rendered allocation");
        *pipe.lock().unwrap().deref_mut() = Some(ra);

        if verify_solution(&allocations, epsilon, max_iter) {
            println!(
                "VERIFICATION SUCCESSFUL (n={}, epsilon={})",
                actual_n,
                epsilon.to_f64().unwrap()
            );
            if let Err(e) = export::serialize_allocations_to_csv(allocations, "sln.csv") {
                println!("Failed to write solution to output sln.csv: {}", e);
            } else {
                println!("Written solution to sln.csv");
            }
        } else {
            println!(
                "VERIFICATION FAILED (n={}, epsilon={})",
                actual_n,
                epsilon.to_f64().unwrap()
            );
        }
    });

    render::render_test(to_allocate);
}
