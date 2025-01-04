use std::{
    fs,
    ops::DerefMut,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::distribution::{create_world, DistributionParams};
use crate::render::RenderAllocation;
use crate::solver::fractal;
use crate::solver::fractal::FractalSettings;
use crate::solver::{verify_solution, Item};
use crate::world::{House, World};
use crate::{distribution, export};
use rand_distr::{uniform::SampleUniform, Distribution, Open01, StandardNormal};
use serde::{de::DeserializeOwned, Serialize};

pub fn run_all_cstat<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let mut w0: World<F>;
    if let Ok(s) = fs::read_to_string("config_cstats.json") {
        w0 = serde_json::from_str(s.as_str()).unwrap();
    } else {
        w0 = distribution::create_world::<F>(school_count, house_count, &Default::default());
        while !w0.validate() {
            w0 = distribution::create_world::<F>(school_count, house_count, &Default::default());
        }
        fs::write(
            "config_cstats.json",
            serde_json::to_string_pretty(&w0).unwrap(),
        )
        .unwrap();
    }

    cstat_repitition(school_count, house_count, settings, Some(&w0.houses));
    cstat_inc_mean(school_count, house_count, settings, Some(&w0.houses));
    cstat_inc_cv(school_count, house_count, settings, Some(&w0.houses));
    cstat_asp_mean(school_count, house_count, settings, Some(&w0.houses));
    cstat_asp_std(school_count, house_count, settings, Some(&w0.houses));
    // cstat_qual_loc(school_count, house_count, settings);
    // cstat_qual_scale(school_count, house_count, settings);
    cstat_qual_separation(
        school_count,
        house_count,
        settings,
        Some(&w0.houses),
        F::from(0.1).unwrap(),
    );
    println!("Finished running comparative statics!");
}

pub fn cstat_repitition<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let params = [DistributionParams::default(); 10];

    run_params(school_count, house_count, &params, houses, settings, "rep");
}

pub fn cstat_inc_mean<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let params = [
        DistributionParams {
            inc_mean: F::from(20000.0).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            inc_mean: F::from(40000.0).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            inc_mean: F::from(80000.0).unwrap(),
            ..Default::default()
        },
    ];

    run_params(
        school_count,
        house_count,
        &params,
        houses,
        settings,
        "inc_mean",
    );
}

pub fn cstat_inc_cv<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let params = [
        DistributionParams {
            inc_cv: F::from(0.5).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            inc_cv: F::from(1.0).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            inc_cv: F::from(2.0).unwrap(),
            ..Default::default()
        },
    ];

    run_params(
        school_count,
        house_count,
        &params,
        houses,
        settings,
        "inc_cv",
    );
}

pub fn cstat_asp_mean<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let params = [
        DistributionParams {
            asp_mean: F::from(0.4).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            asp_mean: F::from(0.5).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            asp_mean: F::from(0.6).unwrap(),
            ..Default::default()
        },
    ];

    run_params(
        school_count,
        house_count,
        &params,
        houses,
        settings,
        "asp_mean",
    );
}

pub fn cstat_asp_std<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let params = [
        DistributionParams {
            asp_std: F::from(0.05).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            asp_std: F::from(0.1).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            asp_std: F::from(0.15).unwrap(),
            ..Default::default()
        },
    ];

    run_params(
        school_count,
        house_count,
        &params,
        houses,
        settings,
        "asp_std",
    );
}

pub fn cstat_qual_loc<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let qual_loc = F::from(-0.13494908716854648).unwrap();

    let params = [
        DistributionParams {
            qual_loc: qual_loc - F::from(0.1).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            qual_loc,
            ..Default::default()
        },
        DistributionParams {
            qual_loc: qual_loc + F::from(0.1).unwrap(),
            ..Default::default()
        },
    ];

    run_params(
        school_count,
        house_count,
        &params,
        houses,
        settings,
        "qual_loc",
    );
}

pub fn cstat_qual_scale<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let qual_scale = F::from(1.1861854805071834).unwrap();
    let params = [
        DistributionParams {
            qual_scale: qual_scale - F::from(0.2).unwrap(),
            ..Default::default()
        },
        DistributionParams {
            qual_scale,
            ..Default::default()
        },
        DistributionParams {
            qual_scale: qual_scale + F::from(0.2).unwrap(),
            ..Default::default()
        },
    ];

    run_params(
        school_count,
        house_count,
        &params,
        houses,
        settings,
        "qual_scale",
    );
}

pub fn cstat_qual_separation<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    settings: &FractalSettings<F>,
    houses: Option<&[House<F>]>,
    sep_tick: F,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    let mut world = distribution::create_world::<F>(school_count, house_count, &Default::default());
    if let Some(houses) = &houses {
        world.houses = houses.to_vec();
    }

    let mut total_quality = F::zero();

    for h in world.houses.iter() {
        total_quality = total_quality + h.quality();
    }

    let mean_quality: F = total_quality / F::from(world.houses.len()).unwrap();

    for i in 0..3 {
        let world_i = world.clone();

        let allocations = fractal::root(
            world_i.households,
            world_i.houses,
            F::zero(),
            settings.clone(),
            None,
        )
        .unwrap();

        if verify_solution(&allocations, settings.epsilon, settings.max_iter) {
            println!(
                "VERIFICATION SUCCESSFUL (n={}, epsilon={})",
                house_count,
                settings.epsilon.to_f64().unwrap()
            );
            if let Err(e) =
                export::serialize_allocations_to_csv(allocations, &format!("qual_sep_{}.csv", i))
            {
                println!("Failed to write solution to output sln.csv: {}", e);
            } else {
                println!("Written solution to sln.csv");
            }
        } else {
            println!(
                "VERIFICATION FAILED (n={}, epsilon={})",
                house_count,
                settings.epsilon.to_f64().unwrap()
            );
        }

        // Update inequality of schools.
        for item in world.houses.iter_mut() {
            // Exacerbate any distance from mean proportional to the distance.
            let diff: F = item.school_quality - mean_quality;
            item.school_quality =
                item.school_quality + diff.abs().sqrt() * diff.signum() * sep_tick;
        }
    }
}

pub fn run_params<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    school_count: usize,
    house_count: usize,
    param_sets: &[DistributionParams<F>],
    houses: Option<&[House<F>]>,
    settings: &FractalSettings<F>,
    prefix: &str,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    for (i, params) in param_sets.iter().enumerate() {
        let mut world = distribution::create_world::<F>(school_count, house_count, params);
        if let Some(houses) = &houses {
            world.houses = houses.to_vec();
        }
        let allocations = fractal::root(
            world.households,
            world.houses,
            F::zero(),
            settings.clone(),
            None,
        )
        .unwrap();

        if verify_solution(&allocations, settings.epsilon, settings.max_iter) {
            println!(
                "VERIFICATION SUCCESSFUL (n={}, epsilon={})",
                house_count,
                settings.epsilon.to_f64().unwrap()
            );
        } else {
            println!(
                "VERIFICATION FAILED (n={}, epsilon={})",
                house_count,
                settings.epsilon.to_f64().unwrap()
            );
        }

        if let Err(e) =
            export::serialize_allocations_to_csv(allocations, &format!("{}_{}.csv", prefix, i))
        {
            println!("Failed to write solution to output sln.csv: {}", e);
        } else {
            println!("Written solution to sln.csv");
        }
    }
}

pub fn run_worlds<F: 'static + Send + num::Float + Serialize + DeserializeOwned>(
    worlds: Vec<World<F>>,
    settings: &FractalSettings<F>,
    prefix: &str,
) where
    F: SampleUniform,
    StandardNormal: Distribution<F>,
    Open01: Distribution<F>,
{
    for (i, world) in worlds.into_iter().enumerate() {
        let house_count = world.houses.len();
        let allocations = fractal::root(
            world.households,
            world.houses,
            F::zero(),
            settings.clone(),
            None,
        )
        .unwrap();

        if verify_solution(&allocations, settings.epsilon, settings.max_iter) {
            println!(
                "VERIFICATION SUCCESSFUL (n={}, epsilon={})",
                house_count,
                settings.epsilon.to_f64().unwrap()
            );
        } else {
            println!(
                "VERIFICATION FAILED (n={}, epsilon={})",
                house_count,
                settings.epsilon.to_f64().unwrap()
            );
        }

        if let Err(e) =
            export::serialize_allocations_to_csv(allocations, &format!("{}_{}.csv", prefix, i))
        {
            println!("Failed to write solution to output sln.csv: {}", e);
        } else {
            println!("Written solution to sln.csv");
        }
    }
}
