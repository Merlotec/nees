use std::cmp::Ordering;

use rand_distr::{uniform::SampleUniform, Beta, Distribution, LogNormal, Normal, Open01, StandardNormal, Uniform};

use crate::world::*;

pub fn create_world<F: num::Float + SampleUniform>(school_count: usize, house_count: usize) -> World<F>
where StandardNormal: Distribution<F>, Open01: Distribution<F> {
    assert!(house_count % school_count == 0);

    let school_capacity = (house_count / school_count) as isize;

    let mut schools = Vec::with_capacity(school_count);
    let mut houses = Vec::with_capacity(house_count);
    let mut households = Vec::with_capacity(house_count);

    let mut rng = rand::thread_rng();

    //let school_quality_distribution = Normal::<F>::new(F::from(0.8).unwrap(), F::from(0.28).unwrap()).unwrap();
    let ability_distribution = Normal::<F>::new(F::zero(), F::one()).unwrap();
    //let aspiration_distribution = Normal::<F>::new(F::from(0.5).unwrap(), F::from(0.15).unwrap()).unwrap();
    let aspiration_distribution = Normal::<F>::new(F::from(0.5).unwrap(), F::from(0.15).unwrap()).unwrap();

    let q_alpha = F::from(10.415578075512496).unwrap();
    let q_beta = F::from(5.458807204519353).unwrap();
    let q_loc = F::from(-0.13494908716854648).unwrap();
    let q_scale = F::from(1.1861854805071834).unwrap();
    let school_quality_distribution = Beta::<F>::new(q_alpha, q_beta).unwrap();

    let mean_household_inc = F::from(40000.0).unwrap();
    let cv = F::from(0.4).unwrap();
    let variance = (cv * mean_household_inc) * (cv * mean_household_inc);

    let sigma_squared = ((variance / (mean_household_inc * mean_household_inc)) + F::one()).ln();
    let sigma = sigma_squared.sqrt();
    let mu = mean_household_inc.ln() - (sigma_squared / F::from(2.0).unwrap());

    let household_income_distribution = LogNormal::<F>::new(mu, sigma).unwrap();

    let location_axis_distribution = Uniform::<F>::new(F::from(-1.0).unwrap(), F::one());

    // Create schools
    for _ in 0..school_count {
        let quality = q_loc + q_scale * school_quality_distribution.sample(&mut rng);
        let x = location_axis_distribution.sample(&mut rng);
        let y = location_axis_distribution.sample(&mut rng);

        let school = School {
            capacity: school_capacity,
            x,
            y,
            quality,
            attainment: F::from(-1.0).unwrap(),
            num_pupils: 0,
        };

        schools.push(school);
    }

    // Sort schools by quality
    schools.sort_by(|a, b| a.quality.partial_cmp(&b.quality).unwrap_or(Ordering::Equal));

    // Create houses
    for _ in 0..house_count {
        let x = location_axis_distribution.sample(&mut rng);
        let y = location_axis_distribution.sample(&mut rng);

        let house = House::new(x, y, None, F::zero());
        houses.push(house);
    }

    let mut allocated_houses = Vec::with_capacity(houses.len());

    for (sc, school) in schools.iter().enumerate() {
        houses.sort_by(|a, b| {
            let a_dis = (a.x - school.x).powi(2) + (a.y - school.y).powi(2);
            let b_dis = (b.x - school.x).powi(2) + (b.y - school.y).powi(2);
            b_dis
                .partial_cmp(&a_dis)
                .unwrap_or(Ordering::Equal)
        });

        let n = std::cmp::min(school.capacity as usize, houses.len());
        for _ in 0..n {
            let mut h = houses.pop().unwrap();
            h.set_school(sc, school.quality);
            allocated_houses.push(h);
        }
    }

    assert_eq!(allocated_houses.len(), house_count);

    // Create households
    for i in 0..house_count {
        let income = household_income_distribution.sample(&mut rng);
        let ability = ability_distribution.sample(&mut rng);
        let aspiration = aspiration_distribution.sample(&mut rng).clamp(F::zero(), F::one());

        let household = Household::new(i, income, ability, aspiration);
        households.push(household);
    }

    World::new(households, schools, allocated_houses)
}