use serde::Serialize;

use crate::{solver::{Agent, Allocation, Item}, world::{House, Household}};

#[derive(Serialize)]
pub struct AllocationRow<F: num::Float> {
    quality: F, 
    price: F,
    income: F,
    aspiration: F,
    agent_id: usize,
}

pub fn serialize_allocations_to_csv<F: num::Float>(
    allocations: Vec<Allocation<F, Household<F>, House<F>>>,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: num::Float + Serialize,
{
    let mut writer = csv::Writer::from_path(file_path)?;

    // Write the allocations to the CSV file.
    for allocation in allocations {
        writer.serialize(AllocationRow {
            quality: allocation.quality(),
            price: allocation.price(),
            income: allocation.agent().income(),
            aspiration: allocation.agent().aspiration,
            agent_id: allocation.agent().agent_id(),
        })?;
    }

    writer.flush()?;
    Ok(())
}