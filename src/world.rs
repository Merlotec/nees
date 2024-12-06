use serde::{Deserialize, Serialize};
use crate::solver::Agent;
use crate::solver::Item;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Household<F: num::Float> {
    pub id: usize,
    pub income: F,
    pub ability: F,
    pub aspiration: F,
}

impl<F: num::Float> Household<F> {
    /// Debug information about the household.
    pub fn new(id: usize, income: F, ability: F, aspiration: F) -> Self {
        Self {
            id,
            income,
            ability,
            aspiration,
        }
    }
}

impl<F: num::Float> Agent for Household<F> {
    type FloatType = F;
    /// Returns the item ID.
    fn agent_id(&self) -> usize {
        self.id
    }

    /// Returns the income of the agent.
    fn income(&self) -> F {
        self.income
    }

    /// Utility function used for this agent.
    fn utility(&self, price: F, quality: F) -> F {
        // Additively separable utility function.
        ((self.income - price) / F::from(5000.0).unwrap()).powf(F::one() - self.aspiration) + quality.powf(self.aspiration)
        //(((F::one() - self.aspiration) * (self.income - price) / F::from(100.0).unwrap()).powf(F::from(0.5).unwrap()) + (self.aspiration * quality).powf(F::from(0.5).unwrap())).powf(F::from(2.0).unwrap())
    }


}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct School<F: num::Float> {
    pub capacity: isize, // Capacity of the school.
    pub x: F,
    pub y: F,
    pub quality: F,

    // Endogenous variables
    pub attainment: F,
    pub num_pupils: i32, // Will be different from size if school is not full.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct House<F: num::Float> {
    pub x: F,
    pub y: F,

    /// The (best) school allocated to this house.
    pub school: Option<usize>, // None means no school, Some(-2) means invalid house.

    /// Store quality here to improve cache locality and avoid having to look up quality from school.
    pub school_quality: F,
}

impl<F: num::Float> House<F> {
    pub fn new(x: F, y: F, school: Option<usize>, school_quality: F) -> Self {
        Self {
            x,
            y,
            school,
            school_quality,
        }
    }

    // Checks if the house is valid.
    // fn is_valid(&self) -> bool {
    //     self.school != Some(-2)
    // }

    //Sets the school and its quality.
    pub fn set_school(&mut self, school: usize, quality: F) {
        self.school = Some(school);
        self.school_quality = quality;
    }
}

impl<F: num::Float> Item for House<F> {
    type FloatType = F;

    /// Returns the quality of the school.
    fn quality(&self) -> F {
        self.school_quality
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct World<F: num::Float> {
    pub households: Vec<Household<F>>,
    pub schools: Vec<School<F>>,
    pub houses: Vec<House<F>>,
}

impl<F: num::Float> World<F> {
    /// Creates a new world instance.
    pub fn new(households: Vec<Household<F>>, schools: Vec<School<F>>, houses: Vec<House<F>>) -> Self {
        World {
            households,
            schools,
            houses,
        }
    }


    /// Checks that the households and schools have valid values.
    /// Essentially, aspiration is between 0 and 1, educational quality is positive,
    /// and income is positive for all agents.
    pub fn validate(&self) -> bool {
        self.households.iter().all(|h| {
            h.aspiration < F::from(0.97).unwrap()
            && h.income > F::from(1.0).unwrap()
            && self.schools.iter().all(|s| s.quality > F::from(0.03).unwrap())
        })
    }
}
