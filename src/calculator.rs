use std::collections::BTreeMap;

use crate::descriptor::Descriptor;
use crate::system::System;
use crate::Error;

use crate::calculators::CalculatorBase;

pub struct Calculator {
    implementation: Box<dyn CalculatorBase>,
    parameters: String,
}

impl Calculator {
    pub fn new(name: &str, parameters: String) -> Result<Calculator, Error> {
        let creator = match REGISTERED_CALCULATORS.get(name) {
            Some(creator) => creator,
            None => {
                return Err(Error::InvalidParameter(
                    format!("unknwon calculator with name '{}'", name)
                ));
            }
        };

        return Ok(Calculator {
            implementation: creator(&parameters)?,
            parameters: parameters,
        })
    }

    /// Get the name associated with this Calculator
    pub fn name(&self) -> String {
        self.implementation.name()
    }

    /// Get the parameters used to create this Calculator in a string.
    ///
    /// Currently the string is formatted as JSON, but this could change in the
    /// future.
    pub fn parameters(&self) -> &str {
        &self.parameters
    }

    /// Compute the descriptor for all the given systems and store it in `descriptor`
    pub fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        self.implementation.compute(systems, descriptor);
    }
}

/// Registration of calculator implementations
use crate::calculators::{DummyCalculator, SortedDistances};
type CalculatorCreator = fn(&str) -> Result<Box<dyn CalculatorBase>, Error>;

macro_rules! add_calculator {
    ($map :expr, $name :literal, $type :ty) => (
        $map.insert($name, (|json| {
            let value = serde_json::from_str::<$type>(json)?;
            Ok(Box::new(value))
        }) as CalculatorCreator);
    );
}

lazy_static::lazy_static!{
    pub static ref REGISTERED_CALCULATORS: BTreeMap<&'static str, CalculatorCreator> = {
        let mut map = BTreeMap::new();
        add_calculator!(map, "dummy_calculator", DummyCalculator);
        add_calculator!(map, "sorted_distances", SortedDistances);
        return map;
    };
}
