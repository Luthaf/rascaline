use std::collections::BTreeMap;

use crate::{SimpleSystem, descriptor::{Descriptor, Indexes, IndexValue, IndexesBuilder}};
use crate::systems::System;
use crate::Error;

use crate::calculators::CalculatorBase;

pub struct Calculator {
    implementation: Box<dyn CalculatorBase>,
    parameters: String,
}

/// List of pre-selected indexes on which the user wants to run a calculation
#[derive(Clone, Debug)]
pub enum SelectedIndexes<'a> {
    /// Default, all indexes
    All,
    /// Only the list of selected indexes
    Some(Indexes),
    /// Internal use: list of selected indexes as passed through the C API
    #[doc(hidden)]
    FromC(&'a [IndexValue]),
}

impl<'a> SelectedIndexes<'a> {
    fn into_features(self, calculator: &dyn CalculatorBase) -> Result<Indexes, Error> {
        let indexes = match self {
            SelectedIndexes::All => calculator.features(),
            SelectedIndexes::Some(indexes) => indexes,
            SelectedIndexes::FromC(list) => {
                let mut builder = IndexesBuilder::new(calculator.features_names());

                if list.len() % builder.size() != 0 {
                    return Err(Error::InvalidParameter(format!(
                        "wrong size for partial features list, expected a multiple of {}, got {}",
                        builder.size(), list.len()
                    )))
                }

                for chunk in list.chunks(builder.size()) {
                    builder.add(chunk);
                }
                builder.finish()
            }
        };

        calculator.check_features(&indexes);
        return Ok(indexes);
    }

    fn into_samples(
        self,
        calculator: &dyn CalculatorBase,
        systems: &mut [Box<dyn System>],
    ) -> Result<Indexes, Error> {
        let indexes = match self {
            SelectedIndexes::All => {
                let samples = calculator.samples();
                samples.indexes(systems)
            },
            SelectedIndexes::Some(indexes) => indexes,
            SelectedIndexes::FromC(list) => {
                let samples = calculator.samples();
                let mut builder = IndexesBuilder::new(samples.names());

                if list.len() % builder.size() != 0 {
                    return Err(Error::InvalidParameter(format!(
                        "wrong size for partial samples list, expected a multiple of {}, got {}",
                        builder.size(), list.len()
                    )))
                }

                for chunk in list.chunks(builder.size()) {
                    builder.add(chunk);
                }
                builder.finish()
            }
        };

        calculator.check_samples(&indexes, systems);
        return Ok(indexes);
    }
}

/// Parameters specific to a single call to `compute`
pub struct CalculationOptions<'a> {
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    pub use_native_system: bool,
    /// List of selected samples on which to run the computation
    pub selected_samples: SelectedIndexes<'a>,
    /// List of selected features on which to run the computation
    pub selected_features: SelectedIndexes<'a>,
}

impl<'a> Default for CalculationOptions<'a> {
    fn default() -> CalculationOptions<'a> {
        CalculationOptions {
            use_native_system: false,
            selected_samples: SelectedIndexes::All,
            selected_features: SelectedIndexes::All,
        }
    }
}

impl From<Box<dyn CalculatorBase>> for Calculator {
    fn from(implementation: Box<dyn CalculatorBase>) -> Calculator {
        let parameters = implementation.get_parameters();
        Calculator {
            implementation: implementation,
            parameters: parameters,
        }
    }
}

impl Calculator {
    /// Create a new calculator with the given `name` and `parameters`.
    ///
    /// The list of available calculators and the corresponding parameters are
    /// in the main documentation. The `parameters` should be formatted as JSON.
    ///
    /// # Errors
    ///
    /// This function returns an error if there is no registered calculator with
    /// the given `name`, or if the parameters are invalid for this calculator.
    pub fn new(name: &str, parameters: String) -> Result<Calculator, Error> {
        let creator = match REGISTERED_CALCULATORS.get(name) {
            Some(creator) => creator,
            None => {
                return Err(Error::InvalidParameter(
                    format!("unknown calculator with name '{}'", name)
                ));
            }
        };

        return Ok(Calculator {
            implementation: creator(&parameters)?,
            parameters: parameters,
        })
    }

    /// Get the name of this calculator
    pub fn name(&self) -> String {
        self.implementation.name()
    }

    /// Get the parameters used to create this calculator in a string, formatted
    /// as JSON.
    pub fn parameters(&self) -> &str {
        &self.parameters
    }

    /// Compute the descriptor for all the given `systems` and store it in
    /// `descriptor`
    ///
    /// This function computes the full descriptor, using all samples and all
    /// features.
    #[allow(clippy::shadow_unrelated)]
    pub fn compute(
        &mut self,
        systems: &mut [Box<dyn System>],
        descriptor: &mut Descriptor,
        options: CalculationOptions,
    ) -> Result<(), Error> {
        let mut native_systems;
        let systems = if options.use_native_system {
            native_systems = to_native_systems(systems);
            &mut native_systems
        } else {
            systems
        };

        let features = options.selected_features.into_features(&*self.implementation)?;
        let samples = options.selected_samples.into_samples(&*self.implementation, systems)?;

        let samples_builder = self.implementation.samples();
        if self.implementation.compute_gradients() {
            let gradients = samples_builder
                .gradients_for(systems, &samples)
                .expect("this samples definition do not support gradients");
            descriptor.prepare_gradients(samples, gradients, features);
        } else {
            descriptor.prepare(samples, features);
        }

        self.implementation.compute(systems, descriptor);
        return Ok(());
    }
}

fn to_native_systems(systems: &mut [Box<dyn System>]) -> Vec<Box<dyn System>> {
    let mut native_systems = Vec::with_capacity(systems.len());
    for system in systems.iter() {
        native_systems.push(Box::new(SimpleSystem::from(&**system)) as Box<dyn System>);
    }
    return native_systems;
}

/// Registration of calculator implementations
use crate::calculators::{DummyCalculator, SortedDistances};
use crate::calculators::{SphericalExpansion, SphericalExpansionParameters};
use crate::calculators::{SoapPowerSpectrum, PowerSpectrumParameters};
type CalculatorCreator = fn(&str) -> Result<Box<dyn CalculatorBase>, Error>;

macro_rules! add_calculator {
    ($map :expr, $name :literal, $type :ty) => (
        $map.insert($name, (|json| {
            let value = serde_json::from_str::<$type>(json)?;
            Ok(Box::new(value))
        }) as CalculatorCreator);
    );
    ($map :expr, $name :literal, $type :ty, $parameters :ty) => (
        $map.insert($name, (|json| {
            let parameters = serde_json::from_str::<$parameters>(json)?;
            Ok(Box::new(<$type>::new(parameters)))
        }) as CalculatorCreator);
    );
}

lazy_static::lazy_static!{
    pub static ref REGISTERED_CALCULATORS: BTreeMap<&'static str, CalculatorCreator> = {
        let mut map = BTreeMap::new();
        add_calculator!(map, "dummy_calculator", DummyCalculator);
        add_calculator!(map, "sorted_distances", SortedDistances);
        add_calculator!(map, "spherical_expansion", SphericalExpansion, SphericalExpansionParameters);
        add_calculator!(map, "soap_power_spectrum", SoapPowerSpectrum, PowerSpectrumParameters);
        return map;
    };
}
