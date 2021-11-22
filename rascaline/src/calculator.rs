use std::{collections::BTreeMap, convert::TryFrom};

use crate::{SimpleSystem, descriptor::{Descriptor, Indexes, IndexesBuilder}};
use crate::systems::System;
use crate::Error;

use crate::calculators::CalculatorBase;

pub struct Calculator {
    implementation: Box<dyn CalculatorBase>,
    parameters: String,
}

/// List of pre-selected indexes on which the user wants to run a calculation
#[derive(Clone, Debug)]
pub enum SelectedIndexes {
    /// Default, all indexes
    All,
    /// Only the list of selected indexes. The indexes can contains the same
    /// variable as the full set of indexes, or only a subset of them. In the
    /// latter case, all entries from the full set of indexes matching the set
    /// of variables specified will be used.
    Subset(Indexes),
}

impl SelectedIndexes {
    /// Transform a set of selected indexes into actual indexes usable as
    /// feature indexes by a calculator
    fn into_features(self, calculator: &dyn CalculatorBase) -> Result<Indexes, Error> {
        let indexes = match self {
            SelectedIndexes::All => calculator.features(),
            SelectedIndexes::Subset(indexes) => {
                let default_features = calculator.features();
                if indexes.names() == default_features.names() {
                    calculator.check_features(&indexes)?;
                    indexes
                } else {
                    let mut variables_to_match = Vec::new();
                    for variable in indexes.names() {
                        let i = match default_features.names().iter().position(|&v| v == variable) {
                            Some(index) => index,
                            None => {
                                return Err(Error::InvalidParameter(format!(
                                    "'{}' in requested features is not part of the features of this calculator",
                                    variable
                                )))
                            }
                        };
                        variables_to_match.push(i);
                    }

                    let mut filtered = IndexesBuilder::new(default_features.names());
                    for selected in indexes.iter() {
                        for features in default_features.iter() {
                            // only take the features if they match
                            let mut matches = true;
                            for (i, &v) in variables_to_match.iter().enumerate() {
                                if selected[i] != features[v] {
                                    matches = false;
                                    break;
                                }
                            }

                            if matches {
                                filtered.add(features);
                            }
                        }
                    }

                    filtered.finish()
                }
            },
        };

        return Ok(indexes);
    }

    /// Transform a set of selected indexes into actual indexes usable as sample
    /// indexes by a calculator for the given set of systems
    fn into_samples(
        self,
        calculator: &dyn CalculatorBase,
        systems: &mut [Box<dyn System>],
    ) -> Result<Indexes, Error> {
        let indexes = match self {
            SelectedIndexes::All => {
                calculator.samples_builder().samples(systems)?
            },
            SelectedIndexes::Subset(indexes) => {
                let default_samples = calculator.samples_builder().samples(systems)?;
                if indexes.names() == default_samples.names() {
                    calculator.check_samples(&indexes, systems)?;
                    indexes
                } else {
                    let mut variables_to_match = Vec::new();
                    for variable in indexes.names() {
                        let i = match default_samples.names().iter().position(|&v| v == variable) {
                            Some(index) => index,
                            None => {
                                return Err(Error::InvalidParameter(format!(
                                    "'{}' in requested samples is not part of the samples of this calculator",
                                    variable
                                )))
                            }
                        };
                        variables_to_match.push(i);
                    }

                    let mut filtered = IndexesBuilder::new(default_samples.names());
                    for selected in indexes.iter() {
                        for sample in default_samples.iter() {
                            // only take the samples if they match
                            let mut matches = true;
                            for (i, &v) in variables_to_match.iter().enumerate() {
                                if selected[i] != sample[v] {
                                    matches = false;
                                    break;
                                }
                            }

                            if matches {
                                filtered.add(sample);
                            }
                        }
                    }

                    filtered.finish()
                }
            },
        };

        return Ok(indexes);
    }
}

/// Parameters specific to a single call to `compute`
pub struct CalculationOptions {
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    pub use_native_system: bool,
    /// List of selected samples on which to run the computation
    pub selected_samples: SelectedIndexes,
    /// List of selected features on which to run the computation
    pub selected_features: SelectedIndexes,
}

impl Default for CalculationOptions {
    fn default() -> CalculationOptions {
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

    /// Does this calculator computes gradients?
    pub fn gradients(&self) -> bool {
        self.implementation.compute_gradients()
    }

    /// Get the default set of features for this calculator
    pub fn default_features(&self) -> Indexes {
        self.implementation.features()
    }

    /// Compute the descriptor for all the given `systems` and store it in
    /// `descriptor`
    ///
    /// This function computes the full descriptor, using all samples and all
    /// features.
    #[time_graph::instrument(name = "Calculator::compute")]
    pub fn compute(
        &mut self,
        systems: &mut [Box<dyn System>],
        descriptor: &mut Descriptor,
        options: CalculationOptions,
    ) -> Result<(), Error> {
        let mut native_systems;
        let systems = if options.use_native_system {
            native_systems = Vec::with_capacity(systems.len());
            for system in systems {
                native_systems.push(Box::new(SimpleSystem::try_from(&**system)?) as Box<dyn System>);
            }
            &mut native_systems
        } else {
            systems
        };

        let features = options.selected_features.into_features(&*self.implementation)?;
        let samples = options.selected_samples.into_samples(&*self.implementation, systems)?;

        time_graph::spanned!("Calculator::prepare", {
            let builder = self.implementation.samples_builder();
            if self.implementation.compute_gradients() {
                let gradients = builder
                    .gradients_for(systems, &samples)?
                    .expect("this samples definition do not support gradients");
                descriptor.prepare_gradients(samples, gradients, features);
            } else {
                descriptor.prepare(samples, features);
            }
        });

        self.implementation.compute(systems, descriptor)?;
        return Ok(());
    }
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
            Ok(Box::new(<$type>::new(parameters)?))
        }) as CalculatorCreator);
    );
}

// this code is included in the calculator tutorial, the tags below indicate the
// first/last line to include
// [calculator-registration]
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
// [calculator-registration]


#[cfg(test)]
mod tests {
    use super::SelectedIndexes;

    use crate::calculators::{CalculatorBase, DummyCalculator};
    use crate::descriptor::{IndexesBuilder, IndexValue};

    #[test]
    fn selected_features() {
        let calculator = DummyCalculator {
            cutoff: 3.4, delta: 0, name: "".into(), gradients: false,
        };

        let selected = SelectedIndexes::All;
        let indexes = selected.into_features(&calculator).unwrap();
        let expected = calculator.features();
        assert_eq!(indexes, expected);

        // full specification of the rows
        let mut selected = IndexesBuilder::new(vec!["index_delta", "x_y_z"]);
        selected.add(&[IndexValue::from(0), IndexValue::from(1)]);
        let expected = selected.finish();

        let selected = SelectedIndexes::Subset(expected.clone());
        let indexes = selected.into_features(&calculator).unwrap();
        assert_eq!(indexes, expected);

        // partial specification of the rows
        let mut selected = IndexesBuilder::new(vec!["index_delta"]);
        selected.add(&[IndexValue::from(0)]);

        let selected = SelectedIndexes::Subset(selected.finish());
        let indexes = selected.into_features(&calculator).unwrap();
        assert_eq!(indexes, expected);
    }

    #[test]
    fn selected_samples() {
        let calculator = DummyCalculator {
            cutoff: 3.4, delta: 0, name: "".into(), gradients: false,
        };
        let mut systems = crate::systems::test_utils::test_systems(&["water"]);

        let selected = SelectedIndexes::All;
        let indexes = selected.into_samples(&calculator, &mut systems).unwrap();
        let expected = calculator.samples_builder().samples(&mut systems).unwrap();
        assert_eq!(indexes, expected);

        // full specification of the rows
        let mut selected = IndexesBuilder::new(vec!["structure", "center"]);
        selected.add(&[IndexValue::from(0), IndexValue::from(2)]);
        selected.add(&[IndexValue::from(0), IndexValue::from(0)]);
        let expected = selected.finish();

        let selected = SelectedIndexes::Subset(expected.clone());
        let indexes = selected.into_samples(&calculator, &mut systems).unwrap();
        assert_eq!(indexes, expected);

        // partial specification of the rows
        let mut selected = IndexesBuilder::new(vec!["center"]);
        selected.add(&[IndexValue::from(2)]);
        selected.add(&[IndexValue::from(0)]);

        let selected = SelectedIndexes::Subset(selected.finish());
        let indexes = selected.into_samples(&calculator, &mut systems).unwrap();
        assert_eq!(indexes, expected);
    }
}
