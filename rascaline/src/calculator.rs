use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::sync::Arc;

use once_cell::sync::Lazy;

use equistore::{Labels, LabelsBuilder, LabelValue};
use equistore::{TensorBlock, TensorMap};
use ndarray::ArrayD;

use crate::{SimpleSystem, System, Error};

use crate::calculators::CalculatorBase;

pub struct Calculator {
    implementation: Box<dyn CalculatorBase>,
    parameters: String,
}

/// Rules to select labels (either samples or properties) on which the user
/// wants to run a calculation
#[derive(Clone, Copy, Debug)]
pub enum LabelsSelection<'a> {
    /// Default, use all possible labels
    All,
    /// Select a subset of labels, using the same selection criterion for all
    /// keys in the final `TensorMap`.
    ///
    /// If the inner `Labels` contains the same variables as the full set of
    /// labels, then only entries from the full set that also appear in this
    /// selection will be used.
    ///
    /// If the inner `Labels` contains a subset of the variables of the full set
    /// of labels, then only entries from the full set which match one of the
    /// entry in this selection for all of the selection variable will be used.
    Subset(&'a Labels),
    /// Use a predefined subset of labels, with different entries for different
    /// keys of the final `TensorMap`.
    ///
    /// For each key, the corresponding labels are fetched out of the inner
    /// `TensorMap`. The inner `TensorMap` must have the same set of keys as the
    /// full calculation.
    Predefined(&'a TensorMap),
}

impl<'a> LabelsSelection<'a> {
    fn select<'call, F, G, H>(
        &self,
        label_kind: &str,
        keys: &Labels,
        get_default_names: F,
        get_default_labels: G,
        get_from_block: H,
    ) -> Result<Vec<Arc<Labels>>, Error>
        where F: FnOnce() -> Vec<&'call str>,
              G: FnOnce(&Labels) -> Result<Vec<Arc<Labels>>, Error>,
              H: Fn(&TensorBlock) -> Arc<Labels>,
    {
        assert_ne!(keys.count(), 0);

        match self {
            LabelsSelection::All => {
                return get_default_labels(keys);
            },
            LabelsSelection::Subset(selection) => {
                let default_labels = get_default_labels(keys)?;
                let default_names = get_default_names();

                let mut results = Vec::new();
                if selection.names() == default_names {
                    for labels in default_labels {
                        let mut builder = LabelsBuilder::new(default_names.clone());
                        for entry in selection.iter() {
                            if labels.contains(entry) {
                                builder.add(entry);
                            }
                        }
                        results.push(Arc::new(builder.finish()));
                    }
                } else {
                    let mut variables_to_match = Vec::new();
                    for variable in selection.names() {
                        let i = match default_names.iter().position(|&v| v == variable) {
                            Some(index) => index,
                            None => {
                                return Err(Error::InvalidParameter(format!(
                                    "'{}' in {} selection is not one of the {} of this calculator",
                                    variable, label_kind, label_kind
                                )))
                            }
                        };
                        variables_to_match.push(i);
                    }

                    for labels in default_labels {
                        let mut builder = LabelsBuilder::new(default_names.clone());
                        for entry in labels.iter() {
                            for selected in selection.iter() {
                                let mut matches = true;
                                for (i, &v) in variables_to_match.iter().enumerate() {
                                    if selected[i] != entry[v] {
                                        matches = false;
                                        break;
                                    }
                                }

                                if matches {
                                    builder.add(entry);
                                }
                            }
                        }
                        results.push(Arc::new(builder.finish()));
                    }

                }

                return Ok(results);
            },
            LabelsSelection::Predefined(tensor) => {
                if tensor.keys().names() != keys.names() {
                    return Err(Error::InvalidParameter(format!(
                        "invalid key names in predefined {}: expected [{}], but got [{}]",
                        label_kind,
                        keys.names().join(","),
                        tensor.keys().names().join(", ")
                    )));
                }
                let default_names = get_default_names();

                let mut results = Vec::new();
                for key in keys {
                    let mut selection = LabelsBuilder::new(keys.names());
                    selection.add(key);
                    let block = tensor.block(&selection.finish()).expect("could not find a block in predefined selection");
                    let labels = get_from_block(block);
                    if labels.names() != default_names {
                        return Err(Error::InvalidParameter(format!(
                            "invalid predefined {} names: expected [{}], but got [{}]",
                            label_kind,
                            default_names.join(","),
                            labels.names().join(", ")
                        )));
                    }

                    results.push(labels);
                }

                return Ok(results);
            }
        }
    }
}

/// Parameters specific to a single call to `compute`
#[derive(Debug, Clone, Copy)]
pub struct CalculationOptions<'a> {
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    pub use_native_system: bool,
    /// Selection of samples on which to run the computation
    pub selected_samples: LabelsSelection<'a>,
    /// Selection of properties to compute for the samples
    pub selected_properties: LabelsSelection<'a>,
}

impl<'a> Default for CalculationOptions<'a> {
    fn default() -> CalculationOptions<'a> {
        CalculationOptions {
            use_native_system: false,
            selected_samples: LabelsSelection::All,
            selected_properties: LabelsSelection::All,
        }
    }
}

impl From<Box<dyn CalculatorBase>> for Calculator {
    fn from(implementation: Box<dyn CalculatorBase>) -> Calculator {
        let parameters = implementation.parameters();
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

    /// Get the set of keys this calculator would produce for the given systems
    pub fn default_keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        self.implementation.keys(systems)
    }

    /// Compute the descriptor for all the given `systems` and store it in
    /// `descriptor`
    ///
    /// This function computes the full descriptor, using all samples and all
    /// features.
    pub fn compute(
        &mut self,
        systems: &mut [Box<dyn System>],
        options: CalculationOptions,
    ) -> Result<TensorMap, Error> {
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

        let mut tensor = time_graph::spanned!("Calculator::prepare", {
            // TODO: allow selecting a subset of keys?
            let keys = self.implementation.keys(systems)?;

            let samples = options.selected_samples.select(
                "samples",
                &keys,
                || self.implementation.samples_names(),
                |keys| self.implementation.samples(keys, systems),
                |block| Arc::clone(&block.values().samples)
            )?;

            let gradient_samples = self.implementation.gradient_samples(&keys, &samples, systems)?;

            // no selection on the components
            let components = self.implementation.components(&keys);

            let properties = options.selected_properties.select(
                "properties",
                &keys,
                || self.implementation.properties_names(),
                |keys| Ok(self.implementation.properties(keys)),
                |block| Arc::clone(&block.values().properties),
            )?;

            assert_eq!(keys.count(), samples.len());
            assert_eq!(keys.count(), components.len());
            assert_eq!(keys.count(), properties.len());

            let mut spatial_component = LabelsBuilder::new(vec!["gradient_direction"]);
            spatial_component.add(&[LabelValue::new(0)]);
            spatial_component.add(&[LabelValue::new(1)]);
            spatial_component.add(&[LabelValue::new(2)]);
            let spatial_component = Arc::new(spatial_component.finish());

            let mut blocks = Vec::new();
            for (block_i, ((samples, mut components), properties)) in samples.into_iter().zip(components).zip(properties).enumerate() {
                let shape = shape_from_labels(
                    &samples, &components, &properties
                );
                let mut new_block = TensorBlock::new(
                    ArrayD::from_elem(shape, 0.0),
                    samples,
                    components.clone(),
                    Arc::clone(&properties),
                )?;

                if let Some(ref gradient_samples) = gradient_samples {
                    let gradient_samples = &gradient_samples[block_i];
                    assert_eq!(gradient_samples.names(), ["sample", "structure", "atom"]);

                    // add the x/y/z component for gradients
                    components.insert(0, Arc::clone(&spatial_component));
                    let shape = shape_from_labels(
                        gradient_samples, &components, &properties
                    );

                    new_block.add_gradient(
                        "positions",
                        ArrayD::from_elem(shape, 0.0),
                        Arc::clone(gradient_samples),
                        components,
                    )?;
                }

                blocks.push(new_block);
            }

            TensorMap::new(keys, blocks)?
        });

        self.implementation.compute(systems, &mut tensor)?;

        return Ok(tensor);
    }
}

fn shape_from_labels(samples: &Labels, components: &[Arc<Labels>], properties: &Labels) -> Vec<usize> {
    let mut shape = vec![0; components.len() + 2];
    shape[0] = samples.count();
    let mut i = 1;
    for component in components {
        shape[i] = component.count();
        i += 1;
    }
    shape[i] = properties.count();

    return shape;
}


// Registration of calculator implementations
use crate::calculators::DummyCalculator;
use crate::calculators::SortedDistances;
use crate::calculators::{SphericalExpansion, SphericalExpansionParameters};
use crate::calculators::{SoapPowerSpectrum, PowerSpectrumParameters};
use crate::calculators::{LodeSphericalExpansion, LodeSphericalExpansionParameters};
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
static REGISTERED_CALCULATORS: Lazy<BTreeMap<&'static str, CalculatorCreator>> = Lazy::new(|| {
    let mut map = BTreeMap::new();
    add_calculator!(map, "dummy_calculator", DummyCalculator);
    add_calculator!(map, "sorted_distances", SortedDistances);
    add_calculator!(map, "spherical_expansion", SphericalExpansion, SphericalExpansionParameters);
    add_calculator!(map, "soap_power_spectrum", SoapPowerSpectrum, PowerSpectrumParameters);
    add_calculator!(map, "lode_spherical_expansion", LodeSphericalExpansion, LodeSphericalExpansionParameters);
    return map;
});
// [calculator-registration]
