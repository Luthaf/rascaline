use std::collections::BTreeMap;
use std::convert::TryFrom;

use log::warn;
use metatensor::c_api::MTS_INVALID_PARAMETER_ERROR;
use once_cell::sync::Lazy;

use metatensor::{Labels, LabelsBuilder};
use metatensor::{TensorBlockRef, TensorBlock, TensorMap};
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

fn map_selection_error<'a>(
    default_names: &'a [&str],
    selection_names: &'a [&str],
    label_kind: &'a str
) -> impl FnOnce(metatensor::Error) -> Error + 'a{
    return move |err| {
        match err.code {
            Some(MTS_INVALID_PARAMETER_ERROR) => {
                for name in selection_names {
                    if !default_names.contains(name) {
                        return Error::InvalidParameter(format!(
                            "'{}' in {} selection is not part of the {} of this calculator",
                            name, label_kind, label_kind
                        ));
                    }
                }
                // it was some other error, bubble it up
                Error::from(err)
            }
            _ => Error::from(err)
        }
    };
}

impl<'a> LabelsSelection<'a> {
    fn select<'call, F, G, H>(
        &self,
        label_kind: &str,
        keys: &Labels,
        get_default_names: F,
        get_default_labels: G,
        get_from_block: H,
    ) -> Result<Vec<Labels>, Error>
        where F: FnOnce() -> Vec<&'call str>,
              G: FnOnce(&Labels) -> Result<Vec<Labels>, Error>,
              H: Fn(TensorBlockRef<'_>) -> Labels,
    {
        if keys.count() == 0 {
            return Ok(vec![]);
        }

        match *self {
            LabelsSelection::All => {
                return get_default_labels(keys);
            },
            LabelsSelection::Subset(selection) => {
                let default_labels = get_default_labels(keys)?;
                let default_names = get_default_names();

                let mut results = Vec::new();
                for labels in default_labels {
                    let mut builder = LabelsBuilder::new(default_names.clone());

                    // better error message in case of un-matched names
                    let matches = labels.select(selection)
                        .map_err(map_selection_error(&default_names, &selection.names(), label_kind))?;

                    for entry in matches {
                        builder.add(&labels[entry as usize]);
                    }
                    results.push(builder.finish());
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
                for key in keys {
                    if !tensor.keys().contains(key) {
                        let key_print = keys.names().iter()
                            .zip(key)
                            .map(|(n, v)| format!("{}={}", n, v))
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(Error::InvalidParameter(format!(
                            "expected a block for ({}) in predefined {} selection",
                            key_print,
                            label_kind,
                        )));
                    }
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
                            default_names.join(", "),
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
#[allow(clippy::doc_markdown)]
pub struct CalculationOptions<'a> {
    /// List of gradients that should be computed. If this list is empty no
    /// gradients are computed.
    ///
    /// The following gradients are available:
    ///
    /// - ``"positions"``, for gradients of the representation with respect to
    ///   atomic positions. Positions gradients are computed as
    ///
    ///   $$ \frac{\partial \langle q \vert A_i \rangle}
    ///           {\partial \mathbf{r_j}} $$
    ///
    ///   where $\langle q \vert A_i \rangle$ is the representation around
    ///   atom $i$ and $\mathbf{r_j}$ is the position vector of the
    ///   atom $j$.
    ///
    ///   **Note**: Position gradients of an atom are computed with respect to
    ///   all other atoms within the representation. To recover the force one
    ///   has to accumulate all pairs associated with atom $i$.
    ///
    /// - ``"strain"``, for gradients of the representation with respect to
    ///   strain. These gradients are typically used to compute the virial, and
    ///   from there the pressure acting on a system. To compute them, we
    ///   pretend that all the positions $\mathbf{r}$ and unit cell $\mathbf{H}$
    ///   have been scaled by a strain matrix $\epsilon$:
    ///
    ///   $$
    ///      \mathbf r &\rightarrow \mathbf r \left(\mathbb{1} + \epsilon \right)\\
    ///      \mathbf H &\rightarrow \mathbf H \left(\mathbb{1} + \epsilon \right)
    ///   $$
    ///
    ///   and then take the gradients of the representation with respect to this
    ///   matrix:
    ///
    ///   $$ \frac{\partial \langle q \vert A_i \rangle} {\partial \mathbf{\epsilon}} $$
    ///
    /// - ``"cell"``, for gradients of the representation with respect to the
    ///    system's cell parameters. These gradients are computed at fixed positions,
    ///    and often not what you want when computing gradients explicitly (they are
    ///    mainly used in ``featomic.torch`` to integrate with backward
    ///    propagation).
    ///
    ///    $$ \left. \frac{\partial \langle q \vert A_i \rangle} {\partial \mathbf{H}} \right |_\mathbf{r} $$
    pub gradients: &'a[&'a str],
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    pub use_native_system: bool,
    /// Selection of samples on which to run the computation
    pub selected_samples: LabelsSelection<'a>,
    /// Selection of properties to compute for the samples
    pub selected_properties: LabelsSelection<'a>,
    /// Selection for the keys to include in the output. If this is `None`, the
    /// default set of keys (as determined by the calculator) will be used. Note
    /// that this default set of keys can depend on which systems we are running
    /// the calculation on.
    pub selected_keys: Option<&'a Labels>,
}

impl<'a> Default for CalculationOptions<'a> {
    fn default() -> CalculationOptions<'a> {
        CalculationOptions {
            gradients: &[],
            use_native_system: false,
            selected_samples: LabelsSelection::All,
            selected_properties: LabelsSelection::All,
            selected_keys: None,
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

    /// Get the all radial cutoffs used by this Calculator's neighbors lists
    /// (which can be an empty list)
    pub fn cutoffs(&self) -> &[f64] {
        self.implementation.cutoffs()
    }

    #[time_graph::instrument(name="Calculator::prepare")]
    fn prepare(&mut self, systems: &mut [Box<dyn System>], options: CalculationOptions) -> Result<TensorMap, Error> {
        let default_keys = self.implementation.keys(systems)?;

        let keys = match options.selected_keys {

            Some(selection) => {
                if selection.is_empty() {
                    return Err(Error::InvalidParameter("selected keys can not be empty".into()));
                } else if default_keys.names() == selection.names() {
                    selection.clone()
                } else {
                    let mut builder = LabelsBuilder::new(default_keys.names());
                    let matches = default_keys.select(selection)
                        .map_err(map_selection_error(&default_keys.names(), &selection.names(), "keys"))?;
                    for entry in matches {
                        builder.add(&default_keys[entry as usize]);
                    }
                    builder.finish()
                }
            }
            None => default_keys,
        };

        let samples = options.selected_samples.select(
            "samples",
            &keys,
            || self.implementation.sample_names(),
            |keys| self.implementation.samples(keys, systems),
            |block| block.samples(),
        )?;

        for &parameter in options.gradients {
            if parameter == "positions" || parameter == "strain" || parameter == "cell" {
                continue;
            }

            return Err(Error::InvalidParameter(format!(
                "unexpected gradient \"{}\", should be one of \"positions\", \"cell\", or \"strain\"",
                parameter
            )));
        }

        let positions_gradient_samples = if options.gradients.contains(&"positions") {
            if !self.implementation.supports_gradient("positions") {
                return Err(Error::InvalidParameter(format!(
                    "the {} calculator does not support gradients with respect to positions",
                    self.name()
                )));
            }

            Some(self.implementation.positions_gradient_samples(&keys, &samples, systems)?)
        } else {
            None
        };

        let cell_gradient_samples = if options.gradients.contains(&"cell") {
            if !self.implementation.supports_gradient("cell") {
                return Err(Error::InvalidParameter(format!(
                    "the {} calculator does not support gradients with respect to cell",
                    self.name()
                )));
            }

            if std::env::var("FEATOMIC_NO_WARN_CELL_GRADIENTS").is_err() {
                // TODO: remove this warning around November 2024 (~6 months
                // after this change)
                warn!(
                    "The meaning of \"cell\" gradients has changed recently, \
                    you likely want \"strain\" gradients instead. Please review \
                    the documentation carefully."
                );
            }

            let mut cell_gradient_samples = Vec::new();
            for samples in &samples {
                let mut builder = LabelsBuilder::new(vec!["sample"]);
                for sample_i in 0..samples.count() {
                    builder.add(&[sample_i]);
                }
                cell_gradient_samples.push(builder.finish());
            }
            Some(cell_gradient_samples)
        } else {
            None
        };

        let strain_gradient_samples = if options.gradients.contains(&"strain") {
            if !self.implementation.supports_gradient("strain") {
                return Err(Error::InvalidParameter(format!(
                    "the {} calculator does not support gradients with respect to strain",
                    self.name()
                )));
            }

            let mut strain_gradient_samples = Vec::new();
            for samples in &samples {
                let mut builder = LabelsBuilder::new(vec!["sample"]);
                for sample_i in 0..samples.count() {
                    builder.add(&[sample_i]);
                }
                strain_gradient_samples.push(builder.finish());
            }
            Some(strain_gradient_samples)
        } else {
            None
        };

        // no selection on the components
        let components = self.implementation.components(&keys);

        let properties = options.selected_properties.select(
            "properties",
            &keys,
            || self.implementation.property_names(),
            |keys| Ok(self.implementation.properties(keys)),
            |block| block.properties(),
        )?;

        assert_eq!(keys.count(), samples.len());
        assert_eq!(keys.count(), components.len());
        assert_eq!(keys.count(), properties.len());

        let xyz = Labels::new(["xyz"], &[[0], [1], [2]]);
        let abc = Labels::new(["abc"], &[[0], [1], [2]]);
        let xyz_1 = Labels::new(["xyz_1"], &[[0], [1], [2]]);
        let xyz_2 = Labels::new(["xyz_2"], &[[0], [1], [2]]);

        let mut blocks = Vec::new();
        for (block_i, ((samples, components), properties)) in samples.into_iter().zip(components).zip(properties).enumerate() {
            let shape = shape_from_labels(
                &samples, &components, &properties
            );
            let mut new_block = TensorBlock::new(
                ArrayD::from_elem(shape, 0.0),
                &samples,
                &components,
                &properties,
            )?;

            if let Some(ref gradient_samples) = positions_gradient_samples {
                let gradient_samples = &gradient_samples[block_i];
                assert_eq!(gradient_samples.names(), ["sample", "system", "atom"]);

                // add the x/y/z component for gradients
                let mut components = components.clone();
                components.insert(0, xyz.clone());
                let shape = shape_from_labels(
                    gradient_samples, &components, &properties
                );

                new_block.add_gradient(
                    "positions",
                    TensorBlock::new(
                        ArrayD::from_elem(shape, 0.0),
                        gradient_samples,
                        &components,
                        &properties
                    ).expect("generated invalid gradient")
                ).expect("generated invalid gradient");
            }

            if let Some(ref gradient_samples) = cell_gradient_samples {
                let gradient_samples = &gradient_samples[block_i];

                // add the components for cell gradients
                let mut components = components.clone();
                components.insert(0, abc.clone());
                components.insert(0, xyz.clone());
                let shape = shape_from_labels(
                    gradient_samples, &components, &properties
                );

                new_block.add_gradient(
                    "cell",
                    TensorBlock::new(
                        ArrayD::from_elem(shape, 0.0),
                        gradient_samples,
                        &components,
                        &properties
                    ).expect("generated invalid gradient")
                ).expect("generated invalid gradient");
            }

            if let Some(ref gradient_samples) = strain_gradient_samples {
                let gradient_samples = &gradient_samples[block_i];

                // add the components for strain gradients
                let mut components = components;
                components.insert(0, xyz_1.clone());
                components.insert(0, xyz_2.clone());
                let shape = shape_from_labels(
                    gradient_samples, &components, &properties
                );

                new_block.add_gradient(
                    "strain",
                    TensorBlock::new(
                        ArrayD::from_elem(shape, 0.0),
                        gradient_samples,
                        &components,
                        &properties
                    ).expect("generated invalid gradient")
                ).expect("generated invalid gradient");
            }

            blocks.push(new_block);
        }

        return Ok(TensorMap::new(keys, blocks)?);
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

        let mut tensor = self.prepare(systems, options)?;

        if tensor.keys().count() > 0 {
            self.implementation.compute(systems, &mut tensor)?;
        }

        return Ok(tensor);
    }
}

fn shape_from_labels(samples: &Labels, components: &[Labels], properties: &Labels) -> Vec<usize> {
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
use crate::calculators::AtomicComposition;
use crate::calculators::DummyCalculator;
use crate::calculators::SortedDistances;
use crate::calculators::NeighborList;
use crate::calculators::{SphericalExpansionByPair, SphericalExpansionParameters};
use crate::calculators::SphericalExpansion;
use crate::calculators::{SoapRadialSpectrum, RadialSpectrumParameters};
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
    add_calculator!(map, "atomic_composition", AtomicComposition);
    add_calculator!(map, "dummy_calculator", DummyCalculator);
    add_calculator!(map, "neighbor_list", NeighborList);
    add_calculator!(map, "sorted_distances", SortedDistances);

    add_calculator!(map, "spherical_expansion_by_pair", SphericalExpansionByPair, SphericalExpansionParameters);
    add_calculator!(map, "spherical_expansion", SphericalExpansion, SphericalExpansionParameters);
    add_calculator!(map, "soap_radial_spectrum", SoapRadialSpectrum, RadialSpectrumParameters);
    add_calculator!(map, "soap_power_spectrum", SoapPowerSpectrum, PowerSpectrumParameters);

    add_calculator!(map, "lode_spherical_expansion", LodeSphericalExpansion, LodeSphericalExpansionParameters);
    return map;
});
// [calculator-registration]
