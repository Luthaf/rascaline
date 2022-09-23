use std::sync::Arc;

use equistore::{EmptyArray, TensorBlock, TensorMap};
use equistore::{LabelValue, Labels, LabelsBuilder};

use crate::calculators::CalculatorBase;
use crate::{CalculationOptions, Calculator, LabelsSelection};
use crate::{Error, System};

use super::SphericalExpansionParameters;
use super::{CutoffFunction, RadialBasis, RadialScaling, SphericalExpansion};

use crate::labels::AtomCenteredSamples;
use crate::labels::{CenterSingleNeighborsSpeciesKeys, KeysBuilder};
use crate::labels::{SamplesBuilder, SpeciesFilter};

/// Parameters for the SOAP radial spectrum calculator.
///
/// The SOAP radial spectrum represent each atom by the radial average of the density
/// of its neighbors. It is very similar to a radial distribution function
/// `g(r)`. It is a 2-body representation, only containing information about the
/// distances between atoms.
///
/// See [this review article](https://doi.org/10.1063/1.5090481) for more
/// information on the SOAP representations.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct RadialSpectrumParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use
    pub max_radial: usize,
    /// Width of the atom-centered gaussian creating the atomic density
    pub atomic_gaussian_width: f64,
    /// Weight of the center atom contribution to the features.
    /// If `1` the center atom contribution is weighted the same as any other
    /// contribution.
    pub center_atom_weight: f64,
    /// radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
    /// radial scaling can be used to reduce the importance of neighbor atoms
    /// further away from the center, usually improving the performance of the
    /// model
    #[serde(default)]
    pub radial_scaling: RadialScaling,
}

/// Calculator implementing the Radial
/// spectrum representation of atomistic systems.
pub struct SoapRadialSpectrum {
    parameters: RadialSpectrumParameters,
    spherical_expansion: Calculator,
}

impl std::fmt::Debug for SoapRadialSpectrum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl SoapRadialSpectrum {
    pub fn new(parameters: RadialSpectrumParameters) -> Result<SoapRadialSpectrum, Error> {
        let expansion_parameters = SphericalExpansionParameters {
            cutoff: parameters.cutoff,
            max_radial: parameters.max_radial,
            max_angular: 0,
            atomic_gaussian_width: parameters.atomic_gaussian_width,
            center_atom_weight: parameters.center_atom_weight,
            radial_basis: parameters.radial_basis.clone(),
            cutoff_function: parameters.cutoff_function,
            radial_scaling: parameters.radial_scaling,
        };

        let spherical_expansion = SphericalExpansion::new(expansion_parameters)?;

        return Ok(SoapRadialSpectrum {
            parameters: parameters,
            spherical_expansion: Calculator::from(
                Box::new(spherical_expansion) as Box<dyn CalculatorBase>
            ),
        });
    }

    /// Construct a `TensorMap` containing the set of samples/properties we want the
    /// spherical expansion to compute.
    fn selected_spx_labels(descriptor: &TensorMap) -> TensorMap {
        assert_eq!(
            descriptor.keys().names(),
            ["species_center", "species_neighbor"]
        );

        let mut keys_builder = LabelsBuilder::new(vec![
            "spherical_harmonics_l",
            "species_center",
            "species_neighbor",
        ]);
        let mut blocks = Vec::new();
        for (&[center, neighbor], block) in descriptor.keys().iter_fixed_size().zip(descriptor.blocks()) {
            // spherical_harmonics_l is always 0
            keys_builder.add(&[LabelValue::new(0), center, neighbor]);

            blocks.push(
                TensorBlock::new(
                    EmptyArray::new(vec![
                        block.values().samples.count(),
                        block.values().properties.count(),
                    ]),
                    Arc::clone(&block.values().samples),
                    Vec::new(),
                    Arc::clone(&block.values().properties),
                ).expect("invalid TensorBlock")
            );
        }

        return TensorMap::new(keys_builder.finish(), blocks).expect("invalid TensorMap");
    }
}

impl CalculatorBase for SoapRadialSpectrum {
    fn name(&self) -> String {
        "radial spectrum".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<equistore::Labels, Error> {
        let builder = CenterSingleNeighborsSpeciesKeys {
            cutoff: self.parameters.cutoff,
            self_pairs: true,
        };
        return builder.keys(systems);
    }

    fn samples_names(&self) -> Vec<&str> {
        AtomCenteredSamples::samples_names()
    }

    fn samples(
        &self,
        keys: &equistore::Labels,
        systems: &mut [Box<dyn System>],
    ) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor"]);
        let mut result = Vec::new();
        for [species_center, species_neighbor] in keys.iter_fixed_size() {
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
                self_pairs: true,
            };

            result.push(builder.samples(systems)?);
        }

        return Ok(result);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" | "cell" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([species_center, species_neighbor], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
                self_pairs: true,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &equistore::Labels) -> Vec<Vec<Arc<Labels>>> {
        return vec![vec![]; keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &equistore::Labels) -> Vec<Arc<Labels>> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        for n in 0..self.parameters.max_radial {
            properties.add(&[n]);
        }

        let properties = Arc::new(properties.finish());

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SoapRadialSpectrum::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["species_center", "species_neighbor"]);
        let mut gradients = Vec::new();
        if descriptor.blocks()[0].gradient("positions").is_some() {
            gradients.push("positions");
        }
        if descriptor.blocks()[0].gradient("cell").is_some() {
            gradients.push("cell");
        }

        let selected = SoapRadialSpectrum::selected_spx_labels(descriptor);
        let options = CalculationOptions {
            gradients: &gradients,
            selected_samples: LabelsSelection::Predefined(&selected),
            selected_properties: LabelsSelection::Predefined(&selected),
            ..Default::default()
        };

        let spherical_expansion = self.spherical_expansion.compute(
            systems,
            options,
        ).expect("failed to compute spherical expansion");

        for ((_, mut block), (_, block_spx)) in
            descriptor.iter_mut().zip(spherical_expansion.iter())
        {
            let array = block.values_mut().data.as_array_mut();
            let array_spx = block_spx.values().data.as_array();
            let shape = array_spx.shape();
            // shape[1] is the m component
            debug_assert_eq!(shape[1], 1);
            let array_spx_reshaped = array_spx.view().into_shape(
                (shape[0], shape[2])
            ).expect("wrong shape");
            array.assign(&array_spx_reshaped);

            if let Some(gradient) = block.gradient_mut("positions") {
                let gradient_spx = block_spx.gradient("positions").expect("missing spherical expansion gradients");
                debug_assert_eq!(gradient.samples, gradient_spx.samples);

                let array = gradient.data.as_array_mut();
                let array_spx = gradient_spx.data.as_array();
                let shape = array_spx.shape();
                // shape[2] is the m component
                debug_assert_eq!(shape[2], 1);

                let array_spx_reshaped = array_spx.view().into_shape(
                    (shape[0], shape[1], shape[3])
                ).expect("wrong shape");
                array.assign(&array_spx_reshaped);
            }

            if let Some(gradient) = block.gradient_mut("cell") {
                let gradient_spx = block_spx.gradient("cell").expect("missing spherical expansion gradients");
                debug_assert_eq!(gradient.samples, gradient_spx.samples);

                let array = gradient.data.as_array_mut();
                let array_spx = gradient_spx.data.as_array();
                let shape = array_spx.shape();
                // shape[2] is the m component
                debug_assert_eq!(shape[3], 1);

                let array_spx_reshaped = array_spx.view().into_shape(
                    (shape[0], shape[1], shape[2], shape[4])
                ).expect("wrong shape");
                array.assign(&array_spx_reshaped);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use equistore::LabelValue;

    use crate::systems::test_utils::{test_system, test_systems};
    use crate::Calculator;

    use super::*;
    use crate::calculators::CalculatorBase;

    fn parameters() -> RadialSpectrumParameters {
        RadialSpectrumParameters {
            cutoff: 3.5,
            max_radial: 6,
            atomic_gaussian_width: 0.3,
            center_atom_weight: 1.0,
            radial_basis: RadialBasis::Gto { splined_radial_integral: true, spline_accuracy: 1e-8 },
            radial_scaling: RadialScaling::None {},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters()).unwrap()
        ) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(descriptor.keys().count(), 4);
        assert!(descriptor.keys().contains(&[LabelValue::new(1), LabelValue::new(1)]));
        assert!(descriptor.keys().contains(&[LabelValue::new(1), LabelValue::new(-42)]));
        assert!(descriptor.keys().contains(&[LabelValue::new(-42), LabelValue::new(1)]));
        assert!(descriptor.keys().contains(&[LabelValue::new(-42), LabelValue::new(-42)]));
    }

    #[test]
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters()).unwrap()
        ) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 5e-5,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }

    #[test]
    fn finite_differences_cell() {
        let calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters()).unwrap()
        ) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 5e-5,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_cell(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters()).unwrap()
        ) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let mut properties = LabelsBuilder::new(vec!["n"]);
        properties.add(&[LabelValue::new(0)]);
        properties.add(&[LabelValue::new(3)]);
        properties.add(&[LabelValue::new(4)]);
        properties.add(&[LabelValue::new(1)]);

        let mut samples = LabelsBuilder::new(vec!["structure", "center"]);
        samples.add(&[LabelValue::new(0), LabelValue::new(1)]);
        samples.add(&[LabelValue::new(0), LabelValue::new(0)]);
        samples.add(&[LabelValue::new(1), LabelValue::new(0)]);

        crate::calculators::tests_utils::compute_partial(
            calculator,
            &mut systems,
            &samples.finish(),
            &properties.finish(),
        );
    }
}
