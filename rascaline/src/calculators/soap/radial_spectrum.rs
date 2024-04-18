use metatensor::{EmptyArray, TensorBlock, TensorMap};
use metatensor::{LabelValue, Labels, LabelsBuilder};

use crate::calculators::CalculatorBase;
use crate::{CalculationOptions, Calculator, LabelsSelection};
use crate::{Error, System};

use super::SphericalExpansionParameters;
use super::{CutoffFunction, RadialScaling, SphericalExpansion};
use crate::calculators::radial_basis::RadialBasis;

use crate::labels::AtomCenteredSamples;
use crate::labels::{CenterSingleNeighborsTypesKeys, KeysBuilder};
use crate::labels::{SamplesBuilder, AtomicTypeFilter};

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
    /// Weight of the central atom contribution to the
    /// features. If `1` the center atom contribution is weighted the same
    /// as any other contribution. If `0` the central atom does not
    /// contribute to the features at all.
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
            ["center_type", "neighbor_type"]
        );

        let mut keys_builder = LabelsBuilder::new(vec![
            "o3_lambda",
            "o3_sigma",
            "center_type",
            "neighbor_type",
        ]);
        let mut blocks = Vec::new();
        for (&[center, neighbor], block) in descriptor.keys().iter_fixed_size().zip(descriptor.blocks()) {
            // o3_lambda is always 0, o3_sigma always 1
            keys_builder.add(&[LabelValue::new(0), LabelValue::new(1), center, neighbor]);

            let block = block.data();
            blocks.push(
                TensorBlock::new(
                    EmptyArray::new(vec![block.samples.count(), block.properties.count()]),
                    &block.samples,
                    &[],
                    &block.properties,
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

    fn cutoffs(&self) -> &[f64] {
        self.spherical_expansion.cutoffs()
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<metatensor::Labels, Error> {
        let builder = CenterSingleNeighborsTypesKeys {
            cutoff: self.parameters.cutoff,
            self_pairs: true,
        };
        return builder.keys(systems);
    }

    fn sample_names(&self) -> Vec<&str> {
        AtomCenteredSamples::sample_names()
    }

    fn samples(
        &self,
        keys: &metatensor::Labels,
        systems: &mut [Box<dyn System>],
    ) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type", "neighbor_type"]);
        let mut result = Vec::new();
        for [center_type, neighbor_type] in keys.iter_fixed_size() {
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_pairs: true,
            };

            result.push(builder.samples(systems)?);
        }

        return Ok(result);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" | "cell" | "strain" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type", "neighbor_type"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([center_type, neighbor_type], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_pairs: true,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &metatensor::Labels) -> Vec<Vec<Labels>> {
        return vec![vec![]; keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &metatensor::Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        for n in 0..self.parameters.max_radial {
            properties.add(&[n]);
        }
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SoapRadialSpectrum::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["center_type", "neighbor_type"]);
        let mut gradients = Vec::new();
        if descriptor.block_by_id(0).gradient("positions").is_some() {
            gradients.push("positions");
        }
        if descriptor.block_by_id(0).gradient("cell").is_some() {
            gradients.push("cell");
        }
        if descriptor.block_by_id(0).gradient("strain").is_some() {
            gradients.push("strain");
        }

        let selected = SoapRadialSpectrum::selected_spx_labels(descriptor);
        let options = CalculationOptions {
            gradients: &gradients,
            selected_samples: LabelsSelection::Predefined(&selected),
            selected_properties: LabelsSelection::Predefined(&selected),
            selected_keys: Some(selected.keys()),
            ..Default::default()
        };

        let spherical_expansion = self.spherical_expansion.compute(
            systems,
            options,
        ).expect("failed to compute spherical expansion");

        for ((_, mut block), (_, block_spx)) in
            descriptor.iter_mut().zip(spherical_expansion.iter())
        {
            let array = block.values_mut().to_array_mut();
            let array_spx = block_spx.values().to_array();
            let shape = array_spx.shape();
            // shape[1] is the m component
            debug_assert_eq!(shape[1], 1);
            let array_spx_reshaped = array_spx.view().into_shape(
                (shape[0], shape[2])
            ).expect("wrong shape");
            array.assign(&array_spx_reshaped);

            if let Some(mut gradient) = block.gradient_mut("positions") {
                let gradient_spx = block_spx.gradient("positions").expect("missing spherical expansion gradients");
                debug_assert_eq!(gradient.samples(), gradient_spx.samples());

                let array = gradient.values_mut().to_array_mut();
                let array_spx = gradient_spx.values().to_array();
                let shape = array_spx.shape();
                // shape[2] is the m component
                debug_assert_eq!(shape[2], 1);

                let array_spx_reshaped = array_spx.view().into_shape(
                    (shape[0], shape[1], shape[3])
                ).expect("wrong shape");
                array.assign(&array_spx_reshaped);
            }

            if let Some(mut gradient) = block.gradient_mut("cell") {
                let gradient_spx = block_spx.gradient("cell").expect("missing spherical expansion gradients");
                debug_assert_eq!(gradient.samples(), gradient_spx.samples());

                let array = gradient.values_mut().to_array_mut();
                let array_spx = gradient_spx.values().to_array();
                let shape = array_spx.shape();
                // shape[2] is the m component
                debug_assert_eq!(shape[3], 1);

                let array_spx_reshaped = array_spx.view().into_shape(
                    (shape[0], shape[1], shape[2], shape[4])
                ).expect("wrong shape");
                array.assign(&array_spx_reshaped);
            }

            if let Some(mut gradient) = block.gradient_mut("strain") {
                let gradient_spx = block_spx.gradient("strain").expect("missing spherical expansion gradients");
                debug_assert_eq!(gradient.samples(), gradient_spx.samples());

                let array = gradient.values_mut().to_array_mut();
                let array_spx = gradient_spx.values().to_array();
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
    use metatensor::LabelValue;

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
            radial_basis: RadialBasis::splined_gto(1e-8),
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
    fn finite_differences_strain() {
        let calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters()).unwrap()
        ) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 5e-5,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_strain(calculator, &system, options);
    }

    #[test]
    fn finite_differences_cell() {
        let calculator = Calculator::from(Box::new(SoapRadialSpectrum::new(
            RadialSpectrumParameters {
                cutoff: 15.0,
                atomic_gaussian_width: 0.5,
                max_radial: 3,
                ..parameters()
            }
        ).unwrap()) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-5,
            epsilon: 1e-9,
        };
        crate::calculators::tests_utils::finite_differences_cell(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters()).unwrap()
        ) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let properties = Labels::new(["n"], &[
            [0],
            [3],
            [4],
            [1],
        ]);

        let samples = Labels::new(["system", "atom"], &[
            [1, 0],
            [0, 1],
            [0, 0],
        ]);

        let keys = Labels::new(["center_type", "neighbor_type"], &[
            [1, 1],
            [9, 1], // not part of the default keys
            [-42, 1],
            [1, -42],
            [1, 6],
            [-42, -42],
            [6, 1],
            [6, 6],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties,
        );
    }
}
