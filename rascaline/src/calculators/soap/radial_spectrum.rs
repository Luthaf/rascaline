use std::sync::Arc;

use equistore::{TensorMap, TensorBlock, EmptyArray};
use equistore::{LabelsBuilder, Labels, LabelValue};

use crate::calculators::CalculatorBase;
use crate::{CalculationOptions, Calculator, LabelsSelection};
use crate::{Error, System};

use super::SphericalExpansionParameters;
use super::{SphericalExpansion, RadialBasis, CutoffFunction, RadialScaling};

use crate::labels::{SpeciesFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{KeysBuilder, CenterSingleNeighborsSpeciesKeys};


/// Parameters for Radial spectrum calculator.
///
/// In the Radial spectrum, represents the spherical expansion with l=0.
/// This calculator return the same result (``TensorMap``) that you would get using the spherical expansion with l=0,
/// but it has no key ``spherical_harmonics_l`` nor the blocks have the component ``spherical_harmonics_m``.
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
    /// Should we also compute gradients of the feature?
    pub gradients: bool,
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
            gradients: parameters.gradients,
            radial_basis: parameters.radial_basis,
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
    fn selected_spx_labels(&self, descriptor: &TensorMap) -> TensorMap {
        assert_eq!(descriptor.keys().names(), ["species_center", "species_neighbor"]);

        let mut keys_builder = LabelsBuilder::new(vec!["spherical_harmonics_l", "species_center", "species_neighbor"]);
        let mut blocks = Vec::new();
        for (&[center, neighbor], block) in descriptor.keys().iter_fixed_size().zip(descriptor.blocks()) {
            keys_builder.add(&[LabelValue::new(0),center, neighbor]);

            blocks.push(TensorBlock::new(
                EmptyArray::new(vec![block.values().samples.count(), block.values().properties.count()]),
                Arc::clone(&block.values().samples),
                Vec::new(),
                Arc::clone(&block.values().properties),
            ).expect("invalid TensorBlock"));
        }

        return TensorMap::new(keys_builder.finish(), blocks).expect("invalid TensorMap")
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

    fn samples(&self, keys: &equistore::Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
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

    fn gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Option<Vec<Arc<Labels>>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor"]);
        assert_eq!(keys.count(), samples.len());

        if !self.parameters.gradients {
            return Ok(None);
        }

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

        return Ok(Some(gradient_samples));
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
        assert_eq!(descriptor.keys().names(), [ "species_center", "species_neighbor"]);
        let selected = self.selected_spx_labels(descriptor);
        let options = CalculationOptions {
            selected_samples: LabelsSelection::Predefined(&selected),
            selected_properties: LabelsSelection::Predefined(&selected),
            ..Default::default()
        };

        let spherical_expansion = self.spherical_expansion.compute(
            systems,
            options,
        ).expect("failed to compute spherical expansion");

        for ( (_,mut block_d),(_, block_spx)) in descriptor.iter_mut().zip(spherical_expansion.iter()) {
            let arr = block_d.values_mut().data.as_array_mut();
            let spx = block_spx.values().data.as_array();
            let dim = spx.shape();
            let tmp = spx.view().into_shape((dim[0],dim[2])).expect("wrong shape");
            arr.assign(&tmp);

            if self.parameters.gradients {
                let gradients = block_d.gradient_mut("positions").expect("missing radial spectrum gradients");
                let spx_grad = block_spx.gradient("positions").expect("missing spherical expansion gradients");
                debug_assert_eq!(gradients.samples, spx_grad.samples);
                let arr_grad_data = gradients.data.as_array_mut();
                let spx_grad_data = spx_grad.data.as_array();
                let dim_grad = spx_grad_data.shape();
                let tmp_grad = spx_grad_data.view().into_shape((dim_grad[0],dim_grad[1],dim_grad[3])).expect("wrong shape");
                arr_grad_data.assign(&tmp_grad);
            }
        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use equistore::LabelValue;

    use crate::systems::test_utils::{test_systems, test_system};
    use crate::Calculator;

    use super::*;
    use crate::calculators::CalculatorBase;

    fn parameters(gradients: bool) -> RadialSpectrumParameters {
        RadialSpectrumParameters {
            cutoff: 3.5,
            max_radial: 6,
            atomic_gaussian_width: 0.3,
            center_atom_weight: 1.0,
            gradients: gradients,
            radial_basis: RadialBasis::Gto {},
            radial_scaling: RadialScaling::None {},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SoapRadialSpectrum::new(
            parameters(false)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(descriptor.keys().count(), 4);
        assert!(descriptor.keys().contains(
            &[LabelValue::new(1), LabelValue::new(1)]
        ));
        assert!(descriptor.keys().contains(
            &[LabelValue::new(1), LabelValue::new(-42), ]
        ));

        assert!(descriptor.keys().contains(
            &[LabelValue::new(-42), LabelValue::new(1)]
        ));
        assert!(descriptor.keys().contains(
            &[LabelValue::new(-42), LabelValue::new(-42)]
        ));
    }

    #[test]
    fn finite_differences() {
        let calculator = Calculator::from(Box::new(SoapRadialSpectrum::new(
            parameters(true)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 5e-5,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_difference(calculator, system, options);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SoapRadialSpectrum::new(
            parameters(false)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let mut properties = LabelsBuilder::new(vec![ "n"]);
        properties.add(&[ LabelValue::new(0)]);
        properties.add(&[ LabelValue::new(3)]);
        properties.add(&[ LabelValue::new(4)]);
        properties.add(&[ LabelValue::new(1)]);

        let mut samples = LabelsBuilder::new(vec!["structure", "center"]);
        samples.add(&[LabelValue::new(0), LabelValue::new(1)]);
        samples.add(&[LabelValue::new(0), LabelValue::new(0)]);
        samples.add(&[LabelValue::new(1), LabelValue::new(0)]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &samples.finish(), &properties.finish()
        );
    }

    #[test]
    fn center_atom_weight() {
        let system = &mut test_systems(&["CH"]);

        let mut parameters = parameters(false);
        parameters.cutoff = 0.5;
        parameters.center_atom_weight = 1.0;

        let mut calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters.clone()).unwrap(),
        ) as Box<dyn CalculatorBase>);
        let descriptor = calculator.compute(system, Default::default()).unwrap();

        parameters.center_atom_weight = 0.5;
        let mut calculator = Calculator::from(Box::new(
            SoapRadialSpectrum::new(parameters).unwrap(),
        ) as Box<dyn CalculatorBase>);

        let descriptor_scaled = calculator.compute(system, Default::default()).unwrap();

        for (block, block_scaled) in descriptor.blocks().iter().zip(descriptor_scaled.blocks()) {
            assert_eq!(block.values().data.as_array(), 4.0 * block_scaled.values().data.as_array());
        }
    }
}
