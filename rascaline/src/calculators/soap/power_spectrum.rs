use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use ndarray::parallel::prelude::*;

use equistore::{TensorMap, TensorBlock, EmptyArray};
use equistore::{LabelsBuilder, Labels, LabelValue};

use crate::calculators::CalculatorBase;
use crate::{CalculationOptions, Calculator, LabelsSelection};
use crate::{Error, System};

use super::SphericalExpansionParameters;
use super::{SphericalExpansion, RadialBasis, CutoffFunction, RadialScaling};

use crate::labels::{SpeciesFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{KeysBuilder, CenterTwoNeighborsSpeciesKeys};


/// Parameters for SOAP power spectrum calculator.
///
/// In the SOAP power spectrum, each sample represents rotationally-averaged
/// atomic density correlations, built on top of the spherical expansion. Each
/// sample is a vector indexed by `n1, n2, l`, where `n1` and `n2` are radial
/// basis indexes and `l` is the angular index:
///
/// `< n1 n2 l | X_i > = \sum_m < n1 l m | X_i > < n2 l m | X_i >`
///
/// where the `< n l m | X_i >` are the spherical expansion coefficients.
///
/// See [this review article](https://doi.org/10.1063/1.5090481) for more
/// information on the SOAP representations.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct PowerSpectrumParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use
    pub max_radial: usize,
    /// Number of spherical harmonics to use
    pub max_angular: usize,
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

/// Calculator implementing the Smooth Overlap of Atomic Position (SOAP) power
/// spectrum representation of atomistic systems.
pub struct SoapPowerSpectrum {
    parameters: PowerSpectrumParameters,
    spherical_expansion: Calculator,
}

impl std::fmt::Debug for SoapPowerSpectrum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl SoapPowerSpectrum {
    pub fn new(parameters: PowerSpectrumParameters) -> Result<SoapPowerSpectrum, Error> {
        let expansion_parameters = SphericalExpansionParameters {
            cutoff: parameters.cutoff,
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            atomic_gaussian_width: parameters.atomic_gaussian_width,
            center_atom_weight: parameters.center_atom_weight,
            radial_basis: parameters.radial_basis,
            cutoff_function: parameters.cutoff_function,
            radial_scaling: parameters.radial_scaling,
        };

        let spherical_expansion = SphericalExpansion::new(expansion_parameters)?;

        return Ok(SoapPowerSpectrum {
            parameters: parameters,
            spherical_expansion: Calculator::from(
                Box::new(spherical_expansion) as Box<dyn CalculatorBase>
            ),
        });
    }

    /// Construct a `TensorMap` containing the set of samples/properties we want the
    /// spherical expansion to compute.
    ///
    /// For each block, samples will contain the same set of samples as the power
    /// spectrum, even if a neighbor species might not be around, since that
    /// simplifies the accumulation loops quite a lot.
    fn selected_spx_labels(&self, descriptor: &TensorMap) -> TensorMap {
        assert_eq!(descriptor.keys().names(), ["species_center", "species_neighbor_1", "species_neighbor_2"]);

        let mut requested = HashMap::new();
        let mut requested_spherical_harmonics_l = BTreeSet::new();
        for (&[center, neighbor_1, neighbor_2], block) in descriptor.keys().iter_fixed_size().zip(descriptor.blocks()) {
            let values = block.values();
            for &[l, n1, n2] in values.properties.iter_fixed_size() {
                requested_spherical_harmonics_l.insert(l.usize());

                let (_, properties) = requested.entry([l, center, neighbor_1]).or_insert_with(|| {
                    (BTreeSet::new(), BTreeSet::new())
                });
                properties.insert([n1]);

                let (_, properties) = requested.entry([l, center, neighbor_2]).or_insert_with(|| {
                    (BTreeSet::new(), BTreeSet::new())
                });
                properties.insert([n2]);
            }
        }

        for (&[center, neighbor_1, neighbor_2], block) in descriptor.keys().iter_fixed_size().zip(descriptor.blocks()) {
            let values = block.values();
            for &l in &requested_spherical_harmonics_l {
                let (samples_1, _) = requested.get_mut(&[l.into(), center, neighbor_1])
                    .expect("missing entry while constructing spherical expansion selection");

                for sample in &*values.samples {
                    samples_1.insert(sample);
                }

                let (samples_2, _) = requested.get_mut(&[l.into(), center, neighbor_2])
                    .expect("missing entry while constructing spherical expansion selection");

                for sample in &*values.samples {
                    samples_2.insert(sample);
                }
            }
        }

        let mut keys_builder = LabelsBuilder::new(vec!["spherical_harmonics_l", "species_center", "species_neighbor"]);
        let mut blocks = Vec::new();
        for (key, (samples, properties)) in requested {
            keys_builder.add(&key);

            let mut samples_builder = LabelsBuilder::new(vec!["structure", "center"]);
            for entry in samples {
                samples_builder.add(entry);
            }
            let samples = Arc::new(samples_builder.finish());

            let mut properties_builder = LabelsBuilder::new(vec!["n"]);
            for entry in properties {
                properties_builder.add(&entry);
            }
            let properties = Arc::new(properties_builder.finish());

            blocks.push(TensorBlock::new(
                EmptyArray::new(vec![samples.count(), properties.count()]),
                samples,
                Vec::new(),
                properties,
            ).expect("invalid TensorBlock"));
        }

        // if the user selected only a subset of l entries, make sure there are
        // empty blocks for the corresponding keys in the spherical expansion
        // selection
        let mut missing_keys = BTreeSet::new();
        for &[center, neighbor_1, neighbor_2] in descriptor.keys().iter_fixed_size() {
            for spherical_harmonics_l in 0..=(self.parameters.max_angular) {
                if !requested_spherical_harmonics_l.contains(&spherical_harmonics_l) {
                    missing_keys.insert([spherical_harmonics_l.into(), center, neighbor_1]);
                    missing_keys.insert([spherical_harmonics_l.into(), center, neighbor_2]);
                }
            }
        }
        for key in missing_keys {
            keys_builder.add(&key);

            let samples = Arc::new(LabelsBuilder::new(vec!["structure", "center"]).finish());
            let properties = Arc::new(LabelsBuilder::new(vec!["n"]).finish());
            blocks.push(TensorBlock::new(
                EmptyArray::new(vec![samples.count(), properties.count()]),
                samples,
                Vec::new(),
                properties,
            ).expect("invalid TensorBlock"));
        }

        return TensorMap::new(keys_builder.finish(), blocks).expect("invalid TensorMap")
    }

    /// Pre-compute the correspondance between samples of the spherical
    /// expansion & the power spectrum, both for values and gradients.
    ///
    /// For example, the key `center, neighbor_1, neighbor_2 = 1, 6, 8` will
    /// have a very different set of samples from `c, n_1, n_2 = 1, 6, 6`; but
    /// both will use the spherical expansion `center, neighbor = 1, 6`.
    ///
    /// This function returns the list of spherical expansion sample indexes
    /// corresponding to the requested samples in `descriptor` for each block.
    fn samples_mapping(
        descriptor: &TensorMap,
        spherical_expansion: &TensorMap
    ) -> HashMap<Vec<LabelValue>, SamplesMapping> {
        let mut mapping = HashMap::new();
        for (key, block) in descriptor.iter() {
            let species_center = key[0];
            let species_neighbor_1 = key[1];
            let species_neighbor_2 = key[2];

            let mut values_mapping = Vec::new();
            let values = block.values();
            assert!(values.properties.count() > 0);

            // the spherical expansion samples are the same for all
            // `spherical_harmonics_l` values, so we only need to compute it for
            // the first one.
            let first_l = values.properties[0][0];

            let block_id_1 = spherical_expansion.keys().position(&[
                first_l, species_center, species_neighbor_1
            ]).expect("missing block in spherical expansion");
            let spx_block_1 = &spherical_expansion.blocks()[block_id_1];
            let spx_samples_1 = &spx_block_1.values().samples;

            let block_id_2 = spherical_expansion.keys().position(&[
                first_l, species_center, species_neighbor_2
            ]).expect("missing block in spherical expansion");
            let spx_block_2 = &spherical_expansion.blocks()[block_id_2];
            let spx_samples_2 = &spx_block_2.values().samples;

            values_mapping.reserve(values.samples.count());
            for sample in &*values.samples {
                let sample_1 = spx_samples_1.position(sample).expect("missing spherical expansion sample");
                let sample_2 = spx_samples_2.position(sample).expect("missing spherical expansion sample");
                values_mapping.push((sample_1, sample_2));
            }

            let mut gradient_mapping = Vec::new();
            if let Some(gradient) = block.gradient("positions") {
                let spx_gradient_1 = spx_block_1.gradient("positions").expect("missing spherical expansion gradients");
                let spx_gradient_2 = spx_block_2.gradient("positions").expect("missing spherical expansion gradients");

                gradient_mapping.reserve(gradient.samples.count());

                for gradient_sample in gradient.samples.iter() {
                    gradient_mapping.push((
                        spx_gradient_1.samples.position(gradient_sample),
                        spx_gradient_2.samples.position(gradient_sample),
                    ));
                }
            }

            mapping.insert(key.to_vec(), SamplesMapping {
                values: values_mapping,
                gradients: gradient_mapping
            });
        }

        return mapping;
    }

    /// Get the list of spherical expansion to combine when computing a single
    /// block (associated with the given key) of the power spectrum.
    fn spx_properties_to_combine<'a>(
        key: &[LabelValue],
        properties: &Labels,
        spherical_expansion: &'a TensorMap,
    ) -> Vec<SpxPropertiesToCombine<'a>> {
        let species_center = key[0];
        let species_neighbor_1 = key[1];
        let species_neighbor_2 = key[2];

        let mut spx_to_combine = Vec::with_capacity(properties.count());
        for &[l, n1, n2] in properties.iter_fixed_size() {
            let block_1 = spherical_expansion.keys().position(
                &[l, species_center, species_neighbor_1]
            ).expect("missing first neighbor species block in spherical expansion");
            let block_1 = &spherical_expansion.blocks()[block_1];

            let block_2 = spherical_expansion.keys().position(
                &[l, species_center, species_neighbor_2]
            ).expect("missing second neighbor species block in spherical expansion");
            let block_2 = &spherical_expansion.blocks()[block_2];

            let values_1 = block_1.values().data.as_array();
            let values_2 = block_2.values().data.as_array();

            // both blocks should had the same number of m components
            debug_assert_eq!(values_1.shape()[1], values_2.shape()[1]);

            let property_1 = block_1.values().properties.position(&[n1]).expect("missing n1");
            let property_2 = block_2.values().properties.position(&[n2]).expect("missing n2");

            spx_to_combine.push(SpxPropertiesToCombine {
                spherical_harmonics_l: l.usize(),
                property_1,
                property_2,
                block_1,
                block_2,
            });
        }

        return spx_to_combine;
    }
}


/// Data about the two spherical expansion block that will get combined to
/// produce a single (l, n1, n2) property in a single power spectrum block
struct SpxPropertiesToCombine<'a> {
    /// value of l
    spherical_harmonics_l: usize,
    /// position of n1 in the first spherical expansion properties
    property_1: usize,
    /// position of n2 in the second spherical expansion properties
    property_2: usize,
    /// TODO
    block_1: &'a TensorBlock,
    /// TODO
    block_2: &'a TensorBlock,
}

/// Indexes of the spherical expansion samples/rows corresponding to each power
/// spectrum row.
struct SamplesMapping {
    /// Mapping for the values.
    values: Vec<(usize, usize)>,
    /// Mapping for the gradients.
    ///
    /// Some samples might not be defined in both of the spherical expansion
    /// blocks being considered, for examples when dealing with two different
    /// neighbor species, only one the sample corresponding to the right
    /// neighbor species will be `Some`.
    gradients: Vec<(Option<usize>, Option<usize>)>,
}

impl CalculatorBase for SoapPowerSpectrum {
    fn name(&self) -> String {
        "SOAP power spectrum".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<equistore::Labels, Error> {
        let builder = CenterTwoNeighborsSpeciesKeys {
            cutoff: self.parameters.cutoff,
            self_pairs: true,
            symmetric: true,
        };
        return builder.keys(systems);
    }

    fn samples_names(&self) -> Vec<&str> {
        AtomCenteredSamples::samples_names()
    }

    fn samples(&self, keys: &equistore::Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor_1", "species_neighbor_2"]);
        let mut result = Vec::new();
        for [species_center, species_neighbor_1, species_neighbor_2] in keys.iter_fixed_size() {

            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                // we only want center with both neighbor species present
                species_neighbor: SpeciesFilter::AllOf(
                    [
                        species_neighbor_1.i32(),
                        species_neighbor_2.i32()
                    ].iter().copied().collect()
                ),
                self_pairs: true,
            };

            result.push(builder.samples(systems)?);
        }

        return Ok(result);
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor_1", "species_neighbor_2"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([species_center, species_neighbor_1, species_neighbor_2], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                // gradients samples should contain either neighbor species
                species_neighbor: SpeciesFilter::OneOf(vec![
                    species_neighbor_1.i32(),
                    species_neighbor_2.i32()
                ]),
                self_pairs: true,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            "cell" => true,
            _ => false,
        }
    }

    fn components(&self, keys: &equistore::Labels) -> Vec<Vec<Arc<Labels>>> {
        return vec![vec![]; keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["l", "n1", "n2"]
    }

    fn properties(&self, keys: &equistore::Labels) -> Vec<Arc<Labels>> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        for l in 0..=self.parameters.max_angular {
            for n1 in 0..self.parameters.max_radial {
                for n2 in 0..self.parameters.max_radial {
                    properties.add(&[l, n1, n2]);
                }
            }
        }

        let properties = Arc::new(properties.finish());

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SoapPowerSpectrum::compute")]
    #[allow(clippy::too_many_lines)]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        let mut gradients = Vec::new();
        if descriptor.blocks()[0].gradient("positions").is_some() {
            gradients.push("positions");
        }
        if descriptor.blocks()[0].gradient("cell").is_some() {
            gradients.push("cell");
        }

        let selected = self.selected_spx_labels(descriptor);
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

        let samples_mapping = SoapPowerSpectrum::samples_mapping(descriptor, &spherical_expansion);

        for (key, mut block) in descriptor.iter_mut() {
            let species_neighbor_1 = key[1];
            let species_neighbor_2 = key[2];

            let values = block.values();
            let properties_to_combine = SoapPowerSpectrum::spx_properties_to_combine(
                key,
                &values.properties,
                &spherical_expansion,
            );

            let mapping = samples_mapping.get(key).expect("missing sample mapping");

            block.values_mut().data.as_array_mut()
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .zip_eq(&mapping.values)
                .for_each(|(mut values, &(spx_sample_1, spx_sample_2))| {
                    for (property_i, spx) in properties_to_combine.iter().enumerate() {
                        let values_1 = spx.block_1.values().data.as_array();
                        let values_2 = spx.block_2.values().data.as_array();

                        let mut sum = 0.0;

                        for m in 0..(2 * spx.spherical_harmonics_l + 1) {
                            // unsafe is required to remove the bound checking
                            // in release mode (`uget` still checks bounds in
                            // debug mode)
                            unsafe {
                                sum += values_1.uget([spx_sample_1, m, spx.property_1])
                                     * values_2.uget([spx_sample_2, m, spx.property_2]);
                            }
                        }

                        if species_neighbor_1 != species_neighbor_2 {
                            // We only store values for `species_neighbor_1 <
                            // species_neighbor_2` because the values are the
                            // same for pairs `species_neighbor_1 <->
                            // species_neighbor_2` and `species_neighbor_2 <->
                            // species_neighbor_1`. To ensure the final kernels
                            // are correct, we have to multiply the
                            // corresponding values.
                            sum *= std::f64::consts::SQRT_2;
                        }

                        unsafe {
                            *values.uget_mut(property_i) = sum / f64::sqrt((2 * spx.spherical_harmonics_l + 1) as f64);
                        }
                    }
                });


            // gradients with respect to the atomic positions
            if let Some(gradients) = block.gradient_mut("positions") {
                gradients.data.as_array_mut()
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip_eq(gradients.samples.par_iter())
                    .zip_eq(&mapping.gradients)
                    .for_each(|((mut values, gradient_sample), &(spx_grad_sample_1, spx_grad_sample_2))| {
                        for (property_i, spx) in properties_to_combine.iter().enumerate() {
                            let spx_values_1 = spx.block_1.values().data.as_array();
                            let spx_values_2 = spx.block_2.values().data.as_array();

                            let spx_gradient_1 = spx.block_1.gradient("positions").expect("missing spherical expansion gradients");
                            let spx_gradient_1 = spx_gradient_1.data.as_array();
                            let spx_gradient_2 = spx.block_2.gradient("positions").expect("missing spherical expansion gradients");
                            let spx_gradient_2 = spx_gradient_2.data.as_array();

                            let sample_i = gradient_sample[0].usize();
                            let (spx_sample_1, spx_sample_2) = mapping.values[sample_i];

                            let mut sum = [0.0, 0.0, 0.0];
                            if let Some(grad_sample_1) = spx_grad_sample_1 {
                                for m in 0..(2 * spx.spherical_harmonics_l + 1) {
                                    // SAFETY: see same loop for values
                                    unsafe {
                                        let value_2 = spx_values_2.uget([spx_sample_2, m, spx.property_2]);
                                        for d in 0..3 {
                                            sum[d] += value_2 * spx_gradient_1.uget([grad_sample_1, d, m, spx.property_1]);
                                        }
                                    }
                                }
                            }

                            if let Some(grad_sample_2) = spx_grad_sample_2 {
                                for m in 0..(2 * spx.spherical_harmonics_l + 1) {
                                    // SAFETY: see same loop for values
                                    unsafe {
                                        let value_1 = spx_values_1.uget([spx_sample_1, m, spx.property_1]);
                                        for d in 0..3 {
                                            sum[d] += value_1 * spx_gradient_2.uget([grad_sample_2, d, m, spx.property_2]);
                                        }
                                    }
                                }
                            }

                            if species_neighbor_1 != species_neighbor_2 {
                                // see above
                                for d in 0..3 {
                                    sum[d] *= std::f64::consts::SQRT_2;
                                }
                            }

                            let normalization = f64::sqrt((2 * spx.spherical_harmonics_l + 1) as f64);
                            for d in 0..3 {
                                unsafe {
                                    *values.uget_mut([d, property_i]) = sum[d] / normalization;
                                }
                            }
                        }
                    });
            }


            // gradients with respect to the cell parameters
            if let Some(gradients) = block.gradient_mut("cell") {
                gradients.data.as_array_mut()
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip_eq(gradients.samples.par_iter())
                    .for_each(|(mut values, gradient_sample)| {
                        for (property_i, spx) in properties_to_combine.iter().enumerate() {
                            let spx_values_1 = spx.block_1.values().data.as_array();
                            let spx_values_2 = spx.block_2.values().data.as_array();

                            let spx_gradient_1 = spx.block_1.gradient("cell").expect("missing spherical expansion gradients");
                            let spx_gradient_1 = spx_gradient_1.data.as_array();
                            let spx_gradient_2 = spx.block_2.gradient("cell").expect("missing spherical expansion gradients");
                            let spx_gradient_2 = spx_gradient_2.data.as_array();

                            let sample_i = gradient_sample[0].usize();
                            let (spx_sample_1, spx_sample_2) = mapping.values[sample_i];

                            let mut sum = [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                            ];
                            for m in 0..(2 * spx.spherical_harmonics_l + 1) {
                                // SAFETY: see same loop for values
                                unsafe {
                                    let value_2 = spx_values_2.uget([spx_sample_2, m, spx.property_2]);
                                    for d1 in 0..3 {
                                        for d2 in 0..3 {
                                            // TODO: ensure that gradient samples are 0..nsamples
                                            sum[d1][d2] += value_2 * spx_gradient_1.uget([spx_sample_1, d1, d2, m, spx.property_1]);
                                        }
                                    }
                                }
                            }

                            for m in 0..(2 * spx.spherical_harmonics_l + 1) {
                                // SAFETY: see same loop for values
                                unsafe {
                                    let value_1 = spx_values_1.uget([spx_sample_1, m, spx.property_1]);
                                    for d1 in 0..3 {
                                        for d2 in 0..3 {
                                            // TODO: ensure that gradient samples are 0..nsamples
                                            sum[d1][d2] += value_1 * spx_gradient_2.uget([spx_sample_2, d1, d2, m, spx.property_2]);
                                        }
                                    }
                                }
                            }

                            if species_neighbor_1 != species_neighbor_2 {
                                // see above
                                for d1 in 0..3 {
                                    for d2 in 0..3 {
                                        sum[d1][d2] *= std::f64::consts::SQRT_2;
                                    }
                                }
                            }

                            let normalization = f64::sqrt((2 * spx.spherical_harmonics_l + 1) as f64);

                            for d1 in 0..3 {
                                for d2 in 0..3 {
                                    unsafe {
                                        *values.uget_mut([d1, d2, property_i]) = sum[d1][d2] / normalization;
                                    }
                                }
                            }
                        }
                    });
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

    fn parameters() -> PowerSpectrumParameters {
        PowerSpectrumParameters {
            cutoff: 3.5,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            center_atom_weight: 1.0,
            radial_basis: RadialBasis::Gto { splined_radial_integral: true, spline_accuracy: 1e-8 },
            radial_scaling: RadialScaling::None {},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(descriptor.keys().count(), 6);
        assert!(descriptor.keys().contains(
            &[LabelValue::new(1), LabelValue::new(1), LabelValue::new(1)]
        ));
        assert!(descriptor.keys().contains(
            &[LabelValue::new(1), LabelValue::new(-42), LabelValue::new(1)]
        ));
        assert!(descriptor.keys().contains(
            &[LabelValue::new(1), LabelValue::new(-42), LabelValue::new(-42)]
        ));

        assert!(descriptor.keys().contains(
            &[LabelValue::new(-42), LabelValue::new(1), LabelValue::new(1)]
        ));
        assert!(descriptor.keys().contains(
            &[LabelValue::new(-42), LabelValue::new(-42), LabelValue::new(1)]
        ));
        assert!(descriptor.keys().contains(
            &[LabelValue::new(-42), LabelValue::new(-42), LabelValue::new(-42)]
        ));

        // exact values for power spectrum are regression-tested in
        // `rascaline/tests/soap-power-spectrum.rs`
    }

    #[test]
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

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
        let calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

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
        let calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let mut properties = LabelsBuilder::new(vec!["l", "n1", "n2"]);
        properties.add(&[LabelValue::new(0), LabelValue::new(0), LabelValue::new(1)]);
        properties.add(&[LabelValue::new(3), LabelValue::new(3), LabelValue::new(3)]);
        properties.add(&[LabelValue::new(2), LabelValue::new(4), LabelValue::new(3)]);
        properties.add(&[LabelValue::new(1), LabelValue::new(4), LabelValue::new(4)]);
        properties.add(&[LabelValue::new(5), LabelValue::new(1), LabelValue::new(0)]);
        properties.add(&[LabelValue::new(1), LabelValue::new(1), LabelValue::new(2)]);

        let mut samples = LabelsBuilder::new(vec!["structure", "center"]);
        samples.add(&[LabelValue::new(0), LabelValue::new(1)]);
        samples.add(&[LabelValue::new(0), LabelValue::new(2)]);
        samples.add(&[LabelValue::new(1), LabelValue::new(0)]);
        samples.add(&[LabelValue::new(1), LabelValue::new(2)]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &samples.finish(), &properties.finish()
        );
    }

    #[test]
    fn center_atom_weight() {
        let system = &mut test_systems(&["CH"]);

        let mut parameters = parameters();
        parameters.cutoff = 0.5;
        parameters.center_atom_weight = 1.0;

        let mut calculator = Calculator::from(Box::new(
            SoapPowerSpectrum::new(parameters.clone()).unwrap(),
        ) as Box<dyn CalculatorBase>);
        let descriptor = calculator.compute(system, Default::default()).unwrap();

        parameters.center_atom_weight = 0.5;
        let mut calculator = Calculator::from(Box::new(
            SoapPowerSpectrum::new(parameters).unwrap(),
        ) as Box<dyn CalculatorBase>);

        let descriptor_scaled = calculator.compute(system, Default::default()).unwrap();

        for (block, block_scaled) in descriptor.blocks().iter().zip(descriptor_scaled.blocks()) {
            assert_eq!(block.values().data.as_array(), 4.0 * block_scaled.values().data.as_array());
        }
    }
}
