use std::collections::{BTreeSet, HashMap};

use ndarray::parallel::prelude::*;

use metatensor::{TensorMap, TensorBlock, EmptyArray};
use metatensor::{LabelsBuilder, Labels, LabelValue};

use crate::calculators::CalculatorBase;
use crate::{CalculationOptions, Calculator, LabelsSelection};
use crate::{Error, System};

use super::SphericalExpansionParameters;
use super::{SphericalExpansion, CutoffFunction, RadialScaling};
use crate::calculators::radial_basis::RadialBasis;

use crate::labels::{AtomicTypeFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{KeysBuilder, CenterTwoNeighborsTypesKeys};


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
    /// Weight of the central atom contribution to the
    /// features. If `1.0` the center atom contribution is weighted the same
    /// as any other contribution. If `0.0` the central atom does not
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
            radial_basis: parameters.radial_basis.clone(),
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

    /// Construct a `TensorMap` containing the set of samples/properties we want
    /// the spherical expansion calculator to compute.
    ///
    /// For each block, samples will contain the same set of samples as the
    /// power spectrum, even if a neighbor type might not be around, since
    /// that simplifies the accumulation loops quite a lot.
    fn selected_spx_labels(&self, descriptor: &TensorMap) -> TensorMap {
        assert_eq!(descriptor.keys().names(), ["center_type", "neighbor_1_type", "neighbor_2_type"]);

        // first, go over the requested power spectrum properties and group them
        // depending on the neighbor_type
        let mut requested_by_key = HashMap::new();
        let mut requested_o3_lambda = BTreeSet::new();
        for (&[center, neighbor_1, neighbor_2], block) in descriptor.keys().iter_fixed_size().zip(descriptor.blocks()) {
            for &[l, n1, n2] in block.properties().iter_fixed_size() {
                requested_o3_lambda.insert(l.usize());

                let (_, properties) = requested_by_key
                    .entry([l, 1.into(), center, neighbor_1])
                    .or_insert_with(|| (BTreeSet::new(), BTreeSet::new()));
                properties.insert([n1]);

                let (_, properties) = requested_by_key
                    .entry([l, 1.into(), center, neighbor_2])
                    .or_insert_with(|| (BTreeSet::new(), BTreeSet::new()));
                properties.insert([n2]);
            }
        }

        // make sure all the expected blocks are there, even if the power
        // spectrum does not contain e.g. l=3 at all. The corresponding blocks
        // will have an empty set of properties
        for &[center, neighbor_1, neighbor_2] in descriptor.keys().iter_fixed_size() {
            for &l in &requested_o3_lambda {
                requested_by_key
                    .entry([l.into(), 1.into(), center, neighbor_1])
                    .or_insert_with(|| (BTreeSet::new(), BTreeSet::new()));

                requested_by_key
                    .entry([l.into(), 1.into(), center, neighbor_2])
                    .or_insert_with(|| (BTreeSet::new(), BTreeSet::new()));
            }
        }

        // Then, loop over the requested power spectrum, and accumulate the
        // samples we want to compute.
        for (&[_, _, requested_center, requested_neighbor], (samples, _)) in &mut requested_by_key {
            for (key, block) in descriptor {
                let center = key[0];
                let neighbor_1 = key[1];
                let neighbor_2 = key[2];

                if center != requested_center {
                    continue;
                }

                if !(requested_neighbor == neighbor_1 || requested_neighbor == neighbor_2) {
                    continue;
                }

                for &sample in block.samples().iter_fixed_size::<2>() {
                    samples.insert(sample);
                }
            }
        }

        let mut keys_builder = LabelsBuilder::new(vec!["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        let mut blocks = Vec::new();
        for (key, (samples, properties)) in requested_by_key {
            keys_builder.add(&key);

            let mut samples_builder = LabelsBuilder::new(vec!["system", "atom"]);
            for entry in samples {
                samples_builder.add(&entry);
            }
            let samples = samples_builder.finish();

            let mut properties_builder = LabelsBuilder::new(vec!["n"]);
            for entry in properties {
                properties_builder.add(&entry);
            }
            let properties = properties_builder.finish();

            blocks.push(TensorBlock::new(
                EmptyArray::new(vec![samples.count(), properties.count()]),
                &samples,
                &[],
                &properties,
            ).expect("invalid TensorBlock"));
        }

        // if the user selected only a subset of l entries, make sure there are
        // empty blocks for the corresponding keys in the spherical expansion
        // selection
        let mut missing_keys = BTreeSet::new();
        for &[center, neighbor_1, neighbor_2] in descriptor.keys().iter_fixed_size() {
            for o3_lambda in 0..=(self.parameters.max_angular) {
                if !requested_o3_lambda.contains(&o3_lambda) {
                    missing_keys.insert([o3_lambda.into(), 1.into(), center, neighbor_1]);
                    missing_keys.insert([o3_lambda.into(), 1.into(), center, neighbor_2]);
                }
            }
        }
        for key in missing_keys {
            keys_builder.add(&key);

            let samples = Labels::empty(vec!["system", "atom"]);
            let properties = Labels::empty(vec!["n"]);
            blocks.push(TensorBlock::new(
                EmptyArray::new(vec![samples.count(), properties.count()]),
                &samples,
                &[],
                &properties,
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
        for (key, block) in descriptor {
            let center_type = key[0];
            let neighbor_1_type = key[1];
            let neighbor_2_type = key[2];

            let block_data = block.data();
            if block_data.properties.count() == 0 {
                // no properties to compute, we don't really care about sample
                // mapping and we can not compute the real one (there is no l to
                // find the corresponding spx block), so we'll create a dummy
                // sample mapping / gradient sample mapping
                let mut values_mapping = Vec::new();
                for i in 0..block_data.samples.count() {
                    values_mapping.push((i, i));
                }

                let mut gradient_mapping = Vec::new();
                if let Some(gradient) = block.gradient("positions") {
                    let gradient = gradient.data();
                    for i in 0..gradient.samples.count() {
                        gradient_mapping.push((Some(i), Some(i)));
                    }
                }

                mapping.insert(key.to_vec(), SamplesMapping {
                    values: values_mapping,
                    gradients: gradient_mapping,
                });
                continue;
            }

            let mut values_mapping = Vec::new();

            // the spherical expansion samples are the same for all
            // `o3_lambda` values, so we only need to compute it for
            // the first one.
            let first_l = block_data.properties[0][0];

            let block_id_1 = spherical_expansion.keys().position(&[
                first_l, 1.into(), center_type, neighbor_1_type
            ]).expect("missing block in spherical expansion");
            let spx_block_1 = &spherical_expansion.block_by_id(block_id_1);
            let spx_samples_1 = spx_block_1.samples();

            let block_id_2 = spherical_expansion.keys().position(&[
                first_l, 1.into(), center_type, neighbor_2_type
            ]).expect("missing block in spherical expansion");
            let spx_block_2 = &spherical_expansion.block_by_id(block_id_2);
            let spx_samples_2 = spx_block_2.samples();

            values_mapping.reserve(block_data.samples.count());
            for sample in &*block_data.samples {
                let sample_1 = spx_samples_1.position(sample).expect("missing spherical expansion sample");
                let sample_2 = spx_samples_2.position(sample).expect("missing spherical expansion sample");
                values_mapping.push((sample_1, sample_2));
            }

            let mut gradient_mapping = Vec::new();
            if let Some(gradient) = block.gradient("positions") {
                let spx_gradient_1 = spx_block_1.gradient("positions").expect("missing spherical expansion gradients");
                let spx_gradient_2 = spx_block_2.gradient("positions").expect("missing spherical expansion gradients");

                let gradient_samples = gradient.samples();
                gradient_mapping.reserve(gradient_samples.count());

                let spx_gradient_1_samples = spx_gradient_1.samples();
                let spx_gradient_2_samples = spx_gradient_2.samples();

                for &[sample, system, atom] in gradient_samples.iter_fixed_size() {
                    // The "sample" dimension in the power spectrum gradient
                    // samples do not necessarily matches the "sample" dimension
                    // in the spherical expansion gradient samples. We use the
                    // sample mapping for the values to create what would be the
                    // right gradient sample for spx, and then lookup its
                    // position in the spx gradient samples
                    let (spx_1_sample, spx_2_sample) = values_mapping[sample.usize()];

                    let mapping_1 = spx_gradient_1_samples.position(
                        &[spx_1_sample.into(), system, atom]
                    );
                    let mapping_2 = spx_gradient_2_samples.position(
                        &[spx_2_sample.into(), system, atom]
                    );

                    // at least one of the spx block should contribute to the
                    // gradients
                    debug_assert!(mapping_1.is_some() || mapping_2.is_some());

                    gradient_mapping.push((mapping_1, mapping_2));
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
        spherical_expansion: &HashMap<&[LabelValue], SphericalExpansionBlock<'a>>,
    ) -> Vec<SpxPropertiesToCombine<'a>> {
        let center_type = key[0];
        let neighbor_1_type = key[1];
        let neighbor_2_type = key[2];

        return properties.par_iter().map(|property| {
            let l = property[0];
            let n1 = property[1];
            let n2 = property[2];

            let key_1: &[_] = &[l, 1.into(), center_type, neighbor_1_type];
            let block_1 = spherical_expansion.get(&key_1)
            .expect("missing first neighbor type block in spherical expansion");

            let key_2: &[_] = &[l, 1.into(), center_type, neighbor_2_type];
            let block_2 = spherical_expansion.get(&key_2)
                .expect("missing first neighbor type block in spherical expansion");

            // both blocks should had the same number of m components
            debug_assert_eq!(block_1.values.shape()[1], block_2.values.shape()[1]);

            let property_1 = block_1.properties.position(&[n1]).expect("missing n1");
            let property_2 = block_2.properties.position(&[n2]).expect("missing n2");

            let o3_lambda = l.usize();

            // For consistency with a full Clebsch-Gordan product we need to add
            // a `-1^l / sqrt(2 l + 1)` factor to the power spectrum invariants
            let normalization = if o3_lambda % 2 == 0 {
                f64::sqrt((2 * o3_lambda + 1) as f64)
            } else {
                -f64::sqrt((2 * o3_lambda + 1) as f64)
            };

            SpxPropertiesToCombine {
                o3_lambda,
                normalization,
                property_1,
                property_2,
                spx_1: block_1.clone(),
                spx_2: block_2.clone(),
            }
        }).collect();
    }
}


/// Data about the two spherical expansion block that will get combined to
/// produce a single (l, n1, n2) property in a single power spectrum block
struct SpxPropertiesToCombine<'a> {
    /// value of l
    o3_lambda: usize,
    /// normalization factor $-1^l * \sqrt{2 l + 1}$
    normalization: f64,
    /// position of n1 in the first spherical expansion properties
    property_1: usize,
    /// position of n2 in the second spherical expansion properties
    property_2: usize,
    /// first spherical expansion block
    spx_1: SphericalExpansionBlock<'a>,
    /// second spherical expansion block
    spx_2: SphericalExpansionBlock<'a>,
}

/// Data from a single spherical expansion block
#[derive(Debug, Clone)]
struct SphericalExpansionBlock<'a> {
    properties: Labels,
    /// spherical expansion values
    values: &'a ndarray::ArrayD<f64>,
    /// spherical expansion position gradients
    positions_gradients: Option<&'a ndarray::ArrayD<f64>>,
    /// spherical expansion cell gradients
    cell_gradients: Option<&'a ndarray::ArrayD<f64>>,
}

/// Indexes of the spherical expansion samples/rows corresponding to each power
/// spectrum row.
struct SamplesMapping {
    /// Mapping for the values: if the row `i` of the power spectrum is a
    /// combination of the rows `j` and `k` of two spherical expansion blocks,
    /// then this vector will contain `(j, k)` at index `i`
    values: Vec<(usize, usize)>,
    /// Mapping for the gradients, with a similar layout as the `values`
    ///
    /// Some samples might not be defined in both of the spherical expansion
    /// blocks being considered, for examples when dealing with two different
    /// neighbor types, only one the sample corresponding to the right
    /// neighbor type will be `Some`.
    gradients: Vec<(Option<usize>, Option<usize>)>,
}

impl CalculatorBase for SoapPowerSpectrum {
    fn name(&self) -> String {
        "SOAP power spectrum".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        self.spherical_expansion.cutoffs()
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<metatensor::Labels, Error> {
        let builder = CenterTwoNeighborsTypesKeys {
            cutoff: self.parameters.cutoff,
            self_pairs: true,
            symmetric: true,
        };
        return builder.keys(systems);
    }

    fn sample_names(&self) -> Vec<&str> {
        AtomCenteredSamples::sample_names()
    }

    fn samples(&self, keys: &metatensor::Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type", "neighbor_1_type", "neighbor_2_type"]);
        let mut result = Vec::new();
        for [center_type, neighbor_1_type, neighbor_2_type] in keys.iter_fixed_size() {

            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                // we only want center with both neighbor types present
                neighbor_type: AtomicTypeFilter::AllOf(
                    [
                        neighbor_1_type.i32(),
                        neighbor_2_type.i32()
                    ].iter().copied().collect()
                ),
                self_pairs: true,
            };

            result.push(builder.samples(systems)?);
        }

        return Ok(result);
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type", "neighbor_1_type", "neighbor_2_type"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([center_type, neighbor_1_type, neighbor_2_type], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                // gradients samples should contain either neighbor types
                neighbor_type: AtomicTypeFilter::OneOf(vec![
                    neighbor_1_type.i32(),
                    neighbor_2_type.i32()
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

    fn components(&self, keys: &metatensor::Labels) -> Vec<Vec<Labels>> {
        return vec![vec![]; keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["l", "n_1", "n_2"]
    }

    fn properties(&self, keys: &metatensor::Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        for l in 0..=self.parameters.max_angular {
            for n1 in 0..self.parameters.max_radial {
                for n2 in 0..self.parameters.max_radial {
                    properties.add(&[l, n1, n2]);
                }
            }
        }
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SoapPowerSpectrum::compute")]
    #[allow(clippy::too_many_lines)]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        let mut gradients = Vec::new();
        if descriptor.block_by_id(0).gradient("positions").is_some() {
            gradients.push("positions");
        }
        if descriptor.block_by_id(0).gradient("cell").is_some() {
            gradients.push("cell");
        }

        let selected = self.selected_spx_labels(descriptor);

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
        let samples_mapping = SoapPowerSpectrum::samples_mapping(descriptor, &spherical_expansion);

        let spherical_expansion = spherical_expansion.iter().map(|(key, block)| {
            let spx_block = SphericalExpansionBlock {
                properties: block.properties(),
                values: block.values().to_array(),
                positions_gradients: block.gradient("positions").map(|g| g.values().to_array()),
                cell_gradients: block.gradient("cell").map(|g| g.values().to_array()),
            };

            (key, spx_block)
        }).collect();

        for (key, mut block) in descriptor {
            let neighbor_1_type = key[1];
            let neighbor_2_type = key[2];

            let mut block_data = block.data_mut();
            let properties_to_combine = SoapPowerSpectrum::spx_properties_to_combine(
                key,
                &block_data.properties,
                &spherical_expansion,
            );

            let mapping = samples_mapping.get(key).expect("missing sample mapping");

            block_data.values.as_array_mut()
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .zip_eq(&mapping.values)
                .for_each(|(mut values, &(spx_sample_1, spx_sample_2))| {
                    for (property_i, spx) in properties_to_combine.iter().enumerate() {
                        let SpxPropertiesToCombine { spx_1, spx_2, ..} = spx;

                        let mut sum = 0.0;

                        for m in 0..(2 * spx.o3_lambda + 1) {
                            // unsafe is required to remove the bound checking
                            // in release mode (`uget` still checks bounds in
                            // debug mode)
                            unsafe {
                                sum += spx_1.values.uget([spx_sample_1, m, spx.property_1])
                                     * spx_2.values.uget([spx_sample_2, m, spx.property_2]);
                            }
                        }

                        if neighbor_1_type != neighbor_2_type {
                            // We only store values for `neighbor_1_type <
                            // neighbor_2_type` because the values are the
                            // same for pairs `neighbor_1_type <->
                            // neighbor_2_type` and `neighbor_2_type <->
                            // neighbor_1_type`. To ensure the final kernels
                            // are correct, we have to multiply the
                            // corresponding values.
                            sum *= std::f64::consts::SQRT_2;
                        }

                        unsafe {
                            *values.uget_mut(property_i) = sum / spx.normalization;
                        }
                    }
                });

            // gradients with respect to the atomic positions
            if let Some(mut gradient) = block.gradient_mut("positions") {
                let gradient = gradient.data_mut();

                gradient.values.to_array_mut()
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip_eq(gradient.samples.par_iter())
                    .zip_eq(&mapping.gradients)
                    .for_each(|((mut values, gradient_sample), &(spx_grad_sample_1, spx_grad_sample_2))| {
                        for (property_i, spx) in properties_to_combine.iter().enumerate() {
                            let SpxPropertiesToCombine { spx_1, spx_2, ..} = spx;

                            let spx_1_gradient = spx_1.positions_gradients.expect("missing spherical expansion gradients");
                            let spx_2_gradient = spx_2.positions_gradients.expect("missing spherical expansion gradients");

                            let sample_i = gradient_sample[0].usize();
                            let (spx_sample_1, spx_sample_2) = mapping.values[sample_i];

                            let mut sum = [0.0, 0.0, 0.0];
                            if let Some(grad_sample_1) = spx_grad_sample_1 {
                                for m in 0..(2 * spx.o3_lambda + 1) {
                                    // SAFETY: see same loop for values
                                    unsafe {
                                        let value_2 = spx_2.values.uget([spx_sample_2, m, spx.property_2]);
                                        for d in 0..3 {
                                            sum[d] += value_2 * spx_1_gradient.uget([grad_sample_1, d, m, spx.property_1]);
                                        }
                                    }
                                }
                            }

                            if let Some(grad_sample_2) = spx_grad_sample_2 {
                                for m in 0..(2 * spx.o3_lambda + 1) {
                                    // SAFETY: see same loop for values
                                    unsafe {
                                        let value_1 = spx_1.values.uget([spx_sample_1, m, spx.property_1]);
                                        for d in 0..3 {
                                            sum[d] += value_1 * spx_2_gradient.uget([grad_sample_2, d, m, spx.property_2]);
                                        }
                                    }
                                }
                            }

                            if neighbor_1_type != neighbor_2_type {
                                // see above
                                for d in 0..3 {
                                    sum[d] *= std::f64::consts::SQRT_2;
                                }
                            }

                            for d in 0..3 {
                                unsafe {
                                    *values.uget_mut([d, property_i]) = sum[d] / spx.normalization;
                                }
                            }
                        }
                    });
            }

            // gradients with respect to the cell parameters
            if let Some(mut gradient) = block.gradient_mut("cell") {
                let gradient = gradient.data_mut();

                gradient.values.to_array_mut()
                    .axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .zip_eq(gradient.samples.par_iter())
                    .for_each(|(mut values, gradient_sample)| {
                        for (property_i, spx) in properties_to_combine.iter().enumerate() {
                            let SpxPropertiesToCombine { spx_1, spx_2, ..} = spx;

                            let spx_1_gradient = spx_1.cell_gradients.expect("missing spherical expansion gradients");
                            let spx_2_gradient = spx_2.cell_gradients.expect("missing spherical expansion gradients");

                            let sample_i = gradient_sample[0].usize();
                            let (spx_sample_1, spx_sample_2) = mapping.values[sample_i];

                            let mut sum = [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                            ];
                            for m in 0..(2 * spx.o3_lambda + 1) {
                                // SAFETY: see same loop for values
                                unsafe {
                                    let value_2 = spx_2.values.uget([spx_sample_2, m, spx.property_2]);
                                    for d1 in 0..3 {
                                        for d2 in 0..3 {
                                            sum[d1][d2] += value_2 * spx_1_gradient.uget([spx_sample_1, d1, d2, m, spx.property_1]);
                                        }
                                    }
                                }
                            }

                            for m in 0..(2 * spx.o3_lambda + 1) {
                                // SAFETY: see same loop for values
                                unsafe {
                                    let value_1 = spx_1.values.uget([spx_sample_1, m, spx.property_1]);
                                    for d1 in 0..3 {
                                        for d2 in 0..3 {
                                            sum[d1][d2] += value_1 * spx_2_gradient.uget([spx_sample_2, d1, d2, m, spx.property_2]);
                                        }
                                    }
                                }
                            }

                            if neighbor_1_type != neighbor_2_type {
                                // see above
                                for d1 in 0..3 {
                                    for d2 in 0..3 {
                                        sum[d1][d2] *= std::f64::consts::SQRT_2;
                                    }
                                }
                            }

                            for d1 in 0..3 {
                                for d2 in 0..3 {
                                    unsafe {
                                        *values.uget_mut([d1, d2, property_i]) = sum[d1][d2] / spx.normalization;
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
    use metatensor::LabelValue;

    use crate::systems::test_utils::{test_systems, test_system};
    use crate::Calculator;

    use super::*;
    use crate::calculators::CalculatorBase;

    fn parameters() -> PowerSpectrumParameters {
        PowerSpectrumParameters {
            cutoff: 2.5,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            center_atom_weight: 1.0,
            radial_basis: RadialBasis::splined_gto(1e-8),
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

        let system = test_system("ethanol");
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

        let system = test_system("ethanol");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-5,
            max_relative: 5e-4,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_cell(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["methane"]);

        let properties = Labels::new(["l", "n_1", "n_2"], &[
            [0, 0, 1],
            [3, 3, 3],
            [2, 4, 3],
            [1, 4, 4],
            [5, 1, 0],
            [1, 1, 2],
        ]);

        let samples = Labels::new(["system", "atom"], &[
            [0, 2],
            [0, 1],
        ]);

        let keys = Labels::new(["center_type", "neighbor_1_type", "neighbor_2_type"], &[
            [1, 1, 1],
            [6, 6, 6],
            [1, 8, 6], // not part of the default keys
            [1, 6, 6],
            [1, 1, 6],
            [6, 1, 1],
            [6, 1, 6],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn compute_partial_per_key() {
        let keys = Labels::new(["center_type", "neighbor_1_type", "neighbor_2_type"], &[
            [1, 1, 1],
            [1, 1, 6],
            [1, 6, 6],
            [6, 1, 1],
            [6, 1, 6],
            [6, 6, 6],
        ]);

        let empty_block = metatensor::TensorBlock::new(
            EmptyArray::new(vec![1, 0]),
            &Labels::single(),
            &[],
            &Labels::new::<i32, 3>(["l", "n_1", "n_2"], &[]),
        ).unwrap();

        let blocks = vec![
            // H, H-H
            metatensor::TensorBlock::new(
                EmptyArray::new(vec![1, 1]),
                &Labels::single(),
                &[],
                &Labels::new(["l", "n_1", "n_2"], &[[2, 0, 0]]),
            ).unwrap(),
            // H, C-H
            empty_block.as_ref().try_clone().unwrap(),
            // H, C-C
            empty_block.as_ref().try_clone().unwrap(),
            // C, H-H
            empty_block.as_ref().try_clone().unwrap(),
            // C, C-H
            metatensor::TensorBlock::new(
                EmptyArray::new(vec![1, 1]),
                &Labels::single(),
                &[],
                &Labels::new(["l", "n_1", "n_2"], &[[3, 0, 0]]),
            ).unwrap(),
            // C, C-C
            empty_block,
        ];
        let selection = metatensor::TensorMap::new(keys, blocks).unwrap();

        let options = CalculationOptions {
            selected_properties: LabelsSelection::Predefined(&selection),
            ..Default::default()
        };

        let mut calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["methane"]);
        let descriptor = calculator.compute(&mut systems, options).unwrap();


        assert_eq!(descriptor.keys(), selection.keys());

        assert_eq!(descriptor.block_by_id(0).values().as_array().shape(), [4, 1]);
        assert_eq!(descriptor.block_by_id(1).values().as_array().shape(), [4, 0]);
        assert_eq!(descriptor.block_by_id(2).values().as_array().shape(), [4, 0]);
        assert_eq!(descriptor.block_by_id(3).values().as_array().shape(), [1, 0]);
        assert_eq!(descriptor.block_by_id(4).values().as_array().shape(), [1, 1]);
        assert_eq!(descriptor.block_by_id(5).values().as_array().shape(), [1, 0]);
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
            assert_eq!(block.values().as_array(), 4.0 * block_scaled.values().as_array());
        }
    }
}
