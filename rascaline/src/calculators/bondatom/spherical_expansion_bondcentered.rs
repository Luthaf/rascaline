use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::cell::RefCell;

use ndarray::s;
use rayon::prelude::*;

use metatensor::{LabelsBuilder, Labels, LabelValue};
use metatensor::TensorMap;

use crate::{Error, System};

use crate::labels::{SamplesBuilder, AtomicTypeFilter, BondCenteredSamples};
use crate::labels::{KeysBuilder, TwoCentersSingleNeighborsTypesKeys};

use crate::calculators::{CalculatorBase,GradientsOptions};
use crate::calculators::{split_tensor_map_by_system, array_mut_for_system};
use crate::calculators::soap::{CutoffFunction, RadialScaling};
use crate::calculators::radial_basis::RadialBasis;
use crate::calculators::soap::{
    SoapRadialIntegralParameters,
    SoapRadialIntegralCache,
};

use crate::systems::BATripletNeighborList;
use super::{canonical_vector_for_single_triplet,ExpansionContribution,RawSphericalExpansion,RawSphericalExpansionParameters};

use super::assert_feature_gate;

/// Parameters for spherical expansion calculator for bond-centered neighbor densities.
///
/// (The spherical expansion is at the core of representations in the SOAP
/// (Smooth Overlap of Atomic Positions) family. See [this review
/// article](https://doi.org/10.1063/1.5090481) for more information on the SOAP
/// representation, and [this paper](https://doi.org/10.1063/5.0044689) for
/// information on how it is implemented in rascaline.)
///
/// This calculator is only needed to characterize local environments that are centered
/// on a pair of atoms rather than a single one.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct SphericalExpansionForBondsParameters {
    /// Spherical cutoffs to use for atomic environments
    pub(super) cutoffs: [f64;2],
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density
    pub atomic_gaussian_width: f64,
    /// Weight of the central atom contribution to the
    /// features. If `1` the center atom contribution is weighted the same
    /// as any other contribution. If `0` the central atom does not
    /// contribute to the features at all.
    pub center_atoms_weight: f64,
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
    /// radial scaling can be used to reduce the importance of neighbor atoms
    /// further away from the center, usually improving the performance of the
    /// model
    #[serde(default)]
    pub radial_scaling: RadialScaling,
}

impl SphericalExpansionForBondsParameters {
    /// Validate all the parameters
    pub fn validate(&self) -> Result<(), Error> {
        assert_feature_gate();
        self.cutoff_function.validate()?;
        self.radial_scaling.validate()?;

        // try constructing a radial integral
        SoapRadialIntegralCache::new(self.radial_basis.clone(), SoapRadialIntegralParameters {
            max_radial: self.max_radial,
            max_angular: self.max_angular,
            atomic_gaussian_width: self.atomic_gaussian_width,
            cutoff: self.third_cutoff(),
        })?;

        return Ok(());
    }
    pub fn bond_cutoff(&self) -> f64 {
        self.cutoffs[0]
    }
    pub fn third_cutoff(&self) -> f64 {
        self.cutoffs[1]
    }

    fn decompose(self) -> (RawSphericalExpansionParameters,f64,f64){
        let (bond_cutoff,center_atoms_weight) = (self.bond_cutoff(),self.center_atoms_weight);
        (
        RawSphericalExpansionParameters{
            cutoff: self.third_cutoff(),
            max_radial: self.max_radial,
            max_angular: self.max_angular,
            atomic_gaussian_width: self.atomic_gaussian_width,
            radial_basis: self.radial_basis,
            cutoff_function: self.cutoff_function,
            radial_scaling: self.radial_scaling,
        },
        bond_cutoff,
        center_atoms_weight,
    )}
    fn recompose(expansion_params: RawSphericalExpansionParameters, bond_cutoff: f64, center_atoms_weight: f64) -> Self {
        Self{
            cutoffs: [bond_cutoff, expansion_params.cutoff],
            max_radial: expansion_params.max_radial,
            max_angular: expansion_params.max_angular,
            atomic_gaussian_width: expansion_params.atomic_gaussian_width,
            radial_basis: expansion_params.radial_basis,
            cutoff_function: expansion_params.cutoff_function,
            radial_scaling: expansion_params.radial_scaling,
            center_atoms_weight,
        }
    }

}


/// The actual calculator used to compute SOAP-like spherical expansion coefficients for bond-centered environments
/// In other words, the spherical expansion of the neighbor density function centered on the center of a bond,
/// 'after' rotating the system so that the bond is aligned with the z axis.
///
/// This radial+angular decomposition yields coefficients with labels `n` (radial), and `l` and `m` (angular)
/// as a Calculator, it yields tonsorblocks of with individual values of `l`
/// and individual atomic types for center_1, center_2, and neighbor.
/// Each block has components for each possible value of `m`, and properties for each value of `n`.
/// a given sample corresponds to a single center bond (a pair of center atoms) within a given structure.
pub struct SphericalExpansionForBonds {
    /// The object in charge of computing the vectors and distances
    /// between the bond and the lone atom of the BA triplet (after rotating the system to put the bond in it canonical orientation)
    distance_calculator: BATripletNeighborList,
    /// actual spherical expansion object
    raw_expansion: RawSphericalExpansion,
    /// a weight multiplier for expansion coefficients from self-contributions
    center_atoms_weight: f64,
}

impl SphericalExpansionForBonds {
    /// Create a new `SphericalExpansion` calculator with the given parameters
    pub fn new(parameters: SphericalExpansionForBondsParameters) -> Result<Self, Error> {
        parameters.validate()?;
        let cutoffs = parameters.cutoffs.clone();
        let (exp_params, _bond_cut, center_weight) = parameters.decompose();

        return Ok(Self {
            center_atoms_weight: center_weight,
            raw_expansion: RawSphericalExpansion::new(exp_params),
            distance_calculator: BATripletNeighborList{
                cutoffs,
            },
        });
    }


    /// a smart-ish way to obtain the coefficients of all bond expansions:
    /// this function's API is designed to be resource-efficient for both SphericalExpansionForBondType and
    /// SphericalExpansionForBonds, while being computationally efficient for the underlying BANeighborList calculator.
    pub(super) fn get_coefficients_for<'a>(
        &'a self, system: &'a System,
        s1: i32, s2: i32, s3_list: &'a Vec<i32>,
        do_gradients: GradientsOptions,
    ) -> Result<impl Iterator<Item = (usize, bool, std::rc::Rc<RefCell<ExpansionContribution>>)> + 'a, Error> {

        let types = system.types().unwrap();


        let pre_iter = s3_list.iter().flat_map(|s3|{
            self.distance_calculator.get_per_system_per_type_enumerated(system,s1,s2,*s3).unwrap().into_iter()
        }).flat_map(|(triplet_i,triplet)| {
            let invert: &'static [bool] = {
                if s1==s2 {&[false,true]}
                else if types[triplet.atom_i] == s1 {&[false]}
                else {&[true]}
            };
            invert.iter().map(move |invert|(triplet_i,triplet,*invert))
        }).collect::<Vec<_>>();

        let contribution = std::rc::Rc::new(RefCell::new(
            self.raw_expansion.make_contribution_buffer(do_gradients.any())
        ));

        let mut mtx_cache = BTreeMap::new();
        let mut dmtx_cache = BTreeMap::new();

        return Ok(pre_iter.into_iter().map(move |(triplet_i,triplet,invert)| {
            let mut vector = canonical_vector_for_single_triplet(&triplet, invert, false, &mut mtx_cache, &mut dmtx_cache).unwrap();
            let weight = if triplet.is_self_contrib {self.center_atoms_weight} else {1.0};
            self.raw_expansion.compute_coefficients(&mut *contribution.borrow_mut(), vector.vect,weight,None);
            (triplet_i, invert, contribution.clone())
        }));

    }

}

impl CalculatorBase for SphericalExpansionForBonds {
    fn name(&self) -> String {
        "spherical expansion".into()
    }

    fn cutoffs(&self) -> &[f64] {
        &self.distance_calculator.cutoffs
    }

    fn parameters(&self) -> String {
        let params = SphericalExpansionForBondsParameters::recompose(
            (*self.raw_expansion.parameters()).clone(),
            self.distance_calculator.bond_cutoff(),
            self.center_atoms_weight,
        );
        serde_json::to_string(&params).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        let builder = TwoCentersSingleNeighborsTypesKeys {
            cutoffs: self.distance_calculator.cutoffs,
            self_contributions: true,
            raw_triplets: &self.distance_calculator,
        };
        let keys = builder.keys(systems)?;

        let mut builder = LabelsBuilder::new(vec!["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);
        for &[center_1_type, center_2_type, neighbor_type] in keys.iter_fixed_size() {
            for o3_lambda in 0..=self.raw_expansion.parameters().max_angular {
                builder.add(&[o3_lambda.into(), center_1_type, center_2_type, neighbor_type]);
            }
        }

        return Ok(builder.finish());
    }

    fn sample_names(&self) -> Vec<&str> {
        BondCenteredSamples::sample_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);

        // only compute the samples once for each `atom_type, neighbor_type`,
        // and re-use the results across `o3_lambda`.
        let mut samples_per_type = BTreeMap::new();
        for [_, center_1_type, center_2_type, neighbor_type] in keys.iter_fixed_size() {
            if samples_per_type.contains_key(&(center_1_type, center_2_type, neighbor_type)) {
                continue;
            }

            let builder = BondCenteredSamples {
                cutoffs: self.distance_calculator.cutoffs,
                center_1_type: AtomicTypeFilter::Single(center_1_type.i32()),
                center_2_type: AtomicTypeFilter::Single(center_2_type.i32()),
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_contributions: true,
                raw_triplets: &self.distance_calculator,
            };

            samples_per_type.insert((center_1_type, center_2_type, neighbor_type), builder.samples(systems)?);
        }

        let mut result = Vec::new();
        for [_, center_1_type, center_2_type, neighbor_type] in keys.iter_fixed_size() {
            let samples = samples_per_type.get(
                &(center_1_type, center_2_type, neighbor_type)
            ).expect("missing samples");

            result.push(samples.clone());
        }

        return Ok(result);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        false // for now, discontinuities are a pain
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([_, center_1_type, center_2_type, neighbor_type], samples) in keys.iter_fixed_size().zip(samples) {
            // TODO: we don't need to rebuild the gradient samples for different
            // o3_lambda
            let builder = BondCenteredSamples {
                cutoffs: self.distance_calculator.cutoffs,
                center_1_type: AtomicTypeFilter::Single(center_1_type.i32()),
                center_2_type: AtomicTypeFilter::Single(center_2_type.i32()),
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_contributions: true,
                raw_triplets: &self.distance_calculator,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        assert_eq!(keys.names(), ["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);

        // only compute the components once for each `o3_lambda`,
        // and re-use the results across `atom_type, neighbor_type`.
        let mut component_by_l = BTreeMap::new();
        for [o3_lambda, _, _, _] in keys.iter_fixed_size() {
            if component_by_l.contains_key(o3_lambda) {
                continue;
            }

            let mut component = LabelsBuilder::new(vec!["spherical_harmonics_m"]);
            for m in -o3_lambda.i32()..=o3_lambda.i32() {
                component.add(&[LabelValue::new(m)]);
            }

            let components = vec![component.finish()];
            component_by_l.insert(*o3_lambda, components);
        }

        let mut result = Vec::new();
        for [o3_lambda, _, _, _] in keys.iter_fixed_size() {
            let components = component_by_l.get(o3_lambda).expect("missing samples");
            result.push(components.clone());
        }
        return result;
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        for n in 0..self.raw_expansion.parameters().max_radial {
            properties.add(&[n]);
        }
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_feature_gate();
        assert_eq!(descriptor.keys().names(), ["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);
        if descriptor.blocks().len() == 0 {
            return Ok(());
        }

        let max_angular = self.raw_expansion.parameters().max_angular;
        let l_slices: Vec<_> = (0..=max_angular).map(|l|{
            let lsize = l*l;
            let msize = 2*l+1;
            lsize..lsize+msize
        }).collect();

        let do_gradients = GradientsOptions {
            positions: descriptor.block_by_id(0).gradient("positions").is_some(),
            cell: descriptor.block_by_id(0).gradient("cell").is_some(),
            strain: descriptor.block_by_id(0).gradient("strain").is_some(),
        };
        if do_gradients.positions {
            assert!(self.supports_gradient("positions"));
        }
        if do_gradients.cell {
            assert!(self.supports_gradient("cell"));
        }

        let radial_selection = descriptor.blocks().iter().map(|b|{
            let prop = b.properties();
            assert_eq!(prop.names(), ["n"]);
            prop.iter_fixed_size().map(|&[n]|n.i32()).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        // first, create some partial-key -> block lookup tables to avoid linear searches within blocks later

        // {(s1,s2,s3) -> i_s3}
        let mut s1s2s3_to_is3: BTreeMap<(i32,i32,i32),usize> = BTreeMap::new();
        // {(s1,s2) -> [i_s3->(s3,[l->i_block])]}
        let mut s1s2_to_block_ids: BTreeMap<(i32,i32),Vec<(i32,Vec<usize>)>> = BTreeMap::new();

        for (block_i, &[l, s1,s2,s3]) in descriptor.keys().iter_fixed_size().enumerate(){
            let s1=s1.i32();
            let s2=s2.i32();
            let s3=s3.i32();
            let l=l.usize();
            let s1s2_blocks = s1s2_to_block_ids.entry((s1,s2))
                .or_insert_with(Vec::new);
            let l_blocks = match s1s2s3_to_is3.entry((s1,s2,s3)) {
                Entry::Occupied(i_s3_e) => {
                    let (s3_b, l_blocks) = & mut s1s2_blocks[*i_s3_e.get()];
                    debug_assert_eq!(s3_b,&s3);
                    l_blocks
                },
                Entry::Vacant(i_s3_e) => {
                    let i_s3 = s1s2_blocks.len();
                    i_s3_e.insert(i_s3);
                    s1s2_blocks.push((s3,vec![usize::MAX;max_angular+1]));
                    &mut s1s2_blocks[i_s3].1
                },
            };
            l_blocks[l] = block_i;
        }

        #[cfg(debug_assertions)]{
            for block in descriptor.blocks() {
                assert_eq!(block.samples().names(), ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"]);
            }
        }
        let mut descriptors_by_system = split_tensor_map_by_system(descriptor, systems.len());

        systems.par_iter_mut()
            .zip_eq(&mut descriptors_by_system)
            .try_for_each(|(system, descriptor)|
        {
            //system.compute_triplet_neighbors(self.parameters.bond_cutoff(), self.parameters.third_cutoff())?;
            self.distance_calculator.ensure_computed_for_system(system)?;
            let triplets = self.distance_calculator.get_for_system(system)?;
            let types = system.types()?;

            for ((s1,s2),s1s2_blocks) in s1s2_to_block_ids.iter() {
                let (s3_list,per_s3_blocks): (Vec<i32>,Vec<&Vec<_>>) = s1s2_blocks.iter().map(
                    |(s3,blocks)|(*s3,blocks)
                ).unzip();
                // half-assume that blocks that share s1,s2,s3 have the same sample list
                #[cfg(debug_assertions)]{
                    for (_s3,s3blocks) in s1s2_blocks.iter(){
                        debug_assert!(s3blocks.len()>0);
                        let mut s3goodblocks = s3blocks.iter().filter(|b_i|(**b_i)!=usize::MAX);
                        let first_goodblock = s3goodblocks.next();
                        debug_assert!(first_goodblock.is_some());

                        let samples_n = descriptor.block_by_id(*first_goodblock.unwrap()).samples().size();
                        for lblock in s3goodblocks {
                            debug_assert_eq!(descriptor.block_by_id(*lblock).samples().size(), samples_n);
                        }
                    }
                }
                // {bond_i->(i_s3,sample_i)}
                let mut s3_samples = vec![];
                let mut sample_lut: BTreeMap<(usize,usize,[i32;3]),Vec<(usize,usize)>> = BTreeMap::new();

                // also assume that the systems are in order in the samples
                for (i_s3, s3blocks) in per_s3_blocks.into_iter().enumerate() {
                    let first_good_block = s3blocks.iter().filter(|b_i|**b_i!=usize::MAX).next().unwrap();
                    let samples = descriptor.block_by_id(*first_good_block).samples();
                    for (sample_i, &[_system_i,atom_i,atom_j,cell_shift_a, cell_shift_b, cell_shift_c]) in samples.iter_fixed_size().enumerate(){
                        match sample_lut.entry(
                            (atom_i.usize(),atom_j.usize(),[cell_shift_a.i32(),cell_shift_b.i32(),cell_shift_c.i32()])
                        ) {
                            Entry::Vacant(e) => {
                                e.insert(vec![(i_s3,sample_i)]);
                            },
                            Entry::Occupied(mut e) => {
                                e.get_mut().push((i_s3,sample_i));
                            },
                        }
                    }
                    s3_samples.push(samples);
                }
                for (triplet_i,inverted,contribution) in self.get_coefficients_for(system, *s1, *s2, &s3_list, do_gradients)? {
                    let triplet = &triplets[triplet_i];

                    let contribution = contribution.borrow();
                    let these_samples = match sample_lut.get(
                        &(triplet.atom_i,triplet.atom_j,triplet.bond_cell_shift)
                    ){
                        None => {continue;},
                        Some(a) => a,
                    };

                    for (i_s3,sample_i) in these_samples.iter(){
                        if s3_list[*i_s3] != types[triplet.atom_k] {
                            continue  // this triplet does not contribute to this block
                        }
                        let sample = &s3_samples[*i_s3][*sample_i];
                        let (atom_i,atom_j, ce_sh) = (sample[1].usize(),sample[2].usize(),[sample[3].i32(),sample[4].i32(),sample[5].i32()]);
                        if (!inverted) && (
                            triplet.atom_i != atom_i || triplet.atom_j != atom_j
                            || triplet.bond_cell_shift != ce_sh
                        ){
                            continue;
                        } else if inverted && (
                            triplet.atom_i != atom_j || triplet.atom_j != atom_i
                            || triplet.bond_cell_shift != ce_sh.map(|x|-x)
                        ){
                            continue;
                        }

                        let ret_blocks = &s1s2_blocks[*i_s3].1;
                        for (l,lslice) in l_slices.iter().enumerate() {
                            let block_i = ret_blocks[l];
                            if block_i == usize::MAX {
                                continue;
                            }
                            let mut block = descriptor.block_mut_by_id(block_i);
                            let mut array = array_mut_for_system(block.values_mut());
                            let mut value_slice = array.slice_mut(s![*sample_i,..,..]);
                            let input_slice = contribution.values.slice(s![lslice.clone(),..]);
                            for (n_i,n) in radial_selection[block_i].iter().enumerate() {
                                let mut value_slice = value_slice.slice_mut(s![..,n_i]);
                                value_slice += &input_slice.slice(s![..,*n]);
                            }

                        }
                    }
                }
            }
            Ok::<_, Error>(())
        })?;

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use metatensor::{Labels, TensorBlock, EmptyArray, LabelsBuilder, TensorMap};

    use crate::calculators::bondatom::set_feature_gate;
    use crate::systems::test_utils::test_systems;
    use crate::{Calculator, CalculationOptions, LabelsSelection};
    use crate::calculators::CalculatorBase;

    use super::{SphericalExpansionForBonds, SphericalExpansionForBondsParameters};
    use crate::calculators::soap::{CutoffFunction, RadialScaling};
    use crate::calculators::radial_basis::RadialBasis;


    fn parameters() -> SphericalExpansionForBondsParameters {
        set_feature_gate();
        SphericalExpansionForBondsParameters {
            cutoffs: [3.5,3.5],
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            center_atoms_weight: 10.0,
            radial_basis: RadialBasis::splined_gto(1e-8),
            radial_scaling: RadialScaling::Willatt2018 { scale: 1.5, rate: 0.8, exponent: 2.0},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SphericalExpansionForBonds::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        for l in 0..6 {
            for center_1_type in [1, -42] {
                for center_2_type in [1, -42] {
                    if center_1_type==-42 && center_2_type==-42 {
                        continue;
                    }
                    for neighbor_type in [1, -42] {
                        let block_i = descriptor.keys().position(&[
                            l.into(), center_1_type.into(), center_2_type.into(), neighbor_type.into()
                        ]);
                        assert!(block_i.is_some());
                        let block = &descriptor.block_by_id(block_i.unwrap());
                        let array = block.values().to_array();
                        assert_eq!(array.shape().len(), 3);
                        assert_eq!(array.shape()[1], 2 * l + 1);
                    }
                }
            }
        }

        // exact values for spherical expansion are regression-tested in
        // `rascaline/tests/spherical-expansion.rs`
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SphericalExpansionForBonds::new(
            SphericalExpansionForBondsParameters {
                max_angular: 2,
                ..parameters()
            }
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let properties = Labels::new(["n"], &[
            [0],
            [3],
            [2],
        ]);

        let samples = Labels::new(["system", "first_atom", "second_atom", "cell_shift_a","cell_shift_b","cell_shift_c"], &[
            [0, 0, 2, 0,0,0],
            [0, 0, 1, 0,0,0],
            //[0, 1, 2, 0,0,0],  // excluding this one
        ]);

        let keys = Labels::new(["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"], &[
            // every key that will be generated (in scrambled order) plus one
            [0, -42, 1, -42],
            [0, -42, -42, -42],
            [2, -42, 1, -42],
            [0, 1, -42, -42],
            [0, 1, 1, -42],
            [0, 6, 1, 1], // not part of the default keys
            [1, -42, 1, -42],
            [1, -42, 1, 1],
            [2, -42, -42, -42],
            [1, 1, -42, 1],
            [0, -42, 1, 1],
            [1, 1, 1, -42],
            [2, -42, 1, 1],
            [0, 1, 1, 1],
            [2, 1, -42, -42],
            [1, 1, 1, 1],
            [0, -42, -42, 1],
            [1, -42, -42, 1],
            [2, -42, -42, 1],
            [2, 1, 1, -42],
            [0, 1, -42, 1],
            [2, 1, -42, 1],
            [2, 1, 1, 1],
            [1, -42, -42, -42],
            [1, 1, -42, -42],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn non_existing_samples() {
        let mut calculator = Calculator::from(Box::new(SphericalExpansionForBonds::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let angular_stride = parameters().max_angular +1;
        let mut systems = test_systems(&["water"]);

        // include the three atoms in all blocks, regardless of the
        // atom_type key.
        let block = TensorBlock::new(
            EmptyArray::new(vec![3, 1]),
            &Labels::new(["system", "first_atom", "second_atom", "cell_shift_a","cell_shift_b","cell_shift_c"], &[
                [0, 0, 2, 0,0,0],
                [0, 1, 2, 0,0,0],
                [0, 0, 1, 0,0,0],
            ]),
            &[],
            &Labels::single(),
        ).unwrap();

        let mut keys = LabelsBuilder::new(vec!["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);
        let mut blocks = Vec::new();
        for l in 0..(parameters().max_angular + 1) as isize {
            for center_1_type in [1, -42] {
                for center_2_type in [1, -42] {
                    for neighbor_type in [1, -42] {
                        keys.add(&[l, center_1_type, center_2_type, neighbor_type]);
                        blocks.push(block.as_ref().try_clone().unwrap());
                    }
                }
            }
        }
        let select_all_samples = TensorMap::new(keys.finish(), blocks).unwrap();

        let options = CalculationOptions {
            selected_samples: LabelsSelection::Predefined(&select_all_samples),
            ..Default::default()
        };
        let descriptor = calculator.compute(&mut systems, options).unwrap();

        // get the block for oxygen
        // println!("{:?}", descriptor.keys());
        assert_eq!(descriptor.keys().names(), ["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);
        assert_eq!(descriptor.keys()[2*angular_stride], [0, -42, 1, -42]);  // start with [n, -42, -42, -42], then [n, -42, -42, 1]

        let block = descriptor.block_by_id(2*angular_stride);
        let block = block.data();

        // entries centered on H atoms should be zero
        assert_eq!(
            *block.samples,
            Labels::new(["system", "first_atom", "second_atom", "cell_shift_a","cell_shift_b","cell_shift_c"], &[
                [0, 0, 2, 0,0,0],
                [0, 1, 2, 0,0,0],  // the sample that doesn't exist
                [0, 0, 1, 0,0,0],
            ])
        );
        let array = block.values.as_array();
        assert_eq!(array.index_axis(ndarray::Axis(0), 1), ArrayD::from_elem(vec![1, 6], 0.0));

        // get the block for hydrogen
        assert_eq!(descriptor.keys().names(), ["o3_lambda", "center_1_type", "center_2_type", "neighbor_type"]);
        assert_eq!(descriptor.keys()[5*angular_stride], [0, 1, -42, 1]);

        let block = descriptor.block_by_id(5*angular_stride);
        let block = block.data();

        // entries centered on O atoms should be zero
        assert_eq!(
            *block.samples,
            Labels::new(["system", "first_atom", "second_atom", "cell_shift_a","cell_shift_b","cell_shift_c"], &[
                [0, 0, 2, 0,0,0],
                [0, 1, 2, 0,0,0],
                [0, 0, 1, 0,0,0],
            ])
        );
        let array = block.values.as_array();
        assert_eq!(array.index_axis(ndarray::Axis(0), 0), ArrayD::from_elem(vec![1, 6], 0.0));
    }
}
