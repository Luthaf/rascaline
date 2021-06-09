use std::collections::BTreeSet;
use indexmap::set::IndexSet;

use itertools::Itertools;
use ndarray::{Array2, s};

use super::{Indexes, IndexesBuilder, IndexValue};

pub struct Descriptor {
    /// An array of samples.count() by features.count() values
    pub values: Array2<f64>,
    pub samples: Indexes,
    pub features: Indexes,
    /// Gradients of the descriptor with respect to one atomic position
    pub gradients: Option<Array2<f64>>,
    pub gradients_samples: Option<Indexes>,
}

impl Default for Descriptor {
    fn default() -> Self { Self::new() }
}

impl Descriptor {
    pub fn new() -> Descriptor {
        let indexes = IndexesBuilder::new(vec![]).finish();
        return Descriptor {
            values: Array2::zeros((0, 0)),
            samples: indexes.clone(),
            features: indexes,
            gradients: None,
            gradients_samples: None,
        }
    }

    pub fn prepare(
        &mut self,
        samples: Indexes,
        features: Indexes,
    ) {
        self.samples = samples;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.samples.count(), self.features.count());
        resize_and_reset(&mut self.values, shape);

        self.gradients = None;
        self.gradients_samples = None;
    }

    pub fn prepare_gradients(
        &mut self,
        samples: Indexes,
        gradients: Indexes,
        features: Indexes,
    ) {
        // basic sanity check
        assert_eq!(gradients.names().last(), Some(&"spatial"), "the last index of gradient should be spatial");

        self.samples = samples;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.samples.count(), self.features.count());
        resize_and_reset(&mut self.values, shape);

        let gradient_shape = (gradients.count(), self.features.count());
        self.gradients_samples = Some(gradients);

        if let Some(array) = &mut self.gradients {
            // resize the 'gradient' array if needed, and set the requested initial value
            resize_and_reset(array, gradient_shape);
        } else {
            // create a new gradient array
            let array = Array2::from_elem(gradient_shape, 0.0);
            self.gradients = Some(array);
        }
    }

    #[time_graph::instrument]
    pub fn densify(&mut self, variables: Vec<&str>) {
        if variables.is_empty() {
            return;
        }

        // TODO: return Result and convert this to Error
        assert!(self.features.size() > 0);

        let new_samples = remove_from_samples(&self.samples, &variables);
        let new_gradients_samples = self.gradients_samples.as_ref().map(|indexes| {
            let new_gradients_samples = remove_from_samples(indexes, &variables);

            if new_gradients_samples.new_features != new_samples.new_features {
                let name = if variables.len() == 1 {
                    variables[0].to_owned()
                } else {
                    format!("({})", variables.join(", "))
                };
                panic!("gradient samples contains different values for {} than the samples themselves", name);
            }

            return new_gradients_samples;
        });

        // new feature indexes, add `variables` in the front. This transforms
        // something like [n, l, m] to [species_neighbor, n, l, m]; and fill it
        // with the corresponding values from `new_samples.new_features`,
        // duplicating the `[n, l, m]` block as needed
        let mut feature_names = variables;
        feature_names.extend(self.features.names());
        let mut new_features = IndexesBuilder::new(feature_names);
        for new in new_samples.new_features {
            for feature in self.features.iter() {
                let mut new = new.clone();
                new.extend(feature);
                new_features.add(&new);
            }
        }
        let new_features = new_features.finish();

        let first_feature_tail = self.features.iter().next().expect("missing first feature").to_vec();
        let old_feature_size = self.features.count();

        // copy values themselves as needed
        let mut new_values = Array2::zeros((new_samples.samples.count(), new_features.count()));
        for changed in new_samples.mapping {
            let DensifiedIndex { old_sample_i, new_sample_i, variables } = changed;

            // find in which feature block we need to copy the data
            let mut first_feature = variables;
            first_feature.extend_from_slice(&first_feature_tail);
            let start = new_features.position(&first_feature).expect("missing start of the new feature block");
            let stop = start + old_feature_size;

            let value = self.values.slice(s![old_sample_i, ..]);
            new_values.slice_mut(s![new_sample_i, start..stop]).assign(&value);
        }

        if let Some(gradients) = &self.gradients {
            let new_gradients_samples = new_gradients_samples.expect("missing densified gradients");

            let mut new_gradients = Array2::zeros(
                (new_gradients_samples.samples.count(), new_features.count())
            );

            for changed in new_gradients_samples.mapping {
                let DensifiedIndex { old_sample_i, new_sample_i, variables } = changed;

                // find in which feature block we need to copy the data
                let mut first_feature = variables;
                first_feature.extend_from_slice(&first_feature_tail);
                let start = new_features.position(&first_feature).expect("missing start of the new feature block");
                let stop = start + old_feature_size;

                let value = gradients.slice(s![old_sample_i, ..]);
                new_gradients.slice_mut(s![new_sample_i, start..stop]).assign(&value);
            }

            self.gradients = Some(new_gradients);
            self.gradients_samples = Some(new_gradients_samples.samples);
        }

        self.features = new_features;
        self.samples = new_samples.samples;
        self.values = new_values;
    }
}

fn resize_and_reset(array: &mut Array2<f64>, shape: (usize, usize)) {
    // extract data by replacing array with a temporary value
    let mut tmp = Array2::zeros((0, 0));
    std::mem::swap(array, &mut tmp);

    let mut data = tmp.into_raw_vec();
    data.resize(shape.0 * shape.1, 0.0);

    let mut values = Array2::from_shape_vec(shape, data).expect("wrong array shape");
    values.fill(0.0);
    let _replaced = std::mem::replace(array, values);
}

/// A `DensifiedIndex` contains all the information to reconstruct the new
/// position of the values/gradients associated with a single sample in the
/// initial descriptor
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
struct DensifiedIndex {
    /// Index of the old sample (respectively gradient sample) in the value
    /// (respectively gradients) array
    old_sample_i: usize,
    /// Index of the new sample (respectively gradient sample) in the value
    /// (respectively gradients) array
    new_sample_i: usize,
    /// Values of the variables in the old descriptor. These are part of the
    /// samples in the old descriptor; but part of the features in the new one.
    variables: Vec<IndexValue>,
}

/// Results of removing a set of variables from samples
struct RemovedSampleResult {
    /// New samples, without the variables
    samples: Indexes,
    /// Values taken by the variables in the original samples
    new_features: BTreeSet<Vec<IndexValue>>,
    /// Information about all data that needs to be moved
    mapping: Vec<DensifiedIndex>,
}

/// Remove the given `variables` from the `samples`, returning the updated
/// `samples` and a set of all the values taken by the removed variables.
fn remove_from_samples(samples: &Indexes, variables: &[&str]) -> RemovedSampleResult {
    let variables_positions = variables.iter()
        .map(|v| {
            samples.names()
                .iter()
                .position(|name| name == v)
                // TODO: this function should return a Result and this should be
                // an InvalidParameterError.
                .unwrap_or_else(|| panic!(
                    "can not densify along '{}' which is not present in the samples: [{}]",
                    v, samples.names().join(", ")
                ))
        }).collect::<Vec<_>>();

    let mut mapping = Vec::new();

    // collect all different indexes in maps. Assuming we are densifying
    // along the first index, we want to convert [[2, 3, 0], [1, 3, 0]]
    // to [[3, 0]].
    let mut new_samples = IndexSet::new();
    let mut new_features = BTreeSet::new();

    for (old_sample_i, sample) in samples.iter().enumerate() {
        let mut new_feature = Vec::new();
        for &i in &variables_positions {
            new_feature.push(sample[i]);
        }
        new_features.insert(new_feature.clone());

        let mut new_sample = sample.to_vec();
        // sort and reverse the indexes to ensure the all the calls to `remove`
        // are valid
        for &i in variables_positions.iter().sorted().rev() {
            new_sample.remove(i);
        }
        let (new_sample_i, _) = new_samples.insert_full(new_sample);

        let densified = DensifiedIndex {
            old_sample_i: old_sample_i,
            new_sample_i: new_sample_i,
            variables: new_feature,
        };
        mapping.push(densified);
    }

    let names = samples.names()
        .iter()
        .filter(|&name| !variables.contains(name))
        .cloned()
        .collect();
    let mut builder = IndexesBuilder::new(names);
    for sample in new_samples {
        builder.add(&sample);
    }

    return RemovedSampleResult {
        samples: builder.finish(),
        new_features: new_features,
        mapping: mapping,
    };
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_systems;
    use crate::descriptor::{TwoBodiesSpeciesSamples, StructureSpeciesSamples, SamplesIndexes};
    use ndarray::array;

    fn dummy_features() -> Indexes {
        let mut features = IndexesBuilder::new(vec!["foo", "bar"]);
        features.add(&[IndexValue::from(0), IndexValue::from(-1)]);
        features.add(&[IndexValue::from(4), IndexValue::from(-2)]);
        features.add(&[IndexValue::from(1), IndexValue::from(-5)]);
        return features.finish();
    }

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::indexes::IndexValue::from($value)
        };
    }

    #[test]
    fn prepare() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]).boxed();
        let features = dummy_features();
        let samples = StructureSpeciesSamples.indexes(&mut systems).unwrap();
        descriptor.prepare(samples, features);


        assert_eq!(descriptor.values.shape(), [4, 3]);

        assert_eq!(descriptor.samples.names(), ["structure", "species"]);
        assert_eq!(descriptor.samples[0], [v!(0), v!(1)]);
        assert_eq!(descriptor.samples[1], [v!(0), v!(123456)]);
        assert_eq!(descriptor.samples[2], [v!(1), v!(1)]);
        assert_eq!(descriptor.samples[3], [v!(1), v!(6)]);

        assert!(descriptor.gradients.is_none());
    }

    #[test]
    fn prepare_gradients() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]).boxed();
        let features = dummy_features();
        let (samples, gradients) = StructureSpeciesSamples.with_gradients(&mut systems).unwrap();
        descriptor.prepare_gradients(samples, gradients.unwrap(), features);

        let gradients = descriptor.gradients.unwrap();
        assert_eq!(gradients.shape(), [15, 3]);

        let gradients_samples = descriptor.gradients_samples.as_ref().unwrap();
        assert_eq!(gradients_samples.names(), ["structure", "species", "atom", "spatial"]);

        let expected = [
            [v!(0), v!(1), v!(1)],
            [v!(0), v!(1), v!(2)],
            [v!(0), v!(123456), v!(0)],
            [v!(1), v!(1), v!(0)],
            [v!(1), v!(6), v!(1)]
        ];
        // use a loop to simplify checking the spatial dimension
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(gradients_samples[3 * i][..3], value);
            assert_eq!(gradients_samples[3 * i][3], v!(0));

            assert_eq!(gradients_samples[3 * i + 1][..3], value);
            assert_eq!(gradients_samples[3 * i + 1][3], v!(1));

            assert_eq!(gradients_samples[3 * i + 2][..3], value);
            assert_eq!(gradients_samples[3 * i + 2][3], v!(2));
        }
    }

    #[test]
    fn densify_single_variable() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]).boxed();
        let features = dummy_features();
        let (samples, gradients) = StructureSpeciesSamples.with_gradients(&mut systems).unwrap();
        descriptor.prepare_gradients(samples, gradients.unwrap(), features);

        descriptor.values.assign(&array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]);

        let gradients = descriptor.gradients.as_mut().unwrap();
        gradients.assign(&array![
            [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [-1.0, -2.0, -3.0],
            [4.0, 5.0, 6.0], [0.4, 0.5, 0.6], [-4.0, -5.0, -6.0],
            [7.0, 8.0, 9.0], [0.7, 0.8, 0.9], [-7.0, -8.0, -9.0],
            [10.0, 11.0, 12.0], [0.10, 0.11, 0.12], [-10.0, -11.0, -12.0],
            [13.0, 14.0, 15.0], [0.13, 0.14, 0.15], [-13.0, -14.0, -15.0],
        ]);

        // where the magic happens
        descriptor.densify(vec!["species"]);

        assert_eq!(descriptor.values.shape(), [2, 9]);
        assert_eq!(descriptor.samples.names(), ["structure"]);
        assert_eq!(descriptor.samples[0], [v!(0)]);
        assert_eq!(descriptor.samples[1], [v!(1)]);

        assert_eq!(descriptor.values, array![
            [1.0, 2.0, 3.0, /**/ 0.0, 0.0, 0.0, /**/ 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, /**/ 10.0, 11.0, 12.0, /**/ 0.0, 0.0, 0.0],
        ]);

        let gradients = descriptor.gradients.as_ref().unwrap();
        assert_eq!(gradients.shape(), [15, 9]);
        let gradients_samples = descriptor.gradients_samples.as_ref().unwrap();
        assert_eq!(gradients_samples.names(), ["structure", "atom", "spatial"]);

        let expected = [
            [v!(0), v!(1)],
            [v!(0), v!(2)],
            [v!(0), v!(0)],
            [v!(1), v!(0)],
            [v!(1), v!(1)]
        ];
        // use a loop to simplify checking the spatial dimension
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(gradients_samples[3 * i][..2], value);
            assert_eq!(gradients_samples[3 * i][2], v!(0));

            assert_eq!(gradients_samples[3 * i + 1][..2], value);
            assert_eq!(gradients_samples[3 * i + 1][2], v!(1));

            assert_eq!(gradients_samples[3 * i + 2][..2], value);
            assert_eq!(gradients_samples[3 * i + 2][2], v!(2));
        }

        assert_eq!(*gradients, array![
            [/*H*/ 1.0, 2.0, 3.0,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.1, 0.2, 0.3,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ -1.0, -2.0, -3.0,    /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 4.0, 5.0, 6.0,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.4, 0.5, 0.6,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ -4.0, -5.0, -6.0,    /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ 7.0, 8.0, 9.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.7, 0.8, 0.9],
            [/*H*/ 0.0, 0.0, 0.0,       /*C*/ 0.0, 0.0, 0.0,        /*O*/ -7.0, -8.0, -9.0],
            [/*H*/ 10.0, 11.0, 12.0,    /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.1, 0.11, 0.12,     /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ -10.0, -11.0, -12.0, /*C*/ 0.0, 0.0, 0.0,        /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*C*/ 13.0, 14.0, 15.0,     /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*C*/ 0.13, 0.14, 0.15,     /*O*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*C*/ -13.0, -14.0, -15.0,  /*O*/ 0.0, 0.0, 0.0],
        ]);
    }
    #[test]
    fn densify_multiple_variables() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water"]).boxed();
        let features = dummy_features();
        let (samples, gradients) = TwoBodiesSpeciesSamples::new(3.0).with_gradients(&mut systems).unwrap();
        descriptor.prepare_gradients(samples, gradients.unwrap(), features);

        descriptor.values.assign(&array![
            // H channel around O
            [1.0, 2.0, 3.0],
            // H channel around H1
            [4.0, 5.0, 6.0],
            // O channel around H1
            [7.0, 8.0, 9.0],
            // H channel around H2
            [10.0, 11.0, 12.0],
            // O channel around H2
            [13.0, 14.0, 15.0],
        ]);

        let gradients = descriptor.gradients.as_mut().unwrap();
        gradients.assign(&array![
            // H channel around O, derivatives w.r.t. O
            [1.0, 0.1, -1.0], [2.0, 0.2, -2.0], [3.0, 0.3, -3.0],
            // H channel around O, derivatives w.r.t. H1
            [4.0, 0.4, -4.0], [5.0, 0.5, -5.0], [6.0, 0.6, -6.0],
            // H channel around O, derivatives w.r.t. H2
            [7.0, 0.7, -7.0], [8.0, 0.8, -8.0], [9.0, 0.9, -9.0],
            // H channel around H1, derivatives w.r.t. H1
            [10.0, 0.10, -10.0], [11.0, 0.11, -11.0], [12.0, 0.12, -12.0],
            // H channel around H1, derivatives w.r.t. H2
            [13.0, 0.13, -13.0], [14.0, 0.14, -14.0], [15.0, 0.15, -15.0],
            // O channel around H1, derivatives w.r.t. H1
            [16.0, 0.16, -16.0], [17.0, 0.17, -17.0], [18.0, 0.18, -18.0],
            // O channel around H1, derivatives w.r.t. O
            [19.0, 0.19, -19.0], [20.0, 0.20, -20.0], [21.0, 0.21, -21.0],
            // H channel around H2, derivatives w.r.t. H2
            [22.0, 0.22, -22.0], [23.0, 0.23, -23.0], [24.0, 0.24, -24.0],
            // H channel around H2, derivatives w.r.t. H1
            [25.0, 0.25, -25.0], [26.0, 0.26, -26.0], [27.0, 0.27, -27.0],
            // O channel around H2, derivatives w.r.t. H2
            [28.0, 0.28, -28.0], [29.0, 0.29, -29.0], [30.0, 0.30, -30.0],
            // O channel around H2, derivatives w.r.t. O
            [31.0, 0.31, -31.0], [32.0, 0.32, -32.0], [33.0, 0.33, -33.0],
        ]);

        // where the magic happens
        descriptor.densify(vec!["species_center", "species_neighbor"]);

        assert_eq!(descriptor.values.shape(), [3, 9]);
        assert_eq!(descriptor.samples.names(), ["structure", "center"]);
        assert_eq!(descriptor.samples[0], [v!(0), v!(0)]);
        assert_eq!(descriptor.samples[1], [v!(0), v!(1)]);
        assert_eq!(descriptor.samples[2], [v!(0), v!(2)]);

        assert_eq!(descriptor.values, array![
            /*    H-H             H-O                 O-H             */
            // O in water
            [0.0, 0.0, 0.0,    /**/ 0.0, 0.0, 0.0,    /**/ 1.0, 2.0, 3.0],
            // H1 in water
            [4.0, 5.0, 6.0,    /**/ 7.0, 8.0, 9.0,    /**/ 0.0, 0.0, 0.0],
            // H2 in water
            [10.0, 11.0, 12.0, /**/ 13.0, 14.0, 15.0, /**/ 0.0, 0.0, 0.0],
        ]);

        let gradients = descriptor.gradients.as_ref().unwrap();
        assert_eq!(gradients.shape(), [27, 9]);
        let gradients_samples = descriptor.gradients_samples.as_ref().unwrap();
        assert_eq!(gradients_samples.names(), ["structure", "center", "neighbor", "spatial"]);

        let expected = [
            [v!(0), v!(0), v!(0)],
            [v!(0), v!(0), v!(1)],
            [v!(0), v!(0), v!(2)],
            [v!(0), v!(1), v!(1)],
            [v!(0), v!(1), v!(2)],
            [v!(0), v!(1), v!(0)],
            [v!(0), v!(2), v!(2)],
            [v!(0), v!(2), v!(1)],
            [v!(0), v!(2), v!(0)],
        ];
        // use a loop to simplify checking the spatial dimension
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(gradients_samples[3 * i][..3], value);
            assert_eq!(gradients_samples[3 * i][3], v!(0));

            assert_eq!(gradients_samples[3 * i + 1][..3], value);
            assert_eq!(gradients_samples[3 * i + 1][3], v!(1));

            assert_eq!(gradients_samples[3 * i + 2][..3], value);
            assert_eq!(gradients_samples[3 * i + 2][3], v!(2));
        }

        assert_eq!(*gradients, array![
            /*    H-H                  H-O                  O-H       */
            // O in water, derivatives w.r.t. O
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       1.0, 0.1, -1.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       2.0, 0.2, -2.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       3.0, 0.3, -3.0],
            // O in water, derivatives w.r.t. H1
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       4.0, 0.4, -4.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       5.0, 0.5, -5.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       6.0, 0.6, -6.0],
            // O in water, derivatives w.r.t. H2
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       7.0, 0.7, -7.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       8.0, 0.8, -8.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,       9.0, 0.9, -9.0],
            // H1 in water, derivatives w.r.t. H1
            [10.0, 0.10, -10.0,    16.0, 0.16, -16.0,   0.0, 0.0, 0.0],
            [11.0, 0.11, -11.0,    17.0, 0.17, -17.0,   0.0, 0.0, 0.0],
            [12.0, 0.12, -12.0,    18.0, 0.18, -18.0,   0.0, 0.0, 0.0],
            // H1 in water, derivatives w.r.t. H2
            [13.0, 0.13, -13.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [14.0, 0.14, -14.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [15.0, 0.15, -15.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            // H1 in water, derivatives w.r.t. O
            [0.0, 0.0, 0.0,        19.0, 0.19, -19.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        20.0, 0.20, -20.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        21.0, 0.21, -21.0,   0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. H2
            [22.0, 0.22, -22.0,    28.0, 0.28, -28.0,   0.0, 0.0, 0.0],
            [23.0, 0.23, -23.0,    29.0, 0.29, -29.0,   0.0, 0.0, 0.0],
            [24.0, 0.24, -24.0,    30.0, 0.30, -30.0,   0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. H1
            [25.0, 0.25, -25.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [26.0, 0.26, -26.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [27.0, 0.27, -27.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. O
            [0.0, 0.0, 0.0,        31.0, 0.31, -31.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        32.0, 0.32, -32.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        33.0, 0.33, -33.0,   0.0, 0.0, 0.0],
        ]);
    }
}
