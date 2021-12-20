use std::collections::{BTreeSet, BTreeMap};
use indexmap::set::IndexSet;

use itertools::Itertools;
use ndarray::{Array2, ArrayView2, s};

use log::warn;

use crate::{Error};
use super::{Indexes, IndexesBuilder, IndexValue};

/// A Descriptor contains the representation of atomistic systems, as computed
/// by a [`crate::Calculator`].
#[derive(Clone, Debug)]
pub struct Descriptor {
    /// An array of size `samples.count()` by `features.count()`, containing the
    /// representation of the atomistic systems.
    pub values: Array2<f64>,
    /// Metadata describing the samples (i.e. rows) in the `values` array
    pub samples: Indexes,

    /// An array of size `gradients_samples.count()` by `features.count()`,
    /// containing the gradients of the representation with respect to the
    /// atomic positions.
    pub gradients: Option<Array2<f64>>,
    /// Metadata describing the samples (i.e. rows) in the `gradients` array
    pub gradients_samples: Option<Indexes>,

    /// Metadata describing the features (i.e. columns) in both the `values` and
    /// `gradients` array
    pub features: Indexes,
}

impl Default for Descriptor {
    fn default() -> Self { Self::new() }
}

impl Descriptor {
    /// Create a new empty descriptor
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

    /// Make this descriptor dense along the given `variables`.
    ///
    /// This function "moves" the variables from the samples to the features,
    /// filling the new features with zeros if the corresponding sample is
    /// missing.
    ///
    /// The `requested` parameter defines which set of values taken by the
    /// `variables` should be part of the new features. If it is `None`, this is
    /// the set of values taken by the variables in the samples. Otherwise, it
    /// must be an array with one row for each new feature block, and one column
    /// for each variable.
    ///
    /// For example, take a descriptor containing two samples variables
    /// (`structure` and `species`) and two features (`n` and `l`). Starting
    /// with this descriptor:
    ///
    /// ```text
    ///                       +---+---+---+
    ///                       | n | 0 | 1 |
    ///                       +---+---+---+
    ///                       | l | 0 | 1 |
    /// +-----------+---------+===+===+===+
    /// | structure | species |           |
    /// +===========+=========+   +---+---+
    /// |     0     |    1    |   | 1 | 2 |
    /// +-----------+---------+   +---+---+
    /// |     0     |    6    |   | 3 | 4 |
    /// +-----------+---------+   +---+---+
    /// |     1     |    6    |   | 5 | 6 |
    /// +-----------+---------+   +---+---+
    /// |     1     |    8    |   | 7 | 8 |
    /// +-----------+---------+---+---+---+
    /// ```
    ///
    /// Calling `descriptor.densify(["species"], None)` will move `species` out
    /// of the samples and into the features, producing:
    ///
    /// ```text
    ///             +---------+-------+-------+-------+
    ///             | species |   1   |   6   |   8   |
    ///             +---------+---+---+---+---+---+---+
    ///             |    n    | 0 | 1 | 0 | 1 | 0 | 1 |
    ///             +---------+---+---+---+---+---+---+
    ///             |    l    | 0 | 1 | 0 | 1 | 0 | 1 |
    /// +-----------+=========+===+===+===+===+===+===+
    /// | structure |
    /// +===========+         +---+---+---+---+---+---+
    /// |     0     |         | 1 | 2 | 3 | 4 | 0 | 0 |
    /// +-----------+         +---+---+---+---+---+---+
    /// |     1     |         | 0 | 0 | 5 | 6 | 7 | 8 |
    /// +-----------+---------+---+---+---+---+---+---+
    /// ```
    ///
    /// Notice how there is only one row/sample for each structure now, and how
    /// each value for `species` have created a full block of features. Missing
    /// values (e.g. structure 0/species 8) have been filled with 0.
    #[time_graph::instrument(name="Descriptor::densify")]
    pub fn densify<'a>(
        &mut self,
        variables: &[&str],
        requested: impl Into<Option<ArrayView2<'a, IndexValue>>>,
    ) -> Result<(), Error> {
        self.densify_impl(variables, requested, /*do_gradient*/ true)?;
        return Ok(());
    }

    /// Make this descriptor dense along the given `variables`, only modifying
    /// the values array, and not the gradients array.
    ///
    /// This function behaves similarly to [`Descriptor::densify`], please refer
    /// to its documentation for more information.
    ///
    /// This does not change the gradients themselves, instead returning a
    /// vector containing the changes made to the value array.
    ///
    /// This is an advanced function most users should not need to use, used to
    /// implement backward propagation without having to densify the full
    /// gradient array and waste memory.
    #[time_graph::instrument(name="Descriptor::densify_values")]
    pub fn densify_values<'a>(
        &mut self,
        variables: &[&str],
        requested: impl Into<Option<ArrayView2<'a, IndexValue>>>,
    ) -> Result<DensifiedPositions, Error> {
        self.densify_impl(variables, requested, /*do_gradient*/ false)
    }

    /// Common implementation of `densify` & `densify_value`. This function
    /// returns the vector of new positions for values if `do_gradient` is
    /// false.
    fn densify_impl<'a>(
        &mut self,
        variables: &[&str],
        requested: impl Into<Option<ArrayView2<'a, IndexValue>>>,
        do_gradient: bool
    ) -> Result<DensifiedPositions, Error> {
        if variables.is_empty() || self.features.size() == 0 {
            return Ok(DensifiedPositions::new(0));
        }

        // if the user provided them, extract the set of values to use for the
        // new features.
        let requested_features = if let Some(requested) = requested.into() {
            let shape = requested.shape();
            if shape[1] != variables.len() {
                return Err(Error::InvalidParameter(format!(
                    "provided values in Descriptor::densify must match the \
                    variable size: expected {}, got {}", variables.len(), shape[1]
                )));
            }

            let mut features = BTreeSet::new();
            for value in requested.axis_iter(ndarray::Axis(0)) {
                features.insert(value.to_vec());
            }

            Some(features)
        } else {
            None
        };

        let updated_samples = remove_from_samples(&self.samples, variables, requested_features)?;
        // new features, adding `variables` in the front. This transforms
        // something like [n, l, m] to [species_neighbor, n, l, m]; and fill it
        // with the corresponding values from `new_samples.features`,
        // duplicating the `[n, l, m]` block as needed
        let mut feature_names = variables.to_vec();
        feature_names.extend(self.features.names());
        let mut new_features = IndexesBuilder::new(feature_names);
        for new in updated_samples.features {
            for feature in self.features.iter() {
                let mut new = new.clone();
                new.extend(feature);
                new_features.add(&new);
            }
        }
        let new_features = new_features.finish();
        let new_features_count = new_features.count();

        let feature_block_size = self.features.count();
        // copy values themselves as needed
        let mut new_values = Array2::zeros((updated_samples.samples.count(), new_features_count));
        for (old_sample, new_position) in updated_samples.new_positions.iter().enumerate() {
            if let Some(new_position) = new_position {
                let start = feature_block_size * new_position.features_block;
                let stop = feature_block_size * (new_position.features_block + 1);

                let value = self.values.slice(s![old_sample, ..]);
                new_values.slice_mut(s![new_position.sample, start..stop]).assign(&value);
            }
        }

        self.features = new_features;
        self.samples = updated_samples.samples;
        self.values = new_values;

        if !do_gradient {
            return Ok(updated_samples.new_positions);
        }

        if let Some(ref mut gradients) = self.gradients {
            let gradients_samples = self.gradients_samples.as_ref().expect("missing gradients samples");

            // we need to use indexmap::IndexSet here to get the new positions
            // of the sample as we go over the old samples
            let mut new_gradient_samples = IndexSet::new();
            for gradient_sample in gradients_samples {
                let sample_i = gradient_sample[0].usize();
                let atom = gradient_sample[1];

                if let Some(ref position) = updated_samples.new_positions[sample_i] {
                    new_gradient_samples.insert(
                        (IndexValue::from(position.sample), atom)
                    );
                }
            }

            let mut new_gradients = Array2::zeros(
                (3 * new_gradient_samples.len(), new_features_count)
            );

            for (old_grad_sample_i, gradient_sample) in gradients_samples.iter().enumerate() {
                let sample = gradient_sample[0].usize();
                let atom = gradient_sample[1];
                let spatial = gradient_sample[2].usize();

                if let Some(ref position) = updated_samples.new_positions[sample] {
                    let new_grad_sample_i = new_gradient_samples.get_index_of(
                        &(IndexValue::from(position.sample), atom)
                    ).expect("missing entry in new gradient samples");
                    let new_grad_position = 3 * new_grad_sample_i + spatial;

                    let start = feature_block_size * position.features_block;
                    let stop = feature_block_size * (position.features_block + 1);

                    let value = gradients.slice(s![old_grad_sample_i, ..]);
                    new_gradients.slice_mut(s![new_grad_position, start..stop]).assign(&value);
                }
            }

            let mut builder = IndexesBuilder::new(vec!["sample", "atom", "spatial"]);
            for (sample, atom) in new_gradient_samples {
                builder.add(&[sample, atom, IndexValue::from(0)]);
                builder.add(&[sample, atom, IndexValue::from(1)]);
                builder.add(&[sample, atom, IndexValue::from(2)]);
            }
            self.gradients_samples = Some(builder.finish());
            self.gradients = Some(new_gradients);
        }

        return Ok(DensifiedPositions::new(0));
    }

    /// Initialize this descriptor with the given `samples` and `features`,
    /// allocating memory in the `values` array only. The `values` array is set
    /// to zero.
    ///
    /// This is an advanced function most users should not need to use.
    pub fn prepare(&mut self, samples: Indexes, features: Indexes) {
        self.samples = samples;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.samples.count(), self.features.count());
        resize_and_reset(&mut self.values, shape);

        self.gradients = None;
        self.gradients_samples = None;
    }

    /// Initialize this descriptor with the given `samples`, `gradients_samples`
    /// and `features`, allocating memory in both the `values` and `gradients`
    /// arrays. The arrays are set to zero.
    ///
    /// This is an advanced function most users should not need to use.
    pub fn prepare_gradients(
        &mut self,
        samples: Indexes,
        gradients_samples: Indexes,
        features: Indexes,
    ) {
        // basic sanity check
        debug_assert_eq!(gradients_samples.names(), vec!["sample", "atom", "spatial"]);

        self.samples = samples;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.samples.count(), self.features.count());
        resize_and_reset(&mut self.values, shape);

        let gradient_shape = (gradients_samples.count(), self.features.count());
        self.gradients_samples = Some(gradients_samples);

        if let Some(array) = &mut self.gradients {
            // resize the 'gradient' array if needed, and set the requested initial value
            resize_and_reset(array, gradient_shape);
        } else {
            // create a new gradient array
            let array = Array2::from_elem(gradient_shape, 0.0);
            self.gradients = Some(array);
        }
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

/// A `DensifiedPosition` contains all the information to reconstruct the new
/// position of the values/gradients associated with a single sample in the
/// initial descriptor
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct DensifiedPosition {
    /// Index of the new sample (respectively gradient sample) in the value
    /// (respectively gradients) array
    pub sample: usize,
    /// Index of the feature block in the new array
    pub features_block: usize,
}

/// Mapping from the old position of a sample to the new position after
/// densifying. `None` indicates that this row was not selected by the user
/// (which can happen if the user requests a subset of the values taken by the
/// densified variables)
#[derive(Debug, Clone)]
pub struct DensifiedPositions(Vec<Option<DensifiedPosition>>);

impl DensifiedPositions {
    fn new(size: usize) -> DensifiedPositions {
        DensifiedPositions(vec![None; size])
    }
}

impl std::ops::Deref for DensifiedPositions {
    type Target = [Option<DensifiedPosition>];

    fn deref(&self) -> &[Option<DensifiedPosition>] {
        &*self.0
    }
}

impl std::ops::DerefMut for DensifiedPositions {
    fn deref_mut(&mut self) -> &mut [Option<DensifiedPosition>] {
        &mut *self.0
    }
}


/// Results of removing a set of variables from samples
struct RemovedSamples {
    /// New samples, without the variables
    samples: Indexes,
    /// Values taken by the variables in the original samples
    features: BTreeSet<Vec<IndexValue>>,
    /// mapping containing the position in which each old sample should be
    /// placed in the new values or gradients array. If the dense position is
    /// `None`, that means that this row was not selected by the user (which can
    /// happen if the user requested only a subset of the values taken by the
    /// densified variables)
    new_positions: DensifiedPositions
}

/// Remove the given `variables` from the `samples`, returning the updated
/// `samples`, the set of all the values taken by the removed variables, and the
/// mapping from the old position to the new position in the corresponding
/// values array.
fn remove_from_samples(
    samples: &Indexes,
    variables: &[&str],
    requested: Option<BTreeSet<Vec<IndexValue>>>
) -> Result<RemovedSamples, Error> {
    let mut variables_positions = Vec::new();
    for v in variables {
        let position = samples.names().iter().position(|name| name == v);
        if let Some(position) = position {
            variables_positions.push(position);
        } else {
            return Err(Error::InvalidParameter(format!(
                "can not densify along '{}' which is not present in the samples: [{}]",
                    v, samples.names().join(", ")
            )))
        }
    }

    // collect all different indexes in maps. Assuming we are densifying
    // along the first index, we want to convert [[2, 3, 0], [1, 3, 0]]
    // to [[3, 0]].
    let mut new_features = BTreeSet::new();
    for sample in samples.iter() {
        let mut new_feature = Vec::new();
        for &i in &variables_positions {
            new_feature.push(sample[i]);
        }
        new_features.insert(new_feature);
    }

    // deal with the user requesting a specific set of the features
    let features = if let Some(requested) = requested {
        // check that all features in the dataset are part of the requested ones
        for feature in new_features {
            if !requested.contains(&feature) {
                warn!(
                    "[{}] takes the value [{}] in this descriptor, but it is \
                    not part of the requested features list",
                    variables.iter().join(","),
                    feature.iter().map(|v| v.to_string()).join(",")
                );
            }
        }
        requested
    } else {
        // if no features where requested by the user, use the list we have
        // from the samples
        new_features
    };

    // map from new feature => feature block index
    let features_blocks = features.iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect::<BTreeMap<_, _>>();

    // build the new samples & the mapping from old to new samples
    let mut new_positions = DensifiedPositions::new(samples.count());
    // we need to use indexmap::IndexSet here to get the new positions of the
    // sample as we go over the old samples
    let mut new_samples = IndexSet::new();
    for (old_sample_i, sample) in samples.iter().enumerate() {
        let mut new_feature = Vec::new();
        for &i in &variables_positions {
            new_feature.push(sample[i]);
        }

        let features_block = if let Some(i) = features_blocks.get(&new_feature) {
            *i
        } else {
            // the feature corresponding to the current sample is not part of
            // the new list of features, the user did not request it
            continue;
        };

        let mut new_sample = sample.to_vec();
        // reverse the indexes to ensure the all the calls to `remove` are valid.
        // this works because variables_positions is sorted by construction
        for &i in variables_positions.iter().rev() {
            new_sample.remove(i);
        }
        let (new_sample_i, _) = new_samples.insert_full(new_sample);

        new_positions[old_sample_i] = Some(DensifiedPosition {
            sample: new_sample_i,
            features_block: features_block,
        });
    }

    // finish building the samples
    let names = samples.names()
        .iter()
        .filter(|&name| !variables.contains(name))
        .copied()
        .collect();
    let mut builder = IndexesBuilder::new(names);
    for sample in new_samples {
        builder.add(&sample);
    }

    return Ok(RemovedSamples {
        samples: builder.finish(),
        features: features,
        new_positions: new_positions
    });
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_utils::test_systems;
    use crate::descriptor::{TwoBodiesSpeciesSamples, StructureSpeciesSamples, SamplesBuilder};
    use ndarray::array;

    fn dummy_features() -> Indexes {
        let mut features = IndexesBuilder::new(vec!["foo", "bar"]);
        features.add(&[IndexValue::from(0), IndexValue::from(-1)]);
        features.add(&[IndexValue::from(4), IndexValue::from(-2)]);
        features.add(&[IndexValue::from(1), IndexValue::from(-5)]);
        return features.finish();
    }

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

    #[test]
    fn prepare() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]);
        let features = dummy_features();
        let samples = StructureSpeciesSamples.samples(&mut systems).unwrap();
        descriptor.prepare(samples, features);


        assert_eq!(descriptor.values.shape(), [4, 3]);

        assert_eq!(descriptor.samples.names(), ["structure", "species"]);
        assert_eq!(descriptor.samples[0], [v(0), v(1)]);
        assert_eq!(descriptor.samples[1], [v(0), v(123456)]);
        assert_eq!(descriptor.samples[2], [v(1), v(1)]);
        assert_eq!(descriptor.samples[3], [v(1), v(6)]);

        assert!(descriptor.gradients.is_none());
    }

    #[test]
    fn prepare_gradients() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]);
        let features = dummy_features();
        let (samples, gradients) = StructureSpeciesSamples.with_gradients(&mut systems).unwrap();
        descriptor.prepare_gradients(samples, gradients.unwrap(), features);

        let gradients = descriptor.gradients.unwrap();
        let gradients_samples = descriptor.gradients_samples.as_ref().unwrap();
        assert_eq!(gradients.shape(), [gradients_samples.count(), descriptor.features.count()]);
    }

    #[test]
    fn densify() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water"]);
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
            // O channel around H1, derivatives w.r.t. O
            [16.0, 0.16, -16.0], [17.0, 0.17, -17.0], [18.0, 0.18, -18.0],
            // O channel around H1, derivatives w.r.t. H1
            [19.0, 0.19, -19.0], [20.0, 0.20, -20.0], [21.0, 0.21, -21.0],
            // H channel around H2, derivatives w.r.t. H1
            [22.0, 0.22, -22.0], [23.0, 0.23, -23.0], [24.0, 0.24, -24.0],
            // H channel around H2, derivatives w.r.t. H2
            [25.0, 0.25, -25.0], [26.0, 0.26, -26.0], [27.0, 0.27, -27.0],
            // O channel around H2, derivatives w.r.t. O
            [28.0, 0.28, -28.0], [29.0, 0.29, -29.0], [30.0, 0.30, -30.0],
            // O channel around H2, derivatives w.r.t. H2
            [31.0, 0.31, -31.0], [32.0, 0.32, -32.0], [33.0, 0.33, -33.0],
        ]);

        // where the magic happens
        descriptor.densify(&["species_center", "species_neighbor"], None).unwrap();

        assert_eq!(descriptor.values.shape(), [3, 9]);
        assert_eq!(descriptor.samples.names(), ["structure", "center"]);
        assert_eq!(descriptor.samples[0], [v(0), v(0)]);
        assert_eq!(descriptor.samples[1], [v(0), v(1)]);
        assert_eq!(descriptor.samples[2], [v(0), v(2)]);

        assert_eq!(descriptor.features.names(), ["species_center", "species_neighbor", "foo", "bar"]);
        assert_eq!(descriptor.features[0], [v(1), v(1), v(0), v(-1)]);
        assert_eq!(descriptor.features[1], [v(1), v(1), v(4), v(-2)]);
        assert_eq!(descriptor.features[2], [v(1), v(1), v(1), v(-5)]);
        assert_eq!(descriptor.features[3], [v(1), v(123456), v(0), v(-1)]);
        assert_eq!(descriptor.features[4], [v(1), v(123456), v(4), v(-2)]);
        assert_eq!(descriptor.features[5], [v(1), v(123456), v(1), v(-5)]);
        assert_eq!(descriptor.features[6], [v(123456), v(1), v(0), v(-1)]);
        assert_eq!(descriptor.features[7], [v(123456), v(1), v(4), v(-2)]);
        assert_eq!(descriptor.features[8], [v(123456), v(1), v(1), v(-5)]);

        assert_eq!(descriptor.values, array![
            /*    H-H                    H-O                  O-H      */
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
        assert_eq!(gradients_samples.names(), ["sample", "atom", "spatial"]);

        // TODO: this is not lexicographically sorted but rather sorted in the
        // same order as features are processed. Is this an issue?
        assert_eq!(gradients_samples.iter().collect::<Vec<_>>(), vec![
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],

            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            &[v(1), v(2), v(0)], &[v(1), v(2), v(1)], &[v(1), v(2), v(2)],
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],

            &[v(2), v(1), v(0)], &[v(2), v(1), v(1)], &[v(2), v(1), v(2)],
            &[v(2), v(2), v(0)], &[v(2), v(2), v(1)], &[v(2), v(2), v(2)],
            &[v(2), v(0), v(0)], &[v(2), v(0), v(1)], &[v(2), v(0), v(2)],
        ]);

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
            [10.0, 0.10, -10.0,    19.0, 0.19, -19.0,   0.0, 0.0, 0.0],
            [11.0, 0.11, -11.0,    20.0, 0.20, -20.0,   0.0, 0.0, 0.0],
            [12.0, 0.12, -12.0,    21.0, 0.21, -21.0,   0.0, 0.0, 0.0],
            // H1 in water, derivatives w.r.t. H2
            [13.0, 0.13, -13.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [14.0, 0.14, -14.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [15.0, 0.15, -15.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            // H1 in water, derivatives w.r.t. O
            [0.0, 0.0, 0.0,        16.0, 0.16, -16.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        17.0, 0.17, -17.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        18.0, 0.18, -18.0,   0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. H1
            [22.0, 0.22, -22.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [23.0, 0.23, -23.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            [24.0, 0.24, -24.0,    0.0, 0.0, 0.0,       0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. H2
            [25.0, 0.25, -25.0,    31.0, 0.31, -31.0,   0.0, 0.0, 0.0],
            [26.0, 0.26, -26.0,    32.0, 0.32, -32.0,   0.0, 0.0, 0.0],
            [27.0, 0.27, -27.0,    33.0, 0.33, -33.0,   0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. O
            [0.0, 0.0, 0.0,        28.0, 0.28, -28.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        29.0, 0.29, -29.0,   0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,        30.0, 0.30, -30.0,   0.0, 0.0, 0.0],
        ]);
    }

    #[test]
    fn densify_user_requested() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water"]);
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

        let requested = Array2::from_shape_vec([3, 2], vec![
            v(1), v(1),       // H-H
            v(6), v(1),       // missing
            v(123456), v(1),  // O-H
        ]).unwrap();
        descriptor.densify(&["species_center", "species_neighbor"], requested.view()).unwrap();

        assert_eq!(descriptor.values.shape(), [3, 9]);
        assert_eq!(descriptor.samples.names(), ["structure", "center"]);
        assert_eq!(descriptor.samples[0], [v(0), v(0)]);
        assert_eq!(descriptor.samples[1], [v(0), v(1)]);
        assert_eq!(descriptor.samples[2], [v(0), v(2)]);

        assert_eq!(descriptor.features.names(), ["species_center", "species_neighbor", "foo", "bar"]);
        assert_eq!(descriptor.features[0], [v(1), v(1), v(0), v(-1)]);
        assert_eq!(descriptor.features[1], [v(1), v(1), v(4), v(-2)]);
        assert_eq!(descriptor.features[2], [v(1), v(1), v(1), v(-5)]);
        assert_eq!(descriptor.features[3], [v(6), v(1), v(0), v(-1)]);
        assert_eq!(descriptor.features[4], [v(6), v(1), v(4), v(-2)]);
        assert_eq!(descriptor.features[5], [v(6), v(1), v(1), v(-5)]);
        assert_eq!(descriptor.features[6], [v(123456), v(1), v(0), v(-1)]);
        assert_eq!(descriptor.features[7], [v(123456), v(1), v(4), v(-2)]);
        assert_eq!(descriptor.features[8], [v(123456), v(1), v(1), v(-5)]);

        assert_eq!(descriptor.values, array![
            /*    H-H                 missing              O-H      */
            // O in water
            [0.0, 0.0, 0.0,    /**/ 0.0, 0.0, 0.0, /**/ 1.0, 2.0, 3.0],
            // H1 in water
            [4.0, 5.0, 6.0,    /**/ 0.0, 0.0, 0.0, /**/ 0.0, 0.0, 0.0],
            // H2 in water
            [10.0, 11.0, 12.0, /**/ 0.0, 0.0, 0.0, /**/ 0.0, 0.0, 0.0],
        ]);

        let gradients = descriptor.gradients.as_ref().unwrap();
        assert_eq!(*gradients, array![
            /*    H-H                  missing               O-H       */
            // O in water, derivatives w.r.t. O
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   1.0, 0.1, -1.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   2.0, 0.2, -2.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   3.0, 0.3, -3.0],
            // O in water, derivatives w.r.t. H1
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   4.0, 0.4, -4.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   5.0, 0.5, -5.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   6.0, 0.6, -6.0],
            // O in water, derivatives w.r.t. H2
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   7.0, 0.7, -7.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   8.0, 0.8, -8.0],
            [0.0, 0.0, 0.0,        0.0, 0.0, 0.0,   9.0, 0.9, -9.0],
            // H1 in water, derivatives w.r.t. H1
            [10.0, 0.10, -10.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [11.0, 0.11, -11.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [12.0, 0.12, -12.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            // H1 in water, derivatives w.r.t. H2
            [13.0, 0.13, -13.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [14.0, 0.14, -14.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [15.0, 0.15, -15.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. H2
            [22.0, 0.22, -22.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [23.0, 0.23, -23.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [24.0, 0.24, -24.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            // H2 in water, derivatives w.r.t. H1
            [25.0, 0.25, -25.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [26.0, 0.26, -26.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
            [27.0, 0.27, -27.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
        ]);
    }

    #[test]
    fn densify_values() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water"]);
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

        let mut descriptor_dense = descriptor.clone();
        descriptor_dense.densify(&["species_neighbor"], None).unwrap();

        let new_positions = descriptor.densify_values(&["species_neighbor"], None).unwrap();

        assert_eq!(descriptor.samples, descriptor_dense.samples);
        assert_eq!(descriptor.features, descriptor_dense.features);
        assert_eq!(descriptor.values, descriptor_dense.values);

        // gradients are not modified
        let gradients = descriptor.gradients.as_ref().unwrap();
        assert_eq!(gradients.shape(), [33, 3]);

        let reference_gradients = descriptor_dense.gradients.as_ref().unwrap();
        let reference_gradients_samples = descriptor_dense.gradients_samples.as_ref().unwrap();

        // check that applying the transformation leads to the right array
        let shape = reference_gradients.shape();
        let feature_block_size = gradients.shape()[1];
        let mut dense_gradients = Array2::zeros((shape[0], shape[1]));

        let gradients_samples = descriptor.gradients_samples.as_ref().unwrap();
        for (old_grad_sample_i, gradient_sample) in gradients_samples.iter().enumerate() {
            let sample = gradient_sample[0].usize();
            let atom = gradient_sample[1];
            let spatial = gradient_sample[2];

            if let Some(ref position) = new_positions[sample] {
                let new_grad_position = reference_gradients_samples.position(
                    &[IndexValue::from(position.sample), atom, spatial]
                ).expect("missing entry in reference gradient samples");

                let start = feature_block_size * position.features_block;
                let stop = feature_block_size * (position.features_block + 1);

                let value = gradients.slice(s![old_grad_sample_i, ..]);
                dense_gradients.slice_mut(s![new_grad_position, start..stop]).assign(&value);
            }
        }

        assert_eq!(dense_gradients, reference_gradients);
    }
}
