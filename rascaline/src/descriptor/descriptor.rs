use std::collections::BTreeMap;
use indexmap::set::IndexSet;

use itertools::Itertools;
use ndarray::{Array2, s};

use super::{Indexes, IndexesBuilder, IndexValue};

pub struct Descriptor {
    /// An array of environments.count() by features.count() values
    pub values: Array2<f64>,
    pub environments: Indexes,
    pub features: Indexes,
    /// Gradients of the descriptor with respect to one atomic position
    pub gradients: Option<Array2<f64>>,
    pub gradients_indexes: Option<Indexes>,
}

impl Default for Descriptor {
    fn default() -> Self { Self::new() }
}

impl Descriptor {
    pub fn new() -> Descriptor {
        let indexes = IndexesBuilder::new(vec![]).finish();
        return Descriptor {
            values: Array2::zeros((0, 0)),
            environments: indexes.clone(),
            features: indexes,
            gradients: None,
            gradients_indexes: None,
        }
    }

    pub fn prepare(
        &mut self,
        environments: Indexes,
        features: Indexes,
    ) {
        self.environments = environments;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.environments.count(), self.features.count());
        resize_and_reset(&mut self.values, shape);

        self.gradients = None;
        self.gradients_indexes = None;
    }

    pub fn prepare_gradients(
        &mut self,
        environments: Indexes,
        gradients: Indexes,
        features: Indexes,
    ) {
        // basic sanity check
        assert_eq!(gradients.names().last(), Some(&"spatial"), "the last index of gradient should be spatial");

        self.environments = environments;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.environments.count(), self.features.count());
        resize_and_reset(&mut self.values, shape);

        let gradient_shape = (gradients.count(), self.features.count());
        self.gradients_indexes = Some(gradients);

        if let Some(array) = &mut self.gradients {
            // resize the 'gradient' array if needed, and set the requested initial value
            resize_and_reset(array, gradient_shape);
        } else {
            // create a new gradient array
            let array = Array2::from_elem(gradient_shape, 0.0);
            self.gradients = Some(array);
        }
    }

    pub fn densify(&mut self, variables: Vec<&str>) {
        if variables.is_empty() {
            return;
        }

        let new_environments = remove_from_indexes(&self.environments, &variables);
        let new_gradients = self.gradients_indexes.as_ref().map(|indexes| {
            let gradients = remove_from_indexes(indexes, &variables);

            if gradients.new_features != new_environments.new_features {
                let name = if variables.len() == 1 {
                    variables[0].to_owned()
                } else {
                    format!("({})", variables.join(", "))
                };
                panic!("gradient indexes contains different values for {} than the environment indexes", name);
            }

            return gradients;
        });

        // new feature indexes, add `variable` in the front
        let mut feature_names = variables;
        feature_names.extend(self.features.names());
        let mut new_features = IndexesBuilder::new(feature_names);
        for new in new_environments.new_features {
            for feature in self.features.iter() {
                let mut cleaned = new.clone();
                cleaned.extend(feature);
                new_features.add(&cleaned);
            }
        }
        let new_features = new_features.finish();
        let old_feature_size = self.features.count();

        // copy values as needed
        let mut new_values = Array2::zeros((new_environments.indexes.count(), new_features.count()));
        for (new, old) in new_environments.mapping {
            let value = self.values.slice(s![old, ..]);
            let start = new.feature_block * old_feature_size;
            let stop = (new.feature_block + 1) * old_feature_size;
            new_values.slice_mut(s![new.environment, start..stop]).assign(&value);
        }

        if let Some(self_gradients) = &self.gradients {
            let new_gradients = new_gradients.expect("missing densified gradients");

            let mut gradients = Array2::zeros((new_gradients.indexes.count(), new_features.count()));
            for (new, old) in new_gradients.mapping {
                let value = self_gradients.slice(s![old, ..]);
                let start = new.feature_block * old_feature_size;
                let stop = (new.feature_block + 1) * old_feature_size;
                gradients.slice_mut(s![new.environment, start..stop]).assign(&value);
            }

            self.gradients = Some(gradients);
            self.gradients_indexes = Some(new_gradients.indexes);
        }

        self.features = new_features;
        self.environments = new_environments.indexes;
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
    let _ = std::mem::replace(array, values);
}

/// Representation of an environment/gradient index after a call to `densify`
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
struct DensifiedIndex {
    /// Index of the new environment/gradient in the value/gradients array
    environment: usize,
    /// Index of the feature **block** corresponding to the moved variable
    feature_block: usize,
}

/// Results of removing a set of variables from Indexes
struct RemovedResult {
    /// New Indexes, without the variables
    indexes: Indexes,
    /// Values taken by the variables in the original Index
    ///
    /// This needs to be a IndexSet to keep the same order as in the initial
    /// Indexes.
    new_features: IndexSet<Vec<IndexValue>>,
    /// Mapping from the updated index to the original position
    mapping: BTreeMap<DensifiedIndex, usize>,
}

/// Remove the given `variables` from the `indexes`, returning the updated
/// `indexes` and a set of all the values taken by the removed variables.
fn remove_from_indexes(indexes: &Indexes, variables: &[&str]) -> RemovedResult {
    let variable_indexes = variables.iter()
        .map(|v| {
            indexes.names()
                .iter()
                .position(|name| name == v)
                .unwrap_or_else(|| panic!(
                    "can not densify along '{}' which is not present in the environments: [{}]",
                    v, indexes.names().join(", ")
                ))
        }).collect::<Vec<_>>();

    let mut mapping = BTreeMap::new();

    // collect all different indexes in a set. Assuming we are densifying
    // along the first index, we want to convert [[2, 3, 0], [1, 3, 0]]
    // to [[3, 0]].
    let mut new_indexes = IndexSet::new();
    let mut new_features = IndexSet::new();
    for (old, index) in indexes.iter().enumerate() {
        let mut new_feature = Vec::new();
        for &i in &variable_indexes {
            new_feature.push(index[i]);
        }
        new_features.insert(new_feature.clone());

        let mut new_index = index.to_vec();
        // sort and reverse the indexes to ensure the all the calls to `remove`
        // are valid
        for &i in variable_indexes.iter().sorted().rev() {
            new_index.remove(i);
        }
        new_indexes.insert(new_index);

        let densified = DensifiedIndex{
            environment: new_indexes.len() - 1,
            feature_block: new_features.iter()
                .position(|feature| feature == &new_feature)
                .expect("missing feature that was just added"),
        };
        mapping.insert(densified, old);
    }

    let names = indexes.names()
        .iter()
        .filter(|&name| !variables.contains(name))
        .cloned()
        .collect();
    let mut builder = IndexesBuilder::new(names);
    for env in new_indexes {
        builder.add(&env);
    }

    return RemovedResult {
        indexes: builder.finish(),
        new_features: new_features,
        mapping: mapping,
    };
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::test_systems;
    use crate::descriptor::{AtomSpeciesEnvironment, StructureSpeciesEnvironment, EnvironmentIndexes};
    use ndarray::array;

    fn dummy_features() -> Indexes {
        let mut features = IndexesBuilder::new(vec!["foo", "bar", "baz"]);
        features.add(&[IndexValue::from(0_usize), IndexValue::from(1_isize), IndexValue::from(0.3)]);
        features.add(&[IndexValue::from(4_usize), IndexValue::from(2_isize), IndexValue::from(3.3)]);
        features.add(&[IndexValue::from(1_usize), IndexValue::from(0_isize), IndexValue::from(2.3)]);
        return features.finish();
    }

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
        crate::descriptor::indexes::IndexValue::from($value as f64)
        };
    }

    #[test]
    fn prepare() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]);
        let features = dummy_features();
        let environments = StructureSpeciesEnvironment.indexes(&mut systems.get());
        descriptor.prepare(environments, features);


        assert_eq!(descriptor.values.shape(), [4, 3]);

        assert_eq!(descriptor.environments.names(), ["structure", "species"]);
        assert_eq!(descriptor.environments[0], [v!(0), v!(1)]);
        assert_eq!(descriptor.environments[1], [v!(0), v!(123456)]);
        assert_eq!(descriptor.environments[2], [v!(1), v!(1)]);
        assert_eq!(descriptor.environments[3], [v!(1), v!(6)]);

        assert!(descriptor.gradients.is_none());
    }

    #[test]
    fn prepare_gradients() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]);
        let features = dummy_features();
        let (environments, gradients) = StructureSpeciesEnvironment.with_gradients(&mut systems.get());
        descriptor.prepare_gradients(environments, gradients.unwrap(), features);

        let gradients = descriptor.gradients.unwrap();
        assert_eq!(gradients.shape(), [15, 3]);

        let gradients_indexes = descriptor.gradients_indexes.as_ref().unwrap();
        assert_eq!(gradients_indexes.names(), ["structure", "species", "atom", "spatial"]);

        let expected = [
            [v!(0), v!(1), v!(1)],
            [v!(0), v!(1), v!(2)],
            [v!(0), v!(123456), v!(0)],
            [v!(1), v!(1), v!(0)],
            [v!(1), v!(6), v!(1)]
        ];
        // use a loop to simplify checking the spatial dimension
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(gradients_indexes[3 * i][..3], value);
            assert_eq!(gradients_indexes[3 * i][3], v!(0));

            assert_eq!(gradients_indexes[3 * i + 1][..3], value);
            assert_eq!(gradients_indexes[3 * i + 1][3], v!(1));

            assert_eq!(gradients_indexes[3 * i + 2][..3], value);
            assert_eq!(gradients_indexes[3 * i + 2][3], v!(2));
        }
    }

    #[test]
    fn densify_single_variable() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]);
        let features = dummy_features();
        let (environments, gradients) = StructureSpeciesEnvironment.with_gradients(&mut systems.get());
        descriptor.prepare_gradients(environments, gradients.unwrap(), features);

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
        assert_eq!(descriptor.environments.names(), ["structure"]);
        assert_eq!(descriptor.environments[0], [v!(0)]);
        assert_eq!(descriptor.environments[1], [v!(1)]);

        assert_eq!(descriptor.values, array![
            [1.0, 2.0, 3.0, /**/ 4.0, 5.0, 6.0, /**/ 0.0, 0.0, 0.0],
            [7.0, 8.0, 9.0, /**/ 0.0, 0.0, 0.0, /**/ 10.0, 11.0, 12.0],
        ]);

        let gradients = descriptor.gradients.as_ref().unwrap();
        assert_eq!(gradients.shape(), [15, 9]);
        let gradients_indexes = descriptor.gradients_indexes.as_ref().unwrap();
        assert_eq!(gradients_indexes.names(), ["structure", "atom", "spatial"]);

        let expected = [
            [v!(0), v!(1)],
            [v!(0), v!(2)],
            [v!(0), v!(0)],
            [v!(1), v!(0)],
            [v!(1), v!(1)]
        ];
        // use a loop to simplify checking the spatial dimension
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(gradients_indexes[3 * i][..2], value);
            assert_eq!(gradients_indexes[3 * i][2], v!(0));

            assert_eq!(gradients_indexes[3 * i + 1][..2], value);
            assert_eq!(gradients_indexes[3 * i + 1][2], v!(1));

            assert_eq!(gradients_indexes[3 * i + 2][..2], value);
            assert_eq!(gradients_indexes[3 * i + 2][2], v!(2));
        }

        assert_eq!(*gradients, array![
            [/*H*/ 1.0, 2.0, 3.0,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.1, 0.2, 0.3,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ -1.0, -2.0, -3.0,    /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 4.0, 5.0, 6.0,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.4, 0.5, 0.6,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ -4.0, -5.0, -6.0,    /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*O*/ 7.0, 8.0, 9.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*O*/ 0.7, 0.8, 0.9,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*O*/ -7.0, -8.0, -9.0, /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 10.0, 11.0, 12.0,    /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.1, 0.11, 0.12,     /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ -10.0, -11.0, -12.0, /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.0, 0.0, 0.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ 13.0, 14.0, 15.0],
            [/*H*/ 0.0, 0.0, 0.0,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ 0.13, 0.14, 0.15],
            [/*H*/ 0.0, 0.0, 0.0,       /*O*/ 0.0, 0.0, 0.0,    /*C*/ -13.0, -14.0, -15.0],
        ]);
    }
    #[test]
    fn densify_multiple_variables() {
        let mut descriptor = Descriptor::new();

        let mut systems = test_systems(&["water", "CH"]);
        let features = dummy_features();
        let (environments, gradients) = AtomSpeciesEnvironment::new(3.0).with_gradients(&mut systems.get());
        descriptor.prepare_gradients(environments, gradients.unwrap(), features);

        descriptor.values.assign(&array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
        ]);

        let gradients = descriptor.gradients.as_mut().unwrap();
        gradients.assign(&array![
            [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [-1.0, -2.0, -3.0],
            [4.0, 5.0, 6.0], [0.4, 0.5, 0.6], [-4.0, -5.0, -6.0],
            [7.0, 8.0, 9.0], [0.7, 0.8, 0.9], [-7.0, -8.0, -9.0],
            [10.0, 11.0, 12.0], [0.10, 0.11, 0.12], [-10.0, -11.0, -12.0],
            [13.0, 14.0, 15.0], [0.13, 0.14, 0.15], [-13.0, -14.0, -15.0],
            [16.0, 17.0, 18.0], [0.16, 0.17, 0.18], [-16.0, -17.0, -18.0],
            [19.0, 20.0, 21.0], [0.19, 0.20, 0.21], [-19.0, -20.0, -21.0],
            [22.0, 23.0, 24.0], [0.22, 0.23, 0.24], [-22.0, -23.0, -24.0],
        ]);

        // where the magic happens
        descriptor.densify(vec!["species_center", "species_neighbor"]);

        assert_eq!(descriptor.values.shape(), [5, 15]);
        assert_eq!(descriptor.environments.names(), ["structure", "center"]);
        assert_eq!(descriptor.environments[0], [v!(0), v!(0)]);
        assert_eq!(descriptor.environments[1], [v!(0), v!(1)]);
        assert_eq!(descriptor.environments[2], [v!(0), v!(2)]);
        assert_eq!(descriptor.environments[3], [v!(1), v!(0)]);
        assert_eq!(descriptor.environments[4], [v!(1), v!(1)]);

        assert_eq!(descriptor.values, array![
            /*    O-H             H-H                 H-O                  H-C                C-H     */
            // O in water
            [1.0, 2.0, 3.0,   0.0, 0.0, 0.0,      0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // H in water
            [0.0, 0.0, 0.0,   4.0, 5.0, 6.0,      7.0, 8.0, 9.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // H in water
            [0.0, 0.0, 0.0,   10.0, 11.0, 12.0,   13.0, 14.0, 15.0,   0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // H in CH
            [0.0, 0.0, 0.0,   0.0, 0.0, 0.0,      0.0, 0.0, 0.0,      16.0, 17.0, 18.0,  0.0, 0.0, 0.0],
            // C in CH
            [0.0, 0.0, 0.0,   0.0, 0.0, 0.0,      0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     19.0, 20.0, 21.0],
        ]);

        let gradients = descriptor.gradients.as_ref().unwrap();
        assert_eq!(gradients.shape(), [24, 15]);
        let gradients_indexes = descriptor.gradients_indexes.as_ref().unwrap();
        assert_eq!(gradients_indexes.names(), ["structure", "center", "neighbor", "spatial"]);

        let expected = [
            [v!(0), v!(0), v!(1)],
            [v!(0), v!(0), v!(2)],
            [v!(0), v!(1), v!(2)],
            [v!(0), v!(1), v!(0)],
            [v!(0), v!(2), v!(1)],
            [v!(0), v!(2), v!(0)],
            [v!(1), v!(0), v!(1)],
            [v!(1), v!(1), v!(0)]
        ];
        // use a loop to simplify checking the spatial dimension
        for (i, &value) in expected.iter().enumerate() {
            assert_eq!(gradients_indexes[3 * i][..3], value);
            assert_eq!(gradients_indexes[3 * i][3], v!(0));

            assert_eq!(gradients_indexes[3 * i + 1][..3], value);
            assert_eq!(gradients_indexes[3 * i + 1][3], v!(1));

            assert_eq!(gradients_indexes[3 * i + 2][..3], value);
            assert_eq!(gradients_indexes[3 * i + 2][3], v!(2));
        }

        assert_eq!(*gradients, array![
            /*    O-H                H-H                   H-O                H-C                C-H     */
            // O in water, 1rst H neighbor
            [1.0, 2.0, 3.0,      0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3,      0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0,   0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // O in water, 2nd H neighbor
            [4.0, 5.0, 6.0,      0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6,      0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [-4.0, -5.0, -6.0,   0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // 1rst H in water, H neighbor
            [0.0, 0.0, 0.0,      7.0, 8.0, 9.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,      0.7, 0.8, 0.9,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,    -7.0, -8.0, -9.0,    0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // 1rst H in water, O neighbor
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      10.0, 11.0, 12.0,   0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.10, 0.11, 0.12,   0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,    -10.0, -11.0, -12.0,  0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // 2nd H in water, H neighbor
            [0.0, 0.0, 0.0,   13.0, 14.0, 15.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,   0.13, 0.14, 0.15,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,  -13.0, -14.0, -15.0,   0.0, 0.0, 0.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // 2nd H in water, O neighbor
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,    16.0, 17.0, 18.0,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,    0.16, 0.17, 0.18,      0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,   -16.0, -17.0, -18.0,    0.0, 0.0, 0.0,     0.0, 0.0, 0.0],
            // H in CH
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,    19.0, 20.0, 21.0,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,    0.19, 0.20, 0.21,     0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,   -19.0, -20.0, -21.0,   0.0, 0.0, 0.0],
            // C in CH
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,    0.0, 0.0, 0.0,    22.0, 23.0, 24.0],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,    0.0, 0.0, 0.0,    0.22, 0.23, 0.24],
            [0.0, 0.0, 0.0,     0.0, 0.0, 0.0,      0.0, 0.0, 0.0,    0.0, 0.0, 0.0,   -22.0, -23.0, -24.0],
        ]);
    }
}
