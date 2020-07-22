use ndarray::{Array2, aview0};

use crate::system::System;

mod indexes;
pub use self::indexes::{Indexes, IndexesBuilder};
pub use self::indexes::EnvironmentIndexes;
pub use self::indexes::{StructureIdx, AtomIdx};
pub use self::indexes::{StructureSpeciesIdx, PairSpeciesIdx};

pub struct Descriptor {
    /// An array of environments.count() by features.count() values
    pub values: Array2<f64>,
    pub environments: Indexes,
    pub features: Indexes,
    /// Gradients of the descriptor with respect to one atomic position
    pub gradients: Option<Array2<f64>>,
    pub gradients_indexes: Option<Indexes>,
}

fn resize_array(array: &mut Array2<f64>, shape: (usize, usize)) {
    // extract data by replacing array with a temporary value
    let mut tmp = Array2::zeros((0, 0));
    std::mem::swap(array, &mut tmp);

    let mut data = tmp.into_raw_vec();
    data.resize(shape.0 * shape.1, 0.0);

    // replace the temporary value with the updated array
    let values = Array2::from_shape_vec(shape, data).expect("wrong array shape");
    std::mem::replace(array, values);
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

    pub(crate) fn prepare(
        &mut self,
        environments: impl EnvironmentIndexes,
        features: Indexes,
        systems: &mut [&mut dyn System],
        initial: f64,
    ) {
        self.environments = environments.indexes(systems);
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.environments.count(), self.features.count());
        resize_array(&mut self.values, shape);
        self.values.assign(&aview0(&initial));

        self.gradients = None;
        self.gradients_indexes = None;
    }

    pub(crate) fn prepare_gradients(
        &mut self,
        environments: impl EnvironmentIndexes,
        features: Indexes,
        systems: &mut [&mut dyn System],
        initial: f64,
    ) {
        let (env_idx, grad_idx) = environments.with_gradients(systems);
        let grad_idx = grad_idx.expect(
            "the given environments indexes do not support gradients"
        );
        // basic sanity check
        assert_eq!(grad_idx.names().last(), Some(&"spatial"), "the last index of gradient should be spatial");

        self.environments = env_idx;
        self.features = features;

        // resize the 'values' array if needed, and set the requested initial value
        let shape = (self.environments.count(), self.features.count());
        resize_array(&mut self.values, shape);
        self.values.assign(&aview0(&initial));

        let shape = (grad_idx.count(), self.features.count());
        self.gradients_indexes = Some(grad_idx);

        if let Some(array) = &mut self.gradients {
            // resize the 'gradient' array if needed, and set the requested initial value
            resize_array(array, shape);
            array.assign(&aview0(&initial));
        } else {
            // create a new gradient array
            let array = Array2::from_elem(shape, initial);
            self.gradients = Some(array);
        }
    }
}
