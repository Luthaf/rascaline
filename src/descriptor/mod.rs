use ndarray::Array2;

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
    pub gradient: Option<Array2<f64>>,
    pub grad_envs: Option<Indexes>,
}

impl Descriptor {
    pub fn new(
        environments: impl EnvironmentIndexes,
        features: Indexes,
        systems: &mut [Box<dyn System>]
    ) -> Descriptor {
        let environments = environments.indexes(systems);
        let values = Array2::zeros((environments.count(), features.count()));

        return Descriptor {
            environments: environments,
            features: features,
            values: values,
            gradient: None,
            grad_envs: None,
        }
    }

    pub fn with_gradient(
        environments: impl EnvironmentIndexes,
        features: Indexes,
        systems: &mut [Box<dyn System>]
    ) -> Descriptor {
        let (env_idx, grad_idx) = environments.with_gradients(systems);
        let grad_idx = grad_idx.expect(
            "the given environments indexes do not support gradients"
        );
        // basic sanity check
        assert!(grad_idx.names()[0] == "spatial", "the first index of gradient should be spatial");

        let values = Array2::zeros((env_idx.count(), features.count()));
        let gradient = Array2::zeros((grad_idx.count(), features.count()));

        return Descriptor {
            environments: env_idx,
            features: features,
            values: values,
            gradient: Some(gradient),
            grad_envs: Some(grad_idx),
        }
    }
}
