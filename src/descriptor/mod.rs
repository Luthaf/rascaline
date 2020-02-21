use crate::system::System;

use ndarray::Array2;

mod indexes;
pub use self::indexes::Indexes;
pub use self::indexes::StructureIndexes;
pub use self::indexes::AtomIndexes;
pub use self::indexes::PairIndexes;

pub struct Descriptor {
    /// An array of environments.count() by features.count() values
    values: Array2<f64>,
    /// An array of env_grad.count() by features.count() values
    gradient: Option<Array2<f64>>,

    environments: Box<dyn Indexes>,
    gradient_id: Option<Box<dyn Indexes>>,
    features: Box<dyn Indexes>,
}

impl Descriptor {
    pub fn new(
        mut environments: impl Indexes + 'static,
        mut features: impl Indexes + 'static,
        systems: &[&dyn System]
    ) -> Descriptor {
        environments.initialize(systems);
        features.initialize(systems);

        return Descriptor {
            values: Array2::zeros((environments.count(), features.count())),
            environments: Box::new(environments),
            features: Box::new(features),
            gradient: None,
            gradient_id: None,
        }
    }

    pub fn with_gradient(
        mut environments: impl Indexes + 'static,
        mut features: impl Indexes + 'static,
        systems: &[&dyn System]
    ) -> Descriptor {
        environments.initialize(systems);
        features.initialize(systems);

        let env_grad = environments.gradient().expect(
            "invalid environments indexes without gradient passed to Descriptor::with_gradient"
        );

        let env_count = environments.count();
        let grad_count = env_grad.count();
        let features_count = features.count();

        return Descriptor {
            values: Array2::zeros((env_count, features_count)),
            gradient: Some(Array2::zeros((grad_count, features_count))),
            environments: Box::new(environments),
            features: Box::new(features),
            gradient_id: Some(env_grad),
        }
    }

    pub fn environments(&self) -> &dyn Indexes {
        &*self.environments
    }

    pub fn features(&self) -> &dyn Indexes {
        &*self.features
    }

    pub fn gradient_indexes(&self) -> Option<&dyn Indexes> {
        self.gradient_id.as_ref().map(|g| &**g)
    }

    pub fn values(&self) -> ndarray::ArrayView2<f64> {
        self.values.view()
    }

    pub fn values_mut(&mut self) -> ndarray::ArrayViewMut2<f64> {
        self.values.view_mut()
    }

    pub fn gradient(&self) -> Option<ndarray::ArrayView2<f64>> {
        self.gradient.as_ref().map(|g| g.view())
    }

    pub fn gradient_mut(&mut self) -> Option<ndarray::ArrayViewMut2<f64>> {
        self.gradient.as_mut().map(|g| g.view_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo() {
        unimplemented!()
    }
}
