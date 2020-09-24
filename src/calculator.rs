use std::collections::BTreeMap;

use crate::descriptor::{Descriptor, Indexes, IndexesBuilder};
use crate::system::System;
use crate::Error;

use crate::calculators::CalculatorBase;

pub struct Calculator {
    implementation: Box<dyn CalculatorBase>,
    parameters: String,
}

impl Calculator {
    pub fn new(name: &str, parameters: String) -> Result<Calculator, Error> {
        let creator = match REGISTERED_CALCULATORS.get(name) {
            Some(creator) => creator,
            None => {
                return Err(Error::InvalidParameter(
                    format!("unknwon calculator with name '{}'", name)
                ));
            }
        };

        return Ok(Calculator {
            implementation: creator(&parameters)?,
            parameters: parameters,
        })
    }

    /// Get the name associated with this Calculator
    pub fn name(&self) -> String {
        self.implementation.name()
    }

    /// Get the parameters used to create this Calculator in a string.
    ///
    /// Currently the string is formatted as JSON, but this could change in the
    /// future.
    pub fn parameters(&self) -> &str {
        &self.parameters
    }

    /// Compute the descriptor for all the given `systems` and store it in
    /// `descriptor`
    ///
    /// This function computes the full descriptor, using all samples and all
    /// features.
    pub fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        let features = self.implementation.features();
        let environments = self.implementation.environments();
        if self.implementation.compute_gradients() {
            let (environments, gradients) = environments.with_gradients(systems);
            let gradients = gradients.expect("this environments definition do not support gradients");
            descriptor.prepare_gradients(environments, gradients, features);
        } else {
            let environments = environments.indexes(systems);
            descriptor.prepare(environments, features);
        }

        self.implementation.compute(systems, descriptor);
    }

    /// Compute the descriptor only for the selected samples (all samples if
    /// `None`) and selected features (all features if `None`).
    pub fn compute_partial(
        &mut self,
        systems: &mut [&mut dyn System],
        descriptor: &mut Descriptor,
        samples: Option<Indexes>,
        features: Option<Indexes>
    ) {
        let features = if let Some(features) = features {
            self.implementation.check_features(&features);
            features
        } else {
            self.implementation.features()
        };

        let environments = self.implementation.environments();
        let (samples, gradients) = if let Some(samples) = samples {
            self.implementation.check_environments(&samples, systems);
            let gradients = environments.gradients_for(systems, &samples);
            (samples, gradients)
        } else {
            environments.with_gradients(systems)
        };

        if self.implementation.compute_gradients() {
            let gradients = gradients.expect("this environments definition do not support gradients");
            descriptor.prepare_gradients(samples, gradients, features);
        } else {
            descriptor.prepare(samples, features);
        }

        self.implementation.compute(systems, descriptor);
    }

    pub(crate) fn compute_partial_capi(
        &mut self,
        systems: &mut [&mut dyn System],
        descriptor: &mut Descriptor,
        samples: Option<&[usize]>,
        features: Option<&[usize]>
    ) -> Result<(), Error> {
        let samples = if let Some(list) = samples {
            // TODO: a simpler way to get names directly instead of building the
            // full indexes just to get names ...
            let environments = self.implementation.environments();
            let default = environments.indexes(systems);
            let mut builder = IndexesBuilder::new(default.names());

            if list.len() % builder.size() != 0 {
                return Err(Error::InvalidParameter(format!(
                    "wrong size for partial samples list, expected a multiple of {}, got {}",
                    builder.size(), list.len()
                )))
            }

            for chunk in list.chunks(builder.size()) {
                builder.add(chunk);
            }
            Some(builder.finish())
        } else {
            None
        };

        let features = if let Some(list) = features {
            // TODO: a simpler way to get names directly instead of building the
            // full indexes just to get names ...
            let default = self.implementation.features();
            let mut builder = IndexesBuilder::new(default.names());

            if list.len() % builder.size() != 0 {
                return Err(Error::InvalidParameter(format!(
                    "wrong size for partial features list, expected a multiple of {}, got {}",
                    builder.size(), list.len()
                )))
            }

            for chunk in list.chunks(builder.size()) {
                builder.add(chunk);
            }
            Some(builder.finish())
        } else {
            None
        };

        self.compute_partial(systems, descriptor, samples, features);

        return Ok(());
    }
}

/// Registration of calculator implementations
use crate::calculators::{DummyCalculator, SortedDistances};
type CalculatorCreator = fn(&str) -> Result<Box<dyn CalculatorBase>, Error>;

macro_rules! add_calculator {
    ($map :expr, $name :literal, $type :ty) => (
        $map.insert($name, (|json| {
            let value = serde_json::from_str::<$type>(json)?;
            Ok(Box::new(value))
        }) as CalculatorCreator);
    );
}

lazy_static::lazy_static!{
    pub static ref REGISTERED_CALCULATORS: BTreeMap<&'static str, CalculatorCreator> = {
        let mut map = BTreeMap::new();
        add_calculator!(map, "dummy_calculator", DummyCalculator);
        add_calculator!(map, "sorted_distances", SortedDistances);
        return map;
    };
}
