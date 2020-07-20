use super::Calculator;

use crate::descriptor::{Descriptor, IndexesBuilder, AtomIdx};
use crate::system::System;

#[derive(Debug, Clone)]
#[derive(serde::Deserialize)]
pub struct DummyCalculator {
    cutoff: f64,
    delta: usize,
    name: String,
    gradients: bool,
}

impl Calculator for DummyCalculator {
    fn name(&self) -> String {
        // abusing the name as description
        format!("dummy test calculator with cutoff: {} - delta: {} - name: {} - gradients: {}",
            self.cutoff, self.delta, self.name, self.gradients
        )
    }

    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        let mut features = IndexesBuilder::new(vec!["index + delta", "x + y + z"]);
        features.add(&[1, 0]);
        features.add(&[0, 1]);
        let features = features.finish();

        let environments = AtomIdx::new(3.0);
        if self.gradients {
            descriptor.prepare_gradients(environments, features, systems, 0.0);
        } else {
            descriptor.prepare(environments, features, systems, 0.0);
        }

        for (i, indexes) in descriptor.environments.iter().enumerate() {
            if let &[structure, atom] = indexes {
                descriptor.values[[i, 0]] = (atom + self.delta) as f64;
                let position = systems[structure].positions()[atom];
                descriptor.values[[i, 1]] = position[0] + position[1] + position[2];
            }
        }

        if self.gradients {
            unimplemented!();
        }
    }
}
