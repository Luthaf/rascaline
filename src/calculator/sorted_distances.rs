use ndarray::{Array2, aview1, s};
use std::sync::{Arc, RwLock};

use super::Calculator;

use crate::descriptor::{Descriptor, Indexes, AtomIndexes};
use crate::system::System;
use crate::utils::AtomTypeMap;

#[derive(Debug, Clone)]
#[derive(serde::Deserialize)]
pub struct SortedDistances {
    cutoff: f64,
    max_neighbors: usize,
    padding: f64,
    #[serde(skip)]
    map: Arc<RwLock<AtomTypeMap>>,
}

pub struct SortedDistancesFeature {
    max_neighbors: usize,
    map: Arc<RwLock<AtomTypeMap>>,
    indexes: Vec<[usize; 2]>,
}

impl SortedDistancesFeature {
    pub fn new(max_neighbors: usize) -> SortedDistancesFeature {
        SortedDistancesFeature {
            max_neighbors: max_neighbors,
            map: Arc::default(),
            indexes: Vec::new(),
        }
    }

    pub fn map(&self) -> Arc<RwLock<AtomTypeMap>> {
        return Arc::clone(&self.map);
    }
}

impl Indexes for SortedDistancesFeature {
    fn names(&self) -> &[&'static str] {
        &["α", "neighbor"]
    }

    fn initialize(&mut self, systems: &[&dyn System]) {
        self.map.write().expect("poisonned lock").initialize(systems);
        for alpha in self.map.read().expect("poisonned lock").types() {
            for i in 0..self.max_neighbors {
                self.indexes.push([alpha, i]);
            }
        }
    }

    fn count(&self) -> usize {
        self.indexes.len()
    }

    fn gradient(&self) -> Option<Box<dyn Indexes>> {
        None
    }

    fn value(&self, linear: usize) -> &[usize] {
        &self.indexes[linear]
    }
}

impl Calculator for SortedDistances {
    fn name(&self) -> String {
        "sorted distances vector".into()
    }

    fn compute(&mut self, systems: &[&dyn System]) -> Descriptor {
        let features = SortedDistancesFeature::new(self.max_neighbors);
        self.map = features.map();
        let environment = AtomIndexes::new();

        let mut descriptor = Descriptor::new(environment, features, systems);
        let mut values = descriptor.values_mut();

        let mut start = 0;
        let mut stop;
        let map = self.map.read().expect("poisonned lock");
        for system in systems {
            let cell = system.cell();
            let types = system.types();
            let positions = system.positions();
            let nl = system.neighbors(self.cutoff);

            // Collect all distances in an array of dynamically sized vectors
            let shape = (system.natoms(), map.count());
            let mut distances = Array2::from_elem(shape, Vec::new());
            nl.foreach_pairs(&mut |i, j| {
                let r = cell.distance(&positions[i], &positions[j]);
                let ti = map.get(types[i]);
                let tj = map.get(types[j]);

                distances[[i, tj]].push(r);
                distances[[j, ti]].push(r);
            });

            // Sort, resize to limit to at most `self.max_neighbors` values
            // and pad if needed
            for vec in &mut distances {
                vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                vec.resize(self.max_neighbors, self.padding);
            }

            // Copy the data in the descriptor array
            stop = start + system.natoms();
            let data = values.slice_mut(s![start..stop, ..]);
            let shape = (system.natoms(), map.count(), self.max_neighbors);
            let mut data = data.into_shape(shape).expect("wrong dimensions");

            for i in 0..system.natoms() {
                for alpha in map.types() {
                    data.slice_mut(s![i, alpha, ..]).assign(&aview1(&distances[[i, alpha]]))
                }
            }
            start += stop;
        }

        return descriptor;
    }
}
