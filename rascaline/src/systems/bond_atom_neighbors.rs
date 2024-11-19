use std::cmp::PartialEq;
use std::collections::BTreeMap;

use crate::systems::Pair;
use crate::{Error, System, SystemBase};
use crate::types::Vector3D;


/// Sort a pair and return true if the pair was inverted
#[inline]
fn sort_pair<T: PartialOrd>((i, j): (T, T)) -> ((T, T), bool) {
    if i <= j {
        ((i, j), false)
    } else {
        ((j, i), true)
    }
}


/// This object is a simple representation of a bond-atom triplet (to represent a single neighbor atom to a bond environment)
/// it only makes sense for a given system
#[derive(Debug,Clone,Copy,PartialEq)]
pub struct BATripletInfo{
    /// number of the first atom (the bond's first atom) within the system
    pub atom_i: usize,
    /// number of the second atom (the bond's second atom) within the system
    pub atom_j: usize,
    /// number of the third atom (the neighbor atom) within the system
    pub atom_k: usize,
    /// how many cell boundaries are crossed by the vector between the triplet's first
    /// and second atoms
    pub bond_cell_shift: [i32;3],
    /// how many cell boundaries are crossed by the vector between the triplet's first
    /// and third atoms
    pub third_cell_shift: [i32;3],
    /// wether or not the third atom is the same as one of the first two (and NOT a periodic image thereof)
    pub is_self_contrib: bool,
    /// optional: the vector between first and second atom
    pub bond_vector: Vector3D,
    /// optional: the bector between the bond center and the third atom
    pub third_vector: Vector3D,
}


/// Manages a list of 'neighbors', where one neighbor is the center of a pair of atoms
/// (first and second atom), and the other neighbor is a simple atom (third atom).
/// Both the length of the bond and the distance between neighbors are subjected to a spherical cutoff.
/// This pre-calculator can compute and cache this list within a given system
/// (with two distance vectors per entry: one within the bond and one between neighbors).
/// Then, it can re-enumerate those neighbors, either for a full system, or with restrictions on the atoms or their types.
///
/// This saves memory/computational power by only working with "half" neighbor list
/// This is done by only including one entry for each `i - j` bond, not both `i - j` and `j - i`.
/// The order of i and j is that the atom with the smallest Z (or type ID in general) comes first.
///
/// The two first atoms must not be the same atom, but the third atom may be one of them.
/// (When periodic boundaries arise, the two first atoms may be images of each other.)
#[derive(Debug,Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct BATripletNeighborList {
    // /// Should we compute a full neighbor list (each pair appears twice, once as
    // /// `i-j` and once as `j-i`), or a half neighbor list (each pair only
    // /// appears once, (such that `types_i <= types_j`))
    // pub use_half_enumeration: bool,
    /// Spherical cutoffs to use to determine if two atoms are neighbors
    pub cutoffs: [f64;2],  // bond_, third_cutoff
}

/// the internal function doing the triplet computing itself
fn list_raw_triplets(system: &mut dyn SystemBase, bond_cutoff: f64, third_cutoff: f64) -> Result<Vec<BATripletInfo>,Error> {
    system.compute_neighbors(bond_cutoff)?;
    let bonds = system.pairs()?.to_owned();

    // atoms_cutoff needs to be a bit bigger than the one in the current
    // implementation to be sure we get the same set of neighbors.
    system.compute_neighbors(third_cutoff + bond_cutoff/2.)?;
    let types = system.types()?;

    let reorient_pair =  move |b: Pair| {
        if types[b.first] <= types[b.second] {
            b
        } else {
            Pair{
                first: b.second,
                second: b.first,
                distance: b.distance,
                vector: -b.vector,
                cell_shift_indices: b.cell_shift_indices,  // not corrected because irrelevant here
            }
        }
    };

    let mut ba_triplets = vec![];
    for bond in bonds.into_iter().map(reorient_pair) {
        let halfbond = 0.5 * bond.vector;

        // first, record the self contribution
        {
            let ((pairatom_i,pairatom_j),inverted) = sort_pair((bond.first,bond.second));
            let (halfbond, to_i, to_j) = if inverted {
                (-halfbond, bond.cell_shift_indices, [0;3])
            } else {
                (halfbond, [0;3], bond.cell_shift_indices)
            };

            let mut tri = BATripletInfo{
                atom_i: bond.first, atom_j: bond.second, atom_k: pairatom_i,
                bond_cell_shift: bond.cell_shift_indices,
                third_cell_shift: to_i,
                bond_vector: bond.vector,
                third_vector: -halfbond,
                is_self_contrib: true,
            };
            ba_triplets.push(tri.clone());
            tri.atom_k = pairatom_j;
            tri.third_vector = halfbond;
            tri.third_cell_shift = to_j;
            ba_triplets.push(tri);
        }


        // note: pairs_containing gives the pairs which have that center, as first OR second
        // but the full list of pairs only forms "an upper triangle" of pairs.
        for one_three in system.pairs_containing(bond.first)?.iter().map(|p|reorient_pair(p.clone())) {

            let is_self_contrib = {
                //ASSUMPTION: is_self_contrib means that bond and one_three are the exact same object
                (bond.vector-one_three.vector).norm2() <1E-5 &&
                bond.second == one_three.second
            };
            if is_self_contrib{
                //debug_assert_eq!(&bond as *const Pair, one_three as *const Pair);  // they come from different allocations lol
                debug_assert_eq!(
                    (bond.first, bond.second, bond.cell_shift_indices),
                    (one_three.first, one_three.second, one_three.cell_shift_indices),
                );
                continue;
            }

            // note: since we are looking for neighbors that can be distinguished (a pair of atoms and a lone atoms)
            // we need to ensure undo the anti-double-counting protections from system.pairs, when it comes to self-image pairs
            // hense, two separate if blocks rather than an if/else pair.
            if one_three.first == bond.first {
                let (third,third_vector, to_k) = (one_three.second, one_three.vector - halfbond, one_three.cell_shift_indices);
                if third_vector.norm2() < third_cutoff*third_cutoff {
                    let tri = BATripletInfo{
                        atom_i: bond.first, atom_j: bond.second, atom_k: third,
                        bond_cell_shift: bond.cell_shift_indices,
                        third_cell_shift: to_k,
                        bond_vector: bond.vector,
                        third_vector,
                        is_self_contrib: false,
                    };
                    ba_triplets.push(tri);
                }
            }
            if one_three.second == bond.first {
                let (third,third_vector, to_k) = (one_three.first, -one_three.vector - halfbond, one_three.cell_shift_indices.map(|f|-f));
                if third_vector.norm2() < third_cutoff*third_cutoff {
                    let tri = BATripletInfo{
                        atom_i: bond.first, atom_j: bond.second, atom_k: third,
                        bond_cell_shift: bond.cell_shift_indices,
                        third_cell_shift: to_k,
                        bond_vector: bond.vector,
                        third_vector,
                        is_self_contrib: false,
                    };
                    ba_triplets.push(tri);
                }
            };
        }
    }
    Ok(ba_triplets)
}

impl BATripletNeighborList {
    const CACHE_NAME_ATTR: &'static str = "bond_atom_triplets_cutoffs";
    const CACHE_NAME1: &'static str = "bond_atom_triplets_raw_list";
    const CACHE_NAME2: &'static str = "bond_atom_triplets_types_LUT";
    const CACHE_NAME3: &'static str = "bond_atom_triplets_center_LUT";
    //type CACHE_TYPE1 = TensorBlock;
    //type CACHE_TYPE2 = BTreeMap<(i32,i32,i32),Vec<usize>>;
    //type CACHE_TYPE3 = Vec<Vec<Vec<usize>>>;

    /// get the cutoff distance for the selection of bonds
    pub fn bond_cutoff(&self)-> f64 {
        self.cutoffs[0]
    }
    /// get the cutoff distance for neighbours to the center of a bond
    pub fn third_cutoff(&self)-> f64 {
        self.cutoffs[1]
    }

    /// validate that the cutoffs make sense
    pub fn validate_cutoffs(&self) {
        let (bond_cutoff, third_cutoff) = (self.bond_cutoff(), self.third_cutoff());
        assert!(bond_cutoff > 0.0 && bond_cutoff.is_finite());
        assert!(third_cutoff >= bond_cutoff && third_cutoff.is_finite());
    }

    /// internal function that deletages computing the triplets, but deals with storing them for a given system.
    fn do_compute_for_system(&self, system: &mut System) -> Result<(), Error> {
        let triplets = list_raw_triplets(&mut **system, self.cutoffs[0], self.cutoffs[1])?;

        let types = system.types()?;  // calling this again so the previous borrow expires
        let mut triplets_by_types = BTreeMap::new();
        let mut triplets_by_center = {
            let sz = system.size()?;
            (0..sz).map(|i|vec![vec![];i+1]).collect::<Vec<_>>()//vec![vec![vec![];sz];sz]
        };
        for (triplet_i, triplet) in triplets.iter().enumerate() {
            let ((s1,s2),_) = sort_pair((types[triplet.atom_i],types[triplet.atom_j]));
            triplets_by_types.entry((s1,s2,types[triplet.atom_k]))
                .or_insert_with(Vec::new)
                .push(triplet_i);
            if triplet.atom_i >= triplet.atom_j{
                triplets_by_center[triplet.atom_i][triplet.atom_j].push(triplet_i);
            } else {
                triplets_by_center[triplet.atom_j][triplet.atom_i].push(triplet_i);
            }
            // triplets_by_types.entry((types[triplet.bond.first],types[triplet.bond.second],types[triplet.third]))
            //     .or_insert_with(Vec::new)
            //     .push(triplet_i);
            // if self.use_half_enumeration  {
            //     if sort_pair((types[triplet.bond.first], types[triplet.bond.second])).1 {
            //         triplets_by_center[triplet.bond.second][triplet.bond.first].push(triplet_i);
            //     } else {
            //         triplets_by_center[triplet.bond.first][triplet.bond.second].push(triplet_i);
            //     }
            // } else {
            //     triplets_by_center[triplet.bond.first][triplet.bond.second].push(triplet_i);
            //     triplets_by_center[triplet.bond.second][triplet.bond.first].push(triplet_i);
            // }

        }
        system.store_data(Self::CACHE_NAME2.into(),triplets_by_types);
        system.store_data(Self::CACHE_NAME3.into(),triplets_by_center);
        system.store_data(Self::CACHE_NAME_ATTR.into(),self.cutoffs);
        system.store_data(Self::CACHE_NAME1.into(),triplets);
        Ok(())
    }

    /// check that the precalculator has computed its values for a given system,
    /// and if not, compute them.
    pub fn ensure_computed_for_system(&self, system: &mut System) -> Result<(),Error> {
        self.validate_cutoffs();
        'cached_path: {
            let cutoffs2: &[f64;2] = match system.data(Self::CACHE_NAME_ATTR.into()) {
                Some(cutoff) => cutoff.downcast_ref()
                    .ok_or_else(||Error::Internal("Failed to downcast cache".into()))?,
                None => break 'cached_path,
            };
            if cutoffs2 == &self.cutoffs {
                return Ok(());
            } else {
                break 'cached_path
            }
        }
        // got out of the 'cached' path: need to compute this ourselves
        return self.do_compute_for_system(system);
    }

    /// for a given system, get a reference to all the bond-atom triplets, vectors included
    pub fn get_for_system<'a>(&self, system: &'a System) -> Result<&'a [BATripletInfo], Error>{
        let triplets: &Vec<BATripletInfo> = system.data(&Self::CACHE_NAME1)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;
        Ok(triplets)
    }


    fn get_types_lut<'a>(&self, system: &'a System, s1:i32, s2:i32, s3:i32) -> Result<&'a [usize],Error> {
        let full_lut: &BTreeMap<(i32,i32,i32),Vec<usize>> = system.data(&Self::CACHE_NAME2)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;

        let ((s1,s2),_) = sort_pair((s1,s2));
        Ok(match full_lut.get(&(s1,s2,s3)) {
            None => &[],
            Some(lut) => &lut[..],
        })
    }
    fn get_centers_lut<'a>(&self, system: &'a System, c1:usize, c2:usize) -> Result<&'a [usize], Error> {
        {
            let sz = system.size()?;
            if c1 >= sz || c2 >= sz {
                return Err(Error::InvalidParameter("center ID too high for system".into()));
            }
        }
        let full_lut: &Vec<Vec<Vec<usize>>> = system.data(&Self::CACHE_NAME3)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;
        if c1 >= c2 {
            Ok(&full_lut[c1][c2])
        } else {
            Ok(&full_lut[c2][c1])
        }
    }

    /// for a given system, get a reference to the bond-atom triplets of given set of atomic types.
    /// note: inverting s1 and s2 does not change the result, and the returned triplets may have these types swapped
    pub fn get_per_system_per_type<'a>(
        &self, system: &'a System,
        s1:i32,s2:i32,s3:i32
    ) -> Result<impl Iterator<Item = &'a BATripletInfo> + 'a, Error> {
        let triplets = self.get_for_system(system)?;
        let types_lut = self.get_types_lut(system, s1, s2, s3)?;

        let res = types_lut.iter().map(|triplet_i|{
            triplets.get(*triplet_i).unwrap()
        });
        Ok(res)
    }

    /// for a given system, get a reference to the bond-atom triplets of given set of atomic types.
    /// note: the triplets may be for (c2,c1) rather than (c1,c2)
    pub fn get_per_system_per_center<'a>(
        &self, system: &'a System,
        c1:usize,c2:usize
    ) -> Result<impl Iterator<Item = &'a BATripletInfo> + 'a, Error>{
        let triplets = self.get_for_system(system)?;
        let centers_lut = self.get_centers_lut(system, c1, c2)?;

        let res = centers_lut.iter().map(|triplet_i|{
            triplets.get(*triplet_i).unwrap()
        });
        Ok(res)
    }

    /// for a given system, get a reference to the bond-atom triplets of given set of atomic types.
    /// plus the number of each triplet
    /// note: inverting s1 and s2 does not change the result, and the returned triplets may have these types swapped
    pub fn get_per_system_per_type_enumerated<'a>(
        &self, system: &'a System,
        s1:i32,s2:i32,s3:i32
    ) -> Result<impl Iterator<Item = (usize,&'a BATripletInfo)> + 'a, Error> {
        let triplets = self.get_for_system(system)?;
        let types_lut = self.get_types_lut(system, s1, s2, s3)?;

        let res = types_lut.iter().map(|triplet_i|{
            (*triplet_i,triplets.get(*triplet_i).unwrap())
        });
        Ok(res)
    }

    /// for a given system, get a reference to the bond-atom triplets of given set of atomic types.
    /// plus the number of each triplet
    /// note: the triplets may be for (c2,c1) rather than (c1,c2)
    pub fn get_per_system_per_center_enumerated<'a>(
        &self, system: &'a System,
        c1:usize,c2:usize
    ) -> Result<impl Iterator<Item = (usize,&'a BATripletInfo)> + 'a, Error>{
        let triplets = self.get_for_system(system)?;
        let centers_lut = self.get_centers_lut(system, c1, c2)?;

        let res = centers_lut.iter().map(|triplet_i|{
            (*triplet_i,triplets.get(*triplet_i).unwrap())
        });
        Ok(res)
    }

}



#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use crate::systems::test_utils::test_systems;
    //use crate::Matrix3;
    use super::*;


    fn no_vector(mut t: BATripletInfo) -> BATripletInfo {
        t.bond_vector =Vector3D::new(0.,0.,0.);
        t.third_vector =Vector3D::new(0.,0.,0.);
        t
    }
    fn gen_triplet(atom_i: usize, atom_j: usize, atom_k:usize, is_self_contrib:bool) -> BATripletInfo {
        BATripletInfo{
            atom_i,atom_j,atom_k,
            is_self_contrib,
            bond_vector: Vector3D::new(0.,0.,0.),
            third_vector: Vector3D::new(0.,0.,0.),
            bond_cell_shift: [0;3],
            third_cell_shift: [0;3],
        }
    }

    fn post_process_triplets<'a>(triplets: impl Iterator<Item=&'a BATripletInfo>) -> Vec<BATripletInfo>{
        // needed for now to avoid tripling the number of triplets to take into account
        triplets.filter(|v| v.bond_cell_shift == [0,0,0] && v.third_cell_shift == [0,0,0])
            .map(|v|no_vector(v.clone()))
            .collect::<Vec<_>>()
    }

    #[test]
    fn simple_enum() {
        let mut tsysv = test_systems(&["water"]);
        let precalc = BATripletNeighborList{
            cutoffs: [3.,3.],
        };
        precalc.ensure_computed_for_system(&mut tsysv[0]).unwrap();

        // /// ensure the enumeration is correct
        let triplets = post_process_triplets(precalc.get_for_system(&mut tsysv[0]).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,1,0, true,),
            gen_triplet(0,1,1, true,),
            gen_triplet(0,1,2, false,),
            gen_triplet(0,2,0, true,),
            gen_triplet(0,2,2, true,),
            gen_triplet(0,2,1, false,),
            gen_triplet(1,2,1, true,),
            gen_triplet(1,2,2, true,),
            gen_triplet(1,2,0, false,),
        ]);

        // /// ensure the per-center enumeration is correct
        let triplets = post_process_triplets(precalc.get_per_system_per_center(&mut tsysv[0], 0,1).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,1,0, true,),
            gen_triplet(0,1,1, true,),
            gen_triplet(0,1,2, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_center(&mut tsysv[0], 1,0).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,1,0, true,),
            gen_triplet(0,1,1, true,),
            gen_triplet(0,1,2, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_center(&mut tsysv[0], 0,2).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,2,0, true,),
            gen_triplet(0,2,2, true,),
            gen_triplet(0,2,1, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_center(&mut tsysv[0], 2,0).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,2,0, true,),
            gen_triplet(0,2,2, true,),
            gen_triplet(0,2,1, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_center(&mut tsysv[0], 1,2).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(1,2,1, true,),
            gen_triplet(1,2,2, true,),
            gen_triplet(1,2,0, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_center(&mut tsysv[0], 2,1).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(1,2,1, true,),
            gen_triplet(1,2,2, true,),
            gen_triplet(1,2,0, false,),
        ]);

        // /// ensure the per-type enumeration is correct
        let triplets = post_process_triplets(precalc.get_per_system_per_type(&mut tsysv[0], 1,1, -42).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(1,2,0, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_type(&mut tsysv[0], 1,1, 1).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(1,2,1, true,),
            gen_triplet(1,2,2, true,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_type(&mut tsysv[0], 1,-42, 1).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,1,1, true,),
            gen_triplet(0,1,2, false,),
            gen_triplet(0,2,2, true,),
            gen_triplet(0,2,1, false,),
        ]);
        let triplets = post_process_triplets(precalc.get_per_system_per_type(&mut tsysv[0], -42,1, 1).unwrap().into_iter());
        assert_eq!(triplets, vec![
            gen_triplet(0,1,1, true,),
            gen_triplet(0,1,2, false,),
            gen_triplet(0,2,2, true,),
            gen_triplet(0,2,1, false,),
        ]);

        // ///// deal with the vectors

        let triplets = precalc.get_for_system(&mut tsysv[0]).unwrap();
        let (bondvecs, thirdvecs): (Vec<_>,Vec<_>) = triplets.into_iter()
            // needed for now to avoid tripling the number of triplets to take into account
            .filter(|v| v.bond_cell_shift == [0,0,0] && v.third_cell_shift == [0,0,0])
            .map(|t|(t.bond_vector,t.third_vector))
            .unzip();

        bondvecs.into_iter().map(|v|(v[0],v[1],v[2]))
            .zip(vec![
                (0.0, 0.75545, -0.58895),
                (0.0, 0.75545, -0.58895),
                (0.0, 0.75545, -0.58895),
                (0.0, -0.75545, -0.58895),
                (0.0, -0.75545, -0.58895),
                (0.0, -0.75545, -0.58895),
                (0.0, -1.5109, 0.0),
                (0.0, -1.5109, 0.0),
                (0.0, -1.5109, 0.0),
            ].into_iter())
            .map(|(v1,v2)|{
                assert_ulps_eq!(v1.0,v2.0);
                assert_ulps_eq!(v1.1,v2.1);
                assert_ulps_eq!(v1.2,v2.2);
            }).last();

        thirdvecs.into_iter()
            .map(|v|(v[0],v[1],v[2]))
            .zip(vec![
                (0.0, -0.377725, 0.294475),
                (0.0, 0.377725, -0.294475),
                (0.0, -1.133175, -0.294475),
                (0.0, 0.377725, 0.294475),
                (0.0, -0.377725, -0.294475),
                (0.0, 1.133175, -0.294475),
                (0.0, 0.75545, 0.0),
                (0.0, -0.75545, 0.0),
                (0.0, 0.0, 0.58895),
            ].into_iter())
            .map(|(v1,v2)|{
                assert_ulps_eq!(v1.0,v2.0);
                assert_ulps_eq!(v1.1,v2.1);
                assert_ulps_eq!(v1.2,v2.2);
            }).last();
    }

    // #[test]
    // fn full_enum() {
    //     let mut tsysv = test_systems(&["water"]);
    //     let precalc = BATripletNeighborList{
    //         cutoffs: [6.,6.],
    //         use_half_enumeration: false,
    //     };

    //     precalc.ensure_computed_for_system(&mut tsysv[0]).unwrap();
    //     let triplets = precalc.get_for_system(&mut tsysv[0], false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);

    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 0,1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 1,0,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 0,2,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 2,0,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 1,2,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 2,1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);

    //     // /// ensure the per-type enumeration is correct
    //     let triplets = precalc.get_per_system_per_type(&mut tsysv[0], 1,1, -42,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_type(&mut tsysv[0], 1,1, 1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_type(&mut tsysv[0], 1,-42, 1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_type(&mut tsysv[0], -42,1, 1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    // }
}
