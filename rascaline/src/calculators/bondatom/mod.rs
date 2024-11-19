pub mod spherical_expansion_bondcentered;

mod bond_atom_math;
pub(crate) use bond_atom_math::canonical_vector_for_single_triplet;
use bond_atom_math::{RawSphericalExpansion,RawSphericalExpansionParameters,ExpansionContribution};

//pub use bondatom_neighbor_list::BANeighborList;
pub use spherical_expansion_bondcentered::{
    SphericalExpansionForBonds,
    SphericalExpansionForBondsParameters,
};



const FEATURE_GATE: &'static str = "RASCALINE_EXPERIMENTAL_BOND_ATOM_SPX";
fn get_feature_gate() -> bool {
    use std::env;
    if let Ok(var) = env::var(FEATURE_GATE) {
        if var.len() == 0 {
            false
        } else {
            let var = var.to_lowercase();
            !(&var=="0" || var == "false" || var == "no" || var == "off")
        }
    } else {
        false
    }
}
fn assert_feature_gate() {
    if !get_feature_gate() {
        if !get_feature_gate() {
            unimplemented!("Bond-Atom spherical expansion requires UNSTABLE feature gate: {}", FEATURE_GATE);
        }
    }
}
fn set_feature_gate() {
    use std::env;
    env::set_var(FEATURE_GATE, "true");
}