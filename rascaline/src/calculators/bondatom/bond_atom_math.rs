use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::cell::RefCell;
use thread_local::ThreadLocal;
use log::warn;

use metatensor::TensorBlockRefMut;

use crate::Error;
use crate::types::{Vector3D,Matrix3};
use crate::systems::BATripletInfo;

use crate::calculators::soap::{CutoffFunction, RadialScaling};
use crate::calculators::radial_basis::RadialBasis;
use crate::calculators::soap::{
    SoapRadialIntegralCache,
    SoapRadialIntegralParameters,
};
use crate::math::SphericalHarmonicsCache;


/// for a given vector (`vec`), compute a rotation matrix (`M`) so that `M×vec`
/// is expressed as `(0,0,+z)`
/// currently, this matrix corresponds to a rotatoin expressed as `-z;+y;+z` in euler angles,
/// or as `(x,y,0),theta` in axis-angle representation.
fn rotate_vector_to_z(vec: Vector3D) -> Matrix3 {
    // re-orientation is done through a rotation matrix, computed through the axis-angle and quaternion representations of the rotation
    // axis/angle representation of the rotation: axis is norm(-y,x,0), angle is arctan2( sqrt(x**2+y**2), z)
    // meaning sin(angle) = sqrt((x**2+y**2) /r2); cos(angle) = z/sqrt(r2)

    let (xylen,len) = {
        let xyl = vec[0]*vec[0] + vec[1]*vec[1];
        (xyl.sqrt(), (xyl+vec[2]*vec[2]).sqrt())
    };

    if xylen.abs()<1E-7 {
        if vec[2] < 0. {
            return Matrix3::new([[-1.,0.,0.], [0.,1.,0.], [0.,0.,-1.]])
        }
        else {
            return Matrix3::new([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
        }
    }

    let c = vec[2]/len;
    let s = xylen/len;
    let t = 1. - c;

    let x2 = -vec[1]/xylen;
    let y2 =  vec[0]/xylen;

    let tx = t*x2;
    let sx = s*x2;
    let sy = s*y2;

    return Matrix3::new([
        [tx*x2 +c,  tx*y2,       -sy],
        [tx*y2,     t*y2*y2 + c, sx],
        [sy,       -sx,          c],
    ]);
}


/// returns the derivatives of the reoriention matrix with the three components of the vector to reorient
fn rotate_vector_to_z_derivatives(vec: Vector3D) -> (Matrix3,Matrix3,Matrix3) {

    let (xylen,len) = {
        let xyl = vec[0]*vec[0] + vec[1]*vec[1];
        (xyl.sqrt(), (xyl+vec[2]*vec[2]).sqrt())
    };

    if xylen.abs()<1E-7 {
        let co = 1./len;
        if vec[2] < 0. {
            warn!("trying to get the derivative of a rotation near a breaking point: expect pure jank");
            return (
                //Matrix3::new([[-1.,0.,0.], [0.,1.,0.], [0.,0.,-1.]]) <- the value to derive off of: a +y rotation
                Matrix3::new([[0.,0.,-co], [0.,0.,0.], [co,0.,0.]]),  // +x change -> +y rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,-co], [0.,-co,0.]]),  // +y change -> -x rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]]),  // +z change -> nuthin
            )
        }
        else {
            return (
                //Matrix3::new([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])  <- the value to derive off of
                Matrix3::new([[0.,0.,-co], [0.,0.,0.], [co,0.,0.]]),  // +x change -> -y rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,-co], [0.,co,0.]]),  // +y change -> +x rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]]),  // +z change -> nuthin
            )
        }
    }

    let inv_len = 1./len;
    let inv_len2 = inv_len*inv_len;
    let inv_len3 = inv_len2*inv_len;
    let inv_xy = 1./xylen;
    let inv_xy2 = inv_xy*inv_xy;
    let inv_xy3 = inv_xy2*inv_xy;

    let c = vec[2]/len;  // needed
    let dcdz = 1./len - vec[2]*vec[2]*inv_len3;
    let dcdx = -vec[2]*vec[0]*inv_len3;
    let dcdy = -vec[2]*vec[1]*inv_len3;
    let s = xylen/len;
    let dsdx = vec[0]*inv_len*(inv_xy - xylen*inv_len2);
    let dsdy = vec[1]*inv_len*(inv_xy - xylen*inv_len2);
    let dsdz = -xylen*vec[2]*inv_len3;

    let t = 1. - c;

    let x2 = -vec[1]*inv_xy;
    let dx2dx = vec[1]*vec[0]*inv_xy3;
    let dx2dy = inv_xy * (-1. + vec[1]*vec[1]*inv_xy2);

    let y2 =  vec[0]/xylen;
    let dy2dy = -vec[1]*vec[0]*inv_xy3;
    let dy2dx = inv_xy * (1. - vec[0]*vec[0]*inv_xy2);

    let tx = t*x2;
    let dtxdx = -dcdx*x2 + t*dx2dx;
    let dtxdy = -dcdy*x2 + t*dx2dy;
    let dtxdz = -dcdz*x2;

    //let sx = s*x2;  // needed
    let dsxdx = dsdx*x2 + s*dx2dx;
    let dsxdy = dsdy*x2 + s*dx2dy;
    let dsxdz = dsdz*x2;

    //let sy = s*y2;  //needed
    let dsydx = dsdx*y2 + s*dy2dx;
    let dsydy = dsdy*y2 + s*dy2dy;
    let dsydz = dsdz*y2;

    //let t1 = tx*x2 +c;  // needed
    let dt1dx = dcdx + dtxdx*x2 + tx*dx2dx;
    let dt1dy = dcdy + dtxdy*x2 + tx*dx2dy;
    let dt1dz = dcdz + dtxdz*x2;

    //let t2 = tx*y2;  // needed
    let dt2dx = dtxdx*y2 + tx*dy2dx;
    let dt2dy = dtxdy*y2 + tx*dy2dy;
    let dt2dz = dtxdz*y2;

    //let t3 = t*y2*y2 +c;  // needed
    let dt3dx = -dcdx*y2*y2 + 2.*t*y2*dy2dx +dcdx;
    let dt3dy = -dcdy*y2*y2 + 2.*t*y2*dy2dy +dcdy;
    let dt3dz = -dcdz*y2*y2 +dcdz;

    return (
        // Matrix3::new([
        //     [tx*x2 +c,  tx*y2,       -sy],
        //     [tx*y2,     t*y2*y2 + c, sx],
        //     [sy,       -sx,          c],
        // ]),
        Matrix3::new([
            [dt1dx,  dt2dx, -dsydx],
            [dt2dx,  dt3dx,  dsxdx],
            [dsydx, -dsxdx,  dcdx],
        ]),
        Matrix3::new([
            [dt1dy,  dt2dy, -dsydy],
            [dt2dy,  dt3dy,  dsxdy],
            [dsydy, -dsxdy,  dcdy],
        ]),
        Matrix3::new([
            [dt1dz,  dt2dz, -dsydz],
            [dt2dz,  dt3dz,  dsxdz],
            [dsydz, -dsxdz,  dcdz],
        ]),
    );
}

/// result structure for canonical_vector_for_single_triplet
#[derive(Default,Debug)]
pub(crate) struct VectorResult{
    /// the canonical vector itelf
    pub vect: Vector3D,
    /// gradients of the canonical vector, as an array of three matrices
    /// matrix is [quantity_component,gradient_component]
    /// each matrix corresponds to a different atom to gradiate upon
    pub grads: [Option<(usize,Matrix3)>;3],
}

/// From a list of bond/atom triplets, compute the 'canonical third vector'.
/// Each triplet is composed of two 'neighbors': one is the center of a pair of atoms
/// (first and second atom), and the other is a simple atom (third atom).
/// The third vector of such a triplet is the vector from the center of the atom pair and to the third atom.
/// this third vector becomes canonical when the frame of reference is rotated to express
/// the triplet's bond vector (vector from the first and to the second atom) as (0,0,+z).
///
/// Users can request either a "full" neighbor list (including an entry for both
/// `i-j +k` triplets and `j-i +k` triplets) or save memory/computational by only
/// working with "half" neighbor list (only including one entry for each `i-j +k`
/// bond)
/// When using a half neighbor list, i and j are ordered so the atom with the smallest species comes first.
///
/// The two first atoms must not be the same atom, but the third atom may be one of them,
/// if the `bond_conbtribution` option is active
/// (When periodic boundaries arise, atom which  must not be the same may be images of each other.)
///
/// This sample produces a single property (`"distance"`) with three components
/// (`"vector_direction"`) containing the x, y, and z component of the vector from
/// the center of the triplet's 'bond' to the triplet's 'third atom', in the bond's canonical orientation.
///
/// In addition to the atom indexes, the samples also contain a pair and triplet index,
/// to be able to distinguish between multiple triplets involving the same atoms
/// (which can occur in periodic boundary conditions when the cutoffs are larger than the unit cell).
pub(crate) fn canonical_vector_for_single_triplet(
    triplet: &BATripletInfo,
    invert: bool,
    compute_grad: bool,
    mtx_cache: &mut BTreeMap<(usize,usize,[i32;3],bool),Matrix3>,
    dmtx_cache: &mut BTreeMap<(usize,usize,[i32;3],bool),(Matrix3,Matrix3,Matrix3)>,
) -> Result<VectorResult,Error> {

    let bond_vector = triplet.bond_vector;
    let third_vector = triplet.third_vector;
    let (atom_i,atom_j,bond_vector) = if invert {
        (triplet.atom_j, triplet.atom_i, -bond_vector)
    } else {
        (triplet.atom_i, triplet.atom_j, bond_vector)
    };

    let mut res = VectorResult::default();

    if triplet.is_self_contrib {
        let vec_len = third_vector.norm();
        let vec_len = if third_vector * bond_vector > 0. {
            // third atom on second atom
            vec_len
        } else {
            // third atom on first atom
            -vec_len
        };
        res.vect[2] = vec_len;

        if compute_grad {
            let inv_len = 1./vec_len;

            res.grads[0] = Some((atom_i,Matrix3::new([
                [ -0.25* inv_len * third_vector[0], 0., 0.],
                [ 0., -0.25* inv_len * third_vector[0], 0.],
                [ 0., 0., -0.25* inv_len * third_vector[0]],
            ])));
            res.grads[1] = Some((atom_j,Matrix3::new([
                [ 0.25* inv_len * third_vector[0], 0., 0.],
                [ 0., 0.25* inv_len * third_vector[0], 0.],
                [ 0., 0., 0.25* inv_len * third_vector[0]],
            ])));

        }
    } else {

        let tf_mtx = match mtx_cache.entry((triplet.atom_i,triplet.atom_j,triplet.bond_cell_shift,invert)) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                entry.insert(rotate_vector_to_z(bond_vector)).clone()
            },
        };
        res.vect = tf_mtx * third_vector;

        if compute_grad {

            // for a transformed vector v from an untransformed vector u,
            // dv = TF*du + dTF*u
            // also: the indexing of the gradient array is: i_gradsample, derivation_component, value_component, i_property

            let du_term = -0.5* tf_mtx;
            let (tf_mtx_dx, tf_mtx_dy, tf_mtx_dz) = match dmtx_cache.entry((triplet.atom_i,triplet.atom_j,triplet.bond_cell_shift,invert)) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    entry.insert(rotate_vector_to_z_derivatives(bond_vector)).clone()
                },
            };

            let dmat_term_dx = tf_mtx_dx * third_vector;
            let dmat_term_dy = tf_mtx_dy * third_vector;
            let dmat_term_dz = tf_mtx_dz * third_vector;

            res.grads[0] = Some((atom_i,Matrix3::new([
                [-dmat_term_dx[0] + du_term[0][0], -dmat_term_dy[0] + du_term[0][1], -dmat_term_dz[0] + du_term[0][2]],
                [-dmat_term_dx[1] + du_term[1][0], -dmat_term_dy[1] + du_term[1][1], -dmat_term_dz[1] + du_term[1][2]],
                [-dmat_term_dx[2] + du_term[2][0], -dmat_term_dy[2] + du_term[2][1], -dmat_term_dz[2] + du_term[2][2]],
            ])));
            res.grads[1] = Some((atom_j,Matrix3::new([
                [dmat_term_dx[0] + du_term[0][0], dmat_term_dy[0] + du_term[0][1], dmat_term_dz[0] + du_term[0][2]],
                [dmat_term_dx[1] + du_term[1][0], dmat_term_dy[1] + du_term[1][1], dmat_term_dz[1] + du_term[1][2]],
                [dmat_term_dx[2] + du_term[2][0], dmat_term_dy[2] + du_term[2][1], dmat_term_dz[2] + du_term[2][2]],
            ])));
            res.grads[2] = Some((triplet.atom_k,tf_mtx));
        }
    }
    return Ok(res);
}

/// get the result of canonical_vector_for_single_triplet
/// and store it in a TensorBlock
pub(crate) fn canonical_vector_for_single_triplet_inplace(
    triplet: &BATripletInfo,
    out_block: &mut TensorBlockRefMut,
    sample_i: usize,
    system_i: usize,
    invert: bool,
    mtx_cache: &mut BTreeMap<(usize,usize,[i32;3],bool),Matrix3>,
    dmtx_cache: &mut BTreeMap<(usize,usize,[i32;3],bool),(Matrix3,Matrix3,Matrix3)>,
) -> Result<(),Error> {
    let compute_grad = out_block.gradient_mut("positions").is_some();
    let block_data = out_block.data_mut();
    let array = block_data.values.to_array_mut();

    let res = canonical_vector_for_single_triplet(
        triplet,
        invert,
        compute_grad,
        mtx_cache,
        dmtx_cache
    )?;

    array[[sample_i, 0, 0]] = res.vect[0];
    array[[sample_i, 1, 0]] = res.vect[1];
    array[[sample_i, 2, 0]] = res.vect[2];

    if let Some(mut gradient) = out_block.gradient_mut("positions") {
        let gradient = gradient.data_mut();
        let array = gradient.values.to_array_mut();

        for grad in res.grads {
            if let Some((atom_i, grad_mtx)) = grad {
                let grad_sample_i = gradient.samples.position(&[
                    sample_i.into(), system_i.into(), atom_i.into()
                ]).expect("missing gradient sample");

                array[[grad_sample_i, 0, 0, 0]] = grad_mtx[0][0];
                array[[grad_sample_i, 1, 0, 0]] = grad_mtx[0][1];
                array[[grad_sample_i, 2, 0, 0]] = grad_mtx[0][2];
                array[[grad_sample_i, 0, 1, 0]] = grad_mtx[1][0];
                array[[grad_sample_i, 1, 1, 0]] = grad_mtx[1][1];
                array[[grad_sample_i, 2, 1, 0]] = grad_mtx[1][2];
                array[[grad_sample_i, 0, 2, 0]] = grad_mtx[2][0];
                array[[grad_sample_i, 1, 2, 0]] = grad_mtx[2][1];
                array[[grad_sample_i, 2, 2, 0]] = grad_mtx[2][2];
            }
        }
    }
    Ok(())
}


/// Contribution of a single triplet to the spherical expansion
pub(super) struct ExpansionContribution {
    /// Values of the contribution. The shape is (lm, n), where the lm index
    /// runs over both l and m
    pub values: ndarray::Array2<f64>,
    /// Gradients of the contribution w.r.t. the canonical vector of the triplet.
    /// The shape is (x/y/z, lm, n).
    pub gradients: Option<ndarray::Array3<f64>>,
}

impl ExpansionContribution {
    pub fn new(max_radial: usize, max_angular: usize, do_gradients: bool) -> Self {
        let lm_shape = (max_angular + 1) * (max_angular + 1);
        Self {
            values: ndarray::Array2::from_elem((lm_shape, max_radial), 0.0),
            gradients: if do_gradients {
                Some(ndarray::Array3::from_elem((3, lm_shape, max_radial), 0.0))
            } else {
                None
            }
        }
    }
}

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub(super) struct RawSphericalExpansionParameters {
    /// the cutoff beyond which we neglect neighbors (often called third cutoff)
    pub cutoff: f64,
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density
    pub atomic_gaussian_width: f64,

    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
    /// radial scaling can be used to reduce the importance of neighbor atoms
    /// further away from the center, usually improving the performance of the
    /// model
    pub radial_scaling: RadialScaling,
}

pub(super) struct RawSphericalExpansion{
    parameters: RawSphericalExpansionParameters,
    /// implementation + cached allocation to compute the radial integral for a
    /// single pair
    radial_integral: ThreadLocal<RefCell<SoapRadialIntegralCache>>,
    /// implementation + cached allocation to compute the spherical harmonics
    /// for a single pair
    spherical_harmonics: ThreadLocal<RefCell<SphericalHarmonicsCache>>,
}

impl RawSphericalExpansion {
    pub(super) fn new(parameters: RawSphericalExpansionParameters) -> Self {
        Self{
            parameters,
            radial_integral: ThreadLocal::new(),
            spherical_harmonics: ThreadLocal::new(),
        }
    }

    pub(super) fn parameters(&self) -> &RawSphericalExpansionParameters {
        &self.parameters
    }

    pub(super) fn make_contribution_buffer(&self, do_gradients: bool) -> ExpansionContribution {
        ExpansionContribution::new(
            self.parameters.max_radial,
            self.parameters.max_angular,
            do_gradients,
        )
    }

    /// Compute the product of radial scaling & cutoff smoothing functions
    pub(crate) fn scaling_functions(&self, r: f64) -> f64 {
        let cutoff = self.parameters.cutoff_function.compute(r, self.parameters.cutoff);
        let scaling = self.parameters.radial_scaling.compute(r);
        return cutoff * scaling;
    }

    /// Compute the gradient of the product of radial scaling & cutoff smoothing functions
    pub(crate) fn scaling_functions_gradient(&self, r: f64) -> f64 {
        let cutoff = self.parameters.cutoff_function.compute(r, self.parameters.cutoff);
        let cutoff_grad = self.parameters.cutoff_function.derivative(r, self.parameters.cutoff);

        let scaling = self.parameters.radial_scaling.compute(r);
        let scaling_grad = self.parameters.radial_scaling.derivative(r);

        return cutoff_grad * scaling + cutoff * scaling_grad;
    }


    /// compute the spherical expansion coefficients associated with
    /// a single center->neighbor vector (`vector`),
    /// and store it in a ExpansionContribution buffer (`contribution`).
    /// `gradient_orientation` serves two purposes:
    ///    it tells whether or not the gradient should be computed,
    ///    and it deals with the case where the vector (and the spherical expansion) take place
    ///    in a rotated/scaled/sheared frame of reference: its three vectors contain
    ///    the changes of the vector (in the rotated frame of reference) when adding the
    ///    +x, +y, and +z vectors from the 'real' frame of reference.
    /// `extra_scaling` simply applies a scaling factor to all coefficients
    pub(super) fn compute_coefficients(&self, contribution: &mut ExpansionContribution, vector: Vector3D, extra_scaling: f64, gradient_orientation: Option<(Vector3D,Vector3D,Vector3D)>){
        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = SoapRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
                SoapRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                }
            ).expect("invalid radial integral parameters");
            return RefCell::new(radial_integral);
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsCache::new(self.parameters.max_angular))
        }).borrow_mut();

        let distance = vector.norm();
        let direction = vector/distance;
        // Compute the three factors that appear in the center contribution.
        // Note that this is simply the pair contribution for the special
        // case where the pair distance is zero.
        radial_integral.compute(distance, gradient_orientation.is_some());
        spherical_harmonics.compute(direction, gradient_orientation.is_some());

        let f_scaling = self.scaling_functions(distance) * extra_scaling;

        let (values, gradient_values_o) = (&mut contribution.values, contribution.gradients.as_mut());

        debug_assert_eq!(
            values.shape(),
            [(self.parameters.max_angular+1)*(self.parameters.max_angular+1), self.parameters.max_radial]
        );
        for l in 0..=self.parameters.max_angular {
            let l_offset = l*l;
            let msize = 2*l+1;
            //values.slice_mut(s![l_offset..l_offset+msize, ..]) *= spherical_harmonics.values.slice(l);
            for m in 0..msize {
                let lm = l_offset+m;
                for n in 0..self.parameters.max_radial {
                    values[[lm, n]] = spherical_harmonics.values[lm]
                                     * radial_integral.values[[l,n]]
                                     * f_scaling;
                }
            }
        }

        if let Some((dvdx,dvdy,dvdz)) = gradient_orientation {
            unimplemented!("ööps, gradient not ready yet");
            let gradient_values = gradient_values_o.unwrap();

            let ilen = 1./distance;
            let dlendv = vector*ilen;
            let dlendx = dlendv*dvdx;
            let dlendy = dlendv*dvdy;
            let dlendz = dlendv*dvdz;
            let ddirdx = dvdx*ilen - vector*dlendx*ilen*ilen;
            let ddirdy = dvdy*ilen - vector*dlendy*ilen*ilen;
            let ddirdz = dvdy*ilen - vector*dlendz*ilen*ilen;

            let single_grad = |l,n,m,dlenda,ddirda: Vector3D| {
                f_scaling * (
                    radial_integral.gradients[[l,n]]*dlenda*spherical_harmonics.values[[l as isize,m as isize]]
                    + radial_integral.values[[l,n]]*(
                        spherical_harmonics.gradients[0][[l as isize,m as isize]]*ddirda[0]
                       +spherical_harmonics.gradients[1][[l as isize,m as isize]]*ddirda[1]
                       +spherical_harmonics.gradients[2][[l as isize,m as isize]]*ddirda[2]
                    )
                    // TODO scaling_function_gradient
                )
            };

            for l in 0..=self.parameters.max_angular {
                let l_offset = l*l;
                let msize = 2*l+1;
                for m in 0..(msize) {
                    let lm = l_offset+m;
                    for n in 0..self.parameters.max_radial {
                        gradient_values[[0,lm,n]] = single_grad(l,n,m,dlendx,ddirdx);
                        gradient_values[[1,lm,n]] = single_grad(l,n,m,dlendy,ddirdy);
                        gradient_values[[2,lm,n]] = single_grad(l,n,m,dlendz,ddirdz);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use super::Vector3D;

    use approx::assert_relative_eq;

    use crate::systems::BATripletNeighborList;
    use crate::systems::test_utils::test_systems;
    use super::{RawSphericalExpansion,RawSphericalExpansionParameters};
    use super::canonical_vector_for_single_triplet;
    //use super::super::CalculatorBase;

    #[test]
    fn half_neighbor_list() {
        let pre_calculator = BATripletNeighborList{
            cutoffs: [2.0,2.0],
        };

        let mut systems = test_systems(&["water"]);
        let expected = &[[
            [0.0, 0.0, -0.478948537162397],  // SC 0 1 0
            [0.0, 0.0,  0.478948537162397],  // SC 0 1 1
            [0.0, 0.9289563, -0.7126298],  // 0 1 2
            [0.0, 0.0, -0.478948537162397], // SC 1 0 1
            [0.0, 0.0,  0.478948537162397], // SC 1 0 0
            [0.0, -0.9289563, -0.7126298], // 1 0 2
            [0.0, 0.0, -0.75545],  // SC 1 2 1
            [0.0, 0.0,  0.75545],  // SC 1 2 2
            [0.0, 0.58895, 0.0],  // 1 2 0
        ]];
        for (system,expected) in systems.iter_mut().zip(expected) {
            let mut mtx_cache = BTreeMap::new();
            let mut dmtx_cache = BTreeMap::new();
            pre_calculator.ensure_computed_for_system(system).unwrap();
            let triplets = pre_calculator.get_for_system(system).unwrap()
                .into_iter().filter(|v| v.bond_cell_shift == [0,0,0] && v.third_cell_shift == [0,0,0]);
            for (expected,triplet) in expected.iter().zip(triplets) {
                let res = canonical_vector_for_single_triplet(&triplet, false, false, &mut mtx_cache, &mut dmtx_cache).unwrap();
                assert_relative_eq!(res.vect, Vector3D::new(expected[0],expected[1],expected[2]), max_relative=1e-6);
            }
        }
    }

    #[test]
    fn full_neighbor_list() {
        let pre_calculator = BATripletNeighborList{
            cutoffs: [2.0,2.0],
        };

        let mut systems = test_systems(&["water"]);
        let expected = &[[
            [0.0, 0.0,  0.478948537162397],  // SC 0 1 0
            [0.0, 0.0, -0.478948537162397],  // SC 0 1 1
            [0.0, -0.9289563, 0.7126298],  // 0 1 2
            [0.0, 0.0,  0.478948537162397], // SC 1 0 1
            [0.0, 0.0, -0.478948537162397], // SC 1 0 0
            [0.0, 0.9289563, 0.7126298], // 1 0 2
            [0.0, 0.0,  0.75545],  // SC 1 2 1
            [0.0, 0.0, -0.75545],  // SC 1 2 2
            [0.0, -0.58895, 0.0],  // 1 2 0
        ]];
        for (system,expected) in systems.iter_mut().zip(expected) {
            let mut mtx_cache = BTreeMap::new();
            let mut dmtx_cache = BTreeMap::new();
            pre_calculator.ensure_computed_for_system(system).unwrap();
            let triplets = pre_calculator.get_for_system(system).unwrap()
                .into_iter().filter(|v| v.bond_cell_shift == [0,0,0] && v.third_cell_shift == [0,0,0]);
            for (expected,triplet) in expected.iter().zip(triplets) {
                let res = canonical_vector_for_single_triplet(&triplet, true, false, &mut mtx_cache, &mut dmtx_cache).unwrap();
                assert_relative_eq!(res.vect, Vector3D::new(expected[0],expected[1],expected[2]), max_relative=1e-6);
            }
        }
    }

    // note: the following test does pass, but gradients are disabled because we discovered that
    // the values of this calculator ARE NOT CONTINUOUS around the values of bond_vector == (0,0,-z)
    // ////
    // #[test]
    // fn finite_differences_positions() {
    //     // half neighbor list
    //     let calculator = Calculator::from(Box::new(BANeighborList::Half(HalfBANeighborList{
    //         cutoffs: [2.0,3.0],
    //         bond_contribution: false,
    //     })) as Box<dyn CalculatorBase>);

    //     let system = test_system("water");
    //     let options = crate::calculators::tests_utils::FinalDifferenceOptions {
    //         displacement: 1e-6,
    //         max_relative: 1e-9,
    //         epsilon: 1e-16,
    //     };
    //     crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);

    //   // full neighbor list
    //     let calculator = Calculator::from(Box::new(BANeighborList::Full(FullBANeighborList{
    //         cutoffs: [2.0,3.0],
    //         bond_contribution: false,
    //     })) as Box<dyn CalculatorBase>);
    //     crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    // }

    use super::{RadialBasis,RadialScaling,CutoffFunction};

    #[test]
    fn spherical_expansion() {

        let expected = [
            [[0.16902879658926248, 0.028869505770363096, -0.012939303519269344],
             [0.0, 0.0, -0.0],
             [0.26212372007374773, 0.04923860892292029, -0.02052369607421798],
             [0.0, 0.0, -0.0],
             [0.0, 0.0, -0.0],
             [0.0, 0.0, -0.0],
             [0.2734914300150501, 0.05977378771423668, -0.022198889165475678],
             [0.0, 0.0, -0.0],
             [0.0, 0.0, -0.0]],
            [[0.16902879658926248, 0.028869505770363096, -0.012939303519269344],
             [0.0, 0.0, -0.0],
             [0.0, 0.0, -0.0],
             [0.26212372007374773, 0.04923860892292029, -0.02052369607421798],
             [0.0, 0.0, -0.0],
             [0.0, 0.0, -0.0],
             [-0.13674571500752503, -0.029886893857118332, 0.011099444582737835],
             [0.0, 0.0, -0.0],
             [0.23685052611036728, 0.051765618640947135, -0.019224801953097073]],
            [[0.055690489760816295, 0.06300370381466462, -0.002920081734154629],
             [0.0, 0.0, -0.0],
             [0.06460583443346805, 0.07393339000508466, -0.003326135960304247],
             [0.06460583443346805, 0.07393339000508466, -0.003326135960304247],
             [0.0, 0.0, -0.0],
             [0.0, 0.0, -0.0],
             [0.02647774981023968, 0.030983159900568692, -0.0013111835538098728],
             [0.09172161588286466, 0.10732881425363133, -0.004542073066494842],
             [0.04586080794143233, 0.053664407126815666, -0.002271036533247421]],
            [[0.11139019922564429, 0.05256952008314618, -0.00976648821140304],
             [-0.09132548515017183, -0.044385580585388364, 0.008057500982983558],
             [0.15220914191695306, 0.07397596764231394, -0.013429168304972592],
             [0.015220914191695308, 0.007397596764231396, -0.00134291683049726],
             [-0.014919461593876988, -0.007651634517671902, 0.001329600535188854],
             [-0.1491946159387699, -0.07651634517671901, 0.013296005351888539],
             [0.11700350769036943, 0.06000672829237635, -0.010427180998805892],
             [0.024865769323128322, 0.012752724196119839, -0.0022160008919814237],
             [-0.04351509631547453, -0.022317267343209705, 0.0038780015609674885]],
        ].map(|s|ndarray::arr2(&s));

        let vectors = [
            (0.,  0., 1.),
            (1.,  0., 0.),
            (1.,  0., 1.),
            (0.1,-0.6,1.),
        ].into_iter()
            .map(|(x,y,z)|Vector3D::new(x,y,z))
            .collect::<Vec<_>>();

        let expander = RawSphericalExpansion::new(RawSphericalExpansionParameters{
            cutoff: 3.5,
            max_radial: 3,
            max_angular: 2,
            atomic_gaussian_width: 0.3,
            radial_basis: RadialBasis::splined_gto(1e-8),
            radial_scaling: RadialScaling::Willatt2018 { scale: 1.5, rate: 0.8, exponent: 2.0},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        });
        let mut contrib = expander.make_contribution_buffer(false);

        for (vector,expected) in vectors.into_iter().zip(expected) {
            expander.compute_coefficients(&mut contrib, vector, 1., None);
            assert_relative_eq!(contrib.values, expected, max_relative=1E-6);
        }
    }
}
