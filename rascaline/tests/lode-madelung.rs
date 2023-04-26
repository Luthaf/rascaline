//! Tests of the LODE Spherical Expansion coefficient against
//! the energies of periodic structures with known 1/r potential. For perfect
//! crystals these energies are tabulated and proportional to so called
//! Madelung constants.
//!
//! See for example N. W. Ashcroft and N. D. Mermin, Solid State Physics
//! for reference values and detailed explanations on these constants.

use approx::assert_relative_eq;
use rascaline::calculators::RadialBasis;
use rascaline::calculators::{LodeSphericalExpansionParameters, CalculatorBase, LodeSphericalExpansion};
use rascaline::systems::{System, SimpleSystem, UnitCell};
use rascaline::{Calculator, Matrix3, Vector3D, CalculationOptions};

struct CrystalParameters {
    systems: Vec<Box<dyn System>>,
    charges: Vec<f64>,
    madelung: f64,
}

/// NaCl structure
/// Using a primitive unit cell, the distance between the
/// closest Na-Cl pair is exactly 1. The cubic unit cell
/// in these units would have a length of 2.
fn get_nacl() -> Vec<Box<dyn System>> {
    let cell = Matrix3::new([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]);
    let mut system = SimpleSystem::new(UnitCell::from(cell));
    system.add_atom(11, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(17, Vector3D::new(1.0, 0.0, 0.0));

    vec![Box::new(system) as Box<dyn System>]
}

/// CsCl structure
/// This structure is simple since the primitive unit cell
/// is just the usual cubic cell with side length set to one.
fn get_cscl() -> Vec<Box<dyn System>> {
    let mut system = SimpleSystem::new(UnitCell::cubic(1.0));
    system.add_atom(17, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(55, Vector3D::new(0.5, 0.5, 0.5));

    vec![Box::new(system) as Box<dyn System>]
}

/// ZnS (zincblende) structure
/// As for NaCl, a primitive unit cell is used which makes
/// the lattice parameter of the cubic cell equal to 2.
/// In these units, the closest Zn-S distance is sqrt(3)/2.
fn get_zns() -> Vec<Box<dyn System>> {
    let cell = Matrix3::new([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]);
    let mut system = SimpleSystem::new(UnitCell::from(cell));
    system.add_atom(16, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(30, Vector3D::new(0.5, 0.5, 0.5));

    vec![Box::new(system) as Box<dyn System>]
}


/// ZnS (O4) in wurtzite structure (triclinic cell)
fn get_znso4() -> Vec<Box<dyn System>> {
    let u = 3. / 8.;
    let c = f64::sqrt(1. / u);
    let cell = Matrix3::new([[0.5, -0.5 * f64::sqrt(3.0), 0.0], [0.5, 0.5 * f64::sqrt(3.0), 0.0], [0.0, 0.0, c]]);
    let mut system = SimpleSystem::new(UnitCell::from(cell));
    system.add_atom(16, Vector3D::new(0.5, 0.5 / f64::sqrt(3.0), 0.0));
    system.add_atom(30, Vector3D::new(0.5, 0.5 / f64::sqrt(3.0), u * c));
    system.add_atom(16, Vector3D::new(0.5, -0.5 / f64::sqrt(3.0), 0.5 * c));
    system.add_atom(30, Vector3D::new(0.5, -0.5 / f64::sqrt(3.0), (0.5 + u) * c));

    vec![Box::new(system) as Box<dyn System>]
}

/// Test the agreement with Madelung constant for a variety of
/// atomic_gaussian_width`s and `cutoff`s.
#[test]
fn madelung() {
    let mut crystals = [
        CrystalParameters{systems: get_nacl(), charges: vec![1.0, -1.0], madelung: 1.7476},
        CrystalParameters{systems: get_cscl(), charges: vec![1.0, -1.0], madelung: 2.0 * 1.7626 / f64::sqrt(3.0)},
        CrystalParameters{systems: get_zns(), charges: vec![1.0, -1.0], madelung: 2.0 * 1.6381 / f64::sqrt(3.0)},
        CrystalParameters{systems: get_znso4(), charges: vec![1.0, -1.0, 1.0, -1.0], madelung: 1.6413 / f64::sqrt(3. / 8.)}
    ];

    for cutoff in [0.01_f64, 0.027, 0.074, 0.2] {
        let factor = -1.0 / (4.0 * std::f64::consts::PI * cutoff.powf(2.0)).powf(0.75);
        for atomic_gaussian_width in [0.2, 0.1] {

            for crystal in crystals.iter_mut() {

                let lode_parameters = LodeSphericalExpansionParameters {
                    cutoff,
                    k_cutoff: None,
                    max_radial: 1,
                    max_angular: 0,
                    atomic_gaussian_width,
                    center_atom_weight: 0.0,
                    potential_exponent: 1,
                    radial_basis: RadialBasis::splined_gto(1e-8),
                };

                let mut calculator = Calculator::from(Box::new(LodeSphericalExpansion::new(
                    lode_parameters
                ).unwrap()) as Box<dyn CalculatorBase>);

                let options = CalculationOptions {..Default::default()};

                let descriptor = calculator.compute(&mut crystal.systems, options).unwrap();

                let madelung = factor * (
                    crystal.charges[0] * descriptor.block_by_id(0).values().to_array()[[0, 0, 0]]
                    + crystal.charges[1] * descriptor.block_by_id(1).values().to_array()[[0, 0, 0]]
                );

                assert_relative_eq!(madelung, crystal.madelung, max_relative=8e-2);
            }
        }
    }
}

/// Test the agreement with Madelung constant using parameters for highest
/// accuracy.
#[test]
fn madelung_high_accuracy() {
    let mut crystals = [
        CrystalParameters{systems: get_nacl(), charges: vec![1.0, -1.0], madelung: 1.7476},
        CrystalParameters{systems: get_cscl(),  charges: vec![1.0, -1.0], madelung: 2.0 * 1.7626 / f64::sqrt(3.0)},
        CrystalParameters{systems: get_zns(),  charges: vec![1.0, -1.0], madelung: 2.0 * 1.6381 / f64::sqrt(3.0)},
        CrystalParameters{systems: get_znso4(),  charges: vec![1.0, -1.0, 1.0, -1.0], madelung: 1.6413 / f64::sqrt(3. / 8.)}
    ];

    let cutoff = 0.01_f64;
    let factor = -1.0 / (4.0 * std::f64::consts::PI * cutoff.powf(2.0)).powf(0.75);

    for crystal in crystals.iter_mut() {
        let lode_parameters = LodeSphericalExpansionParameters {
            cutoff,
            k_cutoff: Some(50.),
            max_radial: 1,
            max_angular: 0,
            atomic_gaussian_width: 0.1,
            center_atom_weight: 0.0,
            potential_exponent: 1,
            radial_basis: RadialBasis::splined_gto(1e-8),
        };

        let mut calculator = Calculator::from(Box::new(LodeSphericalExpansion::new(
            lode_parameters
        ).unwrap()) as Box<dyn CalculatorBase>);

        let options = CalculationOptions {..Default::default()};

        let descriptor = calculator.compute(&mut crystal.systems, options).unwrap();

        let madelung = factor * (
            crystal.charges[0] * descriptor.block_by_id(0).values().to_array()[[0, 0, 0]]
            + crystal.charges[1] * descriptor.block_by_id(1).values().to_array()[[0, 0, 0]]
        );

        assert_relative_eq!(madelung, crystal.madelung, max_relative=5e-5);
    }
}
