use rascaline::calculators::{CalculatorBase, SphericalExpansion, SphericalExpansionParameters};
use rascaline::calculators::soap::{RadialBasis, CutoffFunction};

use rascaline::systems::{System, SimpleSystem, UnitCell};
use rascaline::Descriptor;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn testing_system() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::infinite());
    system.add_atom(6, [1.65624000, 4.29873000, 4.95347000].into());
    system.add_atom(1, [0.66542300, 0.09010260, 3.34439000].into());
    system.add_atom(1, [1.48517000, 2.73735000, 2.67172000].into());
    system.add_atom(1, [4.74104000, 0.73324400, 4.66036000].into());
    system.add_atom(8, [3.27751000, 1.44248000, 2.45736000].into());
    system.add_atom(1, [3.15050000, 1.11292000, 1.82185000].into());
    system.add_atom(8, [0.32936900, 4.74779000, 1.78124000].into());
    system.add_atom(1, [4.86120000, 4.00324000, 1.36439000].into());

    system
}

fn spherical_expansion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spherical expansion (per atom)");
    group.noise_threshold(0.05);
    group.measurement_time(std::time::Duration::from_secs(10));

    let system = testing_system();
    let n_centers = system.size().unwrap();
    let systems = &mut [Box::new(system) as Box<dyn System>];

    for &max_radial in black_box(&[2, 8, 14]) {
        for &max_angular in black_box(&[1, 7, 15]) {
            let parameters = SphericalExpansionParameters {
                max_radial,
                max_angular,
                cutoff: 4.5,
                atomic_gaussian_width: 0.5,
                gradients: false,
                radial_basis: RadialBasis::Gto {},
                cutoff_function: CutoffFunction::ShiftedCosine{ width: 0.5 },
            };
            let mut calculator = SphericalExpansion::new(parameters).unwrap();

            let mut descriptor = Descriptor::new();
            let samples = calculator.samples().indexes(systems).unwrap();
            descriptor.prepare(samples, calculator.features());

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    calculator.compute(systems, &mut descriptor).unwrap();
                }
                start.elapsed() / n_centers as u32
            }));
        }
    }
}

fn spherical_expansion_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spherical expansion with gradients (per atom)");
    group.noise_threshold(0.05);
    group.measurement_time(std::time::Duration::from_secs(10));

    let system = testing_system();
    let n_centers = system.size().unwrap();
    let systems = &mut [Box::new(system) as Box<dyn System>];

    for &max_radial in black_box(&[2, 8, 14]) {
        for &max_angular in black_box(&[1, 7, 15]) {
            let parameters = SphericalExpansionParameters {
                max_radial,
                max_angular,
                cutoff: 4.5,
                atomic_gaussian_width: 0.5,
                gradients: true,
                radial_basis: RadialBasis::Gto {},
                cutoff_function: CutoffFunction::ShiftedCosine{ width: 0.5 },
            };
            let mut calculator = SphericalExpansion::new(parameters).unwrap();

            let mut descriptor = Descriptor::new();
            let (samples, gradients) = calculator.samples().with_gradients(systems).unwrap();
            descriptor.prepare_gradients(samples, gradients.unwrap(), calculator.features());

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    calculator.compute(systems, &mut descriptor).unwrap();
                }
                start.elapsed() / n_centers as u32
            }));
        }
    }
}

criterion_group!(all, spherical_expansion, spherical_expansion_gradients);
criterion_main!(all);
