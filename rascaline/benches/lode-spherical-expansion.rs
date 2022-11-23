#![allow(clippy::needless_return)]
use rascaline::{Calculator, System, CalculationOptions};

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime, SamplingMode};
use criterion::{criterion_group, criterion_main};

fn load_systems(path: &str) -> Vec<Box<dyn System>> {
    let systems = rascaline::systems::read_from_file(&format!("benches/data/{}", path))
        .expect("failed to read file");

    return systems.into_iter()
        .map(|s| Box::new(s) as Box<dyn System>)
        .collect()
}

fn run_spherical_expansion(mut group: BenchmarkGroup<WallTime>,
    path: &str,
    gradients: bool,
    test_mode: bool,
) {
    let mut systems = load_systems(path);

    if test_mode {
        // Reduce the time/RAM required to test the benchmarks code.
        // Without this, the process gets killed in github actions CI
        systems.truncate(1);
    }

    let cutoff = 4.0;
    let mut n_centers = 0;
    for system in &mut systems {
        n_centers += system.size().unwrap();
        system.compute_neighbors(cutoff).unwrap();
    }

    for atomic_gaussian_width in &[1.5, 1.0, 0.5] {
        let parameters = format!(r#"{{
            "max_radial": 6,
            "max_angular": 6,
            "cutoff": {cutoff},
            "atomic_gaussian_width": {atomic_gaussian_width},
            "center_atom_weight": 1.0,
            "radial_basis": {{ "Gto": {{}} }},
            "potential_exponent": 1
        }}"#);
        let mut calculator = Calculator::new("lode_spherical_expansion", parameters).unwrap();

        group.bench_function(&format!("gaussian_width = {}", atomic_gaussian_width), |b| b.iter_custom(|repeat| {
            let start = std::time::Instant::now();

            let options = CalculationOptions {
                gradients: if gradients { &["positions"] } else { &[] },
                ..Default::default()
            };

            for _ in 0..repeat {
                calculator.compute(&mut systems, options).unwrap();
            }
            start.elapsed() / n_centers as u32
        }));
    }
}

fn spherical_expansion(c: &mut Criterion) {
    let test_mode = std::env::args().any(|arg| arg == "--test");

    let mut group = c.benchmark_group("LODE spherical expansion (per atom)/Bulk Silicon");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_spherical_expansion(group, "silicon_bulk.xyz", false, test_mode);

    let mut group = c.benchmark_group("LODE spherical expansion (per atom) with gradients/Bulk Silicon");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_spherical_expansion(group, "silicon_bulk.xyz", true, test_mode);
}

criterion_group!(all, spherical_expansion);
criterion_main!(all);
