#![allow(clippy::needless_return)]
use featomic::{Calculator, System, CalculationOptions};
use chemfiles::{Frame, Trajectory};

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime, SamplingMode};
use criterion::{criterion_group, criterion_main};


fn read_systems_from_file(path: &str) -> Vec<Box<dyn System>> {
    let mut trajectory = Trajectory::open(path, 'r').expect("could not open the trajectory");
    let mut frame = Frame::new();
    let mut systems = Vec::new();
    for step in 0..trajectory.nsteps() {
        trajectory.read_step(step, &mut frame).expect("failed to read single frame");
        systems.push((&frame).into());
    }

    systems
}

fn run_soap_power_spectrum(
    mut group: BenchmarkGroup<WallTime>,
    path: &str,
    gradients: bool,
    test_mode: bool,
) {
    let mut systems = read_systems_from_file(&format!("benches/data/{}", path));

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

    for &max_basis in &[1, 7, 14] {
        // keep the memory requirements under control
        if max_basis == 14 {
            systems.truncate(10);
            n_centers = 0;
            for system in &mut systems {
                n_centers += system.size().unwrap();
            }
        }

        let parameters = format!(r#"{{
            "cutoff": {{
                "radius": {cutoff},
                "smoothing": {{
                    "type": "ShiftedCosine",
                    "width": 0.5
                }}
            }},
            "density": {{
                "type": "Gaussian",
                "width": 0.3
            }},
            "basis": {{
                "type": "TensorProduct",
                "max_angular": {max_basis},
                "radial": {{
                    "type": "Gto",
                    "max_radial": {max_basis}
                }}
            }}
        }}"#);

        let mut calculator = Calculator::new("soap_power_spectrum", parameters).unwrap();

        group.bench_function(format!("max_radial = max_angular = {}", max_basis), |b| b.iter_custom(|repeat| {
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

fn soap_power_spectrum(c: &mut Criterion) {
    let test_mode = std::env::args().any(|arg| arg == "--test");

    let mut group = c.benchmark_group("SOAP power spectrum (per atom)/Bulk Silicon");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "silicon_bulk.xyz", false, test_mode);

    let mut group = c.benchmark_group("SOAP power spectrum (per atom) with gradients/Bulk Silicon");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "silicon_bulk.xyz", true, test_mode);

    let mut group = c.benchmark_group("SOAP power spectrum (per atom)/Molecular crystals");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "molecular_crystals.xyz", false, test_mode);

    let mut group = c.benchmark_group("SOAP power spectrum (per atom) with gradients/Molecular crystals");
    group.noise_threshold(0.05);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    run_soap_power_spectrum(group, "molecular_crystals.xyz", true, test_mode);
}


criterion_group!(all, soap_power_spectrum);
criterion_main!(all);
