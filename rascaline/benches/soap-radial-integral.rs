#![allow(clippy::needless_return)]

use rascaline::calculators::radial_integral::RadialIntegral;
use rascaline::calculators::radial_integral::{SplinedRadialIntegral, SplinedRIParameters};

use rascaline::calculators::soap::{GtoParameters, SoapGtoRadialIntegral};

use ndarray::Array2;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use criterion::{BenchmarkGroup, measurement::WallTime};

fn benchmark_radial_integral(
    mut group: BenchmarkGroup<'_, WallTime>,
    benchmark_gradients: bool,
    create_radial_integral: impl Fn(usize, usize) -> Box<dyn RadialIntegral>,
) {
    for &max_angular in black_box(&[1, 7, 15]) {
        for &max_radial in black_box(&[2, 8, 14]) {
            let ri = create_radial_integral(max_angular, max_radial);

            let mut values = Array2::from_elem((max_angular + 1, max_radial), 0.0);
            let mut gradients = Array2::from_elem((max_angular + 1, max_radial), 0.0);

            // multiple random values spanning the whole range [0, cutoff)
            let distances = [
                0.145, 0.218, 0.585, 0.723, 1.011, 1.463, 1.560, 1.704,
                2.109, 2.266, 2.852, 2.942, 3.021, 3.247, 3.859, 4.462,
            ];

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    for &distance in &distances {
                        if benchmark_gradients {
                            ri.compute(distance, values.view_mut(), Some(gradients.view_mut()))
                        } else {
                            ri.compute(distance, values.view_mut(), None)
                        }
                    }
                }
                start.elapsed() / distances.len() as u32
            }));
        }
    }
}

fn gto_radial_integral(c: &mut Criterion) {
    let create_radial_integral = |max_angular, max_radial| {
        let parameters = GtoParameters {
            max_radial,
            max_angular,
            cutoff: 4.5,
            atomic_gaussian_width: 0.5,
        };
        return Box::new(SoapGtoRadialIntegral::new(parameters).unwrap()) as Box<dyn RadialIntegral>;
    };

    let mut group = c.benchmark_group("GTO (per neighbor)");
    group.noise_threshold(0.05);
    benchmark_radial_integral(group, false, create_radial_integral);

    let mut group = c.benchmark_group("GTO with gradients (per neighbor)");
    group.noise_threshold(0.05);
    benchmark_radial_integral(group, true, create_radial_integral);
}

fn splined_gto_radial_integral(c: &mut Criterion) {
    let create_radial_integral = |max_angular, max_radial| {
        let cutoff = 4.5;
        let parameters = GtoParameters {
            max_radial,
            max_angular,
            cutoff,
            atomic_gaussian_width: 0.5,
        };
        let gto = SoapGtoRadialIntegral::new(parameters).unwrap();

        let parameters = SplinedRIParameters {
            max_radial,
            max_angular,
            cutoff,
        };
        let accuracy = 1e-8;
        return Box::new(SplinedRadialIntegral::with_accuracy(parameters, accuracy, gto).unwrap()) as Box<dyn RadialIntegral>;
    };

    let mut group = c.benchmark_group("Splined GTO (per neighbor)");
    group.noise_threshold(0.05);
    benchmark_radial_integral(group, false, create_radial_integral);

    let mut group = c.benchmark_group("Splined GTO with gradients (per neighbor)");
    group.noise_threshold(0.05);
    benchmark_radial_integral(group, true, create_radial_integral);
}

criterion_group!(gto, gto_radial_integral, splined_gto_radial_integral);
criterion_main!(gto);
