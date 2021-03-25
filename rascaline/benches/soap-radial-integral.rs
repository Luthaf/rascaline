use rascaline::calculators::soap::{RadialIntegral, GtoParameters, GtoRadialIntegral};

use ndarray::Array2;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn gto_radial_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("GTO radial integral (per neighbor)");
    group.noise_threshold(0.05);

    for &max_radial in black_box(&[2, 8, 14]) {
        for &max_angular in black_box(&[1, 7, 15]) {
            let parameters = GtoParameters {
                max_radial,
                max_angular,
                cutoff: 4.5,
                atomic_gaussian_width: 0.5,
            };
            let gto: Box<dyn RadialIntegral> = Box::new(GtoRadialIntegral::new(parameters));
            let mut values = Array2::from_elem((max_radial, max_angular + 1), 0.0);

            // multiple random values spanning the whole range [0, cutoff)
            let distances = [
                0.145, 0.218, 0.585, 0.723, 1.011, 1.463, 1.560, 1.704,
                2.109, 2.266, 2.852, 2.942, 3.021, 3.247, 3.859, 4.462,
            ];

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    for &distance in &distances {
                        gto.compute(distance, values.view_mut(), None)
                    }
                }
                start.elapsed() / distances.len() as u32
            }));
        }
    }
}

fn gto_radial_integral_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("GTO radial integral with gradients (per neighbor)");
    group.noise_threshold(0.05);

    for &max_radial in black_box(&[2, 8, 14]) {
        for &max_angular in black_box(&[1, 7, 15]) {
            let parameters = GtoParameters {
                max_radial,
                max_angular,
                cutoff: 4.5,
                atomic_gaussian_width: 0.5,
            };
            let gto: Box<dyn RadialIntegral> = Box::new(GtoRadialIntegral::new(parameters));
            let mut values = Array2::from_elem((max_radial, max_angular + 1), 0.0);
            let mut gradient = Array2::from_elem((max_radial, max_angular + 1), 0.0);

            // multiple random values spanning the whole range [0, cutoff)
            let distances = [
                0.145, 0.218, 0.585, 0.723, 1.011, 1.463, 1.560, 1.704,
                2.109, 2.266, 2.852, 2.942, 3.021, 3.247, 3.859, 4.462,
            ];

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    for &distance in &distances {
                        gto.compute(distance, values.view_mut(), Some(gradient.view_mut()))
                    }
                }
                start.elapsed() / distances.len() as u32
            }));
        }
    }
}

criterion_group!(gto, gto_radial_integral, gto_radial_integral_gradient);
criterion_main!(gto);
