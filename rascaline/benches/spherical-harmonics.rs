use rascaline::Vector3D;
use rascaline::math::{SphericalHarmonics, SphericalHarmonicsArray};

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn spherical_harmonics(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical harmonics (per neighbor)");
    group.noise_threshold(0.05);

    for &max_angular in black_box(&[1, 3, 5, 7, 13, 17, 21, 25]) {
        let mut values = SphericalHarmonicsArray::new(max_angular);
        let mut sph = SphericalHarmonics::new(max_angular);
        let mut directions = [
            // randomly generated directions
            Vector3D::new(-0.762711, -0.145476, -0.630166),
            Vector3D::new(-0.291615, -0.637339, -0.713274),
            Vector3D::new(0.888404, 0.305854, 0.342332),
            Vector3D::new(-0.890056, 0.40123, -0.216367),
            Vector3D::new(-0.975884, -0.0897871, 0.19897),
            Vector3D::new(0.391125, -0.913027, 0.115768),
            Vector3D::new(-0.656982, -0.642407, 0.394572),
            Vector3D::new(0.623778, -0.236985, 0.744808),
            Vector3D::new(0.446324, -0.216075, 0.868393),
            Vector3D::new(-0.811456, 0.40629, -0.42008),
            // a few specific values
            Vector3D::new(0.0, 0.0, 1.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(1.0, 0.0, 0.0),
        ];

        for d in &mut directions {
            *d /= d.norm();
        }

        group.bench_function(format!("l_max = {}", max_angular), |b| b.iter_custom(|repeat| {
            let start = std::time::Instant::now();
            for _ in 0..repeat {
                for &direction in &directions {
                    sph.compute(direction, &mut values, None)
                }
            }
            start.elapsed() / directions.len() as u32
        }));
    }
}

fn spherical_harmonics_with_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical harmonics with gradients (per neighbor)");
    group.noise_threshold(0.05);

    for &max_angular in black_box(&[1, 3, 5, 7, 13, 17, 21, 25]) {
        let mut values = SphericalHarmonicsArray::new(max_angular);
        let mut gradients = [
            SphericalHarmonicsArray::new(max_angular),
            SphericalHarmonicsArray::new(max_angular),
            SphericalHarmonicsArray::new(max_angular),
        ];
        let mut sph = SphericalHarmonics::new(max_angular);
        let mut directions = [
            // randomly generated directions
            Vector3D::new(-0.762711, -0.145476, -0.630166),
            Vector3D::new(-0.291615, -0.637339, -0.713274),
            Vector3D::new(0.888404, 0.305854, 0.342332),
            Vector3D::new(-0.890056, 0.40123, -0.216367),
            Vector3D::new(-0.975884, -0.0897871, 0.19897),
            Vector3D::new(0.391125, -0.913027, 0.115768),
            Vector3D::new(-0.656982, -0.642407, 0.394572),
            Vector3D::new(0.623778, -0.236985, 0.744808),
            Vector3D::new(0.446324, -0.216075, 0.868393),
            Vector3D::new(-0.811456, 0.40629, -0.42008),
            // a few specific values
            Vector3D::new(0.0, 0.0, 1.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(1.0, 0.0, 0.0),
        ];

        for d in &mut directions {
            *d /= d.norm();
        }

        group.bench_function(format!("l_max = {}", max_angular), |b| b.iter_custom(|repeat| {
            let start = std::time::Instant::now();
            for _ in 0..repeat {
                for &direction in &directions {
                    sph.compute(direction, &mut values, Some(&mut gradients))
                }
            }
            start.elapsed() / directions.len() as u32
        }));
    }
}

criterion_group!(benches, spherical_harmonics, spherical_harmonics_with_gradients);
criterion_main!(benches);
