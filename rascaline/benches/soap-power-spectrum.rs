#![allow(clippy::needless_return)]

use flate2::read::GzDecoder;

use rascaline::calculators::CalculatorBase;
use rascaline::calculators::{SoapPowerSpectrum, PowerSpectrumParameters};
use rascaline::calculators::soap::{RadialBasis, CutoffFunction};

use rascaline::system::{System, SimpleSystem, UnitCell};
use rascaline::{Descriptor, Matrix3};

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
use criterion::{black_box, criterion_group, criterion_main};

#[derive(serde::Deserialize)]
struct JsonSystem {
    cell: [[f64; 3]; 3],
    positions: Vec<[f64; 3]>,
    species: Vec<usize>,
}

impl From<JsonSystem> for SimpleSystem {
    fn from(system: JsonSystem) -> SimpleSystem {
        let cell = UnitCell::from(Matrix3::new(system.cell));
        let mut new = SimpleSystem::new(cell);
        for (species, position) in system.species.into_iter().zip(system.positions.into_iter()) {
            new.add_atom(species, position.into());
        }
        return new;
    }
}

fn load_system(path: &str) -> Vec<SimpleSystem> {
    let file = std::fs::File::open(&format!("benches/data/{}", path))
        .expect("failed to open file");

    let data: Vec<JsonSystem> = serde_json::from_reader(GzDecoder::new(file)).expect("failed to parse JSON");

    return data.into_iter().map(|s| s.into()).collect()
}

fn run_soap_power_spectrum(mut group: BenchmarkGroup<WallTime>, path: &str) {
    let mut raw_systems = load_system(path);

    let cutoff = 4.0;

    let mut n_centers = 0;
    let mut systems = Vec::new();
    for system in &mut raw_systems {
        n_centers += system.size();
        system.compute_neighbors(cutoff);
        systems.push(system as &mut dyn System);
    }

    for &max_radial in black_box(&[2, 8, 14]) {
        for &max_angular in black_box(&[1, 7, 15]) {
            let parameters = PowerSpectrumParameters {
                max_radial,
                max_angular,
                cutoff,
                atomic_gaussian_width: 0.3,
                gradients: false,
                radial_basis: RadialBasis::Gto {},
                cutoff_function: CutoffFunction::ShiftedCosine{ width: 0.5 },
            };
            let mut calculator = SoapPowerSpectrum::new(parameters);

            let mut descriptor = Descriptor::new();
            let environments = calculator.environments();
            descriptor.prepare(environments.indexes(&mut systems), calculator.features());

            group.bench_function(&format!("n_max = {}, l_max = {}", max_radial, max_angular), |b| b.iter_custom(|repeat| {
                let start = std::time::Instant::now();
                for _ in 0..repeat {
                    calculator.compute(&mut systems, &mut descriptor);
                }
                start.elapsed() / n_centers as u32
            }));
        }
    }
}

fn soap_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("SOAP power spectrum (per atom)/Bulk Silicon");
    group.noise_threshold(0.05);
    group.measurement_time(std::time::Duration::from_secs(30));

    run_soap_power_spectrum(group, "silicon_bulk.json.gz");

    let mut group = c.benchmark_group("SOAP power spectrum (per atom)/Molecular crystals");
    group.noise_threshold(0.05);
    group.measurement_time(std::time::Duration::from_secs(30));

    run_soap_power_spectrum(group, "molecular_crystals.json.gz");
}


criterion_group!(all, soap_power_spectrum);
criterion_main!(all);
