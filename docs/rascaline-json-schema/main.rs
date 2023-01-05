use std::path::PathBuf;

use schemars::schema::RootSchema;

use rascaline::calculators::SortedDistances;
use rascaline::calculators::SphericalExpansionParameters;
use rascaline::calculators::LodeSphericalExpansionParameters;
use rascaline::calculators::PowerSpectrumParameters;
use rascaline::calculators::RadialSpectrumParameters;
use rascaline::calculators::NeighborList;


macro_rules! generate_schema {
    ($Type: ty) => {
        generate_schema!(stringify!($Type), $Type)
    };
    ($name: expr, $Type: ty) => {
        save_schema($name, schemars::schema_for!($Type))
    };
}

fn save_schema(name: &str, schema: RootSchema) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.push("build");
    path.push("json-schemas");
    std::fs::create_dir_all(&path).expect("failed to create JSON schema directory");

    path.push(format!("{}.json", name));

    let schema = serde_json::to_string_pretty(&schema).expect("failed to create JSON schema");
    std::fs::write(path, schema).expect("failed to save JSON schema to file");
}

fn main() {
    generate_schema!(NeighborList);
    generate_schema!(SortedDistances);
    generate_schema!("SphericalExpansionByPair", SphericalExpansionParameters);
    generate_schema!("SphericalExpansion", SphericalExpansionParameters);
    generate_schema!("LodeSphericalExpansion", LodeSphericalExpansionParameters);
    generate_schema!("SoapPowerSpectrum", PowerSpectrumParameters);
    generate_schema!("SoapRadialSpectrum", RadialSpectrumParameters);
}
