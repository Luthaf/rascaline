use std::path::PathBuf;

use schemars::Schema;

use rascaline::calculators::AtomicComposition;
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

static REFS_TO_RENAME: &[RenameRefInSchema] = &[
    RenameRefInSchema {
        in_code: "SphericalExpansionBasis_for_SoapRadialBasis",
        in_docs: "SphericalExpansionBasis",
    },
    RenameRefInSchema {
        in_code: "SphericalExpansionBasis_for_LodeRadialBasis",
        in_docs: "SphericalExpansionBasis",
    },
    RenameRefInSchema {
        in_code: "SoapRadialBasis",
        in_docs: "RadialBasis",
    },
    RenameRefInSchema {
        in_code: "LodeRadialBasis",
        in_docs: "RadialBasis",
    },
];

#[derive(Clone)]
struct RenameRefInSchema {
    in_code: &'static str,
    in_docs: &'static str,
}

impl schemars::transform::Transform for RenameRefInSchema {
    fn transform(&mut self, schema: &mut Schema) {
        let in_code_reference = format!("#/$defs/{}", self.in_code);
        if let Some(schema_object) = schema.as_object_mut() {
            if let Some(reference) = schema_object.get_mut("$ref") {
                if reference == &in_code_reference {
                    *reference = format!("#/$defs/{}", self.in_docs).into();
                }
            }
        }
        schemars::transform::transform_subschemas(self, schema);
    }
}

fn save_schema(name: &str, mut schema: Schema) {
    let schema_object = schema.as_object_mut().expect("schema should be an object");

    // rename some of the autogenerate names.
    // Step 1: rename the definitions
    for transform in REFS_TO_RENAME {
        if let Some(definitions) = schema_object.get_mut("$defs") {
            let definitions = definitions.as_object_mut().expect("$defs should be an object");
            if let Some(value) = definitions.remove(transform.in_code) {
                assert!(!definitions.contains_key(transform.in_docs));
                definitions.insert(transform.in_docs.into(), value);
            }
        }
    }

    // Step 2: rename the references to these definitions
    for transform in REFS_TO_RENAME {
        schemars::transform::transform_subschemas(&mut transform.clone(), &mut schema);
    }

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
    generate_schema!(AtomicComposition);
    generate_schema!(NeighborList);
    generate_schema!(SortedDistances);
    generate_schema!("SphericalExpansionByPair", SphericalExpansionParameters);
    generate_schema!("SphericalExpansion", SphericalExpansionParameters);
    generate_schema!("LodeSphericalExpansion", LodeSphericalExpansionParameters);
    generate_schema!("SoapPowerSpectrum", PowerSpectrumParameters);
    generate_schema!("SoapRadialSpectrum", RadialSpectrumParameters);
}
