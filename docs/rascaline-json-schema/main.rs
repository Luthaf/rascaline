use std::path::PathBuf;

use schemars::schema::RootSchema;

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

impl schemars::visit::Visitor for RenameRefInSchema {
    fn visit_schema_object(&mut self, schema: &mut schemars::schema::SchemaObject) {
        schemars::visit::visit_schema_object(self, schema);

        let in_code_reference = format!("#/definitions/{}", self.in_code);

        if let Some(reference) = &schema.reference {
            if reference == &in_code_reference {
                schema.reference = Some(format!("#/definitions/{}", self.in_docs));
            }
        }
    }
}

fn save_schema(name: &str, mut schema: RootSchema) {
    for transform in REFS_TO_RENAME {
        if let Some(value) = schema.definitions.remove(transform.in_code) {
            assert!(!schema.definitions.contains_key(transform.in_docs));
            schema.definitions.insert(transform.in_docs.into(), value);

            schemars::visit::visit_root_schema(&mut transform.clone(), &mut schema);
        }
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
