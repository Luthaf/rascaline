use std::collections::BTreeMap;

use ndarray::Array2;

use schemars::schema::{SchemaObject, Schema, SingleOrVec, InstanceType, ObjectValidation, Metadata};

/// A single point entering a spline used for the tabulated radial integrals.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SplinePoint {
    /// Position of the point
    pub position: f64,
    /// Array of values for the tabulated radial integral (the shape should be
    /// `(max_angular + 1) x max_radial`)
    pub values: JsonArray2,
    /// Array of values for the tabulated radial integral (the shape should be
    /// `(max_angular + 1) x max_radial`)
    pub derivatives: JsonArray2,
}

/// A simple wrapper around `ndarray::Array2<f64>` implementing
/// `schemars::JsonSchema`
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct JsonArray2(pub Array2<f64>);

impl std::ops::Deref for JsonArray2 {
    type Target = Array2<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for JsonArray2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


impl schemars::JsonSchema for JsonArray2 {
    fn schema_name() -> String {
        "ndarray::Array".into()
    }

    fn json_schema(_: &mut schemars::gen::SchemaGenerator) -> Schema {
        let mut v = schemars::schema_for_value!(1).schema;
        v.metadata().description = Some("version of the ndarray serialization scheme, should be 1".into());

        let mut dim = schemars::schema_for!(Vec<usize>).schema;
        dim.metadata().description = Some("shape of the array".into());

        let mut data = schemars::schema_for!(Vec<f64>).schema;
        data.metadata().description = Some("data of the array, in row-major order".into());

        let properties = [
            ("v".to_string(), Schema::Object(v)),
            ("dim".to_string(), Schema::Object(dim)),
            ("data".to_string(), Schema::Object(data)),
        ];

        return Schema::Object(SchemaObject {
            metadata: Some(Box::new(Metadata {
                id: None,
                title: Some("ndarray::Array".into()),
                description: Some("Serialization format used by ndarray".into()),
                default: None,
                deprecated: false,
                read_only: false,
                write_only: false,
                examples: vec![],
            })),
            instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::Object))),
            format: None,
            enum_values: None,
            const_value: None,
            subschemas: None,
            number: None,
            string: None,
            array: None,
            object: Some(Box::new(ObjectValidation {
                max_properties: None,
                min_properties: None,
                required: properties.iter().map(|(p, _)| p.clone()).collect(),
                properties: properties.into_iter().collect(),
                pattern_properties: BTreeMap::new(),
                additional_properties: None,
                property_names: None,
            })),
            reference: None,
            extensions: BTreeMap::new(),
        });
    }
 }
