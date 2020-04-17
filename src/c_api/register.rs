use std::collections::BTreeMap;

use crate::calculator::Calculator;
use crate::calculator::SortedDistances;


use crate::calculator::DummyCalculator;

type CalculatorCreator = fn(&str) -> Result<Box<dyn Calculator>, serde_json::Error>;

macro_rules! add_calculator {
    ($map :expr, $name :literal, $type :ty) => (
        $map.insert($name, (|json| {
            match serde_json::from_str::<$type>(json) {
                Ok(value) => Ok(Box::new(value)),
                Err(e) => Err(e)
            }
        }) as CalculatorCreator);
    );
}

lazy_static::lazy_static!{
    pub static ref REGISTERED_CALCULATORS: BTreeMap<&'static str, CalculatorCreator> = {
        let mut map = BTreeMap::new();
        add_calculator!(map, "dummy_calculator", DummyCalculator);
        add_calculator!(map, "sorted_distances", SortedDistances);
        return map;
    };
}
