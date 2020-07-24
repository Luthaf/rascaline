use std::collections::BTreeMap;

use crate::Error;
use crate::calculator::Calculator;
use crate::calculator::SortedDistances;


use crate::calculator::DummyCalculator;

type CalculatorCreator = fn(&str) -> Result<Box<dyn Calculator>, Error>;

macro_rules! add_calculator {
    ($map :expr, $name :literal, $type :ty) => (
        $map.insert($name, (|json| {
            let value = serde_json::from_str::<$type>(json)?;
            Ok(Box::new(value))
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
