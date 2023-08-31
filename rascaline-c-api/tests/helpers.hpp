#ifndef RASCAL_TEST_HELPERS
#define RASCAL_TEST_HELPERS

#include <vector>
#include "metatensor.h"
#include "rascaline.h"

#define CHECK_SUCCESS(__expr__) REQUIRE((__expr__) == 0)

rascal_system_t simple_system();
mts_array_t empty_array(std::vector<size_t> shape);

#endif
