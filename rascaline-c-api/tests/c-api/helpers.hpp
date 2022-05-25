#ifndef RASCAL_TEST_HELPERS
#define RASCAL_TEST_HELPERS

#include <vector>
#include "equistore.h"
#include "rascaline.h"

#define CHECK_SUCCESS(__expr__) CHECK((__expr__) == RASCAL_SUCCESS)

rascal_system_t simple_system();
eqs_array_t empty_array(std::vector<size_t> shape);

#endif
