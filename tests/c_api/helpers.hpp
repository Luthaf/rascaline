#ifndef RASCAL_TEST_HELPERS
#define RASCAL_TEST_HELPERS

#include "rascaline.h"

#define CHECK_SUCCESS(__expr__) CHECK((__expr__) == RASCAL_SUCCESS)

rascal_system_t simple_system();

#endif
