#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <hdf5.h>




static herr_t h5_error_handler(hid_t /*estack*/, void*)
{
    // H5Eprint(estack, stdout);
    return 0;
}

int main(int argc, char* argv[])
{
    H5Eset_auto(H5E_DEFAULT, h5_error_handler, NULL);
    return Catch::Session().run(argc, argv);
}
