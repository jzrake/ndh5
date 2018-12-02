#include <iostream>
#include <fstream>
#include "ndh5.hpp"



herr_t h5_error_handler(hid_t estack, void *unused)
{
    H5Eprint(estack, stdout);
    return 0;
}


int main()
{
    // H5Eset_auto(H5E_DEFAULT, h5_error_handler, NULL);

    // h5::File g;
    // auto f = h5::File("thing.h5", "w");
    // g = std::move(f);


    auto f = h5::File("chkpt.0000.h5", "r");
    std::cout << f.size() << std::endl;


    auto fid = H5Fopen("chkpt.0000.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

    unsigned intent;
    H5Fget_intent(fid, &intent);
    std::cout << H5F_ACC_RDONLY << " " << intent << std::endl;


    // auto op = [] (hid_t, const char *name, const H5L_info_t*, void*)
    // {
    //     std::cout << name << " ";
    //     return herr_t(1);
    // };


    // auto idx = hsize_t(0);

    // for (int n = 0; n < 9; ++n)
    // {
    //     H5Literate(fid, H5_INDEX_NAME, H5_ITER_INC, &idx, op, nullptr);
    //     std::cout << n << std::endl;
    // }
    return 0;
}   
