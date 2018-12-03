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
    H5Eset_auto(H5E_DEFAULT, h5_error_handler, NULL);

    // dset.read<Array>(_|0|100, _|10|50);
    // file.read<Array>("data", _|0|100, _|10|50);
    // file.read<Array>("data", nd::make_selector(_|0|100, _|10|50));

    return 0;
}   
