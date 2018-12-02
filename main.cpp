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
    return 0;
}   
