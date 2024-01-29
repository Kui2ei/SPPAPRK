#include<iostream>
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)
using namespace std;
int main(){
    uint32_t array[2] = {TO_CUDA_T(0x1111222233334444)} ;
    cout<<std::hex<<array[0]<<endl<<std::hex<<array[1]<<endl;
}