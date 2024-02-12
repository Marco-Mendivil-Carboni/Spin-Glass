//Includes

#include "eamsim.cuh" //EA model simulation

#include <time.h> //time utilities library

#include <curand_kernel.h> //cuRAND device functions

//Namespace

namespace mmc //Marco Mendívil Carboni
{

//Constants

//Aliases

using prng = curandStatePhilox4_32_10; //PRNG type

//Enumerations

//Device Functions

//Host Functions

//EA model simulation constructor
eamsim::eamsim(parmap &par) //parameters
{
  //check parameters
  std::string msg = "hello :D"; //message
  logger::record(msg);

  //allocate device memory

  //allocate host memory

  //initialize PRNG
}

//EA model simulation destructor
eamsim::~eamsim()
{
  //deallocate device memory

  //deallocate host memory
}

} //namespace mmc
