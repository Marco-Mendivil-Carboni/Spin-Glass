//Includes

#include "eamsim.cuh" //EA model simulation

#include <time.h> //time utilities library

#include <curand_kernel.h> //cuRAND device functions

//Namespace

namespace mmc //Marco Mendívil Carboni
{

//Constants

static constexpr uint PLTABW = 16; //probability lookup table width

static constexpr uint NSBS = 32; //number of MC steps between swaps

//Aliases

using prng = curandStatePhilox4_32_10; //PRNG type

//Device Functions

//Host Functions

//EA model simulation constructor
eamsim::eamsim(float beta) //inverse temperature
  : eamdat()
  , beta {beta}
{
  //check parameters
  if (!(0.125<=beta&&beta<=2.0))
  {
    throw error("inverse temperature out of range");
  }

  //allocate device memory
  cuda_check(cudaMalloc(&prob,NREP*PLTABW*sizeof(float)));
  //...

  //initialize PRNG

  //record success message
  std::string msg = "eamsim initialized "; //message
  msg += "beta = "+cnfs(beta,5,'0',1)+" ";
  logger::record(msg);
}

//EA model simulation destructor
eamsim::~eamsim()
{
  //deallocate device memory
  cuda_check(cudaFree(prob));
  //...
}

} //namespace mmc
