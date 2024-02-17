//Includes

#include "eamsim.cuh" //EA model simulation

#include <time.h> //time utilities library

#include <curand_kernel.h> //cuRAND device functions

//Namespace

namespace mmc //Marco Mendívil Carboni
{

//Constants

static constexpr uint PLTABW = 14; //probability lookup table width

static constexpr uint NSBS = 32; //number of MC steps between swaps

static constexpr uint NTPB = 256; //number of threads per block

static constexpr uint NBLK = (N*NDIS+NTPB-1)/NTPB; //number of blocks

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
  if (!(0.125<=beta&&beta<=8.0))
  {
    throw error("inverse temperature out of range");
  }

  //allocate device memory
  cuda_check(cudaMalloc(&prob,NREP*PLTABW*sizeof(float)));
  cuda_check(cudaMalloc(&vprng,NTPB*NBLK*sizeof(prng)));
  cuda_check(cudaMalloc(&slattice,NDIS*N*sizeof(uint)));

  //initialize PRNG

  //record success message
  logger::record("eamsim initialized");
  logger::record("beta = "+cnfs(beta,5,'0',3));
}

//EA model simulation destructor
eamsim::~eamsim()
{
  //deallocate device memory
  cuda_check(cudaFree(prob));
  cuda_check(cudaFree(vprng));
  cuda_check(cudaFree(slattice));
}

//initialize lattice multispins
void init_multispins(
  curandGenerator_t gen, //host PRNG
  uint *lattice_h) //lattice host array
{
  //declare auxiliary variables
  uint ranmspin; //random multispin

  //set random lattice multispins
  for (uint i_s = 0; i_s<N; ++i_s) //site index
  {
    curandGenerate(gen,&ranmspin,1);
    lattice_h[i_s] = (lattice_h[i_s]&MASKAJ)|ranmspin&MASKAS;
  }
}

//initialize lattice coupling constants
void init_coupling_constants(
  curandGenerator_t gen, //host PRNG
  uint *lattice_h) //lattice host array
{
  //declare auxiliary variables
  uint ran[3]; //random numbers
  uint Jx[N]; //x coupling constants
  uint Jy[N]; //y coupling constants
  uint Jz[N]; //z coupling constants

  //choose random coupling constants
  for (uint i_s = 0; i_s<N; ++i_s) //site index
  {
    curandGenerate(gen,ran,3);
    Jx[i_s] = ran[0]&1;
    Jy[i_s] = ran[1]&1;
    Jz[i_s] = ran[2]&1;
  }

  //copy coupling constants to lattice
  for (uint x = 0; x<L; ++x) //x index
  {
    uint x1 = (x+L-1)%L; //1st x index
    uint x2 = x; //2nd x index
    for (uint y = 0; y<L; ++y) //y index
    {
      uint y1 = (y+L-1)%L; //1st y index
      uint y2 = y; //2nd y index
      for (uint z = 0; z<L; ++z) //z index
      {
	      uint z1 = (z+L-1)%L; //1st z index
	      uint z2 = z; //2nd z index
	      uint J = //lattice coupling constant
	        MASKJ0*Jx[L*L*z+L*y+x1]|
	        MASKJ1*Jx[L*L*z+L*y+x2]|
	        MASKJ2*Jy[L*L*z+L*y1+x]|
	        MASKJ3*Jy[L*L*z+L*y2+x]|
	        MASKJ4*Jz[L*L*z1+L*y+x]|
	        MASKJ5*Jz[L*L*z2+L*y+x];
        uint i_s = L*L*z+L*y+x; //site index
        lattice_h[i_s] = J|(lattice_h[i_s]&MASKAS);
      }
    }
  }
}

//initialize lattice array
void eamsim::init_lattice()
{
  //initialize host PRNG
  curandGenerator_t gen; //host PRNG
  curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,time(nullptr));

  //initialize every lattice in the host
  for (uint i_l = 0; i_l<NDIS; ++i_l) //lattice index
  {
    init_multispins(gen,&lattice_h[N*i_l]);
    init_coupling_constants(gen,&lattice_h[N*i_l]);
  }

  //copy lattice host array to device
  cuda_check(cudaMemcpy(lattice,lattice_h,NDIS*N*sizeof(uint),
    cudaMemcpyHostToDevice));

  //record success message
  logger::record("lattice array initialized");
}

} //namespace mmc
