//Includes

#include "eamsim.cuh" //EA model simulation

#include <time.h> //time utilities library

#include <curand_kernel.h> //cuRAND device functions

//Constants

static constexpr uint MCSBS = 32; //MC steps between swaps

static constexpr uint NTPB = 256; //number of threads per block
static constexpr uint NBLK = (N*NDIS+NTPB-1)/NTPB; //number of blocks

//Aliases

using prng = curandStatePhilox4_32_10; //PRNG type

//Device Functions

//Global Functions

//initialize PRNG state array
__global__ void init_prng(
  void *vprngs, //void PRNG state array
  uint pseed) //PRNG seed
{
  //calculate thread index
  uint i_t = blockIdx.x*blockDim.x+threadIdx.x; //thread index
  if (i_t>=N*NDIS){ return;}

  //initialize PRNG state
  prng *prngs = static_cast<prng *>(vprngs); //PRNG state array
  curand_init(pseed,i_t,0,&prngs[i_t]);
}

//Host Functions

//EA model simulation constructor
eamsim::eamsim(float beta) //inverse temperature
  : eamdat()
  , beta {beta}
{
  //check parameters
  if (!(0.125<=beta&&beta<=8.0)){ throw error("beta out of range");}
  logger::record("beta = "+cnfs(beta,5,'0',3));

  //allocate device memory
  cuda_check(cudaMalloc(&rbeta,NREP*NDIS*sizeof(ibeta)));
  cuda_check(cudaMalloc(&vprngs,N*NDIS*sizeof(prng)));
  cuda_check(cudaMalloc(&slattice,N*NDIS*sizeof(uint)));

  //allocate host memory
  cuda_check(cudaMallocHost(&rbeta_h,NREP*NDIS*sizeof(ibeta)));

  //initialize replica beta array
  init_rbeta();

  //initialize PRNG state array
  init_prng<<<NTPB,NBLK>>>(vprngs,time(nullptr));

  //record success message
  logger::record("eamsim initialized");
}

//EA model simulation destructor
eamsim::~eamsim()
{
  //deallocate device memory
  cuda_check(cudaFree(rbeta));
  cuda_check(cudaFree(vprngs));
  cuda_check(cudaFree(slattice));

  //deallocate host memory
  cuda_check(cudaFreeHost(rbeta_h));
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
  for (uint xa = 0; xa<L; ++xa) //advanced x index
  {
    uint xr = (xa+L-1)%L; //retarded x index
    for (uint ya = 0; ya<L; ++ya) //advanced y index
    {
      uint yr = (ya+L-1)%L; //retarded y index
      for (uint za = 0; za<L; ++za) //advanced z index
      {
	      uint zr = (za+L-1)%L; //retarded z index
	      uint J = //site's coupling constants
	        (MASKSJ<<0)*Jx[L*L*za+L*ya+xr]|
	        (MASKSJ<<1)*Jx[L*L*za+L*ya+xa]|
	        (MASKSJ<<2)*Jy[L*L*za+L*yr+xa]|
	        (MASKSJ<<3)*Jy[L*L*za+L*ya+xa]|
	        (MASKSJ<<4)*Jz[L*L*zr+L*ya+xa]|
	        (MASKSJ<<5)*Jz[L*L*za+L*ya+xa];
        uint i_s = L*L*za+L*ya+xa; //site index
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
  cuda_check(cudaMemcpy(lattice,lattice_h,N*NDIS*sizeof(uint),
    cudaMemcpyHostToDevice));

  //record success message
  logger::record("lattice array initialized");
}

//run Monte Carlo simulation
void eamsim::run_MC_simulation()
{
  //copy lattice array to shuffled lattice array
  cuda_check(cudaMemcpy(slattice,lattice,N*NDIS*sizeof(uint),
    cudaMemcpyDeviceToDevice));
}

//initialize replica beta array
void eamsim::init_rbeta()
{
  //declare auxiliary variables
  const float bratio = pow(2.0,4/(NREP-1.0)); //beta ratio

  //initialize replica beta host array
  for (uint i_l = 0; i_l<NDIS; ++i_l) //lattice index
  {
    for (uint i_b = 0; i_b<NREP; ++i_b) //beta index
    {
      uint i_r = NREP*i_l+i_b; //replica index
      rbeta_h[i_r].idx = i_b;
      rbeta_h[i_r].beta = pow(bratio,i_b)*beta;
    }
  }

  //copy replica beta host array to device
  cuda_check(cudaMemcpy(rbeta,rbeta_h,NREP*NDIS*sizeof(ibeta),
    cudaMemcpyHostToDevice));
}
