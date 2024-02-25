//Includes

#include "eamsim.cuh" //EA model simulation

#include <time.h> //time utilities library

#include <curand_kernel.h> //cuRAND device functions

//Constants

static constexpr float H = 0.0; //external magnetic field

static constexpr uint NPROB = 14; //number of possible probabilities
static constexpr uint PTABW = 16; //probability lookup table width

static constexpr uint MCSBS = 32; //Monte Carlo steps between shuffles

static constexpr uint NTPB = 256; //number of threads per block
static constexpr uint NBPG = NDIS; //number of blocks per grid

//Aliases

using prng = curandStatePhilox4_32_10; //PRNG type

//Device Functions

//initialize probability lookup table
inline __device__ void init_prob(
  uint prob[NREP][PTABW], //probability lookup table
  const uint i_bt) //block thread index
{
  //initialize all entries to 1
  if (i_bt<PTABW)
  {
    for (uint i_b = 0; i_b<NREP; ++i_b) //beta index
    {
      prob[i_b][i_bt] = UINT_MAX;
    }
  }
}

//compute probability lookup table
inline __device__ void compute_prob(
  uint prob[NREP][PTABW], //probability lookup table
  float *s_rep_beta, //shared replica beta array
  const uint i_bt) //block thread index
{
  //compute all possible probabilities
  if (i_bt<NPROB)
  {
    for (uint i_b = 0; i_b<NREP; ++i_b) //beta index
    {
      float energy = i_bt-6+H-((1+2*H)*(i_bt&1)); //spin energy
      prob[i_b][i_bt] = expf(s_rep_beta[i_b]*2*energy)*UINT_MAX;
    }
  }
}

//shuffle lattice temperature replicas
__device__ void shuffle(
  void *vprngs, //void PRNG state array
  uint *s_rep_idx, //shared replica index array
  float *s_rep_beta, //shared replica beta array
  float *tot_energy, //total energy array
  const uint i_bt, //block thread index
  const uint i_gt, //grid thread index
  bool mode) //shuffle mode
{
  //declare auxiliary variables
  uint i_0; //1st array index
  uint i_1; //2nd array index
  uint max_i_bt; //maximum block thread index
  __shared__ uint s_rai[NREP]; //shared rearranged array index array

  //write shared rearranged array index array
  if (i_bt<NREP){ s_rai[s_rep_idx[i_bt]] = i_bt;}
  __syncthreads ();

  if (mode) //consider even pairs of temperature replicas
  {
    i_0 = s_rai[(i_bt<<1)+0]; i_1 = s_rai[(i_bt<<1)+1]; max_i_bt = NREP/2;
  }
  else //consider odd pairs of temperature replicas
  {
    i_0 = s_rai[(i_bt<<1)+1]; i_1 = s_rai[(i_bt<<1)+2]; max_i_bt = NREP/2-1;
  }

  if (i_bt<max_i_bt) //shuffle pair of temperature replicas
  {
    //generate random number
    prng *prngs = static_cast<prng *>(vprngs); //PRNG state array
    float ran = curand_uniform(&prngs[i_gt]); //random number in (0,1]

    //compute shuffle probability
    float beta_diff = s_rep_beta[i_0]-s_rep_beta[i_1]; //beta difference
    float energy_diff = tot_energy[i_0]-tot_energy[i_1]; //energy difference
    float prob = expf(beta_diff*energy_diff); //shuffle probability

    if (ran<prob) //accept shuffle
    {
      uint tmp_idx = s_rep_idx[i_0]; //temporary index
      s_rep_idx[i_0] = s_rep_idx[i_1]; s_rep_idx[i_1] = tmp_idx;
      float tmp_beta = s_rep_beta[i_0]; //temporary beta
      s_rep_beta[i_0] = s_rep_beta[i_1]; s_rep_beta[i_1] = tmp_beta;
    }
  }
  __syncthreads();
}

//perform skewed sequential sum reduction
inline __device__ void sum_reduce(
  float *tot_energy, //total energy array
  short aux_energy[NREP][NTPB], //auxiliary energy array
  const uint i_bt) //block thread index
{
  //sum auxiliary energies for each temperature replica
  if (i_bt<NREP)
  {
    int sum = 0; //sum of energies
    for (uint i_sl = 0; i_sl<NTPB; ++i_sl) //skewed loop index
    {
      sum += aux_energy[i_bt][(i_sl+i_bt)%NTPB];
    }
    tot_energy[i_bt] = sum;
  }
  __syncthreads();
}

//stencil...

//stencil_swap...

//Global Functions

//initialize PRNG state array
__global__ void init_prng(
  void *vprngs, //void PRNG state array
  uint pseed) //PRNG seed
{
  //calculate grid thread index
  const uint i_gt = blockDim.x*blockIdx.x+threadIdx.x; //grid thread index

  //initialize PRNG state
  prng *prngs = static_cast<prng *>(vprngs); //PRNG state array
  curand_init(pseed,i_gt,0,&prngs[i_gt]);
}

//kernel_warmup...

//kernel_swap...

//rearrange lattice temperature replicas
__global__ void rearrange(
  uint *lattice, //lattice array
  ib_s *repib, //replica index-beta array
  uint *slattice) //shuffled lattice array
{
  //calculate indexes
  const uint i_bt = threadIdx.x; //block thread index
  const uint i_gb = blockIdx.x; //grid block index

  //declare auxiliary variables
  uint smspin; //shuffled multispin
  uint rmspin; //rearranged multispin
  __shared__ uint s_rep_idx[NREP]; //shared replica index array

  //write shared replica index array
  if (i_bt<NREP){ s_rep_idx[i_bt] = repib[NREP*i_gb+i_bt].idx;}
  __syncthreads();

  //update lattice array
  for (uint i_s = i_bt; i_s<N; i_s += NTPB) //site index
  {
    smspin = slattice[N*i_gb+i_s];
    rmspin = 0;
    for (uint i_b = 0; i_b<NREP; ++i_b) //beta index
    {
	    rmspin |= ((smspin>>i_b)&1)<<s_rep_idx[i_b];
    }
    lattice[N*i_gb+i_s] = (lattice[N*i_gb+i_s]&MASKAJ)|rmspin;
  }
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
  cuda_check(cudaMalloc(&repib,NREP*NDIS*sizeof(ib_s)));
  cuda_check(cudaMalloc(&vprngs,NTPB*NBPG*sizeof(prng)));
  cuda_check(cudaMalloc(&slattice,N*NDIS*sizeof(uint)));

  //allocate host memory
  cuda_check(cudaMallocHost(&repib_h,NREP*NDIS*sizeof(ib_s)));

  //initialize replica index-beta array
  init_repib();

  //initialize PRNG state array
  init_prng<<<NTPB,NBPG>>>(vprngs,time(nullptr));

  //record success message
  logger::record("eamsim initialized");
}

//EA model simulation destructor
eamsim::~eamsim()
{
  //deallocate device memory
  cuda_check(cudaFree(repib));
  cuda_check(cudaFree(vprngs));
  cuda_check(cudaFree(slattice));

  //deallocate host memory
  cuda_check(cudaFreeHost(repib_h));
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
void eamsim::run_MC_simulation(std::ofstream &bin_out_f) //binary output file
{
  //copy lattice array to shuffled lattice array
  cuda_check(cudaMemcpy(slattice,lattice,N*NDIS*sizeof(uint),
    cudaMemcpyDeviceToDevice));

  //Monte Carlo steps...

  //rearrange lattice temperature replicas
  rearrange<<<NTPB,NBPG>>>(lattice,repib,slattice);

  //copy lattice array to host
  cuda_check(cudaMemcpy(lattice_h,lattice,N*NDIS*sizeof(uint),
    cudaMemcpyDeviceToHost));

  //write state to binary file
  write_state(bin_out_f);

  //record success message
  logger::record("simulation ended");
}

//initialize replica index-beta array
void eamsim::init_repib()
{
  //declare auxiliary variables
  const float bratio = pow(2.0,4/(NREP-1.0)); //beta ratio

  //initialize replica index-beta host array
  for (uint i_l = 0; i_l<NDIS; ++i_l) //lattice index
  {
    for (uint i_b = 0; i_b<NREP; ++i_b) //beta index
    {
      repib_h[NREP*i_l+i_b].idx = i_b;
      repib_h[NREP*i_l+i_b].beta = pow(bratio,i_b)*beta;
    }
  }

  //copy replica index-beta host array to device
  cuda_check(cudaMemcpy(repib,repib_h,NREP*NDIS*sizeof(ib_s),
    cudaMemcpyHostToDevice));
}
