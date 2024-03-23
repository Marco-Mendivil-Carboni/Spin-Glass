#ifndef MMC_EAMSIM_H
#define MMC_EAMSIM_H

//Includes

#include "util.cuh" //general utilities

#include <curand_kernel.h> //cuRAND device functions

//Constants

static constexpr int L = 16; //lattice size
static constexpr int N = L*L*L; //number of sites

static constexpr int NDIS = 256; //number of disorder realizations
static constexpr int NCP = 2; //number of disorder realization copies
static constexpr int NL = NDIS*NCP; //number of lattices

static constexpr int SBSHFL = 16; //Monte Carlo steps between shuffles
static constexpr int SBMEAS = 1024; //Monte Carlo steps between measurements
static constexpr int SPFILE = 262144; //Monte Carlo steps per file

static constexpr int NSPS = 4; //number of spins per segment
static constexpr int NREP = 24; //number of temperature replicas

static constexpr uint32_t MASKAS = 0x00ffffff; //all spins mask
static constexpr uint32_t MASKSS = 0x00111111; //spin segments mask
static constexpr uint32_t MASKAJ = 0xfc000000; //all coupling constants mask
static constexpr uint32_t MASKSJ = 0x04000000; //single coupling constant mask
static constexpr uint32_t MASKES = 0x0000000f; //energy-spin index mask
static constexpr uint32_t MASKAB = 0xffffffff; //all bits mask

static constexpr int SHIFTMS = 24; //maximum segment shift
static constexpr int SHIFTSJ = 26; //single coupling constant shift

static constexpr int NPROB = 14; //number of possible probabilities
static constexpr int PTABW = 16; //probability lookup table width

static constexpr int NTPB = L*L; //number of threads per block
static constexpr dim3 CBDIM = {L/2,L,2}; //checkerboard block dimensions

//Aliases

using prng = curandStateXORWOW; //PRNG type

//Structures

struct ib_s //index-beta struct
{
  int idx; //index
  float beta; //inverse temperature
};

struct obs_s //observables struct
{
  float e[NCP]; //energy
  float m[NCP]; //magnetization
  float q_0; //overlap value 0
  float q_1_r[3]; //Re overlap values 1
  float q_1_i[3]; //Im overlap values 1
};

//Classes

class eamsim //EA model simulation
{
  public:

  //Functions

  //EA model simulation constructor
  eamsim(float H); //external magnetic field

  //EA model simulation destructor
  ~eamsim();

  //initialize lattice array
  void init_lattice();

  //save state to binary file
  void save_checkpoint(std::ofstream &bin_out_f); //binary output file

  //load state from binary file
  void load_checkpoint(std::ifstream &bin_inp_f); //binary input file

  //run whole simulation
  void run_simulation(std::ofstream &bin_out_f); //binary output file

  private:

  //Parameters and Variables

  const float H; //external magnetic field

  ib_s *repib; //replica index-beta array
  prng *prngs; //PRNG state array
  uint32_t *lattice; //lattice array
  uint32_t *slattice; //shuffled lattice array
  obs_s *obs; //observables array

  ib_s *repib_h; //replica index-beta host array
  uint32_t *lattice_h; //lattice host array
  obs_s *obs_h; //observables host array

  //Functions

  //initialize replica index-beta array
  void init_repib();
};

#endif //MMC_EAMSIM_H
