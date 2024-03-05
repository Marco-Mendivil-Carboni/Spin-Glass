#ifndef MMC_EAMDAT_H
#define MMC_EAMDAT_H

//Includes

#include "util.cuh" //general utilities

//Constants

static constexpr int L = 16; //lattice size
static constexpr int N = L*L*L; //number of sites

static constexpr int NDIS = 256; //number of disorder realizations

static constexpr int SBSHFL = 32; //Monte Carlo steps between shuffles
static constexpr int SBMEAS = 2048; //Monte Carlo steps between measurements
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

static constexpr int NQVAL = 3; //number of overlap values computed

static constexpr int NTPB = L*L; //number of threads per block
static constexpr dim3 CBDIM = {L/2,L,2}; //checkerboard block dimensions
static constexpr int NBPG = NDIS; //number of blocks per grid

//Classes

class eamdat //EA model data
{
  public:

  //Functions

  //EA model data constructor
  eamdat();

  //EA model data destructor
  ~eamdat();

  //write state to binary file
  void write_state(std::ofstream &bin_out_f); //binary output file

  //read state from binary file
  void read_state(std::ifstream &bin_inp_f); //binary input file

  protected:

  //Parameters and Variables

  uint32_t *lattice; //lattice array

  uint32_t *lattice_h; //lattice host array
};

#endif //MMC_EAMDAT_H
