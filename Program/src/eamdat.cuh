#ifndef MMC_EAMDAT_H
#define MMC_EAMDAT_H

//Includes

#include "util.cuh" //general utilities

//Constants

static constexpr uint L = 16; //lattice size
static constexpr uint N = L*L*L; //number of sites

static constexpr uint NDIS = 256; //number of disorder realizations

static constexpr uint SBSHFL = 32; //Monte Carlo steps between shuffles
static constexpr uint SBMEAS = 2048; //Monte Carlo steps between measurements
static constexpr uint SPFILE = 262144; //Monte Carlo steps per file

static constexpr uint NSPS = 4; //number of spins per segment
static constexpr uint NREP = 24; //number of temperature replicas

static constexpr uint MASKAS = 0x00ffffff; //all spins mask
static constexpr uint MASKSS = 0x00111111; //spin segments mask
static constexpr uint MASKAJ = 0xfc000000; //all coupling constants mask
static constexpr uint MASKSJ = 0x04000000; //single coupling constant mask
static constexpr uint MASKES = 0x0000000f; //energy-spin index mask
static constexpr uint MASKAB = 0xffffffff; //all bits mask

static constexpr uint SHIFTMS = 24; //maximum segment shift
static constexpr uint SHIFTSJ = 26; //single coupling constant shift

static constexpr uint NPROB = 14; //number of possible probabilities
static constexpr uint PTABW = 16; //probability lookup table width

static constexpr uint NTPB = L*L; //number of threads per block
static constexpr dim3 CBDIM = {L/2,L,2}; //checkerboard block dimensions
static constexpr uint NBPG = NDIS; //number of blocks per grid

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

  uint *lattice; //lattice array

  uint *lattice_h; //lattice host array
};

#endif //MMC_EAMDAT_H
