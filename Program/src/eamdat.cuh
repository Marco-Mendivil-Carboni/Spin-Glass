#ifndef MMC_EAMDAT_H
#define MMC_EAMDAT_H

//Includes

#include "util.cuh" //general utilities

//Constants

static constexpr uint L = 16; //lattice size
static constexpr uint N = L*L*L; //number of sites

static constexpr uint NDIS = 256; //number of disorder realizations

static constexpr uint NSEG = 6; //number of spin segments
static constexpr uint NSPS = 4; //number of spins per segment
static constexpr uint NREP = 24; //number of temperature replicas

static constexpr uint MASKAS = 0x00ffffff; //all spins mask
static constexpr uint MASKSS = 0x00111111; //spin segments mask
static constexpr uint MASKAJ = 0xfc000000; //all coupling constants mask
static constexpr uint MASKSJ = 0x04000000; //single coupling constant mask
static constexpr uint MASKES = 0x0000000f; //energy-spin index mask

// MASKAS = 00000000111111111111111111111111
// MASKSS = 00000000000100010001000100010001
// MASKAJ = 11111100000000000000000000000000
// MASKSJ = 00000100000000000000000000000000
// MASKES = 00000000000000000000000000001111

static constexpr uint SHIFTS = 24; //--------------------------------------------
static constexpr uint SHIFTJ = 26; //--------------------------------------------
// static constexpr uint SHIFT_J1 = 27;
// static constexpr uint SHIFT_J2 = 28;
// static constexpr uint SHIFT_J3 = 29;
// static constexpr uint SHIFT_J4 = 30;
// static constexpr uint SHIFT_J5 = 31;

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
