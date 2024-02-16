#ifndef MMC_EAMDAT_H
#define MMC_EAMDAT_H

//Includes

#include "util.cuh" //general utilities

//Namespace

namespace mmc //Marco Mendívil Carboni
{

//Constants

static constexpr uint L = 16; //lattice size
static constexpr uint N = L*L*L; //number of sites

static constexpr uint NDIS = 256; //number of disorder realizations

static constexpr uint NSEG = 6; //number of spin segments
static constexpr uint NSPS = 4; //number of spins per segment
static constexpr uint NREP = 24; //number of temperature replicas

static constexpr uint MASK_S  = 0x00111111; //00000000000100010001000100010001
static constexpr uint MASK_AS = 0x00ffffff; //00000000111111111111111111111111
static constexpr uint MASK_J  = 0xfc000000; //11111100000000000000000000000000
static constexpr uint MASK_J0 = 0x04000000; //00000100000000000000000000000000
static constexpr uint MASK_J1 = 0x08000000; //00001000000000000000000000000000
static constexpr uint MASK_J2 = 0x10000000; //00010000000000000000000000000000
static constexpr uint MASK_J3 = 0x20000000; //00100000000000000000000000000000
static constexpr uint MASK_J4 = 0x40000000; //01000000000000000000000000000000
static constexpr uint MASK_J5 = 0x80000000; //10000000000000000000000000000000
static constexpr uint MASK_E  = 0x0000000f; //00000000000000000000000000001111

static constexpr uint SHIFT_M  = 24;
static constexpr uint SHIFT_J0 = 26;
static constexpr uint SHIFT_J1 = 27;
static constexpr uint SHIFT_J2 = 28;
static constexpr uint SHIFT_J3 = 29;
static constexpr uint SHIFT_J4 = 30;
static constexpr uint SHIFT_J5 = 31;

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

} //namespace mmc

#endif //MMC_EAMDAT_H
