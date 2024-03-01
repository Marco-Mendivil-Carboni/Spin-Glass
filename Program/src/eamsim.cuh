#ifndef MMC_EAMSIM_H
#define MMC_EAMSIM_H

//Includes

#include "eamdat.cuh" //EA model data

#include <curand_kernel.h> //cuRAND device functions

//Aliases

using prng = curandState; //PRNG type

//Structures

struct ib_s //index-beta struct
{
  uint idx; //index
  float beta; //inverse temperature
};

//Classes

class eamsim : public eamdat //EA model simulation
{
  public:

  //Functions

  //EA model simulation constructor
  eamsim(
    float beta, //inverse temperature
    float H); //external magnetic field

  //EA model simulation destructor
  ~eamsim();

  //initialize lattice array
  void init_lattice();

  //read last state from binary file
  void read_last_state(std::ifstream &bin_inp_f); //binary input file

  //run whole simulation
  void run_simulation(std::ofstream &bin_out_f); //binary output file

  private:

  //Parameters and Variables

  const float beta; //inverse temperature
  const float H; //external magnetic field

  ib_s *repib; //replica index-beta array
  prng *prngs; //PRNG state array
  uint *slattice; //shuffled lattice array

  ib_s *repib_h; //replica index-beta host array

  //Functions

  //initialize replica index-beta array
  void init_repib();
};

#endif //MMC_EAMSIM_H
