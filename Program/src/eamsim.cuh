#ifndef MMC_EAMSIM_H
#define MMC_EAMSIM_H

//Includes

#include "eamdat.cuh" //EA model data

//Structures

//Classes

class eamsim : public eamdat //EA model simulation
{
  public:

  //Functions

  //EA model simulation constructor
  eamsim(float beta); //inverse temperature

  //EA model simulation destructor
  ~eamsim();

  //initialize lattice array
  void init_lattice();

  private:

  //Parameters and Variables

  const float beta; //inverse temperature

  float *prob; //probability lookup table

  void *vprng; //void PRNG array

  uint *slattice; //shuffled lattice array
};

#endif //MMC_EAMSIM_H
