#ifndef MMC_EAMSIM_H
#define MMC_EAMSIM_H

//Includes

#include "eamdat.cuh" //EA model data

//Structures

struct ibeta //indexed beta
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
  eamsim(float beta); //inverse temperature

  //EA model simulation destructor
  ~eamsim();

  //initialize lattice array
  void init_lattice();

  //run Monte Carlo simulation
  void run_MC_simulation();

  private:

  //Parameters and Variables

  const float beta; //inverse temperature

  ibeta *rbeta; //replica beta array
  void *vprngs; //void PRNG state array
  uint *slattice; //shuffled lattice array

  ibeta *rbeta_h; //replica beta host array

  //Functions

  //initialize replica beta array
  void init_rbeta();
};

#endif //MMC_EAMSIM_H
