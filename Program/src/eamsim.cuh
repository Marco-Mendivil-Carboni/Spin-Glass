#ifndef MMC_EAMSIM_H
#define MMC_EAMSIM_H

//Includes

#include "util.cuh" //general utilities

//Namespace

namespace mmc //Marco Mendívil Carboni
{

//Structures

//Classes

class eamsim //EA model simulation
{
  public:

  //Functions

  //EA model simulation constructor
  eamsim(parmap &par); //parameters

  //EA model simulation destructor
  ~eamsim();

  private:

  //Parameters and Variables

  //Functions
};

} //namespace mmc

#endif //MMC_EAMSIM_H
