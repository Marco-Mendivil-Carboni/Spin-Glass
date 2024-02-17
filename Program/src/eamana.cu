//Includes

#include "eamana.cuh" //EA model analysis

//Constants

//Device Functions

//Host Functions

//EA model analysis constructor
eamana::eamana()
  : eamdat()
{
  //check parameters

  //allocate device memory

  //allocate host memory

  //record success message
  logger::record("eamana initialized");
}

//EA model analysis destructor
eamana::~eamana()
{
  //deallocate device memory

  //deallocate host memory
}
