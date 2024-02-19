//Includes

#include "eamdat.cuh" //EA model data

//Functions

//EA model data constructor
eamdat::eamdat()
{
  //allocate device memory
  cuda_check(cudaMalloc(&lattice,N*NDIS*sizeof(uint)));

  //allocate host memory
  cuda_check(cudaMallocHost(&lattice_h,N*NDIS*sizeof(uint)));

  //record success message
  logger::record("eamdat initialized");
}

//EA model data destructor
eamdat::~eamdat()
{
  //deallocate device memory
  cuda_check(cudaFree(lattice));

  //deallocate host memory
  cuda_check(cudaFreeHost(lattice_h));
}

//write state to binary file
void eamdat::write_state(std::ofstream &bin_out_f) //binary output file
{
  //write lattice host array
  bin_out_f.write(reinterpret_cast<char *>(lattice_h),N*NDIS*sizeof(uint));

  //check filestream
  if (bin_out_f.fail()){ throw error("failed to write state to binary file");}
}

//read state from binary file
void eamdat::read_state(std::ifstream &bin_inp_f) //binary input file
{
  //read lattice host array
  bin_inp_f.read(reinterpret_cast<char *>(lattice_h),N*NDIS*sizeof(uint));

  //copy lattice host array to device
  cuda_check(cudaMemcpy(lattice,lattice_h,N*NDIS*sizeof(uint),
    cudaMemcpyHostToDevice));

  //check filestream
  if (bin_inp_f.fail()){ throw error("failed to read state from binary file");}
}
