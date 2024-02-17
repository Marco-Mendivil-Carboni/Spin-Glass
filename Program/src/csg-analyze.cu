//Includes

#include "eamana.cuh" //EA model analysis

#include <iostream> //standard input/output stream objects

//Functions

//main function
int main(
  const int argc, //argument count
  const char **argv) //argument vector
{
  //check command-line arguments
  if (argc<3){ std::cout<<"no arguments\n"; return EXIT_FAILURE;}
  if (argc>3){ std::cout<<"extra arguments\n"; return EXIT_FAILURE;}

  //declare auxiliary variables
  const std::string sim_dir = argv[1]; //simulation directory
  float beta = std::stof(argv[2]); //inverse temperature
  std::string pathstr; //file path string
  std::string pathpat; //file path pattern
  std::ifstream inp_f; //input file
  std::ofstream out_f; //output file
  uint n_s_f; //number of simulation files

  //main try block
  try
  {
    //initialize analysis
    eamana ana; //analysis
  }
  catch (const error &err) //caught error
  {
    //exit program unsuccessfully
    logger::record(err.what());
    return EXIT_FAILURE;
  }

  //exit program successfully
  return EXIT_SUCCESS;
}
