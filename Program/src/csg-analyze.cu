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
  if (argc<2){ std::cout<<"no arguments\n"; return EXIT_FAILURE;}
  if (argc>2){ std::cout<<"extra arguments\n"; return EXIT_FAILURE;}

  //declare auxiliary variables
  const std::string sim_dir = argv[1]; //simulation directory
  std::ifstream inp_f; //input file
  std::ofstream out_f; //output file
  std::string pathstr; //file path string
  std::string pathpat; //file path pattern
  uint n_s_f; //number of simulation files

  //main try block
  try
  {
    //read parameters and initialize analysis
    pathstr = sim_dir+"/adjustable-parameters.dat";
    inp_f.open(pathstr);
    mmc::check_file(inp_f,pathstr);
    mmc::parmap par(inp_f); //parameters
    inp_f.close();
    mmc::eamana ana(par); //analysis
  }
  catch (const mmc::error &err) //caught error
  {
    //exit program unsuccessfully
    mmc::logger::record(err.what());
    return EXIT_FAILURE;
  }

  //exit program successfully
  return EXIT_SUCCESS;
}
