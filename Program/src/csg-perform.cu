//Includes

#include "eamsim.cuh" //EA model simulation

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
  uint i_s_f; //simulation file index

  //create log file in current working directory
  time_t t_s = time(nullptr); //starting time
  pathstr = std::to_string(t_s)+".log";
  mmc::logger::set_file(pathstr);

  //main try block
  try
  {
    //read parameters and initialize simulation
    pathstr = sim_dir+"/adjustable-parameters.dat";
    inp_f.open(pathstr);
    mmc::check_file(inp_f,pathstr);
    mmc::parmap par(inp_f); //parameters
    inp_f.close();
    mmc::eamsim sim(par); //simulation
  }
  catch (const mmc::error &err) //caught error
  {
    //exit program unsuccessfully
    mmc::logger::record(err.what());
    return EXIT_FAILURE;
  }

  //remove log file
  mmc::logger::set_file("/dev/null");
  pathstr = std::to_string(t_s)+".log";
  std::remove(pathstr.c_str());

  //exit program successfully
  return EXIT_SUCCESS;
}
