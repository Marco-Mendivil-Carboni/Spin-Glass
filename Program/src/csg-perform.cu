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
  if (argc<3){ std::cout<<"missing arguments\n"; return EXIT_FAILURE;}
  if (argc>3){ std::cout<<"extra arguments\n"; return EXIT_FAILURE;}

  //declare auxiliary variables
  const std::string sim_dir = argv[1]; //simulation directory
  float beta = std::stof(argv[2]); //inverse temperature
  std::string pathstr; //file path string
  std::string pathpat; //file path pattern
  std::ifstream inp_f; //input file
  std::ofstream out_f; //output file
  uint i_s_f; //simulation file index

  //create log file in current working directory
  time_t t_s = time(nullptr); //starting time
  pathstr = std::to_string(t_s)+".log";
  logger::set_file(pathstr);

  //main try block
  try
  {
    //initialize simulation
    eamsim sim(beta); //simulation

    //search for previous simulations
    pathpat = sim_dir+"/sim-"+cnfs(beta,5,'0',3)+"-*";
    i_s_f = glob_count(pathpat);

    if (i_s_f==0) //initialize lattice array
    {
      sim.init_lattice();
    }
    else //read lattice array from previous simulation
    {
      pathstr = sim_dir+"/sim-"+cnfs(beta,5,'0',3)+"-";
      pathstr += cnfs(i_s_f-1,2,'0')+".bin";
      inp_f.open(pathstr,std::ios::binary);
      check_file(inp_f,pathstr);
      sim.read_state(inp_f);
    }

    //run Monte Carlo simulation
    pathstr = sim_dir+"/sim-"+cnfs(beta,5,'0',3)+"-";
    pathstr += cnfs(i_s_f,2,'0')+".bin";
    out_f.open(pathstr,std::ios::binary);
    check_file(out_f,pathstr);
    sim.run_MC_simulation(out_f);
  }
  catch (const error &err) //caught error
  {
    //exit program unsuccessfully
    logger::record(err.what());
    return EXIT_FAILURE;
  }

  //remove log file
  logger::set_file("/dev/null");
  pathstr = std::to_string(t_s)+".log";
  std::remove(pathstr.c_str());

  //exit program successfully
  return EXIT_SUCCESS;
}
