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
  if (argc<4){ std::cout<<"missing arguments\n"; return EXIT_FAILURE;}
  if (argc>4){ std::cout<<"extra arguments\n"; return EXIT_FAILURE;}

  //parse command-line arguments
  const std::string sim_dir = argv[1]; //simulation directory
  const float beta = std::stof(argv[2]); //inverse temperature
  const float H = std::stof(argv[3]); //external magnetic field

  //declare auxiliary variables
  std::string simpathbeg = //simulation file path beginning
    sim_dir+"/"+cnfs(beta,5,'0',3)+"-"+cnfs(H,5,'0',3)+"-";
  std::string path; //complete file path
  std::string pattern; //file path pattern
  std::ifstream inp_f; //input file
  std::ofstream out_f; //output file
  uint i_s_f; //simulation file index

  //create log file in current working directory
  time_t t_s = time(nullptr); //starting time
  path = std::to_string(t_s)+".log";
  logger::set_file(path);

  try //main try block
  {
    //initialize simulation
    eamsim sim = eamsim(beta,H); //simulation

    //search for previous simulations
    pattern = simpathbeg+"*";
    i_s_f = glob_count(pattern);

    if (i_s_f==0) //initialize lattice array
    {
      sim.init_lattice();
    }
    else //read last state from previous simulation
    {
      path = simpathbeg+cnfs(i_s_f-1,2,'0')+".bin";
      inp_f.open(path,std::ios::binary);
      check_file(inp_f,path);
      sim.read_last_state(inp_f);
      inp_f.close();
    }

    //run whole simulation
    path = simpathbeg+cnfs(i_s_f,2,'0')+".bin";
    out_f.open(path,std::ios::binary);
    check_file(out_f,path);
    sim.run_simulation(out_f);
    out_f.close();
  }
  catch (const error &err) //caught error
  {
    //exit program unsuccessfully
    logger::record(err.what());
    return EXIT_FAILURE;
  }

  //remove log file
  logger::set_file("/dev/null");
  path = std::to_string(t_s)+".log";
  std::remove(path.c_str());

  //exit program successfully
  return EXIT_SUCCESS;
}
