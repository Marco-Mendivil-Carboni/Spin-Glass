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

  //parse command-line arguments
  const std::string sim_dir = argv[1]; //simulation directory
  const float H = std::stof(argv[2]); //external magnetic field

  //declare auxiliary variables
  std::string simpathbeg = //simulation file path beginning
    sim_dir+"/"+cnfs(H,5,'0',3)+"-";
  std::string path; //complete file path
  std::string pattern; //file path pattern
  std::ifstream inp_f; //input file
  std::ofstream out_f; //output file
  bool prev_sim; //previous simulations

  //create log file in current working directory
  time_t t_s = time(nullptr); //starting time
  path = std::to_string(t_s)+".log";
  logger::set_file(path);

  try //main try block
  {
    //initialize simulation
    eamsim sim = eamsim(H); //simulation

    //look for previous simulations
    pattern = simpathbeg+"obs.dat";
    prev_sim = glob_count(pattern);

    if (!prev_sim) //initialize lattice array
    {
      sim.init_lattice();
    }
    else //load state from binary file
    {
      path = simpathbeg+"chkpt.bin";
      inp_f.open(path,std::ios::binary);
      check_file(inp_f,path);
      sim.load_checkpoint(inp_f);
      inp_f.close();
    }

    //run whole simulation
    path = simpathbeg+"obs.dat";
    out_f.open(path,std::ios::app);
    check_file(out_f,path);
    sim.run_simulation(out_f);
    out_f.close();

    //save state to binary file
    path = simpathbeg+"chkpt.bin";
    out_f.open(path,std::ios::binary);
    check_file(out_f,path);
    sim.save_checkpoint(out_f);
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
