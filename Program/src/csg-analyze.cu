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
  uint n_s_f; //number of simulation files

  try //main try block
  {
    //initialize analysis
    eamana ana; //analysis

    //count the number of simulation files
    pattern = simpathbeg+"*";
    n_s_f = glob_count(pattern);

    //add all simulation files to analysis
    for (uint i_s_f = 0; i_s_f<n_s_f; ++i_s_f) //simulation file index
    {
      path = simpathbeg+cnfs(i_s_f,2,'0')+".bin";
      inp_f.open(path,std::ios::binary);
      check_file(inp_f,path);
      // ana.add_file(inp_f); ...................................................
      inp_f.close();
    }

    //calculate statistics
    // ana.calc_stats(); ........................................................

    //save statistics
    path = simpathbeg+"stats.dat";
    out_f.open(path);
    check_file(out_f,path);
    // ana.save_stats(out_f); ...................................................
    out_f.close();
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
