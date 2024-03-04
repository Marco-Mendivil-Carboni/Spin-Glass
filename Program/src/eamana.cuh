#ifndef MMC_EAMANA_H
#define MMC_EAMANA_H

//Includes

#include "eamdat.cuh" //EA model data

//Classes

class eamana : public eamdat //EA model analysis
{
  public:

  //Functions

  //EA model analysis constructor
  eamana();

  //EA model analysis destructor
  ~eamana();

  //process simulation file
  void process_sim_file(
    std::ofstream &txt_out_f, //text output file
    std::ifstream &bin_inp_f); //binary input file

  private:

  //Parameters and Variables

  float2 *q; //overlap array

  float2 *q_h; //overlap host array

  //Functions

  //write overlap host array to text file
  void write_q_h(std::ofstream &txt_out_f); //text output file
};

#endif //MMC_EAMANA_H
