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

  float *m; //magnetization array
  float2 *q; //overlap array

  float *m_h; //magnetization host array
  float2 *q_h; //overlap host array

  //Functions

  //write observables to text file
  void write_obs(std::ofstream &txt_out_f); //text output file
};

#endif //MMC_EAMANA_H
