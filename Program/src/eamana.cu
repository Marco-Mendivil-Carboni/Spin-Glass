//Includes

#include "eamana.cuh" //EA model analysis

//Global Functions

//compute observables
__global__ void compute_obs(
  float *m, //magnetization array
  float2 *q, //overlap array
  uint32_t *lattice) //lattice array
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = threadIdx.x; //block thread index

  //declare auxiliary variables
  int l_s[NCP]; //local spin array
  double m_v; //magnetization value
  double2 q_v[NQVAL]; //overlap values

  if (i_bt<NREP) //compute observables for each temperature replica
  {
    //initialize observable values
    m_v = 0.0;
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      q_v[i_qv].x = 0.0;
      q_v[i_qv].y = 0.0;
    }

    //iterate over all sites
    for (int i_s = 0; i_s<N; ++i_s) //site index
    {
      //compute local spin array and magnetization value
      for (int i_c = 0; i_c<NCP; ++i_c) //copy index
      {
        int sbit = (lattice[N*(NCP*i_gb+i_c)+i_s]>>i_bt)&1; //spin bit
        l_s[i_c] = 2*sbit-1;
        m_v += l_s[i_c];
      }

      //calculate local overlap, wave number and position
      float l_q = (l_s[0]-l_s[1])*(l_s[2]-l_s[3])/2; //local overlap
      float k = 2*M_PI/L; //wave number
      float x = i_s%L; //x position
      float y = (i_s/L)%L; //y position
      float z = (i_s/L)/L; //z position

      //compute overlap values
      q_v[0].x += l_q*cosf(0);
      q_v[0].y += l_q*sinf(0);
      q_v[1].x += l_q*cosf(k*x);
      q_v[1].y += l_q*sinf(k*x);
      q_v[2].x += l_q*cosf(k*y);
      q_v[2].y += l_q*sinf(k*y);
      q_v[3].x += l_q*cosf(k*z);
      q_v[3].y += l_q*sinf(k*z);
    }

    //normalize observable values
    m_v /= NCP*N;
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      q_v[i_qv].x /= N;
      q_v[i_qv].y /= N;
    }

    //write observable arrays
    m[NREP*i_gb+i_bt] = m_v;
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      q[NQVAL*(NREP*i_gb+i_bt)+i_qv].x = q_v[i_qv].x;
      q[NQVAL*(NREP*i_gb+i_bt)+i_qv].y = q_v[i_qv].y;
    }
  }
}

//Host Functions

//EA model analysis constructor
eamana::eamana()
  : eamdat()
{
  //allocate device memory
  cuda_check(cudaMalloc(&m,NDIS*NREP*sizeof(float)));
  cuda_check(cudaMalloc(&q,NDIS*NREP*NQVAL*sizeof(float2)));

  //allocate host memory
  cuda_check(cudaMallocHost(&m_h,NDIS*NREP*sizeof(float)));
  cuda_check(cudaMallocHost(&q_h,NDIS*NREP*NQVAL*sizeof(float2)));

  //record success message
  logger::record("eamana initialized");
}

//EA model analysis destructor
eamana::~eamana()
{
  //deallocate device memory
  cuda_check(cudaFree(m));
  cuda_check(cudaFree(q));

  //deallocate host memory
  cuda_check(cudaFreeHost(m_h));
  cuda_check(cudaFreeHost(q_h));
}

//process simulation file
void eamana::process_sim_file(
  std::ofstream &txt_out_f, //text output file
  std::ifstream &bin_inp_f) //binary input file
{
  //read all states in the input file
  for (int i_m = 0; i_m<(SPFILE/SBMEAS); ++i_m) //measurement index
  {
    //read state from binary file
    read_state(bin_inp_f);

    //compute observables
    compute_obs<<<NDIS,NTPB>>>(m,q,lattice);

    //copy observable arrays to host
    cuda_check(cudaMemcpy(m_h,m,NDIS*NREP*sizeof(float),
      cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(q_h,q,NDIS*NREP*NQVAL*sizeof(float2),
      cudaMemcpyDeviceToHost));

    //write observables to text file
    write_obs(txt_out_f);
  }

  //record success message
  logger::record("simulation file processed");
}

//write observables to text file
void eamana::write_obs(std::ofstream &txt_out_f) //text output file
{
  for (int i_dr = 0; i_dr<NDIS; ++i_dr) //disorder realization index
  {
    for (int i_b = 0; i_b<NREP; ++i_b) //beta index
    {
      txt_out_f<<cnfs(m_h[NREP*i_dr+i_b],12,' ',6);
      for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
      {
        txt_out_f<<cnfs(q_h[NQVAL*(NREP*i_dr+i_b)+i_qv].x,12,' ',6);
        if (i_qv!=0)
        {
          txt_out_f<<cnfs(q_h[NQVAL*(NREP*i_dr+i_b)+i_qv].y,12,' ',6);
        }
      }
      txt_out_f<<"\n";
    }
    txt_out_f<<"\n";
  }
  txt_out_f<<"\n";
}
