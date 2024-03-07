//Includes

#include "eamana.cuh" //EA model analysis

//Global Functions

//compute overlap array
__global__ void compute_q(
  float2 *q, //overlap array
  uint32_t *lattice) //lattice array
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = threadIdx.x; //block thread index
  const int i_fl = i_gb*NCP; //first lattice index

  //declare auxiliary variables
  __shared__ float s_l_q[N]; //shared local overlap array
  __shared__ float2 s_q[NQVAL]; //shared overlap array

  //compute shared local overlap array
  int l_q_b; //local overlap bit
  for (int i_s = i_bt; i_s<N; i_s += NTPB) //site index
  {
    l_q_b = (lattice[N*i_fl+i_s]^lattice[N*(i_fl+1)+i_s])&1;
    s_l_q[i_s] = 1-(l_q_b<<1);
  }
  __syncthreads();

  if (i_bt==0) //compute overlap array
  {
    //initialize shared overlap array
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      s_q[i_qv].x = 0.0;
      s_q[i_qv].y = 0.0;
    }

    //iterate over all sites
    for (int i_s = 0; i_s<N; ++i_s) //site index
    {
      //calculate position and wave number
      float x = i_s%L; //x position
      float y = (i_s/L)%L; //y position
      float z = (i_s/L)/L; //z position
      float k = 2*M_PI/L; //wave number

      //read local overlap
      float l_q = s_l_q[i_s]; //local overlap

      //compute overlap values
      s_q[0].x += l_q*cosf(0);
      s_q[0].y += l_q*sinf(0);
      s_q[1].x += l_q*cosf(k*x);
      s_q[1].y += l_q*sinf(k*x);
      s_q[2].x += l_q*cosf(k*y);
      s_q[2].y += l_q*sinf(k*y);
      s_q[3].x += l_q*cosf(k*z);
      s_q[3].y += l_q*sinf(k*z);
    }

    //normalize shared overlap array
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      s_q[i_qv].x /= N;
      s_q[i_qv].y /= N;
    }

    //write overlap array
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      q[NQVAL*i_gb+i_qv].x = s_q[i_qv].x;
      q[NQVAL*i_gb+i_qv].y = s_q[i_qv].y;
    }
  }
}

//Host Functions

//EA model analysis constructor
eamana::eamana()
  : eamdat()
{
  //allocate device memory
  cuda_check(cudaMalloc(&q,NDIS*NQVAL*sizeof(float2)));

  //allocate host memory
  cuda_check(cudaMallocHost(&q_h,NDIS*NQVAL*sizeof(float2)));

  //record success message
  logger::record("eamana initialized");
}

//EA model analysis destructor
eamana::~eamana()
{
  //deallocate device memory
  cuda_check(cudaFree(q));

  //deallocate host memory
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

    //compute overlap array
    compute_q<<<NDIS,NTPB>>>(q,lattice);

    //copy overlap array to host
    cuda_check(cudaMemcpy(q_h,q,NDIS*NQVAL*sizeof(float2),
      cudaMemcpyDeviceToHost));

    //write overlap host array to text file
    write_q_h(txt_out_f);
  }
}

//write overlap host array to text file
void eamana::write_q_h(std::ofstream &txt_out_f) //text output file
{
  for (int i_dr = 0; i_dr<NDIS; ++i_dr) //disorder realization index
  {
    for(int i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      txt_out_f<<cnfs(q_h[NQVAL*i_dr+i_qv].x,12,' ',6);
      txt_out_f<<cnfs(q_h[NQVAL*i_dr+i_qv].y,12,' ',6);
    }
    txt_out_f<<"\n";
  }
  txt_out_f<<"\n";
}
