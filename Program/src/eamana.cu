//Includes

#include "eamana.cuh" //EA model analysis

//Global Functions

//compute overlap array
__global__ void compute_q(
  float2 *q, //overlap array
  uint *lattice) //lattice array
{
  //calculate indexes
  const uint i_gb = blockIdx.x; //grid block index
  const uint i_bt = threadIdx.x; //block thread index
  const uint i_l = i_gb<<1; //lattice index

  //declare auxiliary variables
  __shared__ float s_l_q[N]; //shared local overlap array
  __shared__ double2 s_q[NQVAL]; //shared overlap array

  //compute shared local overlap array
  int l_q_b; //local overlap bit
  for (uint i_s = i_bt; i_s<N; i_s += NTPB) //site index
  {
    l_q_b = (lattice[N*i_l+i_s]^lattice[N*(i_l+1)+i_s])&1;
    s_l_q[i_s] = 1-(l_q_b<<1);
  }
  __syncthreads();

  if (i_bt==0) //compute overlap array
  {
    //initialize shared overlap array
    for(uint i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      s_q[i_qv].x = 0.0;
      s_q[i_qv].y = 0.0;
    }

    //iterate over all sites
    for (uint i_s = 0; i_s<N; ++i_s) //site index
    {
      //calculate position and wave number
      float x = i_s%L; //x position
      float y = (i_s/L)%L; //y position
      float z = (i_s/L)/L; //z position
      float k = 2*M_PI/L; //wave number

      //read local overlap
      float l_q = s_l_q[i_s]; //local overlap

      //compute overlap value 0
      s_q[0].x += l_q*cosf(0);
      s_q[0].y += l_q*sinf(0);

      //compute overlap value 1
      s_q[1].x += l_q*cosf(k*x);
      s_q[1].x += l_q*cosf(k*y);
      s_q[1].x += l_q*cosf(k*z);
      s_q[1].y += l_q*sinf(k*x);
      s_q[1].y += l_q*sinf(k*y);
      s_q[1].y += l_q*sinf(k*z);

      //compute overlap value 2
      s_q[2].x += l_q*cosf(k*x+k*y);
      s_q[2].x += l_q*cosf(k*x-k*y);
      s_q[2].x += l_q*cosf(k*x+k*z);
      s_q[2].x += l_q*cosf(k*x-k*z);
      s_q[2].x += l_q*cosf(k*y+k*z);
      s_q[2].x += l_q*cosf(k*y-k*z);
      s_q[2].y += l_q*sinf(k*x+k*y);
      s_q[2].y += l_q*sinf(k*x-k*y);
      s_q[2].y += l_q*sinf(k*x+k*z);
      s_q[2].y += l_q*sinf(k*x-k*z);
      s_q[2].y += l_q*sinf(k*y+k*z);
      s_q[2].y += l_q*sinf(k*y-k*z);
    }

    //normalize overlap value 0
    s_q[0].x /= N;
    s_q[0].y /= N;

    //normalize overlap value 1
    s_q[1].x /= 3*N;
    s_q[1].y /= 3*N;

    //normalize overlap value 2
    s_q[2].x /= 6*N;
    s_q[2].y /= 6*N;

    //write overlap array
    for(uint i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
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
  cuda_check(cudaMalloc(&q,(NDIS/2)*NQVAL*sizeof(float2)));

  //allocate host memory
  cuda_check(cudaMallocHost(&q_h,(NDIS/2)*NQVAL*sizeof(float2)));

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
  for (uint i_m = 0; i_m<(SPFILE/SBMEAS); ++i_m) //measurement index
  {
    //read state from binary file
    read_state(bin_inp_f);

    //compute overlap array
    compute_q<<<NBPG/2,NTPB>>>(q,lattice);

    //copy overlap array to host
    cuda_check(cudaMemcpy(q_h,q,(NDIS/2)*NQVAL*sizeof(float2),
      cudaMemcpyDeviceToHost));

    //write overlap host array to text file
    write_q_h(txt_out_f);
  }
}

//write overlap host array to text file
void eamana::write_q_h(std::ofstream &txt_out_f) //text output file
{
  for (uint i_uq = 0; i_uq<(NDIS/2); ++i_uq) //unique disorder index
  {
    for(uint i_qv = 0; i_qv<NQVAL; ++i_qv) //overlap value index
    {
      txt_out_f<<cnfs(q_h[NQVAL*i_uq+i_qv].x,10,' ',6)<<" ";
      txt_out_f<<cnfs(q_h[NQVAL*i_uq+i_qv].y,10,' ',6)<<" | ";
    }
  }
  txt_out_f<<"\n";
}
