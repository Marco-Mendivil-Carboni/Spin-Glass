//Includes

#include "eamsim.cuh" //EA model simulation

//Device Functions

//initialize probability lookup table
inline __device__ void init_prob(
  uint64_t s_prob[NREP][PTABW], //shared probability lookup table
  const int i_bt) //block thread index
{
  //initialize all entries to 1
  if (i_bt<PTABW)
  {
    for (int i_r = 0; i_r<NREP; ++i_r) //replica index
    {
      s_prob[i_r][i_bt] = UINT_MAX;
    }
  }
  __syncthreads();
}

//compute probability lookup table
inline __device__ void compute_prob(
  uint64_t s_prob[NREP][PTABW], //shared probability lookup table
  float *s_rep_beta, //shared replica beta array
  const float H, //external magnetic field
  const int i_bt) //block thread index
{
  //compute all possible probabilities
  if (i_bt<NPROB)
  {
    for (int i_r = 0; i_r<NREP; ++i_r) //replica index
    {
      float energy = -(i_bt-6-H-((1-2*H)*(i_bt&1))); //spin energy
      s_prob[i_r][i_bt] = expf(s_rep_beta[i_r]*2*energy)*UINT_MAX;
    }
  }
  __syncthreads();
}

//shuffle lattice temperature replicas
inline __device__ void shuffle(
  int *s_rep_idx, //shared replica index array
  float *s_rep_beta, //shared replica beta array
  float *s_tot_sum_e, //shared total energy sum array
  prng *prngs, //PRNG state array
  const int i_bt, //block thread index
  const int i_gt, //grid thread index
  bool mode) //shuffle mode
{
  //declare auxiliary variables
  int i_0; //1st array index
  int i_1; //2nd array index
  int max_i_bt; //maximum block thread index
  __shared__ int s_rai[NREP]; //shared rearranged array index array

  //write shared rearranged array index array
  if (i_bt<NREP){ s_rai[s_rep_idx[i_bt]] = i_bt;}
  __syncthreads ();

  if (mode==0) //consider even pairs of temperature replicas
  {
    i_0 = s_rai[(i_bt<<1)+0]; i_1 = s_rai[(i_bt<<1)+1]; max_i_bt = NREP/2;
  }
  else //consider odd pairs of temperature replicas
  {
    i_0 = s_rai[(i_bt<<1)+1]; i_1 = s_rai[(i_bt<<1)+2]; max_i_bt = NREP/2-1;
  }

  if (i_bt<max_i_bt) //shuffle pair of temperature replicas
  {
    //generate random number
    float ran = curand_uniform(&prngs[i_gt]); //random number in (0,1]

    //compute shuffle probability
    float beta_diff = s_rep_beta[i_0]-s_rep_beta[i_1]; //beta difference
    float energy_diff = s_tot_sum_e[i_0]-s_tot_sum_e[i_1]; //energy difference
    float prob = expf(beta_diff*energy_diff); //shuffle probability

    if (ran<prob) //accept shuffle
    {
      int tmp_idx = s_rep_idx[i_0]; //temporary index
      s_rep_idx[i_0] = s_rep_idx[i_1]; s_rep_idx[i_1] = tmp_idx;
      float tmp_beta = s_rep_beta[i_0]; //temporary beta
      s_rep_beta[i_0] = s_rep_beta[i_1]; s_rep_beta[i_1] = tmp_beta;
    }
  }
  __syncthreads();
}

//perform skewed sequential sum reduction
inline __device__ void sum_reduce(
  float *s_tot_sum, //shared total sum array
  short s_aux_sum[NREP][NTPB], //shared auxiliary sums array
  const int i_bt) //block thread index
{
  //sum auxiliary sums for each temperature replica
  if (i_bt<NREP)
  {
    int sum = 0; //sum of auxiliary sums
    for (int i_sl = 0; i_sl<NTPB; ++i_sl) //skewed loop index
    {
      sum += s_aux_sum[i_bt][(i_sl+i_bt)%NTPB];
    }
    s_tot_sum[i_bt] = sum;
  }
  __syncthreads();
}

//perform Monte Carlo steps
inline __device__ void perform_MC_steps(
  uint32_t *slattice, //shuffled lattice array
  float *s_rep_beta, //shared replica beta array
  const float H, //external magnetic field
  prng *prngs, //PRNG state array
  int n_steps) //number of Monte Carlo steps
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = //block thread index
    CBDIM.x*CBDIM.y*threadIdx.z+CBDIM.x*threadIdx.y+threadIdx.x;
  const int i_gt = CBDIM.x*CBDIM.y*CBDIM.z*i_gb+i_bt; //grid thread index

  //declare auxiliary variables
  __shared__ uint32_t s_slattice[L][L][L]; //shared shuffled lattice array
  __shared__ uint64_t s_prob[NREP][PTABW]; //shared probability lookup table

  //initilize probability lookup table
  init_prob(s_prob,i_bt);

  //compute probability lookup table
  compute_prob(s_prob,s_rep_beta,H,i_bt);

  //write shared shuffled lattice array
  int xt = (L/2*(threadIdx.y&1))+threadIdx.x; //total x index
  int yt = (L/2*(threadIdx.z&1))+(threadIdx.y>>1); //total y index
  for (int zt = 0; zt<L; ++zt) //total z index
  {
    s_slattice[zt][yt][xt] = slattice[N*i_gb+L*L*zt+i_bt];
  }
  __syncthreads();

  //perform all Monte Carlo steps
  for (int step = 0; step<n_steps; ++step) //Monte Carlo step index
  {
    //perform both phases of each update
    for (int phase = 0; phase<2; ++phase) //update phase index
    {
      //calculate shared shuffled lattice indexes
      int xs = (threadIdx.z&1)^(threadIdx.y&1)^phase; //starting x index
      int xc = (threadIdx.x<<1)+xs; //centered x index
      int xr = (xc+L-1)%L; //retarded x index
      int xa = (xc+1)%L; //advanced x index
      int yc = threadIdx.y; //centered y index
      int yr = (yc+L-1)%L; //retarded y index
      int ya = (yc+1)%L; //advanced y index
      for (int oz = 0; oz<L; oz += CBDIM.z) //z index offset
      {
        int zc = oz+threadIdx.z; //centered z index
        int zr = (zc+L-1)%L; //retarded z index
        int za = (zc+1)%L; //advanced z index

        //compute interactions with first neighbours
        uint32_t cmspin = s_slattice[zc][yc][xc]; //centered multispin
        uint32_t int_0 = s_slattice[zc][yc][xr]; //interaction 0 (left)
        uint32_t int_1 = s_slattice[zc][yc][xa]; //interaction 1 (right)
        uint32_t int_2 = s_slattice[zc][yr][xc]; //interaction 2 (down)
        uint32_t int_3 = s_slattice[zc][ya][xc]; //interaction 3 (up)
        uint32_t int_4 = s_slattice[zr][yc][xc]; //interaction 4 (back)
        uint32_t int_5 = s_slattice[za][yc][xc]; //interaction 5 (front)
        int_0 = (MASKAB*((cmspin>>(SHIFTSJ+0))&1))^int_0^cmspin;
        int_1 = (MASKAB*((cmspin>>(SHIFTSJ+1))&1))^int_1^cmspin;
        int_2 = (MASKAB*((cmspin>>(SHIFTSJ+2))&1))^int_2^cmspin;
        int_3 = (MASKAB*((cmspin>>(SHIFTSJ+3))&1))^int_3^cmspin;
        int_4 = (MASKAB*((cmspin>>(SHIFTSJ+4))&1))^int_4^cmspin;
        int_5 = (MASKAB*((cmspin>>(SHIFTSJ+5))&1))^int_5^cmspin;

        //flip every spin in the multispin
        for (int i_ss = 0; i_ss<NSPS; ++i_ss) //segment spin index
        {
          //compute energy-spin index
          uint32_t es_idx = //energy-spin index
            ((int_0>>i_ss)&MASKSS)+
            ((int_1>>i_ss)&MASKSS)+
            ((int_2>>i_ss)&MASKSS)+
            ((int_3>>i_ss)&MASKSS)+
            ((int_4>>i_ss)&MASKSS)+
            ((int_5>>i_ss)&MASKSS);
          es_idx = (es_idx<<1)+((cmspin>>i_ss)&MASKSS);

          //compute spin flips
          uint32_t flip = 0; //spin flips
          for (int shift = 0; shift<SHIFTMS; shift += NSPS) //segment shift
          {
            //generate random unsigned integer
            uint64_t ran = curand(&prngs[i_gt]); //random unsigned integer

            //compute flip probability
            uint64_t prob = //flip probability
              s_prob[shift+i_ss][(es_idx>>shift)&MASKES];

            //update spin flips
            flip |= (ran<prob)<<shift;
          }

          //flip spins
          cmspin ^= (flip<<i_ss);
        }

        //update shared shuffled lattice array
        s_slattice[zc][yc][xc] = cmspin;
      }
      __syncthreads();
    }
  }

  //write shuffled lattice array
  for (int zt = 0; zt<L; ++zt) //total z index
  {
    slattice[N*i_gb+L*L*zt+i_bt] = s_slattice[zt][yt][xt];
  }
  __syncthreads();
}

//perform Parallel Tempering shuffle
inline __device__ void perform_PT_shuffle(
  uint32_t *slattice, //shuffled lattice array
  int *s_rep_idx, //shared replica index array
  float *s_rep_beta, //shared replica beta array
  float *s_tot_sum_e, //shared total energy sum array
  float *s_tot_sum_m, //shared total magnetization sum array
  const float H, //external magnetic field
  prng *prngs, //PRNG state array
  bool mode) //shuffle mode
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = //block thread index
    CBDIM.x*CBDIM.y*threadIdx.z+CBDIM.x*threadIdx.y+threadIdx.x;
  const int i_gt = CBDIM.x*CBDIM.y*CBDIM.z*i_gb+i_bt; //grid thread index

  //declare auxiliary variables
  __shared__ uint32_t s_slattice[L][L][L]; //shared shuffled lattice array
  __shared__ short s_aux_sum[NREP][NTPB]; //shared auxiliary sums array

  //write shared shuffled lattice array
  int xt = (L/2*(threadIdx.y&1))+threadIdx.x; //total x index
  int yt = (L/2*(threadIdx.z&1))+(threadIdx.y>>1); //total y index
  for (int zt = 0; zt<L; ++zt) //total z index
  {
    s_slattice[zt][yt][xt] = slattice[N*i_gb+L*L*zt+i_bt];
  }
  __syncthreads();

  //initialize shared auxiliary sums array to 0
  for (int i_r = 0; i_r<NREP; ++i_r) //replica index
  {
    s_aux_sum[i_r][i_bt] = 0;
  }
  __syncthreads();

  //calculate shared shuffled lattice indexes
  int xs = (threadIdx.z&1)^(threadIdx.y&1); //starting x index
  int xc = (threadIdx.x<<1)+xs; //centered x index
  int xr = (xc+L-1)%L; //retarded x index
  int xa = (xc+1)%L; //advanced x index
  int xm = xc-2*xs+1; //matching x index
  int yc = threadIdx.y; //centered y index
  int yr = (yc+L-1)%L; //retarded y index
  int ya = (yc+1)%L; //advanced y index
  for (int oz = 0; oz<L; oz += CBDIM.z) //z index offset
  {
    int zc = oz+threadIdx.z; //centered z index
    int zr = (zc+L-1)%L; //retarded z index
    int za = (zc+1)%L; //advanced z index

    //compute interactions with first neighbours
    uint32_t cmspin = s_slattice[zc][yc][xc]; //centered multispin
    uint32_t int_0 = s_slattice[zc][yc][xr]; //interaction 0 (left)
    uint32_t int_1 = s_slattice[zc][yc][xa]; //interaction 1 (right)
    uint32_t int_2 = s_slattice[zc][yr][xc]; //interaction 2 (down)
    uint32_t int_3 = s_slattice[zc][ya][xc]; //interaction 3 (up)
    uint32_t int_4 = s_slattice[zr][yc][xc]; //interaction 4 (back)
    uint32_t int_5 = s_slattice[za][yc][xc]; //interaction 5 (front)
    int_0 = (MASKAB*((cmspin>>(SHIFTSJ+0))&1))^int_0^cmspin;
    int_1 = (MASKAB*((cmspin>>(SHIFTSJ+1))&1))^int_1^cmspin;
    int_2 = (MASKAB*((cmspin>>(SHIFTSJ+2))&1))^int_2^cmspin;
    int_3 = (MASKAB*((cmspin>>(SHIFTSJ+3))&1))^int_3^cmspin;
    int_4 = (MASKAB*((cmspin>>(SHIFTSJ+4))&1))^int_4^cmspin;
    int_5 = (MASKAB*((cmspin>>(SHIFTSJ+5))&1))^int_5^cmspin;

    //add energy indexes to shared auxiliary sums array
    for (int i_ss = 0; i_ss<NSPS; ++i_ss) //segment spin index
    {
      uint32_t e_idx = //energy index
        ((int_0>>i_ss)&MASKSS)+
        ((int_1>>i_ss)&MASKSS)+
        ((int_2>>i_ss)&MASKSS)+
        ((int_3>>i_ss)&MASKSS)+
        ((int_4>>i_ss)&MASKSS)+
        ((int_5>>i_ss)&MASKSS);
      for (int shift = 0; shift<SHIFTMS; shift += NSPS) //segment shift
      {
        s_aux_sum[shift+i_ss][i_bt] += (e_idx>>shift)&MASKES;
      }
    }
  }
  __syncthreads();

  //perform sum reduction of energy indexes
  sum_reduce(s_tot_sum_e,s_aux_sum,i_bt);

  //reset shared auxiliary sums array to 0
  for (int i_r = 0; i_r<NREP; ++i_r) //replica index
  {
    s_aux_sum[i_r][i_bt] = 0;
  }
  __syncthreads();

  //calculate shared shuffled lattice indexes
  for (int oz = 0; oz<L; oz += CBDIM.z) //z index offset
  {
    int zc = oz+threadIdx.z; //centered z index

    //read lattice multispins
    uint32_t cmpsin = s_slattice[zc][yc][xc]; //centered multispin
    uint32_t mmpsin = s_slattice[zc][yc][xm]; //matching multispin

    //add spin indexes to shared auxiliary sums array
    for (int i_ss = 0; i_ss<NSPS; ++i_ss) //segment spin index
    {
      for (int shift = 0; shift<SHIFTMS; shift += NSPS) //segment shift
      {
        int i_r = shift+i_ss; //replica index
        s_aux_sum[i_r][i_bt] += ((cmpsin>>i_r)&1)+((mmpsin>>i_r)&1);
      }
    }
  }
  __syncthreads();

  //perform sum reduction of spin indexes
  sum_reduce(s_tot_sum_m,s_aux_sum,i_bt);

  //shift both energies to their physical value and add them
  if (i_bt<NREP)
  {
    s_tot_sum_e[i_bt] = 2*s_tot_sum_e[i_bt]-6*(N/2);
    s_tot_sum_m[i_bt] = 2*s_tot_sum_m[i_bt]-1*N;
    s_tot_sum_e[i_bt] = -(s_tot_sum_e[i_bt]+H*s_tot_sum_m[i_bt]);
  }
  __syncthreads();

  //shuffle lattice temperature replicas
  shuffle(s_rep_idx,s_rep_beta,s_tot_sum_e,prngs,i_bt,i_gt,mode);
}

//Global Functions

//initialize PRNG state array
__global__ void init_prng(
  prng *prngs, //PRNG state array
  int pseed) //PRNG seed
{
  //calculate grid thread index
  const int i_gt = NTPB*blockIdx.x+threadIdx.x; //grid thread index

  //initialize PRNG state
  curand_init(pseed,i_gt,0,&prngs[i_gt]);
}

//run simulation section between measurements
__global__ void run_simulation_section(
  uint32_t *slattice, //shuffled lattice array
  obs_s *obs, //observables array
  ib_s *repib, //replica index-beta array
  const float H, //external magnetic field
  prng *prngs) //PRNG state array
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = //block thread index
    CBDIM.x*CBDIM.y*threadIdx.z+CBDIM.x*threadIdx.y+threadIdx.x;

  //declare auxiliary variables
  __shared__ int s_rep_idx[NREP]; //shared replica index array
  __shared__ float s_rep_beta[NREP]; //shared replica beta array
  __shared__ int s_prev_rep_idx[NREP]; //shared previous replica index array
  __shared__ float s_tot_sum_e[NREP]; //shared total energy sum array
  __shared__ float s_tot_sum_m[NREP]; //shared total magnetization sum array

  //write shared replica index and beta arrays
  if (i_bt<NREP)
  {
    s_rep_idx[i_bt] = repib[NREP*i_gb+i_bt].idx;
    s_rep_beta[i_bt] = repib[NREP*i_gb+i_bt].beta;
  }
  __syncthreads();

  //perform Parallel Tempering shuffles and Monte Carlo steps
  for (int step = 0; step<SBMEAS; step += SBSHFL) //Monte Carlo step index
  {
    bool mode = (step/SBSHFL)&1; //shuffle mode
    if (i_bt<NREP){ s_prev_rep_idx[i_bt] = s_rep_idx[i_bt];}
    perform_PT_shuffle(slattice,s_rep_idx,s_rep_beta,s_tot_sum_e,s_tot_sum_m,H,
      prngs,mode);
    perform_MC_steps(slattice,s_rep_beta,H,prngs,SBSHFL);
  }

  //update replica index-beta array
  if (i_bt<NREP)
  {
    repib[NREP*i_gb+i_bt].idx = s_rep_idx[i_bt];
    repib[NREP*i_gb+i_bt].beta = s_rep_beta[i_bt];
  }

  //write observables array
  if (i_bt<NREP)
  {
    int i_d = i_gb/NCP; //disorder index
    int i_r = s_prev_rep_idx[i_bt]; //replica index
    int i_c = i_gb%NCP; //copy index
    obs[NREP*i_d+i_r].e[i_c] = s_tot_sum_e[i_bt]/N;
    obs[NREP*i_d+i_r].m[i_c] = s_tot_sum_m[i_bt]/N;
  }
}

//rearrange lattice temperature replicas
__global__ void rearrange(
  uint32_t *lattice, //lattice array
  ib_s *repib, //replica index-beta array
  uint32_t *slattice) //shuffled lattice array
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = threadIdx.x; //block thread index

  //declare auxiliary variables
  uint32_t smspin; //shuffled multispin
  uint32_t rmspin; //rearranged multispin
  __shared__ int s_rep_idx[NREP]; //shared replica index array

  //write shared replica index array
  if (i_bt<NREP){ s_rep_idx[i_bt] = repib[NREP*i_gb+i_bt].idx;}
  __syncthreads();

  //update lattice array
  for (int i_s = i_bt; i_s<N; i_s += NTPB) //site index
  {
    smspin = slattice[N*i_gb+i_s];
    rmspin = 0;
    for (int i_r = 0; i_r<NREP; ++i_r) //replica index
    {
      rmspin |= ((smspin>>i_r)&1)<<s_rep_idx[i_r];
    }
    lattice[N*i_gb+i_s] = (lattice[N*i_gb+i_s]&MASKAJ)|rmspin;
  }
}

//compute overlap
__global__ void compute_q(
  obs_s *obs, //observables array
  uint32_t *lattice) //lattice array
{
  //calculate indexes
  const int i_gb = blockIdx.x; //grid block index
  const int i_bt = threadIdx.x; //block thread index

  //declare auxiliary variables
  int l_s[NCP]; //local spin array
  float q_0; //overlap value 0
  float q_1_r[3]; //Re overlap values 1
  float q_1_i[3]; //Im overlap values 1

  //compute overlap values for each temperature replica
  if (i_bt<NREP)
  {
    //initialize overlap values
    q_0 = 0.0;
    for(int i_q = 0; i_q<3; ++i_q) //overlap value index
    {
      q_1_r[i_q] = 0.0;
      q_1_i[i_q] = 0.0;
    }

    //iterate over all sites
    for (int i_s = 0; i_s<N; ++i_s) //site index
    {
      //compute local spin array
      for (int i_c = 0; i_c<NCP; ++i_c) //copy index
      {
        int sbit = (lattice[N*(NCP*i_gb+i_c)+i_s]>>i_bt)&1; //spin bit
        l_s[i_c] = 2*sbit-1;
      }

      //calculate local overlap, wave vector norm and position
      float l_q = l_s[0]*l_s[1]; //local overlap
      float k = 2*M_PI/L; //wave vector norm
      float x = i_s%L; //x position
      float y = (i_s/L)%L; //y position
      float z = (i_s/L)/L; //z position

      //compute overlap values
      q_0 += l_q;
      q_1_r[0] += l_q*cosf(k*x);
      q_1_i[0] += l_q*sinf(k*x);
      q_1_r[1] += l_q*cosf(k*y);
      q_1_i[1] += l_q*sinf(k*y);
      q_1_r[2] += l_q*cosf(k*z);
      q_1_i[2] += l_q*sinf(k*z);
    }

    //write observables array
    obs[NREP*i_gb+i_bt].q_0 = q_0/N;
    for(int i_q = 0; i_q<3; ++i_q) //overlap value index
    {
      obs[NREP*i_gb+i_bt].q_1_r[i_q] = q_1_r[i_q]/N;
      obs[NREP*i_gb+i_bt].q_1_i[i_q] = q_1_i[i_q]/N;
    }
  }
}

//Host Functions

//EA model simulation constructor
eamsim::eamsim(float H) //external magnetic field
  : H {H}
{
  //check parameters
  if (!(0.0<=H&&H<=4.0)){ throw error("H out of range");}
  logger::record("H = "+cnfs(H,6,'0',4));

  //allocate device memory
  cuda_check(cudaMalloc(&repib,NREP*NL*sizeof(ib_s)));
  cuda_check(cudaMalloc(&prngs,NTPB*NL*sizeof(prng)));
  cuda_check(cudaMalloc(&lattice,N*NL*sizeof(uint32_t)));
  cuda_check(cudaMalloc(&slattice,N*NL*sizeof(uint32_t)));
  cuda_check(cudaMalloc(&obs,NREP*NDIS*sizeof(obs_s)));

  //allocate host memory
  cuda_check(cudaMallocHost(&repib_h,NREP*NL*sizeof(ib_s)));
  cuda_check(cudaMallocHost(&lattice_h,N*NL*sizeof(uint32_t)));
  cuda_check(cudaMallocHost(&obs_h,NREP*NDIS*sizeof(obs_s)));

  //initialize replica index-beta array
  init_repib();

  //initialize PRNG state array
  init_prng<<<NL,NTPB>>>(prngs,time(nullptr));

  //record success message
  logger::record("eamsim initialized");
}

//EA model simulation destructor
eamsim::~eamsim()
{
  //deallocate device memory
  cuda_check(cudaFree(repib));
  cuda_check(cudaFree(prngs));
  cuda_check(cudaFree(lattice));
  cuda_check(cudaFree(slattice));
  cuda_check(cudaFree(obs));

  //deallocate host memory
  cuda_check(cudaFreeHost(repib_h));
  cuda_check(cudaFreeHost(lattice_h));
  cuda_check(cudaFreeHost(obs_h));
}

//initialize lattice multispins
void init_multispins(
  curandGenerator_t gen, //host PRNG
  uint32_t *lattice_h) //lattice host array
{
  //declare auxiliary variables
  uint32_t ranmspin; //random multispin

  //set random lattice multispins
  for (int i_s = 0; i_s<N; ++i_s) //site index
  {
    curandGenerate(gen,&ranmspin,1);
    lattice_h[i_s] = (lattice_h[i_s]&MASKAJ)|(ranmspin&MASKAS);
  }
}

//initialize lattice coupling constants
void init_coupling_constants(
  curandGenerator_t gen, //host PRNG
  uint32_t *lattice_h) //lattice host array
{
  //declare auxiliary variables
  uint32_t ran[3]; //random numbers
  uint32_t Jx[N]; //x coupling constants
  uint32_t Jy[N]; //y coupling constants
  uint32_t Jz[N]; //z coupling constants

  //choose random coupling constants
  for (int i_s = 0; i_s<N; ++i_s) //site index
  {
    curandGenerate(gen,ran,3);
    Jx[i_s] = ran[0]&1;
    Jy[i_s] = ran[1]&1;
    Jz[i_s] = ran[2]&1;
  }

  //copy coupling constants to lattice
  for (int xa = 0; xa<L; ++xa) //advanced x index
  {
    int xr = (xa+L-1)%L; //retarded x index
    for (int ya = 0; ya<L; ++ya) //advanced y index
    {
      int yr = (ya+L-1)%L; //retarded y index
      for (int za = 0; za<L; ++za) //advanced z index
      {
        int zr = (za+L-1)%L; //retarded z index
        uint32_t J = //site's coupling constants
          (MASKSJ<<0)*Jx[L*L*za+L*ya+xr]|
          (MASKSJ<<1)*Jx[L*L*za+L*ya+xa]|
          (MASKSJ<<2)*Jy[L*L*za+L*yr+xa]|
          (MASKSJ<<3)*Jy[L*L*za+L*ya+xa]|
          (MASKSJ<<4)*Jz[L*L*zr+L*ya+xa]|
          (MASKSJ<<5)*Jz[L*L*za+L*ya+xa];
        int i_s = L*L*za+L*ya+xa; //site index
        lattice_h[i_s] = J|(lattice_h[i_s]&MASKAS);
      }
    }
  }
}

//initialize lattice array
void eamsim::init_lattice()
{
  //initialize host PRNG
  curandGenerator_t gen; //host PRNG
  curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,time(nullptr));

  //initialize every lattice in the host
  for (int i_l = 0; i_l<NL; ++i_l) //lattice index
  {
    if ((i_l%NCP)==0) //initialize lattice coupling constants
    {
      init_coupling_constants(gen,&lattice_h[N*i_l]);
    }
    else //use the same coupling constants for adjacent lattices
    {
      cuda_check(cudaMemcpy(&lattice_h[N*i_l],&lattice_h[N*(i_l-1)],
        N*sizeof(uint32_t),cudaMemcpyHostToHost));
    }

    //initialize lattice multispins
    init_multispins(gen,&lattice_h[N*i_l]);
  }

  //copy lattice host array to device
  cuda_check(cudaMemcpy(lattice,lattice_h,N*NL*sizeof(uint32_t),
    cudaMemcpyHostToDevice));

  //record success message
  logger::record("lattice array initialized");
}

//save state to binary file
void eamsim::save_checkpoint(std::ofstream &bin_out_f) //binary output file
{
  //write lattice array to binary file
  bin_out_f.write(reinterpret_cast<char *>(lattice_h),N*NL*sizeof(uint32_t));

  //record success message
  logger::record("checkpoint saved");
}

//load state from binary file
void eamsim::load_checkpoint(std::ifstream &bin_inp_f) //binary input file
{
  //read lattice array from binary file
  bin_inp_f.read(reinterpret_cast<char *>(lattice_h),N*NL*sizeof(uint32_t));

  //copy lattice array to device
  cuda_check(cudaMemcpy(lattice,lattice_h,N*NL*sizeof(uint32_t),
    cudaMemcpyHostToDevice));

  //record success message
  logger::record("checkpoint loaded");
}

//run whole simulation
void eamsim::run_simulation(std::ofstream &bin_out_f) //binary output file
{
  //declare and initialize execution time
  float t_e = clock(); //execution time

  //copy lattice array to shuffled lattice array
  cuda_check(cudaMemcpy(slattice,lattice,N*NL*sizeof(uint32_t),
    cudaMemcpyDeviceToDevice));

  //run whole simulation
  for (int step = 0; step<SPFILE; step += SBMEAS) //Monte Carlo step index
  {
    //show simulation progress
    logger::show_prog_pc(100.0*step/SPFILE);

    //run simulation section between measurements
    run_simulation_section<<<NL,CBDIM>>>(slattice,obs,repib,H,prngs);

    //rearrange lattice temperature replicas
    rearrange<<<NL,NTPB>>>(lattice,repib,slattice);

    //compute overlap
    compute_q<<<NDIS,NTPB>>>(obs,lattice);

    //copy observables array to host
    cuda_check(cudaMemcpy(obs_h,obs,NREP*NDIS*sizeof(obs_s),
      cudaMemcpyDeviceToHost));

    //write observables to binary file
    bin_out_f.write(reinterpret_cast<char *>(obs_h),NREP*NDIS*sizeof(obs_s));
  }

  //copy lattice array to host
  cuda_check(cudaMemcpy(lattice_h,lattice,N*NL*sizeof(uint32_t),
    cudaMemcpyDeviceToHost));

  //compute and record execution time
  t_e = (clock()-t_e)/CLOCKS_PER_SEC;
  t_e *= (1e12/SPFILE)/(N*NL*NREP);
  logger::record("t_e = "+cnfs(t_e,5,'0',2)+" ps");

  //record success message
  logger::record("simulation ended");
}

//initialize replica index-beta array
void eamsim::init_repib()
{
  //declare auxiliary variables
  const float max_beta = 2.0; //maximum beta
  const float bratio = pow(2.0,-4/(NREP-1.0)); //beta ratio

  //initialize replica index-beta host array
  for (int i_l = 0; i_l<NL; ++i_l) //lattice index
  {
    for (int i_r = 0; i_r<NREP; ++i_r) //replica index
    {
      repib_h[NREP*i_l+i_r].idx = i_r;
      repib_h[NREP*i_l+i_r].beta = pow(bratio,i_r)*max_beta;
    }
  }

  //copy replica index-beta host array to device
  cuda_check(cudaMemcpy(repib,repib_h,NREP*NL*sizeof(ib_s),
    cudaMemcpyHostToDevice));
}
