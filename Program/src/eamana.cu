//Includes

#include "eamana.cuh" //EA model analysis

//Constants

//Device Functions

//Global Functions

// //...
// __global__ void kernel_compute_q(int rec)
// {
//   const int bidx = threadIdx.x;

//   // sizeof(u_int32_t) * 16 * 16 * 16 = 16 KB
//   __shared__ MSC_DATATYPE l1[SZ_CUBE];
//   __shared__ double qk_real[3][NBETA];
//   __shared__ double qk_imag[3][NBETA];
//   __shared__ double qk2_real[6][NBETA];
//   __shared__ double qk2_imag[6][NBETA];
//   const int lattice_offset0 = SZ_CUBE * (blockIdx.x << 1);
//   const int lattice_offset1 = lattice_offset0 + SZ_CUBE;
//   const double k=2*PI/SZ;

//   for (int offset = 0; offset < SZ_CUBE; offset += TperB) {
//     l1[offset + bidx] =		// xord_word
//       plattice1[lattice_offset0 + offset + bidx] ^
//       plattice1[lattice_offset1 + offset + bidx];
//   }

//   __syncthreads();

//   // is double an overkill?
//   if (bidx < NBETA) {
//     float q0 = 0.0f;
//     for(int j=0;j<3;j++){
//       qk_real[j][bidx] = 0.0f;
//       qk_imag[j][bidx] = 0.0f;
//     }
//     for(int j=0;j<6;j++){
//       qk2_real[j][bidx] = 0.0f;
//       qk2_imag[j][bidx]= 0.0f;
//     }

//     MSC_DATATYPE xor_word;
//     int xor_bit;

//     for (int i = 0; i < SZ_CUBE; i++) {
//       xor_word = l1[i];
//       xor_bit = (xor_word >> bidx) & 0x1;
//       xor_bit = 1 - (xor_bit << 1);	// parallel: +1, reverse: -1

//       double bit=xor_bit;
//       double x= i % SZ;
//       double y= (i / SZ) % SZ;
//       double z= (i / SZ) / SZ;

//       q0 += bit;

//       qk_real[0][bidx] += bit * cos(x*k);
//       qk_real[1][bidx] += bit * cos(y*k);
//       qk_real[2][bidx] += bit * cos(z*k);

//       qk_imag[0][bidx] += bit * sin(x*k);
//       qk_imag[1][bidx] += bit * sin(y*k);
//       qk_imag[2][bidx] += bit * sin(z*k);

//       qk2_real[0][bidx] += bit * cos(x*k + y*k);
//       qk2_real[1][bidx] += bit * cos(x*k - y*k);
//       qk2_real[2][bidx] += bit * cos(x*k + z*k);
//       qk2_real[3][bidx] += bit * cos(x*k - z*k);
//       qk2_real[4][bidx] += bit * cos(y*k + z*k);
//       qk2_real[5][bidx] += bit * cos(y*k - z*k);

//       qk2_imag[0][bidx] += bit * sin(x*k + y*k);
//       qk2_imag[1][bidx] += bit * sin(x*k - y*k);
//       qk2_imag[2][bidx] += bit * sin(x*k + z*k);
//       qk2_imag[3][bidx] += bit * sin(x*k - z*k);
//       qk2_imag[4][bidx] += bit * sin(y*k + z*k);
//       qk2_imag[5][bidx] += bit * sin(y*k - z*k);
//    }

//     pst[rec].q[blockIdx.x][bidx] = q0;
//     for(int j=0;j<3;j++){
//       pst[rec].qk_real[j][blockIdx.x][bidx] = qk_real[j][bidx];
//       pst[rec].qk_imag[j][blockIdx.x][bidx] = qk_imag[j][bidx];
//     }
//     for(int j=0;j<6;j++){
//       pst[rec].qk2_real[j][blockIdx.x][bidx] = qk2_real[j][bidx];
//       pst[rec].qk2_imag[j][blockIdx.x][bidx] = qk2_imag[j][bidx];
//     }
//   }
//   __syncthreads();
// }

//Host Functions

//EA model analysis constructor
eamana::eamana()
  : eamdat()
{
  //check parameters

  //allocate device memory

  //allocate host memory

  //record success message
  logger::record("eamana initialized");
}

//EA model analysis destructor
eamana::~eamana()
{
  //deallocate device memory

  //deallocate host memory
}
