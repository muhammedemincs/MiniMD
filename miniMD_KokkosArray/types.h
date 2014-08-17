/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#ifndef TYPES_H
#define TYPES_H

enum ForceStyle {FORCELJ, FORCEEAM};


#ifdef _OPENMP
#include <Kokkos_OpenMP.hpp>
#else
#include <Kokkos_Threads.hpp>
#endif
#include "Kokkos_View.hpp"
#include "Kokkos_Macros.hpp"

#ifndef DEVICE
#define DEVICE 1
#endif
#if DEVICE==1
#ifdef _OPENMP
typedef Kokkos::OpenMP device_type;
#else
typedef Kokkos::Threads device_type;
#endif
struct double2 {
  double x, y;
};
struct float2 {
  float x, y;
};
struct double4 {
  double x, y, z, w;
};
struct float4 {
  float x, y, z, w;
};
#define KokkosHost(a) a
#define KokkosCUDA(a)
#define MODE 1
#else
#include "Kokkos_Cuda.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
typedef Kokkos::Cuda device_type;
#define KokkosHost(a)
#define KokkosCUDA(a) a
#define MODE 4
#endif


#ifndef PRECISION
#define PRECISION 2
#endif
#if PRECISION==1
typedef float MMD_float;
typedef float2 MMD_float2;
typedef float4 MMD_float4;
#else
typedef double MMD_float;
typedef double2 MMD_float2;
typedef double4 MMD_float4;
#endif
typedef int MMD_int;
typedef int MMD_bigint;




#ifndef PAD
#define PAD 3
#endif
//Choose Layout

//Scalars
typedef Kokkos::View<MMD_float[1] , Kokkos::LayoutLeft, device_type>  tscalar_d ;
typedef tscalar_d::HostMirror  tscalar_d_host ;
typedef Kokkos::View<MMD_int[1] , Kokkos::LayoutLeft, device_type>  tscalar_i ;
typedef tscalar_i::HostMirror  tscalar_i_host ;


//2d float array n*PADDING for positions
typedef Kokkos::View<MMD_float*[PAD] , Kokkos::LayoutRight, device_type>  t_x_array ;
typedef t_x_array::HostMirror  t_x_array_host ;
typedef Kokkos::View<const MMD_float*[PAD] , Kokkos::LayoutRight, device_type>  t_x_array_const ;
typedef Kokkos::View<const MMD_float*[PAD] ,Kokkos::LayoutRight,device_type,Kokkos::MemoryRandomRead >  t_x_array_tex ;

//2d float array n*PADDING for velocities
typedef Kokkos::View<MMD_float*[PAD] , device_type>  t_v_array ;
typedef t_v_array::HostMirror  t_v_array_host ;
typedef Kokkos::View<const MMD_float*[PAD] , device_type>  t_v_array_const ;
typedef Kokkos::View<const MMD_float*[PAD] , device_type,Kokkos::MemoryRandomRead >  t_v_array_tex ;

//2d float array n*PADDING for forces
typedef Kokkos::View<MMD_float*[PAD] , device_type>  t_f_array ;
typedef t_f_array::HostMirror  t_f_array_host ;
typedef Kokkos::View<const MMD_float*[PAD] , device_type>  t_f_array_const ;
typedef Kokkos::View<const MMD_float*[PAD] , device_type,Kokkos::MemoryRandomRead >  t_f_array_tex ;

//Neighbor array and subviews to neighbors of a single atom
typedef Kokkos::View<MMD_int** , device_type >  tvector_neighbors ;
typedef Kokkos::View<MMD_int* , device_type, Kokkos::MemoryUnmanaged >  tvector_neighbors_sub ;
typedef Kokkos::View<const MMD_int* , device_type, Kokkos::MemoryUnmanaged >  tvector_neighbors_const_sub ;

//1d float array
typedef Kokkos::View<MMD_float* , device_type >  tvector_1d ;
typedef tvector_1d::HostMirror  tvector_1d_host ;

//1d int array
typedef Kokkos::View<MMD_int* , device_type >  tvector_1i ;
typedef tvector_1i::HostMirror  tvector_1i_host ;
typedef Kokkos::View<MMD_int* , device_type , Kokkos::MemoryUnmanaged>  tvector_1i_um ;
typedef Kokkos::View<const MMD_int* , device_type , Kokkos::MemoryUnmanaged>  tvector_1i_const_um ;

//2d int array
typedef Kokkos::View<MMD_int** , Kokkos::LayoutRight, device_type >  tvector_2i ;
typedef tvector_2i::HostMirror  tvector_2i_host ;


typedef Kokkos::View<MMD_int[3][3] , device_type >  tvector_procneigh ;
typedef Kokkos::View<MMD_int[3] , device_type >  tvector_1i3 ;


template <class T, int rank = T::Rank >
struct DeepCopyFunctor;

template <class T>
struct DeepCopyFunctor<T, 1> {
  typedef typename T::device_type                   device_type ;
  T x;
  T y;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int ii) const {
    x(ii) = y(ii);
  }
};

template <class T>
struct DeepCopyFunctor<T, 2> {
  typedef typename T::device_type                   device_type ;
  T x;
  T y;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int ii) const {
    int i = ii / y.dimension(1);
    int j = ii % y.dimension(1);
    x(i, j) = y(i, j);
  }
};

template <class T>
struct DeepCopyFunctor<T, 3> {
  typedef typename T::device_type                   device_type ;
  T x;
  T y;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int ii) const {
    int i = ii / (y.dimension(1) * y.dimension(2));
    int j = ii % (y.dimension(1) * y.dimension(2));
    int k = j % y.dimension(2);
    j = j / y.dimension(2);
    x(i, j, k) = y(i, j, k);
  }
};

template <class T>
void deep_copy_grow(T x, T y)
{
  if((x.ptr_on_device() == NULL) || (y.ptr_on_device() == NULL)) return;

  int nthreads = 1;

  for(int i = 0; i < x.Rank; i++) {
    if(x.dimension(i) < y.dimension(i)) return;

    nthreads *= y.dimension(i);
  }

  DeepCopyFunctor<T> f;
  f.x = x;
  f.y = y;
  Kokkos::parallel_for(nthreads, f);
}

void cuda_check_error(char* comment);

//Texture fetchs
#if DEVICE==2
#if PRECISION==1
static __device__ inline MMD_float4 tex1Dfetch_f4(texture<float4, 1> t, const int &i)
{
  return tex1Dfetch(t, i);
}
static __device__ inline MMD_float tex1Dfetch_f1(texture<float, 1> t, const int &i)
{
  return tex1Dfetch(t, i);
}

#if CUDA_ARCH >= 300
static __device__ inline MMD_float4 tex1Dfetch_f4o(const cudaTextureObject_t &t, const int &i)
{
  return tex1Dfetch<MMD_float4>(t, i);
}
static __device__ inline MMD_float tex1Dfetch_f1o(const cudaTextureObject_t &t, const int &i)
{
  return tex1Dfetch<MMD_float>(t, i);
}
#endif
#endif

#if PRECISION==2
static __device__ inline MMD_float4 tex1Dfetch_f4(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t, 2 * i);
  int4 u = tex1Dfetch(t, 2 * i + 1);
  MMD_float4 w;

  w.x = __hiloint2double(v.y, v.x);
  w.y = __hiloint2double(v.w, v.z);
  w.z = __hiloint2double(u.y, u.x);
  w.w = __hiloint2double(u.w, u.z);
  return w;
}

static __device__ inline MMD_float tex1Dfetch_f1(texture<int2, 1> t, int i)
{
  int2 v = tex1Dfetch(t, i);
  MMD_float w;

  w = __hiloint2double(v.y, v.x);
  return w;
}

#endif
#endif

#endif
