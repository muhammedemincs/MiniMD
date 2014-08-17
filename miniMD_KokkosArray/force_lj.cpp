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

#include "stdio.h"
#include "math.h"
#include "force_lj.h"
#include <Kokkos_Atomic.hpp>

#if __CUDA_ARCH__
#ifdef USE_TEXTURE_REFERENCES
#define c_x(a,b) tex1Dfetch_f1(lj_x_tex,3*a+b)
#else
#define c_x(a,b) t_x(a,b)
#endif
#else
#define c_x(a,b) x(a,b)
#endif

#ifdef USE_TEXTURE_REFERENCES
#pragma message "CUDA Use Texture References"
#if DEVICE==2
#if PRECISION==1
texture<float> lj_x_tex;
#endif
#if PRECISION==2
texture<int2> lj_x_tex;
#endif
#endif
#endif


ForceLJ::ForceLJ()
{
  cutforce = 0.0;
  cutforcesq = 0.0;
  use_oldcompute = 0;
  reneigh = 1;
  style = FORCELJ;

  epsilon = 1.0;
  sigma6 = 1.0;
  sigma = 1.0;
}
ForceLJ::~ForceLJ() {}

void ForceLJ::setup()
{
  cutforcesq = cutforce * cutforce;
  f_compute_fullneigh = new ForceComputeFullneighFunctor;
  f_compute_halfneigh = new ForceComputeHalfneighFunctor;
  f_compute_halfneigh_threaded = new ForceComputeHalfneighThreadedFunctor;
}

void ForceLJ::finalise()
{
  delete f_compute_fullneigh;
  delete f_compute_halfneigh;
  delete f_compute_halfneigh_threaded;
}

void ForceLJ::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  //update local values;
  nlocal = atom.nlocal;
  t_x = atom.t_x;
  f = atom.f;
  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;
  maxneighs = neighbor.maxneighs;
  ghost_newton = neighbor.ghost_newton;

  if(neighbor.halfneigh) {
    if(threads->omp_num_threads > 1)
      compute_halfneigh_threaded(atom, neighbor, me);
    else
      KokkosCUDA(compute_halfneigh_threaded(atom, neighbor, me);)
      KokkosHost(compute_halfneigh(atom, neighbor, me);)

    } else compute_fullneigh(atom, neighbor, me);
}

void ForceLJ::compute_halfneigh(Atom &atom, Neighbor &neighbor, int me)
{
  // clear force on own atoms
  ForceZeroFunctor f_forceZero;
  f_forceZero.f = f;
  Kokkos::parallel_for(atom.nlocal + atom.nghost, f_forceZero);

  x = atom.x;
  f = atom.f;
  nlocal = atom.nlocal;
  nmax = atom.nmax;
  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;
  maxneighs = neighbor.maxneighs;

  f_compute_halfneigh->c = *this;
  MMD_float2 energy_virial;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_halfneigh,energy_virial);
  else
    Kokkos::parallel_for(nlocal, *f_compute_halfneigh);

  eng_vdwl = energy_virial.x;
  virial = energy_virial.y;

}

template<int EVFLAG, int GHOST_NEWTON>
KOKKOS_INLINE_FUNCTION
MMD_float2 ForceLJ::compute_halfneighItem(const MMD_int &i) const
{
  MMD_int numneighs = numneigh[i];
  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);
  MMD_float fix = 0;
  MMD_float fiy = 0;
  MMD_float fiz = 0;
  MMD_float energy = 0;
  MMD_float virial = 0;


  for(MMD_int k = 0; k < numneighs; k++) {
    const MMD_int j = neighbors(i, k);
    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    if(rsq < cutforcesq) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6;
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      //Attention this is not thread safe !!!
      if(GHOST_NEWTON || j < nlocal) {
        f(j, 0) -= delx * force;
        f(j, 1) -= dely * force;
        f(j, 2) -= delz * force;
      }

      if(EVFLAG) {
        const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
        energy += epsilon * scale * (4.0 * sr6 * (sr6 - 1.0));
        virial += scale * (delx * delx * force + dely * dely * force + delz * delz * force);
      }
    }
  }

  f(i, 0) += fix;
  f(i, 1) += fiy;
  f(i, 2) += fiz;

  MMD_float2 energy_virial = {energy, virial};
  return energy_virial;
}

void ForceLJ::compute_halfneigh_threaded(Atom &atom, Neighbor &neighbor, int me)
{
  // clear force on own atoms
  ForceZeroFunctor f_forceZero;
  f_forceZero.f = f;
  Kokkos::parallel_for(atom.nlocal + atom.nghost, f_forceZero);

  x = atom.x;
  f = atom.f;
  nlocal = atom.nlocal;
  nmax = atom.nmax;
  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;
  maxneighs = neighbor.maxneighs;

  //Texture via Texture reference
#ifdef USE_TEXTURE_REFERENCES
#if PRECISION==1
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float>();
#endif
#if PRECISION==2
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<int2>();
#endif
    cudaBindTexture(NULL, &lj_x_tex, (void*)x.ptr_on_device(), &chanDesc, x.dimension(0)*x.dimension(1)*sizeof(MMD_float));
#endif

  f_compute_halfneigh_threaded->c = *this;
  MMD_float2 energy_virial;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_halfneigh_threaded,energy_virial);
  else
    Kokkos::parallel_for(nlocal, *f_compute_halfneigh_threaded);

  eng_vdwl = energy_virial.x;
  virial = energy_virial.y;
  return;
}

template<int EVFLAG, int GHOST_NEWTON>
KOKKOS_INLINE_FUNCTION
MMD_float2 ForceLJ::compute_halfneigh_threadedItem(const MMD_int &i) const
{
  MMD_int numneighs = numneigh[i];
  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);
  MMD_float fix = 0;
  MMD_float fiy = 0;
  MMD_float fiz = 0;
  MMD_float energy = 0;
  MMD_float virial = 0;


  for(MMD_int k = 0; k < numneighs; k++) {
    const MMD_int j = neighbors(i, k);
    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    if(rsq < cutforcesq) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2  * sigma6;
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      if(GHOST_NEWTON || j < nlocal) {
        Kokkos::atomic_fetch_add(&f(j, 0) , -delx * force);
        Kokkos::atomic_fetch_add(&f(j, 1) , -dely * force);
        Kokkos::atomic_fetch_add(&f(j, 2) , -delz * force);
      }

      if(EVFLAG) {
        const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
        energy += scale * (4.0 * sr6 * (sr6 - 1.0)) * epsilon;
        virial += scale * (delx * delx * force + dely * dely * force + delz * delz * force);
      }
    }
  }

  Kokkos::atomic_fetch_add(&f(i, 0) , fix);
  Kokkos::atomic_fetch_add(&f(i, 1) , fiy);
  Kokkos::atomic_fetch_add(&f(i, 2) , fiz);

  MMD_float2 energy_virial = {energy, virial};
  return energy_virial;
}


void ForceLJ::compute_fullneigh(Atom &atom, Neighbor &neighbor, int me)
{
  x = atom.x;
  f = atom.f;
  nlocal = atom.nlocal;
  nmax = atom.nmax;
  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;
  maxneighs = neighbor.maxneighs;

  //Texture via Texture reference
#ifdef USE_TEXTURE_REFERENCES
#if PRECISION==1
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float>();
#endif
#if PRECISION==2
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<int2>();
#endif
    cudaBindTexture(NULL, &lj_x_tex, (void*)x.ptr_on_device(), &chanDesc, x.dimension(0)*x.dimension(1)*sizeof(MMD_float));
#endif


  // clear force on own atoms
  ForceZeroFunctor f_forceZero;
  f_forceZero.f = f;
  Kokkos::parallel_for(nlocal, f_forceZero);
  device_type::fence();

  f_compute_fullneigh->c = *this;
  MMD_float2 energy_virial;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_fullneigh,energy_virial);
  else
    Kokkos::parallel_for(nlocal, *f_compute_fullneigh);

  device_type::fence();

  eng_vdwl = energy_virial.x;
  virial = energy_virial.y;
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION
MMD_float2 ForceLJ::compute_fullneighItem(const MMD_int &i) const
{
  const MMD_int numneighs = numneigh[i];
  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);
  MMD_float fix = 0;
  MMD_float fiy = 0;
  MMD_float fiz = 0;
  MMD_float energy = 0;
  MMD_float virial = 0;

  //pragma simd forces vectorization (ignoring the performance objections of the compiler)
  //give hint to compiler that fix, fiy and fiz are used for reduction only

#ifdef USE_SIMD
  #pragma simd reduction (+: fix,fiy,fiz,energy,virial)
#endif
  for(MMD_int k = 0; k < numneighs; k++) {
    const MMD_int j = neighbors(i, k);
    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    if(rsq < cutforcesq) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2  * sigma6;
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      if(EVFLAG) {
        energy += sr6 * (sr6 - 1.0) * epsilon;
        virial += delx * delx * force + dely * dely * force + delz * delz * force;
      }
    }
  }

  f(i, 0) += fix;
  f(i, 1) += fiy;
  f(i, 2) += fiz;

  MMD_float2 energy_virial = {4.0 * energy, 0.5 * virial};
  return energy_virial;
}



