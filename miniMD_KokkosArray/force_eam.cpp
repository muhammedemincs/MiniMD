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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "force_eam.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "memory.h"
//#include "error.h"
#include <Kokkos_Atomic.hpp>
#define MAXLINE 1024

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

struct RhoZeroFunctor {
  typedef t_x_array::device_type  device_type ;
  typedef t_x_array::size_type    MMD_int;

  tvector_1d f;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    f(i) = 0;
  }
};

struct FZeroFunctor {
  typedef t_x_array::device_type  device_type ;
  typedef t_x_array::size_type    MMD_int;

  t_f_array f;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    f(i, 0) = 0;
    f(i, 1) = 0;
    f(i, 2) = 0;
  }
};

//define constant read access to position data uses standard access on host and texture fetch on GPU
#if __CUDA_ARCH__
#ifdef USE_TEXTURE_REFERENCES
#define c_x(a,b) tex1Dfetch_f1(eam_x_tex,3*a+b)
#define c_rhor_spline(a,b) tex1Dfetch_f1(eam_rhor_spline_tex,7*a+b)
#define c_frho_spline(a,b) tex1Dfetch_f1(eam_frho_spline_tex,7*a+b)
#define c_z2r_spline(a,b) tex1Dfetch_f1(eam_z2r_spline_tex,7*a+b)
#else
#define c_x(a,b) t_x(a,b)
#define c_rhor_spline(a,b) t_rhor_spline(a,b)
#define c_frho_spline(a,b) t_frho_spline(a,b)
#define c_z2r_spline(a,b) t_z2r_spline(a,b)
#endif
#else
#define c_x(a,b) x(a,b)
#define c_rhor_spline(a,b) rhor_spline(a,b)
#define c_frho_spline(a,b) frho_spline(a,b)
#define c_z2r_spline(a,b) z2r_spline(a,b)
#endif

#if DEVICE==2
#if PRECISION==1
texture<float> eam_x_tex;
texture<float> eam_rhor_spline_tex;
texture<float> eam_frho_spline_tex;
texture<float> eam_z2r_spline_tex;
#endif
#if PRECISION==2
texture<int2> eam_x_tex;
texture<int2> eam_rhor_spline_tex;
texture<int2> eam_frho_spline_tex;
texture<int2> eam_z2r_spline_tex;
#endif
#endif

/* ---------------------------------------------------------------------- */

ForceEAM::ForceEAM()
{
  cutforce = 0.0;
  cutforcesq = 0.0;
  use_oldcompute = 0;

  nmax = 0;

  style = FORCEEAM;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

ForceEAM::~ForceEAM()
{

}

void ForceEAM::setup()
{
  me = threads->mpi_me;
  coeff("Cu_u6.eam");
  init_style();
  f_compute_fullneighA = new ForceEAMComputeFullneighAFunctor;
  f_compute_fullneighB = new ForceEAMComputeFullneighBFunctor;
  f_compute_halfneighA = new ForceEAMComputeHalfneighAFunctor;
  f_compute_halfneighB = new ForceEAMComputeHalfneighBFunctor;
  f_compute_halfneighC = new ForceEAMComputeHalfneighCFunctor;
  f_compute_halfneigh_threadedA = new ForceEAMComputeHalfneighThreadedAFunctor;
  f_compute_halfneigh_threadedB = new ForceEAMComputeHalfneighThreadedBFunctor;
}

void ForceEAM::finalise()
{
  delete f_compute_fullneighA;
  delete f_compute_fullneighB;
  delete f_compute_halfneighA;
  delete f_compute_halfneighB;
  delete f_compute_halfneighC;
  delete f_compute_halfneigh_threadedA;
  delete f_compute_halfneigh_threadedB;
}

void ForceEAM::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  t_x = atom.t_x;
  if(neighbor.halfneigh) {
    if(threads->omp_num_threads > 1)
      KokkosHost(compute_halfneigh(atom, neighbor, comm, me);)
      KokkosCUDA(compute_halfneigh_threaded(atom, neighbor, comm, me);)
      else
        KokkosHost(compute_halfneigh(atom, neighbor, comm, me);)
        KokkosCUDA(compute_halfneigh_threaded(atom, neighbor, comm, me);)
      } else {
    compute_fullneigh(atom, neighbor, comm, me);
  }

}
/* ---------------------------------------------------------------------- */

void ForceEAM::compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    rho = tvector_1d("pair:rho", nmax);
    fp = tvector_1d("pair:fp", nmax);
    h_fp = Kokkos::create_mirror_view(fp);
  }

  x = atom.x;
  f = atom.f;
  nlocal = atom.nlocal;

  // zero out density

  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;

  ForceZeroFunctor f_forceZero;
  f_forceZero.f = f;
  Kokkos::parallel_for(atom.nlocal + atom.nghost, f_forceZero);

  RhoZeroFunctor f_rhoZero;
  f_rhoZero.f = rho;
  Kokkos::parallel_for(nlocal, f_rhoZero);

  // rho = density at each atom
  // loop over neighbors of my atoms

  f_compute_halfneighA->c = *this;
  Kokkos::parallel_for(nlocal, *f_compute_halfneighA);
  //for (MMD_int i = 0; i < nlocal; i++) printf("Rho: %i %lf\n",i,rho[i]);

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  f_compute_halfneighC->c = *this;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_halfneighC,eng_vdwl);
  else
    Kokkos::parallel_for(nlocal, *f_compute_halfneighC);

  // communicate derivative of embedding function

  communicate(atom, comm);

  MMD_float2 energy_virial;
  f_compute_halfneighB->c = *this;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_halfneighB,energy_virial);
  else
    Kokkos::parallel_for(nlocal, *f_compute_halfneighB);

  eng_vdwl += energy_virial.x;
  virial = energy_virial.y;

  // compute forces on each atom
  // loop over neighbors of my atoms

  //eng_vdwl = evdwl;
}

void ForceEAM::compute_halfneigh_threaded(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    rho = tvector_1d("pair:rho", nmax);
    fp = tvector_1d("pair:fp", nmax);
    h_fp = Kokkos::create_mirror_view(fp);
  }

  x = atom.x;
  f = atom.f;
  nlocal = atom.nlocal;

  // zero out density

  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;

  ForceZeroFunctor f_forceZero;
  f_forceZero.f = f;
  Kokkos::parallel_for(atom.nlocal + atom.nghost, f_forceZero);

  RhoZeroFunctor f_rhoZero;
  f_rhoZero.f = rho;
  Kokkos::parallel_for(nlocal, f_rhoZero);

  // rho = density at each atom
  // loop over neighbors of my atoms
#ifdef USE_TEXTURE_REFERENCES
  KokkosCUDA(
#if PRECISION==1
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float>();
#endif
#if PRECISION==2
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<int2>();
#endif
    cudaBindTexture(NULL, &eam_x_tex, (void*)x.ptr_on_device(), &chanDesc, x.dimension(0)*x.dimension(1)*sizeof(MMD_float));
    cudaBindTexture(NULL, &eam_frho_spline_tex, (void*)frho_spline.ptr_on_device(), &chanDesc, frho_spline.dimension(0)*frho_spline.dimension(1)*sizeof(MMD_float));
    cudaBindTexture(NULL, &eam_rhor_spline_tex, (void*)rhor_spline.ptr_on_device(), &chanDesc, rhor_spline.dimension(0)*rhor_spline.dimension(1)*sizeof(MMD_float));
    cudaBindTexture(NULL, &eam_z2r_spline_tex, (void*)z2r_spline.ptr_on_device(), &chanDesc, z2r_spline.dimension(0)*z2r_spline.dimension(1)*sizeof(MMD_float));
  )
#endif

  f_compute_halfneigh_threadedA->c = *this;
  Kokkos::parallel_for(nlocal, *f_compute_halfneigh_threadedA);
  //for (MMD_int i = 0; i < nlocal; i++) printf("Rho: %i %lf\n",i,rho[i]);

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  f_compute_halfneighC->c = *this;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_halfneighC,eng_vdwl);
  else
    Kokkos::parallel_for(nlocal, *f_compute_halfneighC);

  // communicate derivative of embedding function

  communicate(atom, comm);

  MMD_float2 energy_virial;
  f_compute_halfneigh_threadedB->c = *this;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_halfneigh_threadedB,energy_virial);
  else
    Kokkos::parallel_for(nlocal, *f_compute_halfneigh_threadedB);

  eng_vdwl += energy_virial.x;
  virial = energy_virial.y;

  // compute forces on each atom
  // loop over neighbors of my atoms

  //eng_vdwl = evdwl;
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::compute_halfneighItemA(const MMD_int &i) const
{

  const MMD_float xtmp = x(i, 0);
  const MMD_float ytmp = x(i, 1);
  const MMD_float ztmp = x(i, 2);
  const MMD_int jnum = numneigh[i];
  MMD_float rhoi = 0.0;
  //printf("%i %i\n",i,jnum);

  //#pragma ivdep
  //#pragma vector always
#pragma ivdep
  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i, jj);

    const MMD_float delx = xtmp - x(j, 0);
    const MMD_float dely = ytmp - x(j, 1);
    const MMD_float delz = ztmp - x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    //printf("EAM: %i %i %lf %lf\n",i,j,rsq,cutforcesq);
    if(rsq < cutforcesq) {
      MMD_float p = sqrt(rsq) * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;
      //if(j<40)
      //only possible because just one atomtype, otherwise rhoij!=rhoji
      const MMD_float rhoij = ((rhor_spline(m, 3) * p + rhor_spline(m, 4)) * p + rhor_spline(m, 5)) * p + rhor_spline(m, 6);
      rhoi += rhoij;

      //if(i==0 && j<20) printf("%i %i %lf %lf %lf %lf %lf %lf %i\n",i,j,((rhor_spline(m,3)*p + rhor_spline(m,4))*p + rhor_spline(m,5))*p + rhor_spline(m,6),rhor_spline(m,3),rhor_spline(m,4), rhor_spline(m,5), rhor_spline(m,6),p,m);
      if(j < nlocal) {
        rho[j] += rhoij;
      }

      //printf("RhoIJ: %i %i %lf\n",i,j,rhoij);

    }
  }

  rho[i] += rhoi;
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::compute_halfneigh_threadedItemA(const MMD_int &i) const
{

  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);
  const MMD_int jnum = numneigh[i];
  MMD_float rhoi = 0.0;
  //printf("%i %i\n",i,jnum);

  //#pragma ivdep
  //#pragma vector always
  #pragma ivdep
  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i, jj);

    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    //printf("EAM: %i %i %lf %lf\n",i,j,rsq,cutforcesq);
    if(rsq < cutforcesq) {
      MMD_float p = sqrt(rsq) * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;
      //if(j<40)
      //only possible because just one atomtype, otherwise rhoij!=rhoji
      const MMD_float rhoij = ((c_rhor_spline(m, 3) * p + c_rhor_spline(m, 4)) * p + c_rhor_spline(m, 5)) * p + c_rhor_spline(m, 6);
      rhoi += rhoij;

      //if(i==0 && j<20) printf("%i %i %lf %lf %lf %lf %lf %lf %i\n",i,j,((rhor_spline(m,3)*p + rhor_spline(m,4))*p + rhor_spline(m,5))*p + rhor_spline(m,6),rhor_spline(m,3),rhor_spline(m,4), rhor_spline(m,5), rhor_spline(m,6),p,m);
      if(j < nlocal) {
        Kokkos::atomic_fetch_add(&rho[j], rhoij);
      }

      //printf("RhoIJ: %i %i %lf\n",i,j,rhoij);

    }
  }

  Kokkos::atomic_fetch_add(&rho[i], rhoi);
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION
MMD_float ForceEAM::compute_halfneighItemC(const MMD_int &i) const
{
  MMD_float evdwl = 0;
  MMD_float p = 1.0 * rho[i] * rdrho + 1.0;
  MMD_int m = static_cast<int>(p);
  m = MAX(1, MIN(m, nrho - 1));
  p -= m;
  p = MIN(p, 1.0);
  fp[i] = (c_frho_spline(m, 0) * p + c_frho_spline(m, 1)) * p + c_frho_spline(m, 2);

  if(EVFLAG) {
    evdwl += ((c_frho_spline(m, 3) * p + c_frho_spline(m, 4)) * p + c_frho_spline(m, 5)) * p + c_frho_spline(m, 6);
  }

  return evdwl;
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION MMD_float2 ForceEAM::compute_halfneighItemB(const MMD_int &i) const
{
  MMD_float2 energy_virial = {0, 0};
  const MMD_float xtmp = x(i, 0);
  const MMD_float ytmp = x(i, 1);
  const MMD_float ztmp = x(i, 2);
  MMD_float fx = 0;
  MMD_float fy = 0;
  MMD_float fz = 0;

  const MMD_int jnum = numneigh[i];
  //printf("Hallo %i %i %lf %lf\n",i,numneigh[i],sqrt(cutforcesq),neighbor.cutneigh);

  //#pragma ivdep
  //#pragma vector always
  for(MMD_int jj = 0; jj < jnum; jj++) {
    MMD_int j = neighbors(i, jj);

    const MMD_float delx = xtmp - x(j, 0);
    const MMD_float dely = ytmp - x(j, 1);
    const MMD_float delz = ztmp - x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    if(rsq < cutforcesq) {
      const MMD_float r = sqrt(rsq);
      MMD_float p = r * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;


      // rhoip = derivative of (density at atom j due to atom i)
      // rhojp = derivative of (density at atom i due to atom j)
      // phi = pair potential energy
      // phip = phi'
      // z2 = phi * r
      // z2p = (phi * r)' = (phi' r) + phi
      // psip needs both fp[i] and fp[j] terms since r_ij appears in two
      //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
      //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

      const  MMD_float rhoip = (rhor_spline(m, 0) * p + rhor_spline(m, 1)) * p + rhor_spline(m, 2);
      const  MMD_float z2p = (z2r_spline(m, 0) * p + z2r_spline(m, 1)) * p + z2r_spline(m, 2);
      const  MMD_float z2 = ((z2r_spline(m, 3) * p + z2r_spline(m, 4)) * p + z2r_spline(m, 5)) * p + z2r_spline(m, 6);

      const MMD_float recip = 1.0 / r;
      const MMD_float phi = z2 * recip;
      const MMD_float phip = z2p * recip - phi * recip;
      const MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
      MMD_float fpair = -psip * recip;

      fx += delx * fpair;
      fy += dely * fpair;
      fz += delz * fpair;

      //  	if(i==0&&j<20)
      //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
      if(j < nlocal) {
        f(j, 0) -= delx * fpair;
        f(j, 1) -= dely * fpair;
        f(j, 2) -= delz * fpair;
      } else fpair *= 0.5;

      if(EVFLAG) {
        energy_virial.y += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
      }

      if(j < nlocal) energy_virial.x += phi;
      else energy_virial.x += 0.5 * phi;
    }
  }

  f(i, 0) += fx;
  f(i, 1) += fy;
  f(i, 2) += fz;
  return energy_virial;
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION MMD_float2 ForceEAM::compute_halfneigh_threadedItemB(const MMD_int &i) const
{
  MMD_float2 energy_virial = {0, 0};
  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);
  MMD_float fx = 0;
  MMD_float fy = 0;
  MMD_float fz = 0;

  const MMD_int jnum = numneigh[i];
  //printf("Hallo %i %i %lf %lf\n",i,numneigh[i],sqrt(cutforcesq),neighbor.cutneigh);

  //#pragma ivdep
  //#pragma vector always
  for(MMD_int jj = 0; jj < jnum; jj++) {
    MMD_int j = neighbors(i, jj);

    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    if(rsq < cutforcesq) {
      const MMD_float r = sqrt(rsq);
      MMD_float p = r * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;


      // rhoip = derivative of (density at atom j due to atom i)
      // rhojp = derivative of (density at atom i due to atom j)
      // phi = pair potential energy
      // phip = phi'
      // z2 = phi * r
      // z2p = (phi * r)' = (phi' r) + phi
      // psip needs both fp[i] and fp[j] terms since r_ij appears in two
      //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
      //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

      const  MMD_float rhoip = (c_rhor_spline(m, 0) * p + c_rhor_spline(m, 1)) * p + c_rhor_spline(m, 2);
      const  MMD_float z2p = (c_z2r_spline(m, 0) * p + c_z2r_spline(m, 1)) * p + c_z2r_spline(m, 2);
      const  MMD_float z2 = ((c_z2r_spline(m, 3) * p + c_z2r_spline(m, 4)) * p + c_z2r_spline(m, 5)) * p + c_z2r_spline(m, 6);

      const MMD_float recip = 1.0 / r;
      const MMD_float phi = z2 * recip;
      const MMD_float phip = z2p * recip - phi * recip;
      const MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
      MMD_float fpair = -psip * recip;

      fx += delx * fpair;
      fy += dely * fpair;
      fz += delz * fpair;

      //  	if(i==0&&j<20)
      //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
      if(j < nlocal) {
        Kokkos::atomic_fetch_add(&f(j, 0) , -delx * fpair);
        Kokkos::atomic_fetch_add(&f(j, 1) , -dely * fpair);
        Kokkos::atomic_fetch_add(&f(j, 2) , -delz * fpair);
      } else fpair *= 0.5;

      if(EVFLAG) {
        energy_virial.y += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
      }

      if(j < nlocal) energy_virial.x += phi;
      else energy_virial.x += 0.5 * phi;
    }
  }

  Kokkos::atomic_fetch_add(&f(i, 0) , fx);
  Kokkos::atomic_fetch_add(&f(i, 1) , fy);
  Kokkos::atomic_fetch_add(&f(i, 2) , fz);
  return energy_virial;
}
/* ---------------------------------------------------------------------- */

void ForceEAM::compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    rho = tvector_1d("pair:rho", nmax);
    fp = tvector_1d("pair:fp", nmax);
    h_fp = Kokkos::create_mirror_view(fp);
  }

  device_type::fence();

  x = atom.x;
  f = atom.f;
  nlocal = atom.nlocal;

  numneigh = neighbor.numneigh;
  neighbors = neighbor.neighbors;

  FZeroFunctor f_forceZero;
  f_forceZero.f = f;
  Kokkos::parallel_for(nlocal, f_forceZero);
  device_type::fence();

#ifdef USE_TEXTURE_REFERENCES
  KokkosCUDA(
#if PRECISION==1
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float>();
#endif
#if PRECISION==2
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<int2>();
#endif
    cudaBindTexture(NULL, &eam_x_tex, (void*)x.ptr_on_device(), &chanDesc, x.dimension(0)*x.dimension(1)*sizeof(MMD_float));
    cudaBindTexture(NULL, &eam_frho_spline_tex, (void*)frho_spline.ptr_on_device(), &chanDesc, frho_spline.dimension(0)*frho_spline.dimension(1)*sizeof(MMD_float));
    cudaBindTexture(NULL, &eam_rhor_spline_tex, (void*)rhor_spline.ptr_on_device(), &chanDesc, rhor_spline.dimension(0)*rhor_spline.dimension(1)*sizeof(MMD_float));
    cudaBindTexture(NULL, &eam_z2r_spline_tex, (void*)z2r_spline.ptr_on_device(), &chanDesc, z2r_spline.dimension(0)*z2r_spline.dimension(1)*sizeof(MMD_float));
  )
#endif

  f_compute_fullneighA->c = *this;
  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_fullneighA,eng_vdwl);
  else
    Kokkos::parallel_for(nlocal, *f_compute_fullneighA);

  device_type::fence();
  //cuda_check_error("E");

  // rho = density at each atom
  // loop over neighbors of my atoms


  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  // communicate derivative of embedding function

  communicate(atom, comm);

  MMD_float2 energy_virial;
  f_compute_fullneighB->c = *this;

  if(evflag)
    Kokkos::parallel_reduce(nlocal, *f_compute_fullneighB,energy_virial);
  else
    Kokkos::parallel_for(nlocal, *f_compute_fullneighB);

  device_type::fence();

  eng_vdwl += energy_virial.x;
  virial = energy_virial.y;

  eng_vdwl *= 2;
}



template<int EVFLAG>
KOKKOS_INLINE_FUNCTION MMD_float ForceEAM::compute_fullneighItemA(const MMD_int &i) const
{
  MMD_float evdwl = 0;
  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);
  const MMD_int jnum = numneigh[i];
  MMD_float rhoi = 0;

#pragma ivdep

  for(MMD_int jj = 0; jj < jnum; jj++) {
    MMD_int j = neighbors(i, jj);

    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    if(rsq < cutforcesq) {
      MMD_float p = sqrt(rsq) * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;

      rhoi += ((c_rhor_spline(m, 3) * p + c_rhor_spline(m, 4)) * p + c_rhor_spline(m, 5)) * p + c_rhor_spline(m, 6);
    }
  }

  MMD_float p = 1.0 * rhoi * rdrho + 1.0;
  MMD_int m = static_cast<int>(p);
  m = MAX(1, MIN(m, nrho - 1));
  p -= m;
  p = MIN(p, 1.0);
  fp[i] = (c_frho_spline(m, 0) * p + c_frho_spline(m, 1)) * p + c_frho_spline(m, 2);

  if(EVFLAG) {
    evdwl += ((c_frho_spline(m, 3) * p + c_frho_spline(m, 4)) * p + c_frho_spline(m, 5)) * p + c_frho_spline(m, 6);
  }

  return evdwl;
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION MMD_float2 ForceEAM::compute_fullneighItemB(const MMD_int &i) const
{
  MMD_float2 energy_virial = {0, 0};
  const MMD_float xtmp = c_x(i, 0);
  const MMD_float ytmp = c_x(i, 1);
  const MMD_float ztmp = c_x(i, 2);

  const MMD_int jnum = numneigh[i];

  MMD_float fx = 0.0;
  MMD_float fy = 0.0;
  MMD_float fz = 0.0;

  //#pragma simd reduction (+: fx,fy,fz,avirial)

#pragma ivdep

  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i, jj);

    const MMD_float delx = xtmp - c_x(j, 0);
    const MMD_float dely = ytmp - c_x(j, 1);
    const MMD_float delz = ztmp - c_x(j, 2);
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;
    //printf("EAM: %i %i %lf %lf // %lf %lf\n",i,j,rsq,cutforcesq,fp[i],fp[j]);

    if(rsq < cutforcesq) {
      const MMD_float r = sqrt(rsq);
      MMD_float p = r * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;


      // rhoip = derivative of (density at atom j due to atom i)
      // rhojp = derivative of (density at atom i due to atom j)
      // phi = pair potential energy
      // phip = phi'
      // z2 = phi * r
      // z2p = (phi * r)' = (phi' r) + phi
      // psip needs both fp[i] and fp[j] terms since r_ij appears in two
      //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
      //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

      const MMD_float rhoip = (c_rhor_spline(m, 0) * p + c_rhor_spline(m, 1)) * p + c_rhor_spline(m, 2);
      const MMD_float z2p = (c_z2r_spline(m, 0) * p + c_z2r_spline(m, 1)) * p + c_z2r_spline(m, 2);
      const MMD_float z2 = ((c_z2r_spline(m, 3) * p + c_z2r_spline(m, 4)) * p + c_z2r_spline(m, 5)) * p + c_z2r_spline(m, 6);

      const MMD_float recip = 1.0 / r;
      const MMD_float phi = z2 * recip;
      const MMD_float phip = z2p * recip - phi * recip;
      const MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
      MMD_float fpair = -psip * recip;

      fx += delx * fpair;
      fy += dely * fpair;
      fz += delz * fpair;

      fpair *= 0.5;

      if(EVFLAG) {
        energy_virial.y += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
        energy_virial.x += 0.5 * phi;
      }

    }
  }

  f(i, 0) = fx;
  f(i, 1) = fy;
  f(i, 2) = fz;
  return energy_virial;
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void ForceEAM::coeff(const char* arg)
{
  // read funcfl file if hasn't already been read
  // store filename in Funcfl data struct
  read_file(arg);
  int n = strlen(arg) + 1;
  funcfl.file = new char[n];

  // set setflag and map only for i,i type pairs
  // set mass of atom type if i = j

  //atom->mass = funcfl.mass;
  cutmax = funcfl.cut;

  cutforcesq = cutmax * cutmax;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void ForceEAM::init_style()
{
  // convert read-in file(s) to arrays and spline them

  file2array();
  array2spline();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void ForceEAM::read_file(const char* filename)
{
  Funcfl* file = &funcfl;

  //me = 0;
  FILE* fptr;
  char line[MAXLINE];

  if(me == 0) {
    fptr = fopen(filename, "r");

    if(fptr == NULL) {
      char str[128];
      sprintf(str, "Cannot open EAM potential file %s", filename);
    }
  }

  int tmp;

  if(me == 0) {
    fgets(line, MAXLINE, fptr);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg", &tmp, &file->mass);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg %d %lg %lg",
           &file->nrho, &file->drho, &file->nr, &file->dr, &file->cut);
  }

  //printf("Read: %lf %i %lf %i %lf %lf\n",file->mass,file->nrho,file->drho,file->nr,file->dr,file->cut);
  MPI_Bcast(&file->nrho, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&file->mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->drho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->dr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  mass = file->mass;
  file->frho = tvector_1d_host("pair:frho", file->nrho + 1);
  file->rhor = tvector_1d_host("pair:rhor", file->nr + 1);
  file->zr = tvector_1d_host("pair:zr", file->nr + 1);

  if(me == 0) grab(fptr, file->nrho, file->frho);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->frho.ptr_on_device(), file->nrho, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->frho.ptr_on_device(), file->nrho, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->zr);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->zr.ptr_on_device(), file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->zr.ptr_on_device(), file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->rhor);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->rhor.ptr_on_device(), file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->rhor.ptr_on_device(), file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for(int i = file->nrho; i > 0; i--) file->frho(i) = file->frho(i - 1);

  for(int i = file->nr; i > 0; i--) file->rhor(i) = file->rhor(i - 1);

  for(int i = file->nr; i > 0; i--) file->zr(i) = file->zr(i - 1);

  if(me == 0) fclose(fptr);
}

/* ----------------------------------------------------------------------
   convert read-in funcfl potential(s) to standard array format
   interpolate all file values to a single grid and cutoff
------------------------------------------------------------------------- */

void ForceEAM::file2array()
{
  int k, m;
  double sixth = 1.0 / 6.0;

  // determine max function params from all active funcfl files
  // active means some element is pointing at it via map

  double rmax, rhomax;
  dr = drho = rmax = rhomax = 0.0;

  Funcfl* file = &funcfl;
  dr = MAX(dr, file->dr);
  drho = MAX(drho, file->drho);
  rmax = MAX(rmax, (file->nr - 1) * file->dr);
  rhomax = MAX(rhomax, (file->nrho - 1) * file->drho);

  // set nr,nrho from cutoff and spacings
  // 0.5 is for round-off in divide

  nr = static_cast<int>(rmax / dr + 0.5);
  nrho = static_cast<int>(rhomax / drho + 0.5);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of funcfl files + 1 for zero array

  frho = tvector_1d("frho", nrho + 1);
  h_frho = Kokkos::create_mirror_view(frho);
  // interpolate each file's frho to a single grid and cutoff

  double r, p, cof1, cof2, cof3, cof4;

  for(m = 1; m <= nrho; m++) {
    r = (m - 1) * drho;
    p = r / file->drho + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nrho - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    h_frho[m] = cof1 * file->frho[k - 1] + cof2 * file->frho[k] +
                cof3 * file->frho[k + 1] + cof4 * file->frho[k + 2];
  }


  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of funcfl files

  rhor = tvector_1d("pair:rhor", nr + 1);
  h_rhor = Kokkos::create_mirror_view(rhor);

  // interpolate each file's rhor to a single grid and cutoff

  for(m = 1; m <= nr; m++) {
    r = (m - 1) * dr;
    p = r / file->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    h_rhor[m] = cof1 * file->rhor[k - 1] + cof2 * file->rhor[k] +
                cof3 * file->rhor[k + 1] + cof4 * file->rhor[k + 2];
    //if(m==119)printf("BuildRho: %e %e %e %e %e %e\n",rhor[m],cof1,cof2,cof3,cof4,file->rhor[k]);
  }

  // type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for funcfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of funcfl files

  z2r = tvector_1d("pair:z2r", nr + 1);
  h_z2r = Kokkos::create_mirror_view(z2r);

  // create a z2r array for each file against other files, only for I >= J
  // interpolate zri and zrj to a single grid and cutoff

  double zri, zrj;

  Funcfl* ifile = &funcfl;
  Funcfl* jfile = &funcfl;

  for(m = 1; m <= nr; m++) {
    r = (m - 1) * dr;

    p = r / ifile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, ifile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zri = cof1 * ifile->zr[k - 1] + cof2 * ifile->zr[k] +
          cof3 * ifile->zr[k + 1] + cof4 * ifile->zr[k + 2];

    p = r / jfile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, jfile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zrj = cof1 * jfile->zr[k - 1] + cof2 * jfile->zr[k] +
          cof3 * jfile->zr[k + 1] + cof4 * jfile->zr[k + 2];

    h_z2r[m] = 27.2 * 0.529 * zri * zrj;
  }

  Kokkos::deep_copy(frho, h_frho);
  Kokkos::deep_copy(rhor, h_rhor);
  Kokkos::deep_copy(z2r, h_z2r);

}

/* ---------------------------------------------------------------------- */

void ForceEAM::array2spline()
{
  rdr = 1.0 / dr;
  rdrho = 1.0 / drho;

  //frho_spline = t_x_array_o("pair:frho_spline", nrho + 1, 7);
  frho_spline = EAM_arrays_type("pair:frho_spline", nrho + 1);
  h_frho_spline = Kokkos::create_mirror_view(frho_spline);
  t_frho_spline = EAM_arrays_type_tex(frho_spline);
  //rhor_spline = t_x_array_o("pair:rhor_spline", nr + 1, 7);
  rhor_spline = EAM_arrays_type("pair:rhor_spline", nr + 1);
  h_rhor_spline = Kokkos::create_mirror_view(rhor_spline);
  t_rhor_spline = EAM_arrays_type_tex(rhor_spline);
  //z2r_spline = t_x_array_o("pair:z2r_spline", nr + 1, 7);
  z2r_spline = EAM_arrays_type("pair:z2r_spline", nr + 1);
  h_z2r_spline = Kokkos::create_mirror_view(z2r_spline);
  t_z2r_spline = EAM_arrays_type_tex(z2r_spline);

  interpolate(nrho, drho, h_frho, h_frho_spline);

  interpolate(nr, dr, h_rhor, h_rhor_spline);

  // printf("Rhor: %lf\n",rhor(119));

  interpolate(nr, dr, h_z2r, h_z2r_spline);

  //printf("RhorSpline: %e %e %e %e\n",rhor_spline(119,3),rhor_spline(119,4),rhor_spline(119,5),rhor_spline(119,6));
  //printf("FrhoSpline: %e %e %e %e\n",frho_spline(119,3),frho_spline(119,4),frho_spline(119,5),frho_spline(119,6));
  Kokkos::deep_copy(frho_spline, h_frho_spline);
  Kokkos::deep_copy(rhor_spline, h_rhor_spline);
  Kokkos::deep_copy(z2r_spline, h_z2r_spline);
}

/* ---------------------------------------------------------------------- */

//void ForceEAM::interpolate(MMD_int n, MMD_float delta, tvector_1d_host f, t_x_array_o_host spline)
void ForceEAM::interpolate(MMD_int n, MMD_float delta, tvector_1d_host f, EAM_arrays_type_host spline)
{
  for(int m = 1; m <= n; m++) spline(m, 6) = f[m];

  spline(1, 5) = spline(2, 6) - spline(1, 6);
  spline(2, 5) = 0.5 * (spline(3, 6) - spline(1, 6));
  spline(n - 1, 5) = 0.5 * (spline(n, 6) - spline(n - 2, 6));
  spline(n, 5) = spline(n, 6) - spline(n - 1, 6);

  for(int m = 3; m <= n - 2; m++)
    spline(m, 5) = ((spline(m - 2, 6) - spline(m + 2, 6)) +
                    8.0 * (spline(m + 1, 6) - spline(m - 1, 6))) / 12.0;

  for(int m = 1; m <= n - 1; m++) {
    spline(m, 4) = 3.0 * (spline(m + 1, 6) - spline(m, 6)) -
                   2.0 * spline(m, 5) - spline(m + 1, 5);
    spline(m, 3) = spline(m, 5) + spline(m + 1, 5) -
                   2.0 * (spline(m + 1, 6) - spline(m, 6));
  }

  spline(n, 4) = 0.0;
  spline(n, 3) = 0.0;

  for(int m = 1; m <= n; m++) {
    spline(m, 2) = spline(m, 5) / delta;
    spline(m, 1) = 2.0 * spline(m, 4) / delta;
    spline(m, 0) = 3.0 * spline(m, 3) / delta;
  }
}

/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void ForceEAM::grab(FILE* fptr, MMD_int n, tvector_1d_host list)
{
  char* ptr;
  char line[MAXLINE];

  int i = 0;

  while(i < n) {
    fgets(line, MAXLINE, fptr);
    ptr = strtok(line, " \t\n\r\f");
    list[i++] = atof(ptr);

    while(ptr = strtok(NULL, " \t\n\r\f")) list[i++] = atof(ptr);
  }
}


void ForceEAM::communicate(Atom &atom, Comm &comm)
{

  int iswap;
  tvector_1d buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = 0; iswap < comm.nswap; iswap++) {

    /* pack buffer */

    int size = pack_comm(comm.sendnum[iswap], iswap, comm.buf_send, comm.sendlist);
    device_type::fence();


    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(comm.sendproc[iswap] != me) {
      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(comm.buf_recv.ptr_on_device(), comm.comm_recv_size[iswap], MPI_FLOAT,
                  comm.recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(comm.buf_send.ptr_on_device(), comm.comm_send_size[iswap], MPI_FLOAT,
                 comm.sendproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(comm.buf_recv.ptr_on_device(), comm.comm_recv_size[iswap], MPI_DOUBLE,
                  comm.recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(comm.buf_send.ptr_on_device(), comm.comm_send_size[iswap], MPI_DOUBLE,
                 comm.sendproc[iswap], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);
      buf = comm.buf_recv;
    } else buf = comm.buf_send;

    /* unpack buffer */

    unpack_comm(comm.recvnum[iswap], comm.firstrecv[iswap], buf);
    device_type::fence();
  }
}
/* ---------------------------------------------------------------------- */

int ForceEAM::pack_comm(int n, int iswap, tvector_1d buf, tvector_2i asendlist)
{
  EAMPackCommFunctor f;
  f.fp = fp;
  f.actswap = iswap;
  f.sendlist = asendlist;
  f.actbuf = buf;

  //Kokkos::parallel_for(0,*f_pack_comm);
  Kokkos::parallel_for(n, f);
  device_type::fence();
  return n;
}
/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_comm(int n, int first, tvector_1d buf)
{
  EAMUnpackCommFunctor f;
  f.fp = fp;
  f.first = first;
  f.actbuf = buf;
  Kokkos::parallel_for(n, f);
}

/* ---------------------------------------------------------------------- */

int ForceEAM::pack_reverse_comm(int n, int first, MMD_float* buf)
{
  int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) buf[m++] = h_rho[i];

  return 1;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_reverse_comm(int n, int* list, MMD_float* buf)
{
  int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = list[i];
    h_rho[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

MMD_float ForceEAM::memory_usage()
{
  MMD_int bytes = 2 * nmax * sizeof(MMD_float);
  return bytes;
}


void ForceEAM::bounds(char* str, int nmax, int &nlo, int &nhi)
{
  char* ptr = strchr(str, '*');

  if(ptr == NULL) {
    nlo = nhi = atoi(str);
  } else if(strlen(str) == 1) {
    nlo = 1;
    nhi = nmax;
  } else if(ptr == str) {
    nlo = 1;
    nhi = atoi(ptr + 1);
  } else if(strlen(ptr + 1) == 0) {
    nlo = atoi(str);
    nhi = nmax;
  } else {
    nlo = atoi(str);
    nhi = atoi(ptr + 1);
  }

  if(nlo < 1 || nhi > nmax) printf("Numeric index is out of bounds");
}
