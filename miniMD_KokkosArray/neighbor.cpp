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
#include "stdlib.h"

#include "neighbor.h"
#include <Kokkos_Atomic.hpp>

#define FACTOR 0.999
#define SMALL 1.0e-6

Neighbor::Neighbor()
{
  ncalls = 0;
  max_totalneigh = 0;
  maxneighs = 100;
  nmax = 0;
  atoms_per_bin = 8;
  threads = NULL;
  halfneigh = 0;
  resize = tscalar_i("Resize");
  new_maxneighs = tscalar_i("NewMaxneighs");
  h_resize = Kokkos::create_mirror_view(resize);
  h_new_maxneighs = Kokkos::create_mirror_view(new_maxneighs);
  ghost_newton = 1;
  nbinx = -1;
  nbiny = -1;
  nbinz = -1;
}

Neighbor::~Neighbor()
{
}

void Neighbor::finalise()
{
  delete f_binatoms;
  delete f_build;
  delete f_build_cuda;
}


/* binned neighbor list construction with full Newton's 3rd law
   every pair stored exactly once by some processor
   each owned atom i checks its own bin and other bins in Newton stencil */
#define THREADS_PER_BIN 32

KOKKOS_FUNCTION
void Neighbor::build_Item(const MMD_int &i) const
{
#ifndef KOKKOS_HAVE_CUDA
  /* if necessary, goto next page and add pages */
  MMD_int n = 0;

  // get subview of neighbors of i
  const tvector_neighbors_sub neighbors_i =
  		  Kokkos::subview<tvector_neighbors_sub>(neighbors,i,Kokkos::ALL());
  const tvector_1i_const_um bincount_c = bincount;

  const MMD_float xtmp = x(i, 0);
  const MMD_float ytmp = x(i, 1);
  const MMD_float ztmp = x(i, 2);

  const MMD_int ibin = coord2bin(xtmp, ytmp, ztmp);

  // loop over all bins in neighborhood (includes ibin)
  for(MMD_int k = 0; k < nstencil; k++) {
    const MMD_int jbin = ibin + stencil[k];

    // get subview of jbin
    const tvector_1i_const_um loc_bin =
    		Kokkos::subview<tvector_1i_const_um>(bins,jbin,Kokkos::ALL());
    if(ibin == jbin)
      for(int m = 0; m < bincount_c[jbin]; m++) {
        const MMD_int j = loc_bin[m];

        //for same bin as atom i skip j if i==j and skip atoms "below and to the left" if using halfneighborlists
        if((j == i) || (halfneigh && !ghost_newton && (j < i))  ||
            (halfneigh && ghost_newton && ((j < i) || ((j >= nlocal) &&
                                           ((x(j, 2) < ztmp) || (x(j, 2) == ztmp && x(j, 1) < ytmp) ||
                                            (x(j, 2) == ztmp && x(j, 1)  == ytmp && x(j, 0) < xtmp)))))
          ) continue;

        const MMD_float delx = xtmp - x(j, 0);
        const MMD_float dely = ytmp - x(j, 1);
        const MMD_float delz = ztmp - x(j, 2);
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        if((rsq <= cutneighsq)) neighbors_i[n++] = j;
      }
    else {
      for(int m = 0; m < bincount_c[jbin]; m++) {
        const MMD_int j = loc_bin[m];

        if(halfneigh && !ghost_newton && (j < i)) continue;

        const MMD_float delx = xtmp - x(j, 0);
        const MMD_float dely = ytmp - x(j, 1);
        const MMD_float delz = ztmp - x(j, 2);
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        if((rsq <= cutneighsq)) neighbors_i[n++] = j;
      }
    }
  }

  numneigh[i] = n;

  if(n >= maxneighs) {
    resize(0) = 1;

    if(n >= new_maxneighs(1)) new_maxneighs(1) = n;
  }
#endif
}


#ifdef KOKKOS_HAVE_CUDA
KOKKOS_INLINE_FUNCTION
void Neighbor::build_ItemCuda(Kokkos::Cuda dev) const
{

  const int factor = atoms_per_bin<64?2:1;
  typedef Kokkos::View<MMD_float*[3], Kokkos::LayoutLeft,device_type,Kokkos::MemoryUnmanaged> tv_um_d1;
  tv_um_d1 other_x(dev,(size_t) atoms_per_bin*factor);


  typedef Kokkos::View<int*, Kokkos::LayoutLeft,device_type,Kokkos::MemoryUnmanaged> tv_um_i1;
  tv_um_i1 other_id(dev,(size_t) atoms_per_bin*factor);

  const unsigned int ibin = dev.league_rank()*factor+dev.team_rank()/atoms_per_bin;
  const unsigned int worker_rank = dev.team_rank()%atoms_per_bin;
  const unsigned int worker_count = dev.team_size()/factor;
  for(int ii = 0; ii<bincount[ibin];ii+=worker_count) {

	const int i = ii+worker_rank<bincount[ibin]?bins(ibin, ii+worker_rank):-1;
	int test = (dev.team_barrier_count(i >= 0 && i <= nlocal) == 0);

	if(test) continue;

	int n = 0;
	MMD_float xtmp,ytmp,ztmp;
	if(i>=0 && i<nlocal) {
	  xtmp = x(i,0);
	  ytmp = x(i,1);
	  ztmp = x(i,2);
	}

	{
	  const MMD_int jbin = ibin;

      const MMD_int bincount_current = bincount[jbin];

      for(int jj = worker_rank; jj<bincount_current;jj+=worker_count) {
        const MMD_int j = bins(jbin, jj);
        other_x(jj,0) = x(j,0);
        other_x(jj,1) = x(j,1);
        other_x(jj,2) = x(j,2);
        other_id[jj] = j;
      }
      dev.team_barrier();

      if(i>=0 && i<nlocal) {

        #pragma unroll 4
        for(int m = 0; m < bincount_current; m++) {
          const MMD_int j = other_id[m];

          if((j == i) || (halfneigh && (j < i)))  continue;
          const MMD_float delx = xtmp - other_x(m,0);
          const MMD_float dely = ytmp - other_x(m,1);
          const MMD_float delz = ztmp - other_x(m,2);
          const MMD_float rsq = delx * delx + dely * dely + delz * delz;

          if((rsq <= cutneighsq) && (n < maxneighs)) neighbors(i, n++) = j;
        }
      }
      dev.team_barrier();
	}

	for(MMD_int k = 0; k < nstencil; k++) {
      const MMD_int jbin = ibin + stencil[k];

      if(jbin==ibin) continue;
      const MMD_int bincount_current = bincount[jbin];

      for(int jj = worker_rank; jj<bincount_current;jj+=worker_count) {
        const MMD_int j = bins(jbin, jj);
        other_x(jj,0) = x(j,0);
        other_x(jj,1) = x(j,1);
        other_x(jj,2) = x(j,2);
        other_id[jj] = j;
      }

      dev.team_barrier();

      if(i >= 0 && i < nlocal) {
	    #pragma unroll 8
    	for(int m = 0; m < bincount_current; m++) {
		  const MMD_int j = other_id[m];

  	  	  if(halfneigh && (j < i))  continue;
          const MMD_float delx = xtmp - other_x(m,0);
          const MMD_float dely = ytmp - other_x(m,1);
          const MMD_float delz = ztmp - other_x(m,2);
          const MMD_float rsq = delx * delx + dely * dely + delz * delz;

  	  	  if((rsq <= cutneighsq) && (n < maxneighs)) neighbors(i, n++) = j;
		}
      }

      dev.team_barrier();
	}
    if(i >= 0 && i < nlocal)
      numneigh[i] = n;

    if(n >= maxneighs) {
      resize(0) = 1;

      if(n >= new_maxneighs(1)) new_maxneighs(1) = n;
    }
  }
}
#endif

void Neighbor::build(Atom &atom)
{
  ncalls++;
  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;
  /* extend atom arrays if necessary */

  x = atom.t_x;

  if(nall > nmax) {
    nmax = nall;
    numneigh = tvector_1i("numneigh", nmax);
    h_numneigh = Kokkos::create_mirror_view(numneigh);
    neighbors = tvector_neighbors("neighbors", nmax, maxneighs);
  }

  /* bin local & ghost atoms */
  binatoms(atom,nall);
  count = 0;
  /* loop over each atom, storing neighbors */

  h_resize(0) = 1;

  //timer->stamp_extra_start();

  while(h_resize(0)) {
    h_new_maxneighs(0) = maxneighs;
    h_resize(0) = 0;

    Kokkos::deep_copy(resize, h_resize);
    deep_copy(new_maxneighs, h_new_maxneighs);

#if KOKKOS_HAVE_CUDA
    const int factor = atoms_per_bin<64?2:1;
    Kokkos::ParallelWorkRequest config(mbins/factor,atoms_per_bin*factor);
    f_build_cuda->c = *this;
    Kokkos::parallel_for(config, *f_build_cuda);
#else
    f_build->c = *this;
    Kokkos::parallel_for(nlocal, *f_build);
#endif

    device_type::fence();

    deep_copy(h_resize, resize);

    if(h_resize(0)) {
      deep_copy(h_new_maxneighs, new_maxneighs);
      maxneighs = h_new_maxneighs(0) * 1.2;
      neighbors = tvector_neighbors("neighbors", nmax, maxneighs);
    }
  }

  //timer->stamp_extra_stop(TIME_TEST);
}



void Neighbor::binatoms(Atom &atom, MMD_int count)
{
  x = atom.t_x;

  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  h_resize(0) = 1;


  while(h_resize(0) > 0) {
    h_resize(0) = 0;
    deep_copy(resize, h_resize);

    MemsetZeroFunctor f_zero;
    f_zero.ptr = (void*) bincount.ptr_on_device();
    Kokkos::parallel_for(mbins, f_zero);
    device_type::fence();

    f_binatoms->c = *this;
    Kokkos::parallel_for(count, *f_binatoms);
    device_type::fence();

    deep_copy(h_resize, resize);

    if(h_resize(0)) {
      atoms_per_bin *= 2;
      bins = tvector_2i("bins", mbins, atoms_per_bin);
    }
  }
}

KOKKOS_INLINE_FUNCTION void Neighbor::binatomsItem(const MMD_int &i) const
{
  const MMD_int ibin = coord2bin(x(i, 0), x(i, 1), x(i, 2));

  const MMD_int ac = Kokkos::atomic_fetch_add(&bincount[ibin], 1);

  if(ac < atoms_per_bin) {
    bins(ibin, ac) = i;
  } else resize(0) = 1;
}

/* convert xyz atom coords into local bin #
   take special care to insure ghost atoms with
   coord >= prd or coord < 0.0 are put in correct bins */

KOKKOS_INLINE_FUNCTION int Neighbor::coord2bin(MMD_float x, MMD_float y, MMD_float z) const
{
  MMD_int ix, iy, iz;

  if(x >= xprd)
    ix = (MMD_int)((x - xprd) * bininvx) + nbinx - mbinxlo;
  else if(x >= 0.0)
    ix = (MMD_int)(x * bininvx) - mbinxlo;
  else
    ix = (MMD_int)(x * bininvx) - mbinxlo - 1;

  if(y >= yprd)
    iy = (MMD_int)((y - yprd) * bininvy) + nbiny - mbinylo;
  else if(y >= 0.0)
    iy = (MMD_int)(y * bininvy) - mbinylo;
  else
    iy = (MMD_int)(y * bininvy) - mbinylo - 1;

  if(z >= zprd)
    iz = (MMD_int)((z - zprd) * bininvz) + nbinz - mbinzlo;
  else if(z >= 0.0)
    iz = (MMD_int)(z * bininvz) - mbinzlo;
  else
    iz = (MMD_int)(z * bininvz) - mbinzlo - 1;

  return (iz * mbiny * mbinx + iy * mbinx + ix + 1);
}


/*
setup neighbor binning parameters
bin numbering is global: 0 = 0.0 to binsize
                         1 = binsize to 2*binsize
                         nbin-1 = prd-binsize to binsize
                         nbin = prd to prd+binsize
                         -1 = -binsize to 0.0
coord = lowest and highest values of ghost atom coords I will have
        add in "small" for round-off safety
mbinlo = lowest global bin any of my ghost atoms could fall into
mbinhi = highest global bin any of my ghost atoms could fall into
mbin = number of bins I need in a dimension
stencil() = bin offsets in 1-d sense for stencil of surrounding bins
*/

int Neighbor::setup(Atom &atom)
{
  MMD_int i, j, k, nmax;
  MMD_float coord;
  MMD_int mbinxhi, mbinyhi, mbinzhi;
  MMD_int nextx, nexty, nextz;

  cutneighsq = cutneigh * cutneigh;

  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  /*
  c bins must evenly divide into box size,
  c   becoming larger than cutneigh if necessary
  c binsize = 1/2 of cutoff is near optimal

  if (flag == 0) {
    nbinx = 2.0 * xprd / cutneigh;
    nbiny = 2.0 * yprd / cutneigh;
    nbinz = 2.0 * zprd / cutneigh;
    if (nbinx == 0) nbinx = 1;
    if (nbiny == 0) nbiny = 1;
    if (nbinz == 0) nbinz = 1;
  }
  */

  binsizex = xprd / nbinx;
  binsizey = yprd / nbiny;
  binsizez = zprd / nbinz;
  bininvx = 1.0 / binsizex;
  bininvy = 1.0 / binsizey;
  bininvz = 1.0 / binsizez;

  coord = atom.box.xlo - cutneigh - SMALL * xprd;
  mbinxlo = static_cast<int>(coord * bininvx);

  if(coord < 0.0) mbinxlo = mbinxlo - 1;

  coord = atom.box.xhi + cutneigh + SMALL * xprd;
  mbinxhi = static_cast<int>(coord * bininvx);

  coord = atom.box.ylo - cutneigh - SMALL * yprd;
  mbinylo = static_cast<int>(coord * bininvy);

  if(coord < 0.0) mbinylo = mbinylo - 1;

  coord = atom.box.yhi + cutneigh + SMALL * yprd;
  mbinyhi = static_cast<int>(coord * bininvy);

  coord = atom.box.zlo - cutneigh - SMALL * zprd;
  mbinzlo = static_cast<int>(coord * bininvz);

  if(coord < 0.0) mbinzlo = mbinzlo - 1;

  coord = atom.box.zhi + cutneigh + SMALL * zprd;
  mbinzhi = static_cast<int>(coord * bininvz);

  /* extend bins by 1 in each direction to insure stencil coverage */

  mbinxlo = mbinxlo - 1;
  mbinxhi = mbinxhi + 1;
  mbinx = mbinxhi - mbinxlo + 1;

  mbinylo = mbinylo - 1;
  mbinyhi = mbinyhi + 1;
  mbiny = mbinyhi - mbinylo + 1;

  mbinzlo = mbinzlo - 1;
  mbinzhi = mbinzhi + 1;
  mbinz = mbinzhi - mbinzlo + 1;

  /*
  compute bin stencil of all bins whose closest corner to central bin
  is within neighbor cutoff
  for partial Newton (newton = 0),
  stencil is all surrounding bins including self
  for full Newton (newton = 1),
  stencil is bins to the "upper right" of central bin, does NOT include self
  next(xyz) = how far the stencil could possibly extend
  factor < 1.0 for special case of LJ benchmark so code will create
  correct-size stencil when there are 3 bins for every 5 lattice spacings
  */

  nextx = static_cast<int>(cutneigh * bininvx);

  if(nextx * binsizex < FACTOR * cutneigh) nextx++;

  nexty = static_cast<int>(cutneigh * bininvy);

  if(nexty * binsizey < FACTOR * cutneigh) nexty++;

  nextz = static_cast<int>(cutneigh * bininvz);

  if(nextz * binsizez < FACTOR * cutneigh) nextz++;

  nmax = (2 * nextz + 1) * (2 * nexty + 1) * (2 * nextx + 1);
  stencil = tvector_1i("stencil", nmax);
  h_stencil = Kokkos::create_mirror_view(stencil);
  nstencil = 0;
  MMD_int kstart = -nextz;

  if(halfneigh && ghost_newton) {
    kstart = 0;
    h_stencil(nstencil++) = 0;
  }

  for(k = kstart; k <= nextz; k++) {
    for(j = -nexty; j <= nexty; j++) {
      for(i = -nextx; i <= nextx; i++) {
        if(!ghost_newton || !halfneigh || (k > 0 || j > 0 || (j == 0 && i > 0)))
          if(bindist(i, j, k) < cutneighsq) {
            h_stencil(nstencil++) = k * mbiny * mbinx + j * mbinx + i;
          }
      }
    }
  }

  Kokkos::deep_copy(stencil, h_stencil);
  mbins = mbinx * mbiny * mbinz;
  bincount = tvector_1i("bincount", mbins);
  bins = tvector_2i("bins", mbins, atoms_per_bin);
  f_build = new NeighborBuildFunctor;
  f_build_cuda = new NeighborBuildCudaFunctor;
  f_binatoms = new NeighborBinatomsFunctor;
  return 0;
}

/* compute closest distance between central bin (0,0,0) and bin (i,j,k) */

KOKKOS_INLINE_FUNCTION double Neighbor::bindist(MMD_int i, MMD_int j, MMD_int k)
{
  MMD_float delx, dely, delz;

  if(i > 0)
    delx = (i - 1) * binsizex;
  else if(i == 0)
    delx = 0.0;
  else
    delx = (i + 1) * binsizex;

  if(j > 0)
    dely = (j - 1) * binsizey;
  else if(j == 0)
    dely = 0.0;
  else
    dely = (j + 1) * binsizey;

  if(k > 0)
    delz = (k - 1) * binsizez;
  else if(k == 0)
    delz = 0.0;
  else
    delz = (k + 1) * binsizez;

  return (delx * delx + dely * dely + delz * delz);
}

