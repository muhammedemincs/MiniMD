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

#ifndef NEIGHBOR_H
#define NEIGHBOR_H

#include "atom.h"
#include "threadData.h"
#include "timer.h"
#include "types.h"

class NeighborBuildFunctor;
class NeighborBuildCudaFunctor;
class NeighborBinatomsFunctor;

class Neighbor
{

  public:
    MMD_int every;                       // re-neighbor every this often
    MMD_int nbinx, nbiny, nbinz;         // # of global bins
    MMD_float cutneigh;                 // neighbor cutoff
    MMD_float cutneighsq;               // neighbor cutoff squared
    MMD_int ncalls;                      // # of times build has been called
    MMD_int max_totalneigh;              // largest # of neighbors ever stored

    tvector_1i numneigh;                   // # of neighbors for each atom
    tvector_1i_host h_numneigh;                   // # of neighbors for each atom
    tvector_neighbors neighbors;                  // array of neighbors of each atom
    MMD_int maxneighs;
    tscalar_i new_maxneighs;				   // max number of neighbors per atom
    tscalar_i_host h_new_maxneighs;				   // max number of neighbors per atom
    MMD_int halfneigh;

    MMD_int ghost_newton;
    MMD_int count;
    MMD_int nlocal, nall;

    t_x_array_tex x;
    Neighbor();
    ~Neighbor();
    MMD_int setup(Atom &);               // setup bins based on box and cutoff
    void finalise();
    void build(Atom &);              // create neighbor list


    Timer* timer;

    ThreadData* threads;
    // private:
    MMD_float xprd, yprd, zprd;         // box size

    MMD_int nmax;                        // max size of atom arrays in neighbor
    tvector_1i bincount;                   // # of atoms in each bin
    tvector_2i bins;                       // #
    MMD_int atoms_per_bin;

    MMD_int nstencil;                    // # of bins in stencil
    tvector_1i stencil;                    // stencil list of bin offsets
    tvector_1i_host h_stencil;                    // stencil list of bin offsets

    MMD_int mbins;                       // binning parameters
    MMD_int mbinx, mbiny, mbinz;
    MMD_int mbinxlo, mbinylo, mbinzlo;
    MMD_float binsizex, binsizey, binsizez;
    MMD_float bininvx, bininvy, bininvz;

    tscalar_i resize;
    tscalar_i_host h_resize;

    void binatoms(Atom &, MMD_int count);           // bin all atoms
  private:
    friend class NeighborBuildFunctor;
    NeighborBuildFunctor* f_build;
    friend class NeighborBuildCudaFunctor;
    NeighborBuildCudaFunctor* f_build_cuda;
    KOKKOS_FUNCTION void build_Item(const MMD_int &i) const;           // create neighlist of atom i

#ifdef KOKKOS_HAVE_CUDA
    KOKKOS_FUNCTION void build_ItemCuda(Kokkos::Cuda device) const;           // create neighlist of atom i
#endif

    friend class NeighborBinatomsFunctor;
    NeighborBinatomsFunctor* f_binatoms;
    KOKKOS_FUNCTION void binatomsItem(const MMD_int &i) const;           // bin atom i


    KOKKOS_FUNCTION double bindist(MMD_int, MMD_int, MMD_int);   // distance between binx
    KOKKOS_INLINE_FUNCTION int coord2bin(MMD_float, MMD_float, MMD_float) const;   // mapping atom coord to a bin
};

struct NeighborBuildFunctor {
  typedef t_x_array::device_type  device_type ;
  Neighbor c;

  KOKKOS_INLINE_FUNCTION void operator()(const MMD_int i) const {
    c.build_Item(i);
  }
};

struct NeighborBuildCudaFunctor {
  typedef t_x_array::device_type  device_type ;
  Neighbor c;

  KOKKOS_INLINE_FUNCTION size_t shmem_size() const {
    const int factor = c.atoms_per_bin<64?2:1;
    return c.atoms_per_bin * 4 * sizeof(MMD_float) * factor;
  }
#ifdef KOKKOS_HAVE_CUDA
  KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Cuda device) const {
    c.build_ItemCuda(device);
  }
#else
  KOKKOS_INLINE_FUNCTION void operator()(const MMD_int i) const {
    c.build_Item(i);
  }
#endif
};


struct NeighborBinatomsFunctor {
  typedef t_x_array::device_type  device_type ;
  Neighbor c;

  KOKKOS_INLINE_FUNCTION void operator()(const MMD_int i) const {
    c.binatomsItem(i);
  }
};

struct MemsetZeroFunctor {
  typedef t_x_array::device_type  device_type ;
  void* ptr;
  KOKKOS_INLINE_FUNCTION void operator()(const MMD_int i) const {
    ((MMD_int*)ptr)[i] = 0;
  }
};
#endif
