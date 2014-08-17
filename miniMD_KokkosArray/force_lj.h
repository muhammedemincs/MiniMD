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

#ifndef FORCELJ_H
#define FORCELJ_H

#include "atom.h"
#include "neighbor.h"
#include "threadData.h"
#include "types.h"
#include "force.h"
#include "comm.h"

class ForceComputeFullneighFunctor;
class ForceComputeHalfneighFunctor;
class ForceComputeHalfneighThreadedFunctor;

class ForceLJ : Force
{
  public:
    double sigma6,epsilon;
   
    ForceLJ();
    virtual ~ForceLJ();
    void setup();
    void finalise();
    void compute(Atom &, Neighbor &, Comm &, int);
    void compute_halfneigh(Atom &, Neighbor &, int);
    void compute_halfneigh_threaded(Atom &, Neighbor &, int);

    void compute_fullneigh(Atom &, Neighbor &, int);

  private:
    friend class ForceComputeFullneighFunctor;
    friend class ForceComputeHalfneighFunctor;
    friend class ForceComputeHalfneighThreadedFunctor;

    ForceComputeFullneighFunctor* f_compute_fullneigh;
    template<int EVFLAG>
    KOKKOS_FUNCTION MMD_float2 compute_fullneighItem(const MMD_int &i) const;

    ForceComputeHalfneighFunctor* f_compute_halfneigh;
    template<int EVFLAG, int GHOST_NEWTON>
    KOKKOS_FUNCTION MMD_float2 compute_halfneighItem(const MMD_int &i) const;
    ForceComputeHalfneighThreadedFunctor* f_compute_halfneigh_threaded;
    template<int EVFLAG, int GHOST_NEWTON>
    KOKKOS_FUNCTION MMD_float2 compute_halfneigh_threadedItem(const MMD_int &i) const;
};

struct ForceComputeHalfneighFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float2	value_type;

  ForceLJ c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    if(c.ghost_newton)
      c.compute_halfneighItem<0, 1>(i);

    if(!c.ghost_newton)
      c.compute_halfneighItem<0, 0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy_virial) const {
    MMD_float2 ev_tmp;

    if(c.ghost_newton)
      ev_tmp = c.compute_halfneighItem<1, 1>(i);
    else
      ev_tmp = c.compute_halfneighItem<1, 0>(i);

    energy_virial.x += ev_tmp.x;
    energy_virial.y += ev_tmp.y;
  }
  KOKKOS_FUNCTION static void init(volatile value_type &update) {
    update.x = update.y = 0;
  }
  KOKKOS_FUNCTION static void join(volatile value_type &update ,
                                        const volatile value_type &source) {
    update.x += source.x ;
    update.y += source.y ;
  }
};

struct ForceComputeHalfneighThreadedFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float2	value_type;

  ForceLJ c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    if(c.ghost_newton)
      c.compute_halfneigh_threadedItem<0, 1>(i);

    if(!c.ghost_newton)
      c.compute_halfneigh_threadedItem<0, 0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy_virial) const {
    MMD_float2 ev_tmp;

    if(c.ghost_newton)
      ev_tmp = c.compute_halfneigh_threadedItem<1, 1>(i);
    else
      ev_tmp = c.compute_halfneigh_threadedItem<1, 0>(i);

    energy_virial.x += ev_tmp.x;
    energy_virial.y += ev_tmp.y;
  }
  KOKKOS_FUNCTION static void init(volatile value_type &update) {
    update.x = update.y = 0;
  }
  KOKKOS_FUNCTION static void join(volatile value_type &update ,
                                        const volatile value_type &source) {
    update.x += source.x ;
    update.y += source.y ;
  }
};
struct ForceComputeFullneighFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float2	value_type;
  ForceLJ c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_fullneighItem<0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy_virial) const {
    MMD_float2 ev_tmp = c.compute_fullneighItem<1>(i);
    energy_virial.x += ev_tmp.x;
    energy_virial.y += ev_tmp.y;
  }
  KOKKOS_FUNCTION static void init(volatile value_type &update) {
    update.x = update.y = 0;
  }
  KOKKOS_FUNCTION static void join(volatile value_type &update ,
                                        const volatile value_type &source) {
    update.x += source.x ;
    update.y += source.y ;
  }
};
#endif
