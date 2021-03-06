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

#ifndef FORCE_H_
#define FORCE_H_

#include "ljs.h"
#include "atom.h"
#include "neighbor.h"
#include "comm.h"



class Force
{
  public:
    MMD_float cutforce;
    MMD_float cutforcesq;
    MMD_float eng_vdwl;
    MMD_float mass;
    MMD_int evflag;
    MMD_float virial;
    MMD_int ghost_newton;

    Force() {};
    virtual ~Force() {};
    virtual void setup() {};
    virtual void finalise() {};
    virtual void compute(Atom &, Neighbor &, Comm &, int) {};

    int use_sse;
    int use_oldcompute;
    ThreadData* threads;
    MMD_int reneigh;
    Timer* timer;

    MMD_float epsilon, sigma6, sigma; //Parameters for LJ only

    ForceStyle style;
  protected:

    MMD_int me;
    MMD_int nlocal, nmax;
    t_x_array x;
    t_f_array f;
    t_x_array_tex t_x;
    tvector_1i numneigh;                   // # of neighbors for each atom
    tvector_neighbors neighbors;                  // array of neighbors of each atom
    MMD_int maxneighs;
};

struct ForceZeroFunctor {
  typedef t_f_array::device_type  device_type ;

  t_f_array f;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    f(i, 0) = 0;
    f(i, 1) = 0;
    f(i, 2) = 0;
  }
};

#endif /* FORCE_H_ */
