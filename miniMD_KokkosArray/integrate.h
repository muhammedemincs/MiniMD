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
#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "atom.h"
#include "force.h"
#include "neighbor.h"
#include "comm.h"
#include "thermo.h"
#include "timer.h"
#include "threadData.h"
#include "types.h"

struct FinalIntegrateFunctor;
struct InitialIntegrateFunctor;

class Integrate
{
  public:
    MMD_float dt;
    MMD_float dtforce;
    MMD_bigint ntimes;
    MMD_int nlocal, nmax;

    t_x_array x;
    t_v_array v;
    t_f_array f;

    MMD_float mass;

    int sort_every;
    Integrate();
    ~Integrate();
    void setup();
    void initialIntegrate();
    void finalIntegrate();
    void run(Atom &, Force*, Neighbor &, Comm &, Thermo &, Timer &);
    void finalise();

    ThreadData* threads;

  private:
    friend class FinalIntegrateFunctor;
    friend class InitialIntegrateFunctor;

    FinalIntegrateFunctor* f_finalIntegrate;
    InitialIntegrateFunctor* f_initialIntegrate;
    KOKKOS_FUNCTION void initialIntegrateItem(const int &i) const;
    KOKKOS_FUNCTION void finalIntegrateItem(const int &i) const;
    inline void initialIntegrateFull() const;
    inline void initialIntegrateFullPlain() const;

};

/*#define Functor(a,b,c)
template<class T,
*/
struct InitialIntegrateFunctor {
  typedef t_x_array::device_type                   device_type ;
  Integrate c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.initialIntegrateItem(i);
  }
};

struct FinalIntegrateFunctor {
  typedef t_x_array::device_type                   device_type ;
  Integrate c;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.finalIntegrateItem(i);
  }
};

#endif
