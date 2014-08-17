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
//#define PRINTDEBUG(a) a
#define PRINTDEBUG(a)
#include "stdio.h"
#include "integrate.h"
#include "math.h"
#include "types.h"

Integrate::Integrate()
{
	sort_every = 1000;
}

Integrate::~Integrate()
{
}

void Integrate::setup()
{
  f_initialIntegrate = new InitialIntegrateFunctor;
  f_finalIntegrate = new FinalIntegrateFunctor;

  dtforce = 0.5 * dt;
}

void Integrate::finalise()
{
  delete f_initialIntegrate;
  delete f_finalIntegrate;
}

void Integrate::initialIntegrate()
{
  f_initialIntegrate->c = *this;
  Kokkos::parallel_for(nlocal, *f_initialIntegrate);
}

void Integrate::finalIntegrate()
{
  f_finalIntegrate->c = *this;
  Kokkos::parallel_for(nlocal, *f_finalIntegrate);
}

void Integrate::run(Atom &atom, Force* force, Neighbor &neighbor,
                    Comm &comm, Thermo &thermo, Timer &timer)
{
  mass = atom.mass;
  dtforce = dtforce / mass;
  comm.timer = &timer;
  atom.timer = &timer;
  timer.array[TIME_TEST] = 0.0;

  int next_sort = sort_every;
  for(MMD_bigint n = 0; n < ntimes; n++) {
    x = atom.x;
    v = atom.v;
    f = atom.f;
    nlocal = atom.nlocal;
    nmax = atom.nmax;

    initialIntegrate();
    device_type::fence();

    timer.stamp();

    if((n + 1) % neighbor.every) {
      timer.stamp_extra_start();
      comm.communicate(atom);
      timer.stamp_extra_stop(TIME_TEST);
      timer.stamp(TIME_COMM);
    } else {

      comm.exchange(atom);
      if((next_sort>0) && (n+1>=next_sort)) {
        atom.sort(neighbor);
        next_sort +=  sort_every;
      }
      comm.borders(atom);
      device_type::fence();
      timer.stamp(TIME_COMM);
      neighbor.build(atom);
      device_type::fence();

      timer.stamp(TIME_NEIGH);
    }

    force->evflag = (n + 1) % thermo.nstat == 0;
    force->compute(atom, neighbor, comm, comm.me);
    device_type::fence();


    timer.stamp(TIME_FORCE);

    if(neighbor.halfneigh && neighbor.ghost_newton) {
      timer.stamp_extra_start();
      comm.reverse_communicate(atom);

      timer.stamp_extra_stop(TIME_TEST);
      timer.stamp(TIME_COMM);
    }

    v = atom.v;
    f = atom.f;
    nlocal = atom.nlocal;

    finalIntegrate();
    device_type::fence();

    if(thermo.nstat) thermo.compute(n + 1, atom, neighbor, force, timer, comm);

    device_type::fence();

  }
}

KOKKOS_INLINE_FUNCTION void Integrate::initialIntegrateItem(const MMD_int &i) const
{
  v(i, 0) += dtforce * f(i, 0);
  v(i, 1) += dtforce * f(i, 1);
  v(i, 2) += dtforce * f(i, 2);
  x(i, 0) += dt * v(i, 0);
  x(i, 1) += dt * v(i, 1);
  x(i, 2) += dt * v(i, 2);
}

KOKKOS_INLINE_FUNCTION void Integrate::finalIntegrateItem(const MMD_int &i) const
{
  v(i, 0) += dtforce * f(i, 0);
  v(i, 1) += dtforce * f(i, 1);
  v(i, 2) += dtforce * f(i, 2);
}
