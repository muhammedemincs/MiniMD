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

#ifndef FORCEEAM_H
#define FORCEEAM_H

#include "stdio.h"
#include "atom.h"
#include "neighbor.h"
#include "threadData.h"
#include "types.h"
#include "mpi.h"
#include "comm.h"
#include "force.h"

class ForceEAMComputeHalfneighAFunctor;
class ForceEAMComputeHalfneighBFunctor;
class ForceEAMComputeHalfneighCFunctor;
class ForceEAMComputeHalfneighThreadedAFunctor;
class ForceEAMComputeHalfneighThreadedBFunctor;
class ForceEAMComputeFullneighAFunctor;
class ForceEAMComputeFullneighBFunctor;


class ForceEAM : Force
{
  public:
	typedef Kokkos::View<MMD_float*[7] , Kokkos::LayoutRight, device_type > EAM_arrays_type;
	typedef EAM_arrays_type::HostMirror EAM_arrays_type_host;
	typedef Kokkos::View<const MMD_float*[7] , Kokkos::LayoutRight, device_type, Kokkos::MemoryRandomRead > EAM_arrays_type_tex;

    // public variables so USER-ATC package can access them

    MMD_float cutmax;

    // potentials as array data

    MMD_int nrho, nr;
    tvector_1d frho, rhor, z2r;
    tvector_1d_host h_frho, h_rhor, h_z2r;

    // potentials in spline form used for force computation

    MMD_float dr, rdr, drho, rdrho;
    //t_x_array_o rhor_spline, frho_spline, z2r_spline;
    //t_x_array_o_host h_rhor_spline, h_frho_spline, h_z2r_spline;
    EAM_arrays_type rhor_spline, frho_spline, z2r_spline;
    EAM_arrays_type_host h_rhor_spline, h_frho_spline, h_z2r_spline;
    EAM_arrays_type_tex t_rhor_spline, t_frho_spline, t_z2r_spline;

    ForceEAM();
    virtual ~ForceEAM();
    virtual void compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    virtual void coeff(const char*);
    virtual void setup();
    void finalise();
    void init_style();

    virtual MMD_int pack_comm(int n, int iswap, tvector_1d buf, tvector_2i asendlist);
    virtual void unpack_comm(int n, int first, tvector_1d buf);
    MMD_int pack_reverse_comm(MMD_int, MMD_int, MMD_float*);
    void unpack_reverse_comm(MMD_int, MMD_int*, MMD_float*);
    MMD_float memory_usage();

  protected:
    void compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    void compute_halfneigh_threaded(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    void compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    friend class ForceEAMComputeHalfneighAFunctor;
    friend class ForceEAMComputeHalfneighBFunctor;
    friend class ForceEAMComputeHalfneighCFunctor;
    friend class ForceEAMComputeHalfneighThreadedAFunctor;
    friend class ForceEAMComputeHalfneighThreadedBFunctor;
    friend class ForceEAMComputeFullneighAFunctor;
    friend class ForceEAMComputeFullneighBFunctor;

    ForceEAMComputeHalfneighAFunctor* f_compute_halfneighA;
    KOKKOS_FUNCTION void compute_halfneighItemA(const MMD_int &i) const;

    ForceEAMComputeHalfneighBFunctor* f_compute_halfneighB;
    template<int EVFLAG>
    KOKKOS_FUNCTION MMD_float2 compute_halfneighItemB(const MMD_int &i) const;

    ForceEAMComputeHalfneighCFunctor* f_compute_halfneighC;
    template<int EVFLAG>
    KOKKOS_FUNCTION MMD_float compute_halfneighItemC(const MMD_int &i) const;

    ForceEAMComputeHalfneighThreadedAFunctor* f_compute_halfneigh_threadedA;
    KOKKOS_FUNCTION void compute_halfneigh_threadedItemA(const MMD_int &i) const;

    ForceEAMComputeHalfneighThreadedBFunctor* f_compute_halfneigh_threadedB;
    template<int EVFLAG>
    KOKKOS_FUNCTION MMD_float2 compute_halfneigh_threadedItemB(const MMD_int &i) const;

    ForceEAMComputeFullneighAFunctor* f_compute_fullneighA;
    template<int EVFLAG>
    KOKKOS_FUNCTION MMD_float compute_fullneighItemA(const MMD_int &i) const;

    ForceEAMComputeFullneighBFunctor* f_compute_fullneighB;
    template<int EVFLAG>
    KOKKOS_FUNCTION  MMD_float2 compute_fullneighItemB(const MMD_int &i) const;


    // per-atom arrays

    tvector_1d rho, fp;
    tvector_1d_host h_rho, h_fp;

    // potentials as file data

    MMD_int* map;                   // which element each atom type maps to

    struct Funcfl {
      char* file;
      MMD_int nrho, nr;
      double drho, dr, cut, mass;
      tvector_1d_host frho, rhor, zr;
    };
    Funcfl funcfl;

    void array2spline();
    //void interpolate(MMD_int n, MMD_float delta, tvector_1d_host f, t_x_array_o_host spline);
    void interpolate(MMD_int n, MMD_float delta, tvector_1d_host f, EAM_arrays_type_host spline);

    void grab(FILE*, MMD_int, tvector_1d_host);

    virtual void read_file(const char*);
    virtual void file2array();

    void bounds(char* str, int nmax, int &nlo, int &nhi);

    void communicate(Atom &atom, Comm &comm);
};


struct ForceEAMComputeHalfneighAFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_halfneighItemA(i);
  }
};

struct ForceEAMComputeHalfneighBFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float2	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_halfneighItemB<0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy_virial) const {
    MMD_float2 ev_tmp = c.compute_halfneighItemB<1>(i);
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

struct ForceEAMComputeHalfneighCFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_halfneighItemC<0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy) const {
    energy += c.compute_halfneighItemC<1>(i);
  }
  KOKKOS_FUNCTION static void init(volatile value_type &update) {
    update = 0;
  }
  KOKKOS_FUNCTION static void join(volatile value_type &update ,
                                        const volatile value_type &source) {
    update += source ;
  }

};


struct ForceEAMComputeHalfneighThreadedAFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_halfneigh_threadedItemA(i);
  }
};

struct ForceEAMComputeHalfneighThreadedBFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float2	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_halfneigh_threadedItemB<0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy_virial) const {
    MMD_float2 ev_tmp = c.compute_halfneigh_threadedItemB<1>(i);
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


struct ForceEAMComputeFullneighAFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_fullneighItemA<0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy) const {
    energy += c.compute_fullneighItemA<1>(i);
  }
  KOKKOS_FUNCTION static void init(volatile value_type &update) {
    update = 0;
  }
  KOKKOS_FUNCTION static void join(volatile value_type &update ,
                                        const volatile value_type &source) {
    update += source ;
  }

};

struct ForceEAMComputeFullneighBFunctor {
  typedef t_x_array::device_type                   device_type ;
  typedef MMD_float2	value_type;

  ForceEAM c;

  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    c.compute_fullneighItemB<0>(i);
  }
  KOKKOS_FUNCTION
  void operator()(const MMD_int i, value_type &energy_virial) const {
    MMD_float2 ev_tmp = c.compute_fullneighItemB<1>(i);
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

struct EAMPackCommFunctor {
  typedef t_x_array::device_type                   device_type ;
  int actswap;
  tvector_1d fp;
  tvector_1d actbuf;
  tvector_2i sendlist;
  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    const MMD_int j = sendlist(actswap, i);
    actbuf[i] = fp[j];
  }
};

struct EAMUnpackCommFunctor {
  typedef t_x_array::device_type                   device_type ;
  int first;
  tvector_1d fp;
  tvector_1d actbuf;
  KOKKOS_FUNCTION
  void operator()(const MMD_int i) const {
    fp[i + first] = actbuf[i];
  }
};

