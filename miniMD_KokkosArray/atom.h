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

#ifndef ATOM_H
#define ATOM_H

#include "threadData.h"
#include "types.h"
#include "timer.h"

class Neighbor;
struct Box {
  MMD_float xprd, yprd, zprd;
  MMD_float xlo, xhi;
  MMD_float ylo, yhi;
  MMD_float zlo, zhi;
};

class 	AtomPBCFunctor;
class   AtomPackCommFunctor;
class   AtomUnpackCommFunctor;
class   AtomPackCommPBCFunctor;
class   AtomPackReverseFunctor;
class   AtomUnpackReverseFunctor;
class   AtomSortFunctor;

#define DATA_X 1
#define DATA_V 2
#define DATA_F 4
#define DATA_XOLD 8
#define DATA_VOLD 16
#define DATA_ALL 0xffffffff

class Atom
{
  public:
    MMD_bigint natoms;
    MMD_int nlocal, nghost;
    MMD_int nmax;

    t_x_array x;
    t_x_array_tex t_x;
    t_v_array v;
    t_v_array_tex t_v;
    t_f_array f;
    t_x_array_host h_x;
    t_v_array_host h_v;
    t_f_array_host h_f;

    tvector_1d actbuf;
    MMD_int actswap;
    tvector_2i sendlist;
    int pbc_flags[4];
    MMD_int first;
    ThreadData* threads;
    MMD_float virial, mass;
    int comm_size, reverse_size, border_size;

    struct Box box;

    Atom();
    ~Atom();
    void setup();
    void finalise();
    void addatom(MMD_float, MMD_float, MMD_float, MMD_float, MMD_float, MMD_float);
    void pbc();
    void growarray(MMD_int host_current);
    void growarray(MMD_int host_current, MMD_int newsize);

    void copy(MMD_int, MMD_int);

    void pack_comm(MMD_int, MMD_int , tvector_1d, MMD_int*, tvector_2i);
    void unpack_comm(MMD_int, MMD_int, tvector_1d);

    void pack_reverse(MMD_int, MMD_int, tvector_1d);
    void unpack_reverse(MMD_int, MMD_int , tvector_1d, tvector_2i);

    int pack_border(MMD_int, MMD_float*, MMD_int*);
    int unpack_border(MMD_int, MMD_float*);
    int pack_exchange(MMD_int, MMD_float*);
    int unpack_exchange(MMD_int, MMD_float*);
    int skip_exchange(MMD_float*);

    void upload(int datamask = DATA_ALL);
    void download(int datamask = DATA_ALL);

    void sort(Neighbor & neighbor);
    Timer* timer;

  private:

    tvector_1i binpos;
    tvector_2i bins;
    t_x_array x_copy;
    t_v_array v_copy;

    friend class   AtomPBCFunctor;
    friend class   AtomPackCommFunctor;
    friend class   AtomUnpackCommFunctor;
    friend class   AtomPackCommPBCFunctor;
    friend class   AtomPackReverseFunctor;
    friend class   AtomUnpackReverseFunctor;
    friend class   AtomSortFunctor;

    AtomPBCFunctor* f_pbc;
    KOKKOS_FUNCTION void pbcItem(const MMD_int &i) const;
    AtomPackCommFunctor* f_pack_comm;
    AtomUnpackCommFunctor* f_unpack_comm;
    KOKKOS_FUNCTION void pack_commItem(const MMD_int &i) const;
    KOKKOS_FUNCTION void unpack_commItem(const MMD_int &i) const;

    AtomPackCommPBCFunctor* f_pack_comm_pbc;
    KOKKOS_FUNCTION void pack_commItemPBC(const MMD_int &i) const;

    AtomPackReverseFunctor* f_pack_reverse;
    AtomUnpackReverseFunctor* f_unpack_reverse;
    KOKKOS_FUNCTION void pack_reverseItem(const MMD_int &i) const;
    KOKKOS_FUNCTION void unpack_reverseItem(const MMD_int &i) const;

    AtomSortFunctor* f_sort;
    KOKKOS_FUNCTION void sortItem(const MMD_int &threadId) const;

};

struct AtomPBCFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.pbcItem(i);
  }
};

struct AtomPackCommFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.pack_commItem(i);
  }
};

struct AtomPackCommPBCFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.pack_commItemPBC(i);
  }
};

struct AtomUnpackCommFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.unpack_commItem(i);
  }
};

struct AtomPackReverseFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.pack_reverseItem(i);
  }
};

struct AtomUnpackReverseFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.unpack_reverseItem(i);
  }
};

struct AtomSortFunctor {
  typedef t_x_array::device_type                   device_type ;
  Atom c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.sortItem(i);
  }
};
#endif
