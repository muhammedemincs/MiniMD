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

#ifndef COMM_H
#define COMM_H

#include "atom.h"
#include "threadData.h"
#include "types.h"
#include "timer.h"

namespace Kokkos
{



}
class CommExchangeSendItemAFunctor;
class CommExchangeSendItemBFunctor;
class CommExchangeSendItemCFunctor;
class CommExchangeRecvItemAFunctor;
class CommExchangeRecvItemBFunctor;
class CommBorderSendItemAFunctor;
class CommBorderSendItemBFunctor;
class CommBorderRecvItemFunctor;


class Comm
{
  public:
    Comm();
    ~Comm();
    int setup(double, Atom &);
    void finalize();
    void communicate(Atom &);
    void reverse_communicate(Atom &);
    void exchange(Atom &);
    void exchange_all(Atom &);
    void borders(Atom &);
    void growsend(int);
    void growrecv(int);
    void growlist(int, int);

  public:
    int me;                           // my proc ID
    int nswap;                        // # of swaps to perform
    MMD_float pbc_flag[4];                     // whether any PBC on this swap
    tvector_1i_host pbc_any;                     // whether any PBC on this swap
    tvector_1i_host pbc_flagx;                   // PBC correction in x for this swap
    tvector_1i_host pbc_flagy;                   // same in y
    tvector_1i_host pbc_flagz;                   // same in z
    tvector_1i_host sendnum, recvnum;           // # of atoms to send/recv in each swap
    tvector_1i_host comm_send_size;              // # of values to send in each comm
    tvector_1i_host comm_recv_size;              // # of values to recv in each comm
    tvector_1i_host reverse_send_size;           // # of values to send in each reverse
    tvector_1i_host reverse_recv_size;           // # of values to recv in each reverse
    tvector_1i_host sendproc, recvproc;         // proc to send/recv with at each swap
    tvector_1i_host sendproc_exc, recvproc_exc; // proc to send/recv with at each swap for safe exchange

    tvector_1i_host firstrecv;                   // where to put 1st recv atom in each swap
    tvector_2i sendlist;                   // list of atoms to send in each swap
    tvector_2i_host h_sendlist;                   // list of atoms to send in each swap
    int maxsendlist;
    int maxswap;

    tvector_1d buf_send;                 // send buffer for all comm
    tvector_1d buf_recv;                 // recv buffer for all comm
    tvector_1d buf;                 // send buffer for all comm
    tvector_1d_host h_buf_send;                 // send buffer for all comm
    tvector_1d_host h_buf_recv;                 // recv buffer for all comm
    int maxsend;
    int maxrecv;

    int procneigh[3][3];              // my 6 proc neighbors
    int procgrid[3];                  // # of procs in each dim
    int need[3];                      // how many procs away needed in each dim
    tvector_1d_host slablo, slabhi;           // bounds of slabs to send to other procs

    ThreadData* threads;		    //

    int check_safeexchange;           // if sets give warnings if an atom moves further than subdomain size
    int do_safeexchange;		    // exchange atoms with all subdomains within neighbor cutoff
    Timer* timer;

  private:
    int idim, iswap;
    int nsplit;
    MMD_float lo, hi;
    int copy_size;
    tvector_1i nsend_thread, nsend_thread_startpos;
    tscalar_i nsend_total;
    tscalar_i nrecv_total;
    tscalar_i nholes_total;
    tscalar_i_host h_nsend_total;
    tscalar_i_host h_nrecv_total;
    tscalar_i_host h_nholes_total;
    tvector_1i nrecv_thread;
    tvector_1i nholes_thread, nholes_thread_startpos;
    tvector_2i exc_sendlist_thread;
    tvector_1i send_flag;

    t_x_array x;
    t_x_array_tex t_x;
    t_v_array v;
    t_v_array_tex t_v;
    int nthreads;
    int maxnlocal;
    int nrecv_atoms;
    int nlocal;
    int nfirst;
    int nlast;
    int nstart;
    struct Box box;

    friend class CommExchangeSendItemAFunctor;
    friend class CommExchangeSendItemBFunctor;
    friend class CommExchangeSendItemCFunctor;
    friend class CommExchangeRecvItemAFunctor;
    friend class CommExchangeRecvItemBFunctor;
    CommExchangeSendItemAFunctor* f_exchange_sendItemA;
    CommExchangeSendItemBFunctor* f_exchange_sendItemB;
    CommExchangeSendItemCFunctor* f_exchange_sendItemC;
    CommExchangeRecvItemAFunctor* f_exchange_recvItemA;
    CommExchangeRecvItemBFunctor* f_exchange_recvItemB;
    KOKKOS_FUNCTION
    void exchange_sendItemA(MMD_int i) const; //how many am I going to send
    KOKKOS_FUNCTION
    void exchange_sendItemB(MMD_int i) const; //how many holes to fill
    KOKKOS_FUNCTION
    void exchange_sendItemC(MMD_int i) const; //copy form end to fill holes, pack excbuffer
    KOKKOS_FUNCTION
    void exchange_recvItemA(MMD_int i) const; //figure out which ones to keep, and where to put them
    KOKKOS_FUNCTION
    void exchange_recvItemB(MMD_int i) const; //copy from excbuffer to data


    friend class CommBorderSendItemAFunctor;
    friend class CommBorderSendItemBFunctor;
    friend class CommBorderRecvItemFunctor;
    CommBorderSendItemAFunctor* f_border_sendItemA;
    CommBorderSendItemBFunctor* f_border_sendItemB;
    CommBorderRecvItemFunctor* f_border_recvItem;
    KOKKOS_FUNCTION
    void border_sendItemA(MMD_int i) const; //how many am I going to send,
    KOKKOS_FUNCTION
    void border_sendItemB(MMD_int i) const; //pack into buffer
    KOKKOS_FUNCTION
    void border_recvItem(MMD_int i) const; //unpack buffer


};

struct CommExchangeSendItemAFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.exchange_sendItemA(i);
  }
};

struct CommExchangeSendItemBFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.exchange_sendItemB(i);
  }
};

struct CommExchangeSendItemCFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.exchange_sendItemC(i);
  }
};

struct CommExchangeRecvItemAFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.exchange_recvItemA(i);
  }
};

struct CommExchangeRecvItemBFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.exchange_recvItemB(i);
  }
};

struct CommBorderSendItemAFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.border_sendItemA(i);
  }
};

struct CommBorderSendItemBFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.border_sendItemB(i);
  }
};

struct CommBorderRecvItemFunctor {
  typedef t_x_array::device_type                   device_type ;
  Comm c;
  KOKKOS_INLINE_FUNCTION
  void operator()(const MMD_int i) const {
    c.border_recvItem(i);
  }
};

#endif
