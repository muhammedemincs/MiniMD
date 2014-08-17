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
#include "mpi.h"
#include "comm.h"
#include <Kokkos_Atomic.hpp>

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

Comm::Comm()
{
  maxsend = BUFMIN;
  maxrecv = BUFMIN;
  check_safeexchange = 0;
  do_safeexchange = 0;
  maxswap = 0;

  maxnlocal = 0;

}

Comm::~Comm() {}

/* setup spatial-decomposition communication patterns */

int Comm::setup(double cutneigh, Atom &atom)
{
  f_exchange_sendItemA = new CommExchangeSendItemAFunctor;
  f_exchange_sendItemB = new CommExchangeSendItemBFunctor;
  f_exchange_sendItemC = new CommExchangeSendItemCFunctor;
  f_exchange_recvItemA = new CommExchangeRecvItemAFunctor;
  f_exchange_recvItemB = new CommExchangeRecvItemBFunctor;
  f_border_sendItemA = new CommBorderSendItemAFunctor;
  f_border_sendItemB = new CommBorderSendItemBFunctor;
  f_border_recvItem = new CommBorderRecvItemFunctor;
  nsend_total = tscalar_i("Comm::nsend_total");
  nrecv_total = tscalar_i("Comm::nrecv_total");
  nholes_total = tscalar_i("Comm::nholes_total");
  h_nsend_total = Kokkos::create_mirror_view(nsend_total);
  h_nrecv_total = Kokkos::create_mirror_view(nrecv_total);
  h_nholes_total = Kokkos::create_mirror_view(nholes_total);
  buf_send = tvector_1d("buf_send", maxsend + BUFEXTRA);
  buf_recv = tvector_1d("buf_recv", maxrecv + BUFEXTRA);
  h_buf_send  = Kokkos::create_mirror_view(buf_send);
  h_buf_recv  = Kokkos::create_mirror_view(buf_recv);
  int nprocs;
  int periods[3];
  double prd[3];
  int myloc[3];
  MPI_Comm cartesian;
  double lo, hi;
  int ineed, idim, nbox;

  prd[0] = atom.box.xprd;
  prd[1] = atom.box.yprd;
  prd[2] = atom.box.zprd;

  /* setup 3-d grid of procs */

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  double area[3];

  area[0] = prd[0] * prd[1];
  area[1] = prd[0] * prd[2];
  area[2] = prd[1] * prd[2];

  double bestsurf = 2.0 * (area[0] + area[1] + area[2]);

  // loop thru all possible factorizations of nprocs
  // surf = surface area of a proc sub-domain
  // for 2d, insure ipz = 1

  int ipx, ipy, ipz, nremain;
  double surf;

  ipx = 1;

  while(ipx <= nprocs) {
    if(nprocs % ipx == 0) {
      nremain = nprocs / ipx;
      ipy = 1;

      while(ipy <= nremain) {
        if(nremain % ipy == 0) {
          ipz = nremain / ipy;
          surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

          if(surf < bestsurf) {
            bestsurf = surf;
            procgrid[0] = ipx;
            procgrid[1] = ipy;
            procgrid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }

  if(procgrid[0]*procgrid[1]*procgrid[2] != nprocs) {
    if(me == 0) printf("ERROR: Bad grid of processors\n");

    return 1;
  }

  /* determine where I am and my neighboring procs in 3d grid of procs */

  int reorder = 0;
  periods[0] = periods[1] = periods[2] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 3, procgrid, periods, reorder, &cartesian);
  MPI_Cart_get(cartesian, 3, procgrid, periods, myloc);
  MPI_Cart_shift(cartesian, 0, 1, &procneigh[0][0], &procneigh[0][1]);
  MPI_Cart_shift(cartesian, 1, 1, &procneigh[1][0], &procneigh[1][1]);
  MPI_Cart_shift(cartesian, 2, 1, &procneigh[2][0], &procneigh[2][1]);

  /* lo/hi = my local box bounds */

  atom.box.xlo = myloc[0] * prd[0] / procgrid[0];
  atom.box.xhi = (myloc[0] + 1) * prd[0] / procgrid[0];
  atom.box.ylo = myloc[1] * prd[1] / procgrid[1];
  atom.box.yhi = (myloc[1] + 1) * prd[1] / procgrid[1];
  atom.box.zlo = myloc[2] * prd[2] / procgrid[2];
  atom.box.zhi = (myloc[2] + 1) * prd[2] / procgrid[2];

  /* need = # of boxes I need atoms from in each dimension */

  need[0] = static_cast<int>(cutneigh * procgrid[0] / prd[0] + 1);
  need[1] = static_cast<int>(cutneigh * procgrid[1] / prd[1] + 1);
  need[2] = static_cast<int>(cutneigh * procgrid[2] / prd[2] + 1);

  /* alloc comm memory */

  maxswap = 2 * (need[0] + need[1] + need[2]);

  slablo = tvector_1d_host("slablo", maxswap);
  slabhi = tvector_1d_host("slabhi", maxswap);
  pbc_any = tvector_1i_host("pbc_any", maxswap);
  pbc_flagx = tvector_1i_host("pbc_flagx", maxswap);
  pbc_flagy = tvector_1i_host("pbc_flagy", maxswap);
  pbc_flagz = tvector_1i_host("pbc_flagz", maxswap);
  sendproc = tvector_1i_host("sendproc", maxswap);
  recvproc = tvector_1i_host("recvproc", maxswap);
  sendproc_exc = tvector_1i_host("sendproc_exc", maxswap);
  recvproc_exc = tvector_1i_host("recvproc_exc", maxswap);
  sendnum = tvector_1i_host("sendnum", maxswap);
  recvnum = tvector_1i_host("recvnum", maxswap);
  comm_send_size = tvector_1i_host("comm_send_size", maxswap);
  comm_recv_size = tvector_1i_host("comm_recv_size", maxswap);
  reverse_send_size = tvector_1i_host("reverse_send_size", maxswap);
  reverse_recv_size = tvector_1i_host("reverse_recv_size", maxswap);
  int iswap = 0;

  for(int idim = 0; idim < 3; idim++)
    for(int i = 1; i <= need[idim]; i++, iswap += 2) {
      MPI_Cart_shift(cartesian, idim, i, &sendproc_exc[iswap], &sendproc_exc[iswap + 1]);
      MPI_Cart_shift(cartesian, idim, i, &recvproc_exc[iswap + 1], &recvproc_exc[iswap]);
    }

  MPI_Comm_free(&cartesian);

  firstrecv = tvector_1i_host("firstrecv", maxswap);
  maxsendlist = BUFMIN;
  sendlist = tvector_2i("sendlist", maxswap, maxsendlist);
  h_sendlist = Kokkos::create_mirror_view(sendlist);

  /* setup 4 parameters for each exchange: (spart,rpart,slablo,slabhi)
     sendproc(nswap) = proc to send to at each swap
     recvproc(nswap) = proc to recv from at each swap
     slablo/slabhi(nswap) = slab boundaries (in correct dimension) of atoms
                            to send at each swap
     1st part of if statement is sending to the west/south/down
     2nd part of if statement is sending to the east/north/up
     nbox = atoms I send originated in this box */

  /* set commflag if atoms are being exchanged across a box boundary
     commflag(idim,nswap) =  0 -> not across a boundary
                          =  1 -> add box-length to position when sending
                          = -1 -> subtract box-length from pos when sending */

  nswap = 0;

  for(idim = 0; idim < 3; idim++) {
    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {
      pbc_any[nswap] = 0;
      pbc_flagx[nswap] = 0;
      pbc_flagy[nswap] = 0;
      pbc_flagz[nswap] = 0;

      if(ineed % 2 == 0) {
        sendproc[nswap] = procneigh[idim][0];
        recvproc[nswap] = procneigh[idim][1];
        nbox = myloc[idim] + ineed / 2;
        lo = nbox * prd[idim] / procgrid[idim];

        if(idim == 0) hi = atom.box.xlo + cutneigh;

        if(idim == 1) hi = atom.box.ylo + cutneigh;

        if(idim == 2) hi = atom.box.zlo + cutneigh;

        hi = MIN(hi, (nbox + 1) * prd[idim] / procgrid[idim]);

        if(myloc[idim] == 0) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = 1;

          if(idim == 1) pbc_flagy[nswap] = 1;

          if(idim == 2) pbc_flagz[nswap] = 1;
        }
      } else {
        sendproc[nswap] = procneigh[idim][1];
        recvproc[nswap] = procneigh[idim][0];
        nbox = myloc[idim] - ineed / 2;
        hi = (nbox + 1) * prd[idim] / procgrid[idim];

        if(idim == 0) lo = atom.box.xhi - cutneigh;

        if(idim == 1) lo = atom.box.yhi - cutneigh;

        if(idim == 2) lo = atom.box.zhi - cutneigh;

        lo = MAX(lo, nbox * prd[idim] / procgrid[idim]);

        if(myloc[idim] == procgrid[idim] - 1) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = -1;

          if(idim == 1) pbc_flagy[nswap] = -1;

          if(idim == 2) pbc_flagz[nswap] = -1;
        }
      }

      slablo[nswap] = lo;
      slabhi[nswap] = hi;
      nswap++;
    }
  }

  return 0;
}

void Comm::finalize()
{
  delete f_exchange_sendItemA;
  delete f_exchange_sendItemB;
  delete f_exchange_sendItemC;
  delete f_exchange_recvItemA;
  delete f_exchange_recvItemB;
  delete f_border_sendItemA;
  delete f_border_sendItemB;
  delete f_border_recvItem;
}
/* communication of atom info every timestep */

void Comm::communicate(Atom &atom)
{

  int iswap;
  int pbc_flags[4];
  tvector_1d buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = 0; iswap < nswap; iswap++) {

    /* pack buffer */

    pbc_flags[0] = pbc_any[iswap];
    pbc_flags[1] = pbc_flagx[iswap];
    pbc_flags[2] = pbc_flagy[iswap];
    pbc_flags[3] = pbc_flagz[iswap];
    //timer->stamp_extra_start();

    atom.pack_comm(sendnum[iswap], iswap, buf_send, pbc_flags, sendlist);
    device_type::fence();
    //timer->stamp_extra_stop(TIME_TEST);


    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(sendproc[iswap] != me) {
      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(buf_recv.ptr_on_device(), comm_recv_size[iswap], MPI_FLOAT,
                  recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.ptr_on_device(), comm_send_size[iswap], MPI_FLOAT,
                 sendproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(buf_recv.ptr_on_device(), comm_recv_size[iswap], MPI_DOUBLE,
                  recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.ptr_on_device(), comm_send_size[iswap], MPI_DOUBLE,
                 sendproc[iswap], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);
      buf = buf_recv;
    } else buf = buf_send;

    /* unpack buffer */

    atom.unpack_comm(recvnum[iswap], firstrecv[iswap], buf);
    device_type::fence();
  }
}


/* reverse communication of atom info every timestep */

void Comm::reverse_communicate(Atom &atom)
{
  int iswap;
  tvector_1d buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = nswap - 1; iswap >= 0; iswap--) {

    /* pack buffer */

    atom.pack_reverse(recvnum[iswap], firstrecv[iswap], buf_send);
    device_type::fence();

    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(sendproc[iswap] != me) {
      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(buf_recv.ptr_on_device(), reverse_recv_size[iswap], MPI_FLOAT,
                  sendproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.ptr_on_device(), reverse_send_size[iswap], MPI_FLOAT,
                 recvproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(buf_recv.ptr_on_device(), reverse_recv_size[iswap], MPI_DOUBLE,
                  sendproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.ptr_on_device(), reverse_send_size[iswap], MPI_DOUBLE,
                 recvproc[iswap], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);
      buf = buf_recv;
    } else buf = buf_send;

    /* unpack buffer */

    atom.unpack_reverse(sendnum[iswap], iswap, buf, sendlist);
    device_type::fence();
  }
}

/* exchange:
   move atoms to correct proc boxes
   send out atoms that have left my box, receive ones entering my box
   this routine called before every reneighboring
   atoms exchanged with all 6 stencil neighbors
*/

KOKKOS_INLINE_FUNCTION
void Comm::exchange_sendItemA(MMD_int thread) const
{
  int chunk = (nlocal + nthreads - 1) / nthreads;
  int start = chunk * thread;
  int end = start + chunk;

  if(end > nlocal) end = nlocal;

  int maxsend = exc_sendlist_thread.dimension(1);
  int nsend = 0;

  for(int i = start; i < end; i++) {
    if(x(i, idim) < lo || x(i, idim) >= hi) {

      if(nsend < maxsend)
        exc_sendlist_thread(thread, nsend) = i;

      nsend++;
      send_flag(i) = 0;
    } else
      send_flag(i) = 1;
  }

  nsend_thread_startpos(thread) = Kokkos::atomic_fetch_add(&nsend_total(0), nsend);
  nsend_thread(thread) = nsend;
}

KOKKOS_INLINE_FUNCTION
void Comm::exchange_sendItemB(MMD_int thread) const
{
  int nsend = nsend_thread(thread);
  int nholes = 0;

  for(int k = 0; k < nsend; k++)
    if(exc_sendlist_thread(thread, k) < nlocal - nsend_total(0)) {
      nholes++;
    }

  nholes_thread_startpos(thread) = Kokkos::atomic_fetch_add(&nholes_total(0), nholes);
  nholes_thread(thread) = nholes;
}

KOKKOS_INLINE_FUNCTION
void Comm::exchange_sendItemC(MMD_int thread) const
{
  int holes = 0;
  int j = nlocal;

  while(holes < nholes_thread_startpos(thread)) {
    j--;

    if(send_flag(j)) holes++;
  }

  int nsend = nsend_thread(thread);
  int nsend_startpos = nsend_thread_startpos(thread);

  for(int k = 0; k < nsend; k++) {
    int i = exc_sendlist_thread(thread, k);

    buf_send((nsend_startpos + k) * 6 + 0) = t_x(i, 0);
    buf_send((nsend_startpos + k) * 6 + 1) = t_x(i, 1);
    buf_send((nsend_startpos + k) * 6 + 2) = t_x(i, 2);
    buf_send((nsend_startpos + k) * 6 + 3) = t_v(i, 0);
    buf_send((nsend_startpos + k) * 6 + 4) = t_v(i, 1);
    buf_send((nsend_startpos + k) * 6 + 5) = t_v(i, 2);


    j--;

    if(i < nlocal - nsend_total(0)) {
      while(!send_flag[j]) j--;

      x(i, 0) = t_x(j, 0);
      x(i, 1) = t_x(j, 1);
      x(i, 2) = t_x(j, 2);
      v(i, 0) = t_v(j, 0);
      v(i, 1) = t_v(j, 1);
      v(i, 2) = t_v(j, 2);
    }
  }
}

KOKKOS_INLINE_FUNCTION
void Comm::exchange_recvItemA(MMD_int thread) const
{
  int chunk = (nsend_total(0) + nthreads - 1) / nthreads;
  int start = chunk * thread;
  int end = start + chunk;

  if(end > nsend_total(0)) end = nsend_total(0);

  int nrecv = 0;

  for(int i = start; i < end; i++) {
    if(buf_recv(6 * i + idim) >= lo && buf_recv(6 * i + idim) < hi)
      nrecv++;
  }

  nsend_thread_startpos(thread) = Kokkos::atomic_fetch_add(&nrecv_total(0), nrecv);
}

KOKKOS_INLINE_FUNCTION
void Comm::exchange_recvItemB(MMD_int thread) const
{
  int chunk = (nsend_total(0) + nthreads - 1) / nthreads;
  int start = chunk * thread;
  int end = start + chunk;

  if(end > nsend_total(0)) end = nsend_total(0);

  int j = nlocal + nsend_thread_startpos(thread);

  for(int i = start; i < end; i++) {
    if(buf_recv(6 * i + idim) >= lo && buf_recv(6 * i + idim) < hi) {
      x(j, 0) = buf_recv(6 * i);
      x(j, 1) = buf_recv(6 * i + 1);
      x(j, 2) = buf_recv(6 * i + 2);
      v(j, 0) = buf_recv(6 * i + 3);
      v(j, 1) = buf_recv(6 * i + 4);
      v(j, 2) = buf_recv(6 * i + 5);

      j++;
    }
  }
}

void Comm::exchange(Atom &atom)
{
  if(do_safeexchange)
    return exchange_all(atom);

  MPI_Request request;
  MPI_Status status;
  nthreads = threads->omp_num_threads;
  /* enforce PBC */

  atom.pbc();

  /* loop over dimensions */

  for(idim = 0; idim < 3; idim++) {

    /* only exchange if more than one proc in this dimension */

    if(procgrid[idim] == 1) continue;

    /* fill buffer with atoms leaving my box
       when atom is deleted, fill it in with last atom */

    int nsend = 0;

    if(idim == 0) {
      lo = atom.box.xlo;
      hi = atom.box.xhi;
    } else if(idim == 1) {
      lo = atom.box.ylo;
      hi = atom.box.yhi;
    } else {
      lo = atom.box.zlo;
      hi = atom.box.zhi;
    }

    //tvector_2d_host x = atom.h_x;
    x = atom.x;
    v = atom.v;
    t_x = atom.t_x;
    t_v = atom.t_v;
    nlocal = atom.nlocal;

    if(nlocal > maxnlocal) {
      send_flag = tvector_1i("Comm::SendFlag", nlocal);
      maxnlocal = nlocal;
    }

    if(nsend_thread.dimension(0) < threads->omp_num_threads) {
      exc_sendlist_thread = tvector_2i("Comm::exc_sendlist_thread", threads->omp_num_threads, maxsend / 6);
      nsend_thread_startpos = tvector_1i("Comm::nsend_thread_startpos", threads->omp_num_threads);
      nsend_thread = tvector_1i("Comm::nsend_thread", threads->omp_num_threads);
      nholes_thread_startpos = tvector_1i("Comm::nholes_thread_startpos", threads->omp_num_threads);
      nholes_thread = tvector_1i("Comm::nholes_thread", threads->omp_num_threads);
    }

    h_nsend_total(0) = 0;
    Kokkos::deep_copy(nsend_total, h_nsend_total);
    f_exchange_sendItemA->c = *this;
    Kokkos::parallel_for(threads->omp_num_threads, *f_exchange_sendItemA);
    device_type::fence();
    Kokkos::deep_copy(h_nsend_total, nsend_total);
    nsend = h_nsend_total(0) * 6;

    if(nsend >= maxsend || h_nsend_total(0) >= exc_sendlist_thread.dimension(1)) {

      growsend(nsend);
      exc_sendlist_thread = tvector_2i("Comm::exc_sendlist_thread", threads->omp_num_threads, h_nsend_total(0));
      h_nsend_total(0) = 0;
      Kokkos::deep_copy(nsend_total, h_nsend_total);
      f_exchange_sendItemA->c = *this;
      Kokkos::parallel_for(threads->omp_num_threads, *f_exchange_sendItemA);
      device_type::fence();
      Kokkos::deep_copy(h_nsend_total, nsend_total);
      nsend = h_nsend_total(0) * 6;
    }

    h_nholes_total(0) = 0;
    Kokkos::deep_copy(nholes_total, h_nholes_total);
    f_exchange_sendItemB->c = *this;
    Kokkos::parallel_for(threads->omp_num_threads, *f_exchange_sendItemB);

    f_exchange_sendItemC->c = *this;
    Kokkos::parallel_for(threads->omp_num_threads, *f_exchange_sendItemC);

    nlocal -= h_nsend_total(0);

    atom.nlocal = nlocal;

    /* send/recv atoms in both directions
       only if neighboring procs are different */
    int nrecv1, nrecv2;
    MPI_Send(&nsend, 1, MPI_INT, procneigh[idim][0], 0, MPI_COMM_WORLD);
    MPI_Recv(&nrecv1, 1, MPI_INT, procneigh[idim][1], 0, MPI_COMM_WORLD, &status);
    int nrecv = nrecv1;

    if(procgrid[idim] > 2) {
      MPI_Send(&nsend, 1, MPI_INT, procneigh[idim][1], 0, MPI_COMM_WORLD);
      MPI_Recv(&nrecv2, 1, MPI_INT, procneigh[idim][0], 0, MPI_COMM_WORLD, &status);
      nrecv += nrecv2;
    }

    if(nrecv > maxrecv) growrecv(nrecv);

    if(sizeof(MMD_float) == 4) {
      MPI_Irecv(buf_recv.ptr_on_device(), nrecv1, MPI_FLOAT, procneigh[idim][1], 0,
                MPI_COMM_WORLD, &request);
      MPI_Send(buf_send.ptr_on_device(), nsend, MPI_FLOAT, procneigh[idim][0], 0, MPI_COMM_WORLD);
    } else {
      MPI_Irecv(buf_recv.ptr_on_device(), nrecv1, MPI_DOUBLE, procneigh[idim][1], 0,
                MPI_COMM_WORLD, &request);
      MPI_Send(buf_send.ptr_on_device(), nsend, MPI_DOUBLE, procneigh[idim][0], 0, MPI_COMM_WORLD);
    }

    MPI_Wait(&request, &status);

    if(procgrid[idim] > 2) {
      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(buf_recv.ptr_on_device() + nrecv1, nrecv2, MPI_FLOAT, procneigh[idim][0], 0,
                  MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.ptr_on_device(), nsend, MPI_FLOAT, procneigh[idim][1], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(buf_recv.ptr_on_device() + nrecv1, nrecv2, MPI_DOUBLE, procneigh[idim][0], 0,
                  MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.ptr_on_device(), nsend, MPI_DOUBLE, procneigh[idim][1], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);
    }

    /* check incoming atoms to see if they are in my box
       if they are, add to my list */

    h_nsend_total(0) = nrecv / 6;
    h_nrecv_total(0) = 0;
    Kokkos::deep_copy(nsend_total, h_nsend_total);
    Kokkos::deep_copy(nrecv_total, h_nrecv_total);
    nlocal = atom.nlocal;
    f_exchange_recvItemA->c = *this;
    Kokkos::parallel_for(threads->omp_num_threads, *f_exchange_recvItemA);
    Kokkos::deep_copy(h_nrecv_total, nrecv_total);

    if(nlocal + h_nrecv_total(0) >= atom.nmax) {
      atom.growarray(0, nlocal + h_nrecv_total(0));
      x = atom.x;
      v = atom.v;
      t_x = atom.t_x;
      t_v = atom.t_v;
    }

    f_exchange_recvItemB->c = *this;
    Kokkos::parallel_for(threads->omp_num_threads, *f_exchange_recvItemB);

    nlocal += h_nrecv_total(0);
    atom.nlocal = nlocal;

  }


}

void Comm::exchange_all(Atom &atom)
{
#if DEVICE==1
  int i, m, n, idim, nsend, nrecv, nrecv1, nrecv2, nlocal;
  double lo, hi, value;

  MPI_Request request;
  MPI_Status status;

  /* enforce PBC */

  atom.h_x = atom.x;
  atom.h_v = atom.v;

  atom.pbc();

  /* loop over dimensions */
  int iswap = 0;

  for(idim = 0; idim < 3; idim++) {

    /* only exchange if more than one proc in this dimension */

    if(procgrid[idim] == 1) {
      iswap += 2 * need[idim];
      continue;
    }

    /* fill buffer with atoms leaving my box
    *        when atom is deleted, fill it in with last atom */

    i = nsend = 0;

    if(idim == 0) {
      lo = atom.box.xlo;
      hi = atom.box.xhi;
    } else if(idim == 1) {
      lo = atom.box.ylo;
      hi = atom.box.yhi;
    } else {
      lo = atom.box.zlo;
      hi = atom.box.zhi;
    }

    t_x_array x = atom.x;

    nlocal = atom.nlocal;

    while(i < nlocal) {
      if(x(i, idim) < lo || x(i, idim) >= hi) {
        if(nsend > maxsend) growsend(nsend);

        nsend += atom.pack_exchange(i, &buf_send[nsend]);
        atom.copy(nlocal - 1, i);
        nlocal--;
      } else i++;
    }

    atom.nlocal = nlocal;

    /* send/recv atoms in both directions
    *        only if neighboring procs are different */
    for(int ineed = 0; ineed < 2 * need[idim]; ineed += 1) {
      if(ineed < procgrid[idim] - 1) {
        MPI_Send(&nsend, 1, MPI_INT, sendproc_exc[iswap], 0, MPI_COMM_WORLD);
        MPI_Recv(&nrecv, 1, MPI_INT, recvproc_exc[iswap], 0, MPI_COMM_WORLD, &status);

        if(nrecv > maxrecv) growrecv(nrecv);

        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(buf_recv.ptr_on_device(), nrecv, MPI_FLOAT, recvproc_exc[iswap], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(buf_send.ptr_on_device(), nsend, MPI_FLOAT, sendproc_exc[iswap], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(buf_recv.ptr_on_device(), nrecv, MPI_DOUBLE, recvproc_exc[iswap], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(buf_send.ptr_on_device(), nsend, MPI_DOUBLE, sendproc_exc[iswap], 0, MPI_COMM_WORLD);
        }

        MPI_Wait(&request, &status);

        /* check incoming atoms to see if they are in my box
        *        if they are, add to my list */

        n = atom.nlocal;
        m = 0;

        while(m < nrecv) {
          value = buf_recv[m + idim];

          if(value >= lo && value < hi)
            m += atom.unpack_exchange(n++, &buf_recv[m]);
          else m += atom.skip_exchange(&buf_recv[m]);
        }

        atom.nlocal = n;
      }

      iswap += 1;

    }
  }

#endif
}

/* borders:
   make lists of nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a communicate (so don't need to explicitly
     call communicate routine on reneighboring timestep)
   this routine is called before every reneighboring
*/

KOKKOS_INLINE_FUNCTION
void Comm::border_sendItemA(MMD_int thread) const
{
  int chunk = (nlast - nfirst + nthreads - 1) / nthreads;
  int start = chunk * thread + nfirst;
  int end = start + chunk + nfirst;

  if(end > nlast) end = nlast;

  int maxsend = exc_sendlist_thread.dimension(1);
  int nsend = 0;

  for(int i = start; i < end; i++) {
    if(x(i, idim) >= lo && x(i, idim) <= hi) {
      if(nsend < maxsend)
        exc_sendlist_thread(thread, nsend) = i;

      nsend++;
    }
  }

  nsend_thread_startpos(thread) = Kokkos::atomic_fetch_add(&nsend_total(0), nsend);
  nsend_thread(thread) = nsend;
}

KOKKOS_INLINE_FUNCTION
void Comm::border_sendItemB(MMD_int thread) const
{
  int nsend = nsend_thread(thread);
  int nsend_startpos = nsend_thread_startpos(thread);

  for(int k = 0; k < nsend; k++) {
    const int kk = k + nsend_startpos;
    const int i = exc_sendlist_thread(thread, k);

    if(pbc_flag[0] == 0) {
      buf_send(kk * 3 + 0) = t_x(i, 0);
      buf_send(kk * 3 + 1) = t_x(i, 1);
      buf_send(kk * 3 + 2) = t_x(i, 2);
    } else {
      buf_send(kk * 3 + 0) = t_x(i, 0) + pbc_flag[1] * box.xprd;
      buf_send(kk * 3 + 1) = t_x(i, 1) + pbc_flag[2] * box.yprd;
      buf_send(kk * 3 + 2) = t_x(i, 2) + pbc_flag[3] * box.zprd;
    }

    sendlist(iswap, kk) = i;
  }
}

KOKKOS_INLINE_FUNCTION
void Comm::border_recvItem(MMD_int thread) const
{
  int chunk = (nrecv_total(0) + nthreads - 1) / nthreads;
  int start = chunk * thread;
  int end = start + chunk;

  if(end > nrecv_total(0)) end = nrecv_total(0);

  for(int k = start; k < end; k++) {
    x(k + nstart, 0) = buf(k * 3 + 0);
    x(k + nstart, 1) = buf(k * 3 + 1);
    x(k + nstart, 2) = buf(k * 3 + 2);
  }
}


void Comm::borders(Atom &atom)
{
  int ineed, nsend, nrecv;
  MPI_Request request;
  MPI_Status status;

  box = atom.box;
  /* erase all ghost atoms */

  atom.nghost = 0;

  /* do swaps over all 3 dimensions */

  iswap = 0;

  if(nsend_thread.dimension(0) < threads->omp_num_threads) {
    exc_sendlist_thread = tvector_2i("Comm::exc_sendlist_thread", threads->omp_num_threads, maxsend / 6);
    nsend_thread_startpos = tvector_1i("Comm::nsend_thread_startpos", threads->omp_num_threads);
    nsend_thread = tvector_1i("Comm::nsend_thread", threads->omp_num_threads);
    nholes_thread_startpos = tvector_1i("Comm::nholes_thread_startpos", threads->omp_num_threads);
    nholes_thread = tvector_1i("Comm::nholes_thread", threads->omp_num_threads);
  }

  for(idim = 0; idim < 3; idim++) {
    nlast = 0;

    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {

      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in list for use in future timesteps
      x = atom.x;
      t_x = atom.t_x;
      lo = slablo[iswap];
      hi = slabhi[iswap];
      pbc_flag[0] = pbc_any[iswap];
      pbc_flag[1] = pbc_flagx[iswap];
      pbc_flag[2] = pbc_flagy[iswap];
      pbc_flag[3] = pbc_flagz[iswap];

      if(ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom.nlocal + atom.nghost;
      }

      h_nsend_total(0) = 0;
      Kokkos::deep_copy(nsend_total, h_nsend_total);
      f_border_sendItemA->c = *this;
      Kokkos::parallel_for(threads->omp_num_threads, *f_border_sendItemA);
      device_type::fence();
      Kokkos::deep_copy(h_nsend_total, nsend_total);
      nsend = h_nsend_total(0);

      if(nsend * 3 >= maxsend || h_nsend_total(0) >= exc_sendlist_thread.dimension(1)) {

        growsend(nsend * 3);
        exc_sendlist_thread = tvector_2i("Comm::exc_sendlist_thread", threads->omp_num_threads, h_nsend_total(0));
        h_nsend_total(0) = 0;
        Kokkos::deep_copy(nsend_total, h_nsend_total);
        f_border_sendItemA->c = *this;
        Kokkos::parallel_for(threads->omp_num_threads, *f_border_sendItemA);
        device_type::fence();
        Kokkos::deep_copy(h_nsend_total, nsend_total);
        nsend = h_nsend_total(0);
      }

      if(nsend > maxsendlist) growlist(iswap, nsend);

      f_border_sendItemB->c = *this;
      Kokkos::parallel_for(threads->omp_num_threads, *f_border_sendItemB);


      /* swap atoms with other proc
      put incoming ghosts at end of my atom arrays
      if swapping with self, simply copy, no messages */

      if(sendproc[iswap] != me) {

        MPI_Send(&nsend, 1, MPI_INT, sendproc[iswap], 0, MPI_COMM_WORLD);
        MPI_Recv(&nrecv, 1, MPI_INT, recvproc[iswap], 0, MPI_COMM_WORLD, &status);

        if(nrecv * atom.border_size > maxrecv) growrecv(nrecv * atom.border_size);

        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(buf_recv.ptr_on_device(), nrecv * atom.border_size, MPI_FLOAT,
                    recvproc[iswap], 0, MPI_COMM_WORLD, &request);
          MPI_Send(buf_send.ptr_on_device(), nsend * atom.border_size, MPI_FLOAT,
                   sendproc[iswap], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(buf_recv.ptr_on_device(), nrecv * atom.border_size, MPI_DOUBLE,
                    recvproc[iswap], 0, MPI_COMM_WORLD, &request);
          MPI_Send(buf_send.ptr_on_device(), nsend * atom.border_size, MPI_DOUBLE,
                   sendproc[iswap], 0, MPI_COMM_WORLD);
        }

        MPI_Wait(&request, &status);
        buf = buf_recv;
      } else {
        nrecv = nsend;
        buf = buf_send;
      }

      /* unpack buffer */

      nstart = atom.nlocal + atom.nghost;
      h_nrecv_total(0) = nrecv;
      Kokkos::deep_copy(nrecv_total, h_nrecv_total);

      if(nstart + nrecv >= atom.nmax) {
        atom.growarray(0, nstart + nrecv);
        x = atom.x;
      }

      f_border_recvItem->c = *this;
      Kokkos::parallel_for(threads->omp_num_threads, *f_border_recvItem);
      device_type::fence();

      /* set all pointers & counters */

      sendnum[iswap] = nsend;
      recvnum[iswap] = nrecv;
      comm_send_size[iswap] = nsend * atom.comm_size;
      comm_recv_size[iswap] = nrecv * atom.comm_size;
      reverse_send_size[iswap] = nrecv * atom.reverse_size;
      reverse_recv_size[iswap] = nsend * atom.reverse_size;
      firstrecv[iswap] = atom.nlocal + atom.nghost;
      atom.nghost += nrecv;
      iswap++;
    }
  }

  /* insure buffers are large enough for reverse comm */

  int max1, max2;
  max1 = max2 = 0;

  for(iswap = 0; iswap < nswap; iswap++) {
    max1 = MAX(max1, reverse_send_size[iswap]);
    max2 = MAX(max2, reverse_recv_size[iswap]);
  }

  if(max1 > maxsend) growsend(max1);

  if(max2 > maxrecv) growrecv(max2);

}

/* realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA */

void Comm::growsend(int n)
{

  int maxsendnew = static_cast<int>(BUFFACTOR * n);

  if(maxsendnew <= maxsend) return;

  maxsend = maxsendnew;
  tvector_1d tmpbuf_send("buf_send", maxsend + BUFEXTRA);
  tvector_1d_host h_tmpbuf_send = Kokkos::create_mirror_view(tmpbuf_send);

  if(h_buf_send.ptr_on_device() != NULL)
    for(int i = 0; i < h_buf_send.dimension(0); i++)
      h_tmpbuf_send(i) = h_buf_send(i);

  buf_send = tmpbuf_send;
  h_buf_send = h_tmpbuf_send;

}

/* free/malloc the size of the recv buffer as needed with BUFFACTOR */

void Comm::growrecv(int n)
{

  maxrecv = static_cast<int>(BUFFACTOR * n);
  tvector_1d tmpbuf_recv("buf_recv", maxrecv + BUFEXTRA);
  tvector_1d_host h_tmpbuf_recv = Kokkos::create_mirror_view(tmpbuf_recv);

  if(h_buf_recv.ptr_on_device() != NULL)
    for(int i = 0; i < h_buf_recv.dimension(0); i++)
      h_tmpbuf_recv(i) = h_buf_recv(i);

  buf_recv = tmpbuf_recv;
  h_buf_recv = h_tmpbuf_recv;
}

/* realloc the size of the iswap sendlist as needed with BUFFACTOR */

void Comm::growlist(int iswap, int n)
{
  maxsendlist = static_cast<int>(BUFFACTOR * n);
  Kokkos::deep_copy(h_sendlist, sendlist);
  sendlist = tvector_2i("sendlist", maxswap, maxsendlist);
  tvector_2i_host h_tmpsendlist = Kokkos::create_mirror_view(sendlist);

  if(h_sendlist.ptr_on_device() != NULL)
    deep_copy_grow(h_tmpsendlist, h_sendlist);

  h_sendlist = h_tmpsendlist;
  Kokkos::deep_copy(sendlist, h_sendlist);
}
