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
#include "string.h"
#include "stdlib.h"
#include "mpi.h"
#include "atom.h"
#include "neighbor.h"

#define DELTA 20000


void cuda_check_error(char* comment)
{
#if DEVICE==1
  printf("ERROR %s in %s:%i\n", comment, __FILE__, __LINE__);
#endif
#if DEVICE==2
  printf("ERROR-CUDA %s %s in %s:%i\n", comment, cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
#endif

}

Atom::Atom()
{
  natoms = 0;
  nlocal = 0;
  nghost = 0;
  nmax = 0;

  comm_size = 3;
  reverse_size = 3;
  border_size = 3;

#ifdef MINIMD_COMD_DEFAULTS
  mass = 63.55;
#else
  mass = 1.0;
#endif
}

void Atom::setup()
{
  f_pbc = new AtomPBCFunctor;
  f_pack_comm = new AtomPackCommFunctor;
  f_pack_comm_pbc = new AtomPackCommPBCFunctor;
  f_unpack_comm = new AtomUnpackCommFunctor;
  f_pack_reverse = new AtomPackReverseFunctor;
  f_unpack_reverse = new AtomUnpackReverseFunctor;
  f_sort = new AtomSortFunctor;
}

void Atom::finalise()
{
  delete f_pbc;
  delete f_pack_comm;
  delete f_pack_comm_pbc;
  delete f_unpack_comm;
  delete f_pack_reverse;
  delete f_unpack_reverse;
  delete f_sort;
}

Atom::~Atom()
{
}

void Atom::growarray(MMD_int host_current)
{
  nmax += DELTA;

  if(host_current) {

    x = t_x_array("X", nmax);
    v = t_v_array("V", nmax);
    f = t_f_array("F", nmax);

    t_x_array_host xnew = Kokkos::create_mirror_view(x);
    t_v_array_host vnew = Kokkos::create_mirror_view(v);
    t_f_array_host fnew = Kokkos::create_mirror_view(f);

    deep_copy_grow(xnew, h_x);
    deep_copy_grow(vnew, h_v);
    deep_copy_grow(fnew, h_f);

    h_x = xnew;
    h_v = vnew;
    h_f = fnew;
  } else {

    t_x_array xnew("X", nmax);
    t_v_array vnew("V", nmax);
    t_f_array fnew("F", nmax);

    deep_copy_grow(xnew, x);
    deep_copy_grow(vnew, v);
    deep_copy_grow(fnew, f);

    x = xnew;
    v = vnew;
    f = fnew;

    h_x = Kokkos::create_mirror_view(x);
    h_v = Kokkos::create_mirror_view(v);
    h_f = Kokkos::create_mirror_view(f);
  }
  t_x = t_x_array_tex(x);
  t_v = t_v_array_tex(v);
}

void Atom::growarray(MMD_int host_current, MMD_int newsize)
{
  if(newsize <= nmax) return;

  nmax = newsize;
  growarray(host_current);
}

void Atom::addatom(MMD_float x_in, MMD_float y_in, MMD_float z_in,
                   MMD_float vx_in, MMD_float vy_in, MMD_float vz_in)
{
  if(nlocal == nmax) {
    growarray(1);
  }

  h_x(nlocal, 0) = x_in;
  h_x(nlocal, 1) = y_in;
  h_x(nlocal, 2) = z_in;
  h_v(nlocal, 0) = vx_in;
  h_v(nlocal, 1) = vy_in;
  h_v(nlocal, 2) = vz_in;

  nlocal++;
}

/* enforce PBC
   order of 2 tests is important to insure lo-bound <= coord < hi-bound
   even with round-off errors where (coord +/- epsilon) +/- period = bound */

KOKKOS_INLINE_FUNCTION
void Atom::pbcItem(const MMD_int &i) const
{
  if(x(i, 0) < 0.0) x(i, 0) += box.xprd;

  if(x(i, 0) >= box.xprd) x(i, 0) -= box.xprd;

  if(x(i, 1) < 0.0) x(i, 1) += box.yprd;

  if(x(i, 1) >= box.yprd) x(i, 1) -= box.yprd;

  if(x(i, 2) < 0.0) x(i, 2) += box.zprd;

  if(x(i, 2) >= box.zprd) x(i, 2) -= box.zprd;

}

void Atom::pbc()
{

  f_pbc->c = *this;

  Kokkos::parallel_for(nlocal, *f_pbc);
}

void Atom::copy(MMD_int i, MMD_int j)
{
  h_x(j, 0) = h_x(i, 0);
  h_x(j, 1) = h_x(i, 1);
  h_x(j, 2) = h_x(i, 2);
  h_v(j, 0) = h_v(i, 0);
  h_v(j, 1) = h_v(i, 1);
  h_v(j, 2) = h_v(i, 2);
}

void Atom::pack_comm(MMD_int n, MMD_int iswap, tvector_1d buf, int* act_pbc_flags, tvector_2i asendlist)
{
  actswap = iswap;
  sendlist = asendlist;
  pbc_flags[0] = act_pbc_flags[0];
  pbc_flags[1] = act_pbc_flags[1];
  pbc_flags[2] = act_pbc_flags[2];
  pbc_flags[3] = act_pbc_flags[3];

  actbuf = buf;

  if(pbc_flags[0] == 0) {
    f_pack_comm->c = *this;
    //Kokkos::parallel_for(0,*f_pack_comm);
    Kokkos::parallel_for(n, *f_pack_comm);
  } else {
    f_pack_comm_pbc->c = *this;
    //Kokkos::parallel_for(0,*f_pack_comm_pbc);
    Kokkos::parallel_for(n, *f_pack_comm_pbc);
  }
}

KOKKOS_INLINE_FUNCTION void Atom::pack_commItem(const MMD_int &i) const
{
  const MMD_int j = sendlist(actswap, i);
  actbuf[3 * i] = t_x(j, 0);
  actbuf[3 * i + 1] = t_x(j, 1);
  actbuf[3 * i + 2] = t_x(j, 2);
}

KOKKOS_INLINE_FUNCTION void Atom::pack_commItemPBC(const MMD_int &i) const
{
  const MMD_int j = sendlist(actswap, i);;
  actbuf[3 * i] = t_x(j, 0) + pbc_flags[1] * box.xprd;
  actbuf[3 * i + 1] = t_x(j, 1) + pbc_flags[2] * box.yprd;
  actbuf[3 * i + 2] = t_x(j, 2) + pbc_flags[3] * box.zprd;
}

void Atom::unpack_comm(MMD_int n, MMD_int actfirst, tvector_1d buf)
{
  first = actfirst;
  actbuf = buf;

  f_unpack_comm->c = *this;
  //Kokkos::parallel_for(0,*f_unpack_comm);
  Kokkos::parallel_for(n, *f_unpack_comm);
}

KOKKOS_INLINE_FUNCTION void Atom::unpack_commItem(const MMD_int &i) const
{
  x(first + i, 0) = actbuf[3 * i];
  x(first + i, 1) = actbuf[3 * i + 1];
  x(first + i, 2) = actbuf[3 * i + 2];
}

void Atom::pack_reverse(MMD_int n, MMD_int actfirst, tvector_1d buf)
{
  first = actfirst;
  actbuf = buf;

  f_pack_reverse->c = *this;

  //Kokkos::parallel_for(0,*f_pack_reverse);
  Kokkos::parallel_for(n, *f_pack_reverse);
}

KOKKOS_INLINE_FUNCTION void Atom::pack_reverseItem(const MMD_int &i) const
{
  actbuf[3 * i] = f(first + i, 0);
  actbuf[3 * i + 1] = f(first + i, 1);
  actbuf[3 * i + 2] = f(first + i, 2);
}

void Atom::unpack_reverse(MMD_int n, MMD_int iswap, tvector_1d buf, tvector_2i asendlist)
{
  actswap = iswap;
  sendlist = asendlist;
  actbuf = buf;

  f_unpack_reverse->c = *this;

  //Kokkos::parallel_for(0,*f_unpack_reverse);
  Kokkos::parallel_for(n, *f_unpack_reverse);
}

KOKKOS_INLINE_FUNCTION void Atom::unpack_reverseItem(const MMD_int &i) const
{
  MMD_int j = sendlist(actswap, i);
  f(j, 0) += actbuf[3 * i];
  f(j, 1) += actbuf[3 * i + 1];
  f(j, 2) += actbuf[3 * i + 2];
}

int Atom::pack_border(MMD_int i, MMD_float* buf, int* pbc_flags)
{
  int m = 0;

  if(pbc_flags[0] == 0) {
    buf[m++] = h_x(i, 0);
    buf[m++] = h_x(i, 1);
    buf[m++] = h_x(i, 2);
  } else {
    buf[m++] = h_x(i, 0) + pbc_flags[1] * box.xprd;
    buf[m++] = h_x(i, 1) + pbc_flags[2] * box.yprd;
    buf[m++] = h_x(i, 2) + pbc_flags[3] * box.zprd;
  }

  return m;
}

int Atom::unpack_border(MMD_int i, MMD_float* buf)
{
  if(i == nmax) growarray(1);

  int m = 0;
  h_x(i, 0) = buf[m++];
  h_x(i, 1) = buf[m++];
  h_x(i, 2) = buf[m++];
  return m;
}

int Atom::pack_exchange(MMD_int i, MMD_float* buf)
{
  MMD_int m = 0;
  buf[m++] = h_x(i, 0);
  buf[m++] = h_x(i, 1);
  buf[m++] = h_x(i, 2);
  buf[m++] = h_v(i, 0);
  buf[m++] = h_v(i, 1);
  buf[m++] = h_v(i, 2);
  return m;
}

int Atom::unpack_exchange(MMD_int i, MMD_float* buf)
{
  if(i == nmax) growarray(1);

  MMD_int m = 0;
  h_x(i, 0) = buf[m++];
  h_x(i, 1) = buf[m++];
  h_x(i, 2) = buf[m++];
  h_v(i, 0) = buf[m++];
  h_v(i, 1) = buf[m++];
  h_v(i, 2) = buf[m++];
  return m;
}

int Atom::skip_exchange(MMD_float* buf)
{
  return 6;
}

void Atom::upload(int datamask)
{
  if(nlocal == 0) return;

  if(datamask & DATA_X) Kokkos::deep_copy(x, h_x);

  if(datamask & DATA_V) Kokkos::deep_copy(v, h_v);

  if(datamask & DATA_F) Kokkos::deep_copy(f, h_f);
}

void Atom::download(int datamask)
{
  if(nlocal == 0) return;

  if(datamask & DATA_X) Kokkos::deep_copy(h_x, x);

  if(datamask & DATA_V) Kokkos::deep_copy(h_v, v);

  if(datamask & DATA_F) Kokkos::deep_copy(h_f, f);
}
/* realloc a 2-d MMD_float array */

void Atom::sort(Neighbor &neighbor)
{
  neighbor.binatoms(*this,nlocal);
  binpos = neighbor.bincount;
  bins = neighbor.bins;
  tvector_1i_host h_binpos = Kokkos::create_mirror_view(binpos);
  Kokkos::deep_copy(h_binpos,binpos);


  const int mbins = binpos.dimension_0();
  for(int i=1; i<mbins; i++) {
	  h_binpos(i) += h_binpos(i-1);
  }

  Kokkos::deep_copy(binpos,h_binpos);

  if(x_copy.dimension_0()<x.dimension_0()) {
    x_copy = t_x_array("Atom::x",x.dimension_0());
    v_copy = t_v_array("Atom::v",v.dimension_0());
  }

  f_sort->c = *this;

#if DEVICE==1
  int n = mbins;
#endif
#if DEVICE==2
  int n= mbins*32;
#endif
  Kokkos::parallel_for(n, *f_sort);
  device_type::fence();
  t_x_array x_tmp = x;
  t_v_array v_tmp = v;

  x = x_copy;
  v = v_copy;
  x_copy = x_tmp;
  v_copy = v_tmp;
  t_x = t_x_array_tex(x);
  t_v = t_v_array_tex(v);
}

KOKKOS_INLINE_FUNCTION void Atom::sortItem(const MMD_int &threadId) const
{
#if DEVICE==1
  const int mybin = threadId;
  const int myLocalId = 0;
  const int nWorkGroup = 1;
#endif
#if DEVICE==2
  const int mybin = threadId/32;
  const int myLocalId = threadId%32;
  const int nWorkGroup = 32;
#endif

  const int start = mybin>0?binpos(mybin-1):0;
  const int count = binpos(mybin) - start;
  for(int k=myLocalId; k<count; k+=nWorkGroup) {
	const int new_i = start+k;
	const int old_i = bins(mybin,k);
	x_copy(new_i,0) = t_x(old_i,0);
	x_copy(new_i,1) = t_x(old_i,1);
	x_copy(new_i,2) = t_x(old_i,2);
	v_copy(new_i,0) = t_v(old_i,0);
	v_copy(new_i,1) = t_v(old_i,1);
	v_copy(new_i,2) = t_v(old_i,2);
  }
}
