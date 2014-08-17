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
#include "openmp.h"
#include "math.h"

Integrate::Integrate() {sort_every=20;}
Integrate::~Integrate() {}

void Integrate::setup()
{
  dtforce = 0.5 * dt;
}

void Integrate::initialIntegrate()
{
  const int nlocal_ = nlocal;
  const MMD_float dt_ = dt;
  const MMD_float dtforce_ = dtforce;
  const MMD_float* const restrict f_ = f;
  MMD_float* const restrict v_ = v;
  MMD_float* const restrict x_ = x; 
  
  #pragma acc kernels deviceptr(x_,v_,f_)
  for(MMD_int i = 0; i < nlocal_; i++) {
    v_[i * PAD + 0] += dtforce_ * f_[i * PAD + 0];
    v_[i * PAD + 1] += dtforce_ * f_[i * PAD + 1];
    v_[i * PAD + 2] += dtforce_ * f_[i * PAD + 2];
    x_[i * PAD + 0] += dt_ * v_[i * PAD + 0];
    x_[i * PAD + 1] += dt_ * v_[i * PAD + 1];
    x_[i * PAD + 2] += dt_ * v_[i * PAD + 2];
  }
}

void Integrate::finalIntegrate()
{
  const int nlocal_ = nlocal;
  const MMD_float dtforce_ = dtforce;
  const MMD_float* const restrict f_ = f;
  MMD_float* const restrict v_ = v;
  #pragma acc kernels deviceptr(v_,f_)
  for(MMD_int i = 0; i < nlocal_; i++) {
    v_[i * PAD + 0] += dtforce_ * f_[i * PAD + 0];
    v_[i * PAD + 1] += dtforce_ * f_[i * PAD + 1];
    v_[i * PAD + 2] += dtforce_ * f_[i * PAD + 2];
  }
}

void Integrate::run(Atom &atom, Force* force, Neighbor &neighbor,
                    Comm &comm, Thermo &thermo, Timer &timer)
{
  int i, n;

  comm.timer = &timer;
  timer.array[TIME_TEST] = 0.0;

  int check_safeexchange = comm.check_safeexchange;

  mass = atom.mass;
  dtforce = dtforce / mass;
  //Use OpenMP threads only within the following loop containing the main loop.
  //Do not use OpenMP for setup and postprocessing.
  {
    int next_sort = sort_every>0?sort_every:ntimes+1;

    atom.sync_device(atom.d_x,&atom.x[0][0],atom.nmax*3*sizeof(MMD_float));
    atom.sync_device(atom.d_v,&atom.v[0][0],atom.nmax*3*sizeof(MMD_float));

    for(n = 0; n < ntimes; n++) {

      

      //x = &atom.x[0][0];
      //v = &atom.v[0][0];
      //f = &atom.f[0][0];
      x = atom.d_x;
      v = atom.d_v;
      f = atom.d_f;
      xold = &atom.xold[0][0];
      nlocal = atom.nlocal;
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("A %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);
      initialIntegrate();
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("B %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);

      
      timer.stamp();

      if((n + 1) % neighbor.every) {
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));

        comm.communicate(atom);
        //atom.sync_device(atom.d_x,&atom.x[0][0],atom.nmax*3*sizeof(MMD_float));
        
        timer.stamp(TIME_COMM);

      } else {
        atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
        atom.sync_host(&atom.v[0][0],atom.d_v,atom.nmax*3*sizeof(MMD_float));
        {

          timer.stamp_extra_start();
          comm.exchange(atom);
          if(n+1>=next_sort) {
            //atom.sort(neighbor);
            next_sort +=  sort_every;
          }
          comm.borders(atom);
          
          {
            timer.stamp_extra_stop(TIME_TEST);
            timer.stamp(TIME_COMM);
          }

        }

        neighbor.build(atom);

        //atom.sync_device(atom.d_x,&atom.x[0][0],atom.nmax*3*sizeof(MMD_float));
        atom.sync_device(atom.d_v,&atom.v[0][0],atom.nmax*3*sizeof(MMD_float));
        timer.stamp(TIME_NEIGH);
      }
      timer.stamp(TIME_TEST);
      force->evflag = (n + 1) % thermo.nstat == 0;
      force->compute(atom, neighbor, comm, comm.me);
//        atom.sync_host(&atom.f[0][0],atom.d_f,atom.nmax*3*sizeof(MMD_float));

      
      timer.stamp(TIME_FORCE);

      if(neighbor.halfneigh && neighbor.ghost_newton) {
        atom.sync_host(&atom.f[0][0],atom.d_f,atom.nmax*3*sizeof(MMD_float));
        comm.reverse_communicate(atom);

        
        timer.stamp(TIME_COMM);
      }

      //v = &atom.v[0][0];
      //f = &atom.f[0][0];
      v = atom.d_v;
      f = atom.d_f;
      nlocal = atom.nlocal;

      
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("G %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);
      finalIntegrate();
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("H %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);
      if(thermo.nstat) {
        atom.sync_host(&atom.v[0][0],atom.d_v,atom.nmax*3*sizeof(MMD_float));
        thermo.compute(n + 1, atom, neighbor, force, timer, comm);
      }
    }
  } //end OpenMP parallel
        atom.sync_host(&atom.v[0][0],atom.d_v,atom.nmax*3*sizeof(MMD_float));
        atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
}
