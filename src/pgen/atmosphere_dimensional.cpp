//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// Zhaohuan's FU Ori test
// In units where scale height = 1.0
// density at midplane = 1.0
// inital temperature is T = 1.0

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"

static Real heatrate;
static Real opacity;
static bool gravity;
static Real T0;
static Real H0;
static Real rho0;
static Real mol_weight;
static Real t0;
static Real gamma_gas;

void Source(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar);
void DiskInnerOutflowX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterOutflowX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadOuterX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  return;
}

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  gravity = pin->GetOrAddBoolean("problem", "gravity", false);
  T0 = pin->GetReal("radiation", "T_unit");
  H0 = pin->GetReal("radiation", "length_unit");
  rho0 = pin->GetReal("radiation", "density_unit");
  mol_weight = pin->GetReal("radiation", "molecular_weight");
  Real r_ideal = 8.314462618e7/mol_weight;
  t0 = H0/std::sqrt(r_ideal*T0); // timeunit
  heatrate = pin->GetOrAddReal("problem", "heatrate", 0.0);
  opacity = pin->GetOrAddReal("problem", "opacity", 0.0);
  // make heatrate and opacity dimensionless
  heatrate *= t0*t0*t0/H0/H0;
  opacity *= rho0*H0;

  EnrollUserExplicitSourceFunction(Source);
  if(mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerOutflowX1);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, DiskRadInnerX1);
  }
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterOutflowX1);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, DiskRadOuterX1);
  }
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  gamma_gas = peos->GetGamma();
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = exp(-x1*x1*0.5);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS){
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)/(gamma_gas-1.0);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }
  // Now initialize opacity and specific intensity
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    int nfreq = pnrrad->nfreq;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          for (int ifr=0; ifr < nfreq; ++ifr) {
            pnrrad->sigma_s(k,j,i,ifr) = 0.0;
            pnrrad->sigma_a(k,j,i,ifr) = opacity*phydro->u(IDN,k,j,i);
            pnrrad->sigma_pe(k,j,i,ifr) = opacity*phydro->u(IDN,k,j,i);
            pnrrad->sigma_p(k,j,i,ifr) = opacity*phydro->u(IDN,k,j,i);
          }
          for (int n=0; n<pnrrad->n_fre_ang; ++n) {
              // T = 1 everywhere in code units so equilibrium
              // is given by ir = T^4 = 1
              pnrrad->ir(k,j,i,n) = 1.0;
          }
        }
      }
    }
  }
  return;
}

void Source(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar)
{
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        if (gravity) {
          cons(IM1,k,j,i) -= x*prim(IDN,k,j,i)*dt;
        }
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) += heatrate*prim(IDN,k,j,i)*dt;
          if (gravity) {
            cons(IEN,k,j,i) -= x*prim(IDN,k,j,i)*prim(IVX,k,j,i)*dt;
          }
        }
      }
    }
  }
  return;
}

void DiskInnerOutflowX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,is-i) = prim(n,k,j,is);
          if ((n==IVX) && (prim(n,k,j,is-i) > 0.0))
            prim(n,k,j,is-i)=0.0;
        }
      }
    }
  }
  return;
}

void DiskOuterOutflowX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,ie+i) = prim(n,k,j,ie);
          if ((n==IVX) && (prim(n,k,j,ie+i) < 0.0))
            prim(n,k,j,ie+i)=0.0;
        }
      }
    }
  }
  return;
}

void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  int nang=prad->nang;
  int nfreq=prad->nfreq;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            Real& miux=prad->mu(0,k,j,is-i,n);
            if (miux < 0.0)
              ir(k,j,is-i,ang) = ir(k,j,is,n);
            else
              ir(k,j,is-i,ang) = 0.0;
          }
        }
      }
    }
  }
  return;
}

void DiskRadOuterX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  int nang=prad->nang;
  int nfreq=prad->nfreq;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            Real& miux=prad->mu(0,k,j,ie+i,n);
            if (miux > 0.0)
              ir(k,j,ie+i,ang) = ir(k,j,ie,n);
            else
              ir(k,j,ie+i,ang) = 0.0;
          }
        }
      }
    }
  }
  return;
}
