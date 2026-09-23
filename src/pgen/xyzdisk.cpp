//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//  spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// Configure command:
/*
python configure.py --prob=xyzdisk --coord=<cartesian,spherical_polar> -mpi -hdf5 --hdf5_path=<hdf5-path> -cxx='icpc' --cflag="-lmpi -lmpi++"

TODO:
|/| Redo POverRBinary function
| | Check implementation of source terms - combine all energy terms into one step?
|/| Add debug comment flag?
*/


// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <cstdlib>    // srand
#include <cfloat>     // FLT_MIN

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"  // Alpha viscosity

// Gravitational mass header
#include "../Binary.hpp"

static void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
static Real DenProfileCyl(const Real rad, const Real phi, const Real z);
static Real PoverR(const Real rad, const Real phi, const Real z);
static Real PoverRBinary(const Real rad, const Real phi, const Real z);
static void VelProfileCyl(const Real rad, const Real phi, const Real z,
  Real &v1, Real &v2, Real &v3);


// Alpha viscosity function
void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

// User-defined functions
void UserSourceTerms(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, 
      const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, 
      AthenaArray<Real> &cons_scalar);

// "Static" AMR refinement condition
int RefinementCondition(MeshBlock *pmb);

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

// problem parameters which are useful to make global to this file
static Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas;
static Real dfloor;
static Real rin_mu, rin_sigma;
static Real rout_mu, rout_sigma;
static Real alpha;
static Real Tc;
static Real Tdamp;
static Real rs;
Real orbit_t;
static Real orbit_dt;
// Disk geometry variables
//static Real Rmin, Ri, Ro, Rmax;
static Real Rdamp_in, Rdamp_out;
static Real disk_inc;

//static Real thmin, thi, tho, thmax;
static bool cooling, grav, damping;
static bool accreting;
static bool debugmsg;

// binary variables
static Real M1, M2, Mtot;
static Real bin_a, bin_ecc, bin_inc;

// AMR variables
static Real r_csd;
static Real box_x, box_y, box_z;
static bool Is2DSim;

// Initial particle array
Particle ParticleList[N_PARTICLES] = {
    Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };


//Particle ParticleList[2] = {
//    Particle(bin_a*(1.0-ecc), 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
//    Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
//    };


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));


  // Set inner and outer disk edges
  rin_mu = pin->GetOrAddReal("problem", "rin_mu", 0.3);
  rin_sigma = pin->GetOrAddReal("problem", "rin_sigma", 0.1);

  rout_mu = pin->GetOrAddReal("problem", "rout_mu", 3.0);
  rout_sigma = pin->GetOrAddReal("problem", "rout_sigma", 0.1);

  alpha = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  Tc = pin->GetOrAddReal("problem", "Tc", pow(10., -5));
  Tdamp = pin->GetOrAddReal("problem", "Tdamp", pow(10., -3));
  rs = pin->GetOrAddReal("problem", "rsmooth", (mesh_size.x1max-mesh_size.x1min)/128.);

  // Toggles for various source terms.
  cooling = pin->GetOrAddBoolean("problem", "cooling", true);
  grav = pin->GetOrAddBoolean("problem", "grav", true);
  damping = pin->GetOrAddBoolean("problem", "damping", false);
  accreting = pin->GetOrAddBoolean("problem", "accreting", false);

  // Toggle debug messages
  debugmsg = pin->GetOrAddBoolean("problem", "debugmsg", false);


  // Set inner/outer boundaries for damping zones.
  // Define using maximum mesh size so damping zones are consistent across coordinate systems
  // Should these variables size with rin/rout mu, sigma?
  // Rmin = pin->GetOrAddReal("problem", "Rmin", 0.0);
  // Ri =   pin->GetOrAddReal("problem", "Ri", 0.5);
  // Ro =   pin->GetOrAddReal("problem", "Ro", 4.0);
  // Rmax = pin->GetOrAddReal("problem", "Rmax", 5.0);
  Rdamp_in = pin->GetOrAddReal("problem", "Ro", rout_mu);
  Rdamp_out = pin->GetOrAddReal("problem", "Rmax", std::sqrt(std::pow(mesh_size.x1max, 2) + std::pow(mesh_size.x2max, 2) + std::pow(mesh_size.x3max, 2)));
  // Rmax = mesh_size.x1max;
  //Ri = 0.8*mesh_size.x1max;
  //Ro = mesh_size.x1max;

  // UNUSED: theta damping variables
  //thmin = pin->GetOrAddReal("problem", "thmin", 0.0); // 3 scale heights
  //thi =   pin->GetOrAddReal("problem", "thi", 1e-5);   // 2 scale heights
  //tho =   pin->GetOrAddReal("problem", "tho", PI);   // 2 scale heights
  //thmax = pin->GetOrAddReal("problem", "thmax", PI-1e-5); // 3 scale heights

  //thmin = pin->GetOrAddReal("problem", "thmin", 1.27); // 3 scale heights
  //thi =   pin->GetOrAddReal("problem", "thi", 1.37);   // 2 scale heights
  //tho =   pin->GetOrAddReal("problem", "tho", 1.77);   // 2 scale heights
  //thmax = pin->GetOrAddReal("problem", "thmax", 1.87); // 3 scale heights

  // Set up time and dt variables for orbit outputs
  orbit_t = time;
  orbit_dt = pin->GetOrAddReal("problem", "orbit_dt", 0.01);

  // Set binary parameters. Default: equal-mass, circular binary.
  M1 =  pin->GetOrAddReal("problem", "Ma", 1.0);
  M2 =  pin->GetOrAddReal("problem", "Mb", 1.0);
  Mtot = M1 + M2;
  M1 = M1/Mtot;
  M2 = M2/Mtot;
  Mtot = Mtot/Mtot;

  bin_a = pin->GetOrAddReal("problem", "bin_a", 1.0);
  bin_ecc = pin->GetOrAddReal("problem", "bin_ecc", 0.0);
  bin_inc = pin->GetOrAddReal("problem", "bin_inc", 0.0);

  // Set disk inclination
  disk_inc = pin->GetOrAddReal("problem", "disk_inc", 0.0);

  r_csd = pin->GetOrAddReal("problem", "r_csd", 0.4);
  box_x = mesh_size.x1max - mesh_size.x1min;
  box_y = mesh_size.x2max - mesh_size.x2min;
  box_z = mesh_size.x3max - mesh_size.x3min;

  Is2DSim = (mesh_size.nx3 == 1 ? true : false);

  // Enroll user functions
  if (alpha > 0.0) {
    EnrollViscosityCoefficient(AlphaViscosity);
  }
  EnrollUserExplicitSourceFunction(UserSourceTerms);

  // If AMR is active, enroll refinement condition
  if(adaptive==true)
      EnrollUserRefinementCondition(RefinementCondition);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }


  // Set initial particle masses
  ParticleList[0].M = M1;
  ParticleList[1].M = M2;

  // Set initial particle positions
  ParticleList[0].x = 0.0;
  ParticleList[1].x = bin_a*(1.0-bin_ecc);

  // Set initial velocities for binary particles
  Real dx = ParticleList[0].x-ParticleList[1].x;
  Real dy = ParticleList[0].y-ParticleList[1].y; 
  Real dz = ParticleList[0].z-ParticleList[1].z;
  Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
  
  // Calculate specific velocity vs = v/M
  Real vs = std::sqrt( 1.0/Mtot/dist*(1.0+bin_ecc) );
  
  //Mtot = ParticleList[0].M + ParticleList[1].M;
  //Real vo = ParticleList[0].M*std::sqrt(1.0/Mtot/fabs(ParticleList[0].x-ParticleList[1].x)*(1.0+ecc) );
  ParticleList[0].vy = -vs * ParticleList[1].M * cos(bin_inc);
  ParticleList[0].vz = -vs * ParticleList[1].M * sin(bin_inc);
  ParticleList[1].vy = vs * ParticleList[0].M * cos(bin_inc);
  ParticleList[1].vz = vs * ParticleList[0].M * sin(bin_inc);
  //ParticleList[0].vy = vo;
  //ParticleList[1].vy = -vo;
  move_to_com(ParticleList);
  
  
  // Output useful constants at beginning of simulation.
  if (Globals::my_rank == 0) {
    printf("[[[[==== Particle Initial State ====]]]]\n");
    for (int i=0; i<2; ++i)
    {
      printf("Particle %d\n", i);
      ParticleList[i].print_elements();
    }
    printf("\n");
  
  
    printf("======== Binary System Parameters ======\n");
    printf("Masses: %f %f\n", M1, M2);
    printf("Bin. Separation  : %f\n", bin_a);
    printf("Bin. Eccentricity: %f\n", bin_ecc);

    printf("======== Disk Parameters ======\n");
    printf("Den. Slope   :   %f\n", dslope);
    printf("Temp.Slope   :   %f\n", pslope);
    // printf("Rho_Floor0   :   %f\n", rho_floor0);
    printf("Alpha        :   %f\n", alpha);
    printf("Cooling Time :   %f\n", Tc);
    printf("Disk Incl.   :   %f\n", disk_inc);

    printf("======== Source Terms ======\n");
    printf("Cooling      :   %d\n", cooling);
    printf("Binary Grav  :   %d\n", grav);
    printf("Damping      :   %d\n", damping);
    printf("Accretion    :   %d\n", accreting);

    printf("======= AMR Parameters =======\n");
    printf("2D Sim?      :   %d\n", Is2DSim);

    printf("============ Debug ===========\n");
    printf("Debug Msg?   :   %d\n", debugmsg);

  }

  return;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshBlockData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in meshblock class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in MeshBlock constructor.
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // USER DATA DEBUG
  // enroll user data arrays
  int nx1 = ie - is + 1 + 2*NGHOST;
  int nx2 = je - js + 1 + 2*NGHOST;
  int nx3 = ke - ks + 1 + 2*NGHOST;
  AllocateRealUserMeshBlockDataField(3);
  ruser_meshblock_data[0].NewAthenaArray(nx3, nx2, nx1);
  ruser_meshblock_data[1].NewAthenaArray(nx3, nx2, nx1);
  ruser_meshblock_data[2].NewAthenaArray(nx3, nx2, nx1);

  for (int k=0; k<nx3; ++k) {
    for (int j=0; j<nx2; ++j) {
      for (int i=0; i<nx1; ++i) {
        for(int n=0; n<3; ++n) {
          ruser_meshblock_data[n](k,j,i) = 10*(n+1) + i+j+k;
        }
      }
    }
  }
  // ruser_meshblock_data[0](ks,js,is) = 123.0;

  // DEBUG: User variable output
  if (debugmsg)
  {
    printf("USER MESHBLOCK DATA: %f\n", ruser_meshblock_data[0](ks,js,is));
    printf("USER MESHBLOCK DATA (0): %f\n", ruser_meshblock_data[0](0,0,0));
  }
  // enroll user output variables
  AllocateUserOutputVariables(3);
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad, phi, z;
  Real v1, v2, v3;
  
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
  for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
      GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
      // compute initial conditions in cylindrical coordinates
      phydro->u(IDN,k,j,i) = DenProfileCyl(rad,phi,z);
      VelProfileCyl(rad,phi,z,v1,v2,v3);

      phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
      phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
      phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
      if (NON_BAROTROPIC_EOS) {
        //Real p_over_r = PoverR(rad,phi,z);
        Real p_over_r = PoverRBinary(rad,phi,z);
        phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
        phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                   + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
      }
    }
  }}

  return;
}

//----------------------------------------------------------------------------------------
//!\f transform to cylindrical coordinate

static void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (COORDINATE_SYSTEM == "cartesian") {
    Real xp = pco->x1v(i);
    Real yp = pco->x2v(j)*cos(disk_inc) + pco->x3v(k)*sin(disk_inc);
    Real zp = -pco->x2v(j)*sin(disk_inc) + pco->x3v(k)*cos(disk_inc);

    rad=sqrt(xp*xp + yp*yp);
    phi=atan2(yp,xp);
    z=zp;
    
    //rad=sqrt(pco->x1v(i)*pco->x1v(i) + pco->x2v(j)*pco->x2v(j));
    //phi=atan2(pco->x2v(j),pco->x1v(i));
    //z=pco->x3v(k);
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    rad=fabs(pco->x1v(i)*sin(pco->x2v(j)));
    phi=pco->x3v(i);
    z=pco->x1v(i)*cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates

static Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real r = std::max(rad, rs);   // Use max(rad, rs) to prevent power-law from increasing to infinity at rad=0
  Real p_over_r = p0_over_r0;
  //if (NON_BAROTROPIC_EOS) p_over_r = PoverR(r, phi, z);
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRBinary(r, phi, z);
  Real denmid = rho0*pow(r/r0,dslope);
  // Add a Gaussian dropoff to inner and outer edges
  // denmid *= exp(-(rad-rmu)*(rad-rmu)/rsigma/rsigma);
  if (r < rin_mu) {
     denmid *= exp(-(r-rin_mu)*(r-rin_mu)/rin_sigma/rin_sigma);
  }

  if (r > rout_mu) {
    denmid *= exp(-(r-rout_mu)*(r-rout_mu)/rout_sigma/rout_sigma);
  }


  Real dentem = denmid*exp(1.0/p_over_r*(1./std::sqrt(SQR(r)+SQR(z))-1./r));
  //Real dentem = denmid*exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  // DEBUG:  Print den and coords
  //if (rad > 1.0 && rad < 1.1 && phi < 0.1 && phi > -0.1)
  //  printf("%f, %f, %f, %f\n", rad, phi, z, den);
  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! \f  computes pressure/density in cylindrical coordinates

static Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes binary-centered pressure/density in cylindrical coordinates

static Real PoverRBinary(const Real rad, const Real phi, const Real z) {
  Real poverr = 0.0;
  //Real phi_grav = 0.0;
  Real r = std::max(rs, std::sqrt(rad*rad+z*z));
  Real dmin = SIZE_MAX;
  
  // Convert to cartesian coordinates
  Real x = rad*cos(phi);
  Real y = rad*sin(phi);

  // Iterate over particles
  for (int i=0; i<N_PARTICLES; i++)
  {
    Particle pn = ParticleList[i];
    Real dx = x-pn.x;
    Real dy = y-pn.y;
    Real dz = z-pn.z;
    Real d = std::sqrt(dx*dx + dy*dy + dz*dz);
    //Real d2 = std::sqrt(dx*dx + dy*dy + dz*dz);

    dmin = std::min(dmin, d);

    // Calculate gravitational potential
    //phi_grav += -1*pn.M/std::sqrt(dx*dx + dy*dy + dz*dz + 4*rs*rs);

    //Real r = std::sqrt(dx*dx + dy*dy + dz*dz + rs*rs);
    //poverr += p0_over_r0*pow(r/r0, pslope);

  }
  
  dmin = std::max(dmin,rs);
  //poverr = p0_over_r0*pow(rad/r0, pslope);
  
  poverr = p0_over_r0*pow(dmin/r0, pslope);
  //poverr = p0_over_r0*pow(r/r0, pslope);
  //poverr = p0_over_r0*fabs(phi_grav);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates

static void VelProfileCyl(const Real rad, const Real phi, const Real z,
                          Real &v1, Real &v2, Real &v3) {
  Real r = std::max(rad, rs);
  //Real p_over_r = PoverR(r, phi, z);
  Real p_over_r = PoverRBinary(r, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(1.0/r) + (1.0+pslope)
             - pslope*r/std::sqrt(r*r+z*z);
  //Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
  //           - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(1.0/r)*std::sqrt(vel);
  //vel = std::sqrt(gm0/rad)*std::sqrt(vel);
  if (COORDINATE_SYSTEM == "cartesian") {
    v1=-1*vel*sin(phi);
    v2=vel*cos(phi)*cos(disk_inc);
    v3=vel*cos(phi)*sin(disk_inc);
    //v1=-1*vel*sin(phi);
    //v2=vel*cos(phi);
    //v3=0.0;
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    v1=0.0;
    v2=vel;
    v3=0.0;
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    v1=0.0;
    v2=0.0;
    v3=vel;
  }
  return;
}


//----------------------------------------------------------------------------------------
//!\f: Function for alpha-viscosity. It is only enrolled and used if alpha is positive.
//
void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
  Coordinates *pco = pmb->pcoord;
  Real rad, phi, z;
  for (int k=ks; k<=ke; ++k) {
    //Real z = pco->x3v(k);
    for (int j=js; j<=je; ++j) {
      //Real y = pco->x2v(j);
      for (int i=is; i<ie; ++i) {
        //Real x = pco->x1v(i);
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        Real r = std::max(rad, rs);
        // Set constant viscosity
        //phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*PoverR(r, phi, z)/sqrt(1.0/r/r/r);
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*PoverRBinary(r, phi, z)/sqrt(1.0/r/r/r);

      }
    }
  }

}


//----------------------------------------------------------------------------------------
//!\f: User-defined function for the binary gravitational potential.
//
void UserSourceTerms(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, 
      const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, 
      AthenaArray<Real> &cons_scalar ) {

  Real src[NHYDRO];   // Array for gas density/energy changes.
  Real rad, phi, z;   // Cylindrical coordinates
  Real v1, v2, v3;     // Initial velocities

  // Iterate over the entire grid.
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    Real sinx3 = sin(x3);
    Real cosx3 = cos(x3);
    //Real z = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      Real sinx2 = sin(x2);
      Real cosx2 = cos(x2);
      //Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        //Real x = pmb->pcoord->x1v(i);

        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);	
        Real r = std::max(rad, rs);
        Real p_over_r = PoverRBinary(r,phi,z);
        //Real p_over_r = PoverR(r,phi,z);

        Real Facc = 0.0;
        Real Fdamp = 0.0;


        /*================ Binary Gravity and Accretion ====================*/
          
        //Real x = R*sinth*cosphi;
        //Real y = R*sinth*sinphi;
        //Real z = R*costh;

        // Convert to Cartesian coordinates.
        // If the coordinate system is already in cartesian, transfer the existing values
        // to the new variable names.
        Real x, y, z;
        Real dmin = SIZE_MAX;
        if (COORDINATE_SYSTEM == "cartesian")
        {
          x = x1;
          y = x2;
          z = x3;
        }
        else if (COORDINATE_SYSTEM == "spherical_polar")
        {
          x = x1*sinx2*cosx3;
          y = x1*sinx2*sinx3;
          z = x1*cosx2;
        }	

        // Initialize the accelerations from the binary star system.
        Real ax = 0.0, ay = 0.0, az = 0.0;        // Cartesian
        //Real Fr = 0.0, Fth = 0.0, Fph = 0.0;      // Spherical

        // Calcucate the total acceleration from the particles (in Cartesian).
        for (int p=0; p<N_PARTICLES; p++) {
          Particle pn = ParticleList[p];
          Real dx = x-pn.x;
          Real dy = y-pn.y;
          Real dz = z-pn.z;
          Real d = std::sqrt(dx*dx + dy*dy + dz*dz);

          // Record minimum stellar distance
          dmin = std::min(dmin, d);

          // Check if the current cell is accreting
          //if (d < rs)
          //{
          //  IsAccreting = true;
          //}
    
          // Calculate gravitational acceleration using GM/(r*r) 

          //Real F = -1*pn.M/d/d;
          // Real F = (pi.M/(d+dR)-pi.M/(d-dR))/(2*dR);

          // Smoothed gravitational potential
          // Plummer potential
          // Real acc = -1*pn.M*d/pow(d*d+rs*rs, 1.5);

          // Klahr & Kley (2006) potential
          Real acc = (d > rs) ? -1*pn.M/(d*d) : -1*pn.M/(rs*rs) * (4*(d/rs)-3*pow(d/rs, 2.0));
          

          //Fx += F*(x-pn.x)/d;
          //Fy += F*(y-pn.y)/d;
          //Fz += F*(z-pn.z)/d;
          ax += acc*dx/d;
          ay += acc*dy/d;
          az += acc*dz/d;
        }

        // Calculate acceleration components along grid basis vectors.
        // If the coordinate system is spherical, convert the forces back to spherical coordinates.
        Real ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
        if (COORDINATE_SYSTEM == "cartesian")
        {
          ax1 = ax;
          ax2 = ay;
          ax3 = az;
        }
        if (COORDINATE_SYSTEM == "spherical_polar")
        {
                ax1 = ax*sinx2*cosx3 + ay*sinx2*sinx3 + az*cosx2;
                ax2 = ax*cosx2*cosx3 + ay*cosx2*sinx3 - az*sinx2;
                ax3 = -ax*sinx3 + ay*cosx3;
        }

        // Remove the central force, so the gravitational force is only from the binary
        //Fr += gm0/R/R;

        // USER DATA DEBUG: Record data to user arrays for output (before updating values)
        //pmb->ruser_meshblock_data[0](k,j,i) = ax1;
        //pmb->ruser_meshblock_data[1](k,j,i) = ax2;
        //pmb->ruser_meshblock_data[2](k,j,i) = ax3;

        // Update the gas momentum
        src[IM1] = dt*prim(IDN,k,j,i)*ax1;
        src[IM2] = dt*prim(IDN,k,j,i)*ax2;
        src[IM3] = dt*prim(IDN,k,j,i)*ax3;

        cons(IM1,k,j,i) += src[IM1];
        cons(IM2,k,j,i) += src[IM2];
        cons(IM3,k,j,i) += src[IM3];

        // Update the gas energy
        if (NON_BAROTROPIC_EOS) {
          src[IEN] = src[IM1]*prim(IM1,k,j,i) + src[IM2]*prim(IM2,k,j,i) + src[IM3]*prim(IM3,k,j,i);
          cons(IEN,k,j,i) += src[IEN];
        }


        /*=============================== Accretion ============================================*/
        // If the current cell is within the smoothing radius, remove mass according to
        // the mass removal timescale from Tang et al. (2017).
        //
        if (accreting)
        {
          if (dmin < rs)
          {
            Facc = 1.0;
            // Mass removal timescale, t_rem = 2*d^2/(3*nu)
            //Real t_rem = 2.0*rs*rs*std::sqrt(1.0/r/r/r)/alpha/p_over_r/3.0;
            Real t_rem = 1.0;
            Real drho = cons(IDN,k,j,i)/t_rem;

            // Update gas density
            cons(IDN,k,j,i) -= (dt/t_rem)*drho;
            cons(IDN,k,j,i) = std::max(cons(IDN,k,j,i), dfloor);
          }
        }


        /*=============== Disk Cooling ===================*/
        if (cooling)
        {
          // Calculate the cylindrical coordinates and PoverR for each grid

          // Instant Cooling
          /*
                pmb->phydro->u(IEN,k,j,i) = p_over_r*pmb->phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
                pmb->phydro->u(IEN,k,j,i) += 0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i)) + SQR(pmb->phydro->u(IM3,k,j,i)))/pmb->phydro->u(IDN,k,j,i);
          */

          // Cooling the gas with a cooling parameter Tc
          // Use cons() or pmb->phydro->u() ???
          // Calculate the internal energy of the gas E_int = Etot - KE
          Real eint = cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i))
              + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
              
          // Calculate the energy difference dE and the fractional timestep dtr for cooling relaxation.
          // dtr = 2pi/omega*Tc?
          // Real pres_over_r = E_int*(gamma_gas-1.0)/cons(IDN,k,j,i);
          //Real dtr = std::max(Tc*2.0*PI/std::sqrt(gm0/rad/rad/rad), dt);   // Cooling with orbital distance Rcyl
          // Real dtr = std::max(Tc*2.0*PI/std::sqrt(1.0/r/r/r), dt);        // Cooling with orbital distance Rsph
          Real dtr = std::max(Tc*2.0*PI/std::sqrt(1.0/dmin/dmin/dmin), dt);  // Cooling with min. orbtial distance min(Rsph1, Rsph2)
          Real dE = eint - p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
          
          // Update the gas energy by a fraction of dE, determined by the fraction dt/dtr.
          cons(IEN,k,j,i) -= (dt/dtr)*dE;

        } // End of cooling block


        /*================================ m=4 Damping Zone =====================================*/
        
        if (damping) {
          
          // Calculate spherical r and theta coordinates
          Real r_sph, th;
          if (COORDINATE_SYSTEM == "cartesian")
          {
            r_sph = std::sqrt(x*x + y*y + z*z);
            th = acos(z/(r_sph));
          }
          else if (COORDINATE_SYSTEM == "spherical_polar")
          {
            r_sph = x1;
            th = x2;
          }

          // Only apply damping zone to regions > 8 times the binary radius
          if (r_sph > Rdamp_in) {
            Real rho_0 = DenProfileCyl(rad,phi,z);
            VelProfileCyl(rad,phi,z,v1,v2,v3);
            //Real Fdamp = 0.0, Fdamp_r = 0.0, Fdamp_th = 0.0;

            // Set Fdamp as the maximum of the damping values in the radial and theta directions.
            Fdamp = (r_sph*r_sph - Rdamp_in*r_sph)/(Rdamp_out*Rdamp_out - Rdamp_in*Rdamp_out);
            //Fdamp = Fdamp_r;

            // Calculate damping dt
            // Real dtdamp = std::max(dt, Tdamp*2*PI*std::sqrt(r_sph*r_sph*r_sph/1.0));
            Real dtdamp = std::max(dt, Tdamp*2*PI*std::sqrt(Rdamp_in*Rdamp_in*Rdamp_in/1.0));
            // Real dtdamp = std::max(dt, Tdamp*2*PI*std::sqrt(Ro*Ro*Ro/1.0));
            //Real dtdamp = 1e-3;
          
            
            // Calculate the change in hydro quantities as
            // du = u-u_0
            Real drho = cons(IDN,k,j,i)-rho_0;
            Real dm1 =  cons(IM1,k,j,i)-rho_0*v1;
            Real dm2 =  cons(IM2,k,j,i)-rho_0*v2;
            Real dm3 =  cons(IM3,k,j,i)-rho_0*v3;
            //Real dE =   cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) 
                  //  + SQR(cons(IM3,k,j,i)) )/cons(IDN,k,j,i) 
                  //  - p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);

            // Update the conserved variables
            // u = u - du
            
            cons(IDN,k,j,i) -= (dt/dtdamp)*drho*Fdamp;
            cons(IM1,k,j,i) -= (dt/dtdamp)*dm1*Fdamp;
            cons(IM2,k,j,i) -= (dt/dtdamp)*dm2*Fdamp;
            cons(IM3,k,j,i) -= (dt/dtdamp)*dm3*Fdamp;

            Real dE =   cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) 
                    + SQR(cons(IM3,k,j,i)) )/cons(IDN,k,j,i);
                    //- p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
            //Real dE =   cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) 
            //        + SQR(cons(IM3,k,j,i)) )/cons(IDN,k,j,i) 
            //        - p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);

            cons(IEN,k,j,i) -= (dt/dtdamp)*dE*Fdamp;
                  
          }

        }  // End of damping block
        
        // USER DATA DEBUG: Record data to user arrays for output (after updating values)
        pmb->ruser_meshblock_data[0](k-(pmb->ks),j-(pmb->js),i-(pmb->is)) = r;
        pmb->ruser_meshblock_data[1](k-(pmb->ks),j-(pmb->js),i-(pmb->is)) = p_over_r;
        pmb->ruser_meshblock_data[2](k-(pmb->ks),j-(pmb->js),i-(pmb->is)) = Facc;

        //pmb->ruser_meshblock_data[0](k,j,i) = r;
        //pmb->ruser_meshblock_data[1](k,j,i) = Fdamp;
        //pmb->ruser_meshblock_data[2](k,j,i) = Facc;
        
        
      } // End of cell iteration
    }
  }
  //printf("USER MESHBLOCK DATA (SOURCE): %f\n", pmb->ruser_meshblock_data[0](pmb->ks,pmb->js,pmb->is));
  //printf("USER MESHBLOCK DATA (SOURCE,0): %f\n", pmb->ruser_meshblock_data[0](0,0,0));
}


//========================================================================================
//!\f: RefinementCondition: AMR refinement condition, used to determine which Meshblocks need refinement
//========================================================================================
int RefinementCondition(MeshBlock *pmb)
{
  // pmb -> pcoord -> x1v, x2v, x3v, ...
  // Get indices for the MeshBlock
  int mb_is = pmb->is, mb_js = pmb->js, mb_ks = pmb->ks;
  int mb_ie = pmb->ie, mb_je = pmb->je, mb_ke = pmb->ke;

  // Get coordinates of meshblock center
  Real x_mb = 0.5*(pmb->pcoord->x1v(mb_ie) + pmb->pcoord->x1v(mb_is));
  Real y_mb = 0.5*(pmb->pcoord->x2v(mb_je) + pmb->pcoord->x2v(mb_js));
  Real z_mb = 0.5*(pmb->pcoord->x3v(mb_ke) + pmb->pcoord->x3v(mb_ks));

  // Get half-dimensions of meshblock
  Real dx_mb = 0.5*(pmb->pcoord->x1v(mb_ie+1) - pmb->pcoord->x1v(mb_is));
  Real dy_mb = 0.5*(pmb->pcoord->x2v(mb_je+1) - pmb->pcoord->x2v(mb_js));
  //Real dz_mb = 0.0;
  Real dz_mb = 0.5*(pmb->pcoord->x3v(mb_ke+1) - pmb->pcoord->x3v(mb_ks));

  //Real t = pmb->pmy_mesh->time;

  bool RefineBlock = false;
  // Calculate regions of refinement - check if the CSDs are within Meshblock bounds
  /*
  for(int p=0; p<N_PARTICLES; ++p)
  {
    Particle pn = ParticleList[p];
    Real dx = fabs(x_mb-pn.x);
    Real dy = fabs(y_mb-pn.y);
    Real dz = 0.0;
    //Real dz = fabs(z_mb-pn.z);

    //printf("Particle Dist: %f %f\n", dx, dy);

    // Record minimum distance to particle
    // Use Linf norm / Chebyshev distance?
    //dmin = std::max(dx,dy);

    if((dx <= dx_mb) && (dy <= dy_mb) && (dz <= dz_mb))
      RefineBlock = true;

  }

  // Stop refinement if the MeshBlock is smaller than the CSD radius (~0.5a)
  if ((dx_mb < r_csd) || (dy_mb <= r_csd))
    RefineBlock = 0;
  */

  // Refine based off of distance to the origin.
  //Real dist = std::sqrt(x_mb*x_mb + y_mb*y_mb);
  // Only use z direction if the simulation is a 3D sim.
  // Maybe shrink the z-refinement by a factor of h/r?
  Real dist = Is2DSim ? std::sqrt(x_mb*x_mb + y_mb*y_mb) : std::sqrt(x_mb*x_mb + y_mb*y_mb + 5.0*z_mb*z_mb);
  //if (!Is2DSim)
  //  dist = std::sqrt(dist*dist + 1.0*p0_over_r0*z_mb*z_mb);

  if (dist < 2.0*bin_a)
    RefineBlock = true;


  // Debug prints
  
  /*
  if (debugmsg)
  {
    printf("Meshblock center: %f %f %f\n",x_mb, y_mb, z_mb);
    printf("Meshblock half-size: %f %f %f\n", dx_mb, dy_mb, dz_mb);
    printf("Dist. to Origin: %f\n", dist);
    printf("Refined? %d\n", RefineBlock);
    //printf("MB corner: %f %f", )
    //printf("Box corners: %f %f | %f %f\n", pmb->pcoord->x1f(pmb->is), pmb->pcoord->x1f(pmb->ie+1), pmb->pcoord->x2f(pmb->js), pmb->pcoord->x2f(pmb->je+1));
    //printf("MB Bounds: %f %f\n", pmb->x1min, pmb->x1max);
    
  }
  */

  // Return the refinement condition.
  if (RefineBlock)
    return 1;
  else
    return -1;

  return 0;

}

//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
//

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,is-i,j,k);
        prim(IDN,k,j,is-i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        prim(IM1,k,j,is-i) = v1;
        prim(IM2,k,j,is-i) = v2;
        prim(IM3,k,j,is-i) = v3;
        if (NON_BAROTROPIC_EOS)
          //prim(IEN,k,j,is-i) = PoverR(rad, phi, z)*prim(IDN,k,j,is-i);
          prim(IEN,k,j,is-i) = PoverRBinary(rad, phi, z)*prim(IDN,k,j,is-i);
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,ie+i,j,k);
        prim(IDN,k,j,ie+i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        prim(IM1,k,j,ie+i) = v1;
        prim(IM2,k,j,ie+i) = v2;
        prim(IM3,k,j,ie+i) = v3;
        if (NON_BAROTROPIC_EOS)
          //prim(IEN,k,j,ie+i) = PoverR(rad, phi, z)*prim(IDN,k,j,ie+i);
          prim(IEN,k,j,ie+i) = PoverRBinary(rad, phi, z)*prim(IDN,k,j,ie+i);
      }
    }
  }
}

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pco,rad,phi,z,i,js-j,k);
        prim(IDN,k,js-j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        prim(IM1,k,js-j,i) = v1;
        prim(IM2,k,js-j,i) = v2;
        prim(IM3,k,js-j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          //prim(IEN,k,js-j,i) = PoverR(rad, phi, z)*prim(IDN,k,js-j,i);
          prim(IEN,k,js-j,i) = PoverRBinary(rad, phi, z)*prim(IDN,k,js-j,i);
      }
    }
  }
}

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pco,rad,phi,z,i,je+j,k);
        prim(IDN,k,je+j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        prim(IM1,k,je+j,i) = v1;
        prim(IM2,k,je+j,i) = v2;
        prim(IM3,k,je+j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          //prim(IEN,k,je+j,i) = PoverR(rad, phi, z)*prim(IDN,k,je+j,i);
          prim(IEN,k,je+j,i) = PoverRBinary(rad, phi, z)*prim(IDN,k,je+j,i);
      }
    }
  }
}

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pco,rad,phi,z,i,j,ks-k);
        prim(IDN,ks-k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        prim(IM1,ks-k,j,i) = v1;
        prim(IM2,ks-k,j,i) = v2;
        prim(IM3,ks-k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          //prim(IEN,ks-k,j,i) = PoverR(rad, phi, z)*prim(IDN,ks-k,j,i);
          prim(IEN,ks-k,j,i) = PoverRBinary(rad, phi, z)*prim(IDN,ks-k,j,i);
      }
    }
  }
}

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pco,rad,phi,z,i,j,ke+k);
        prim(IDN,ke+k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        prim(IM1,ke+k,j,i) = v1;
        prim(IM2,ke+k,j,i) = v2;
        prim(IM3,ke+k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          //prim(IEN,ke+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ke+k,j,i);
          prim(IEN,ke+k,j,i) = PoverRBinary(rad, phi, z)*prim(IDN,ke+k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//!\f: UserWorkBeforeOutput: User-defined tasks for the Mesh, called once per cycle
//
void Mesh::UserWorkInLoop(void)
{
  Particle_Leapfrog(ParticleList, 2, time, dt);
}

//----------------------------------------------------------------------------------------
//!\f: UserWorkInLoop: User-defined tasks for each MeshBlock
//
void MeshBlock::UserWorkInLoop(void)
{
  // Integrate gravitational bodies
  //if (prev==NULL)
  //{
    //Particle_Leapfrog(ParticleList, 2, pmy_mesh->time, pmy_mesh->dt);

    // Output particle coordinates
    if (Globals::my_rank == 0) {

      //Particle_Leapfrog(ParticleList, 2, pmy_mesh->time, pmy_mesh->dt);
      if (pmy_mesh->time-orbit_t > orbit_dt) {
        printf("particleA %g %g %g\n", ParticleList[0].x, ParticleList[0].y, ParticleList[0].z);
        printf("particleB %g %g %g\n", ParticleList[1].x, ParticleList[1].y, ParticleList[1].z);
        orbit_t += orbit_dt;
      }
    }
  //}

}

//----------------------------------------------------------------------------------------
//!\f: UserWorkBeforeOutput: User-defined tasks before output is written
//
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // USER DATA DEBUG
  // Record user data to output file
  //int il = is - NGHOST, iu = ie + NGHOST;
  //int jl = js - NGHOST, ju = je + NGHOST;
  //int kl = ks - NGHOST, ku = ke + NGHOST;

  // Without ghost zones
  //int il = is, iu = ie;
  //int jl = js, ju = je;
  //int kl = ks, ku = ke;

  // Start from 0?
  int nx1 = ie - is;
  int nx2 = je - js;
  int nx3 = ke - ks;

  // Iterate over cells and output variables, transfer data from user meshblock to output variable array
  for (int n=0; n<3; ++n)
  {
    for (int k=0; k<=nx3; ++k)
    {
      for (int j=0; j<=nx2; ++j)
      {
        for (int i=0; i<=nx1; ++i)
        {
          //user_out_var(n,ks+k,js+j,is+i) = 10*n+i+j+k;
          user_out_var(n,ks+k,js+j,is+i) = ruser_meshblock_data[n](k,j,i);
        }
      }
    }
  }

  // DEBUG: Print user meshblock statements
  if (debugmsg)
  {
    //printf("GRID IDXS             : %d-%d %d-%d %d-%d\n", is,ie, js,je, ks,ke);
    //printf("USER MESHBLOCK DATA (SAVE): %f\n", ruser_meshblock_data[0](ks,js,is));
    //printf("USER MESHBLOCK DATA (0,SAVE): %f\n", ruser_meshblock_data[0](0,0,0));
    //printf("USER OUT VAR          : %f\n", user_out_var(0,ks,js,is));
    //printf("USER OUT VAR (ZEROIDX): %f\n", user_out_var(0,0,0,0));
  }

}
