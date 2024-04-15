//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file polardisk.cpp
//  \brief Initializes stratified Keplerian accretion disk inclined to the simulation plane
//  in spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm,
//  around an eccentric binary to replicate Martin and Lubow (2018).

// Configure command:
/* 
python configure.py --prob=polardisk --coord=spherical_polar --nghost=3 -mpi [hdf5 options] -cxx='icpc' --cflag="-lmpi -lmpi++" 
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
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"    // For alpha viscosity

// Add gravitational masses
#include "../Binary.hpp"

static void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
static Real DenProfileCyl(const Real rad, const Real phi, const Real z);
static Real DenProfileSph(const Real R, const Real th, const Real phi);
static Real PoverR(const Real rad, const Real phi, const Real z);
static Real PoverRSph(const Real R, const Real th, const Real phi);
static void VelProfileCyl(const Real rad, const Real phi, const Real z,
  Real &v1, Real &v2, Real &v3, Real den);
void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
static Real rho_floor(const Real rad, const Real phi, const Real z);

// User-defined functions
void UserSourceTerms(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);
Real UserTimestep(MeshBlock *pmb);

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

// disk geometry variables
static Real gm0, r0;
static Real rin_mu, rin_sigma;
static Real rout_mu, rout_sigma;
static Real inc;

// disk structure variables
static Real rho0, p0_over_r0, gamma_gas;
static Real dfloor;
static Real dslope, pslope;
static Real rho_floor0, rho_floor_slope;
static Real alpha;
static Real Tc;

// binary variables
static Real M1, M2, Mtot;
static Real bin_a, bin_ecc;

// time variables
Real orbit_t;
static Real orbit_dt;
static Real dt_sub;

// simulation variables
static Real xcut;
static Real amp;
static bool HasCrashed; 


// Setup the initial particle array
Particle ParticleList[2] = {
    Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
};


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  // To use the Particle masses, set GM=0 to disable the default central mass.
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Read in variables for exponential decay of density at inner/outer edges
  rin_mu = pin->GetOrAddReal("problem", "rin_mu", mesh_size.x1min);
  rin_sigma = pin->GetOrAddReal("problem", "rin_sigma", mesh_size.x1min/6.);
  rout_mu = pin->GetOrAddReal("problem", "rout_mu", mesh_size.x1max);
  rout_sigma = pin->GetOrAddReal("problem", "rout_sigma", mesh_size.x1max/12.);

  inc = pin->GetOrAddReal("problem", "inc", PI/12);



  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters for pressure floor and pressure slope
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  rho_floor0 = pin->GetOrAddReal("problem", "rho_floor0", pow(10,-5));
  rho_floor_slope = pin->GetOrAddReal("problem", "rho_floor_slope", 0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    gamma_gas = pin->GetReal("hydro","gamma");
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }

  // Alpha-viscosity. Enroll the viscosity function only if alpha is positive.
  alpha = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  if (alpha > 0.0) {
    EnrollViscosityCoefficient(AlphaViscosity);
  }

  // Cooling Time
  Tc = pin->GetOrAddReal("problem", "Tc", pow(10, -5));

  // Read in and rescale binary parameters
  // In code units, masses are scaled so that GMtot = 1.0,
  // so M1 and M2 are normalized so M = M/Mtot.

  M1 =  pin->GetOrAddReal("problem", "Ma", 1.0);
  M2 =  pin->GetOrAddReal("problem", "Mb", 1.0);
  Mtot = M1 + M2;
  M1 = M1/Mtot;
  M2 = M2/Mtot;
  Mtot = Mtot/Mtot;

  bin_a = pin->GetOrAddReal("problem", "bin_a", 0.25);
  bin_ecc = pin->GetOrAddReal("problem", "bin_ecc", 0.0);

  // Initialize orbit output counter
  orbit_t = time;
  orbit_dt = pin->GetOrAddReal("problem", "orbit_dt", 0.01);

  // Set subcycling dt
  dt_sub = pin->GetOrAddReal("problem", "dt_sub", 2*PI*pow(bin_a,1.5)/3000.);

  // Add for new density floor
  xcut = mesh_size.x1min;
  amp = pin->GetOrAddReal("problem", "amp", 0.001);


  // Enroll user functions
  EnrollUserExplicitSourceFunction(UserSourceTerms);
  // EnrollUserTimeStepFunction(UserTimestep);

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


  // Try to read in particle coordinates for a restart file.
  // If one exists, set the particle coordinates to these values.

  std::ifstream ipfile("particle.rst");
  if (ipfile.good()) {
    std::cout << "Particle input file found.  Restarting particle positions." << std::endl;
    // Read in particle positions
    std::string line;
    // Loop for each particle
    while (getline(ipfile,line)) {
      static int pn = 0;
      int n = 0;
      std::size_t s_beg = 0, s_end = 0;
      float p_rst [N_EQ+1];

      // Parse the data and set the Particle data
      do {
        s_end = line.find_first_of(' ', s_beg);
        std::string pval = line.substr(s_beg, s_end-s_beg);
        p_rst[n] = std::stof(pval);
        n++;
        s_beg = s_end+1;
        // std::cout << "Particle " << pn << " Quantity " << n << std::endl;
      } while (s_end < line.size());

      // TODO: Direct Particle assignment is bugged; rewrite or create Particle assignment operator
      // ParticleList[pn] = Particle(p_rst[0], p_rst[1], p_rst[2], p_rst[3], p_rst[4], p_rst[5], p_rst[6]);
      ParticleList[pn].x = p_rst[0];
      ParticleList[pn].y = p_rst[1];
      ParticleList[pn].z = p_rst[2];
      ParticleList[pn].vx = p_rst[3];
      ParticleList[pn].vy = p_rst[4];
      ParticleList[pn].vz = p_rst[5];
      ParticleList[pn].M = p_rst[6];
      // DEBUG: Print restarted particle information 
      if (Globals::my_rank == 0)
      {
        printf("Particle %d: %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", 
	  pn, ParticleList[pn].x, ParticleList[pn].y, ParticleList[pn].z, ParticleList[pn].vx,
	  ParticleList[pn].vy, ParticleList[pn].vz,ParticleList[pn].M);
      }
      pn++;
    }
  }

  // If no restart file exists, initialize the particle velocities 
  //based off their separation and eccentricity.
  else {
    std::cout << "No restart file found.  Initializing to default positions." << std::endl;

    // Move binary to COM frame and set velocity and set particle velocities
    // Only set initial velocity if no restart file is present.  Move this to initialization of ParticleList?

    // NOTE: The current configuration is set for a binary in the simulation XZ-plane (polar alignment).
    // For coplanar alignment, shuffle the axes accordingly or use the commented lines below.


    // Set initial particle masses
    ParticleList[0].M = M1;
    ParticleList[1].M = M2;

    // Set initial particle positions
    //ParticleList[0].z = 0.0;
    //ParticleList[1].z = bin_a*(1.0-bin_ecc);

    ParticleList[0].x = 0.0;                 // Coplanar version
    ParticleList[1].x = bin_a*(1.0-bin_ecc);

    Real dx = ParticleList[0].x-ParticleList[1].x;
    Real dy = ParticleList[0].y-ParticleList[1].y; 
    Real dz = ParticleList[0].z-ParticleList[1].z;
    Real dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Calculate specific velocity vs = v/M
    Real vs = std::sqrt( 1.0/Mtot/dist*(1.0+bin_ecc) );

    //ParticleList[0].vx = vs * ParticleList[1].M;
    //ParticleList[1].vx = -vs * ParticleList[0].M;


    // Coplanar orientation
    ParticleList[0].vy = -vs * ParticleList[1].M;
    ParticleList[1].vy = vs * ParticleList[0].M;

    move_to_com(ParticleList);
     
    printf("Initial Particle Positions:  %f %f\n", ParticleList[0].x, ParticleList[1].x);
    printf("Initial Particle Velocities: %f %f\n", ParticleList[0].vy, ParticleList[1].vy);
    printf("Particle Subcycle Timestep:  %f\n", dt_sub);
  }
  ipfile.close();

  
  // DEBUG: Set output variables and statements
  HasCrashed = false;

  // Output useful constants at beginning of simulation.
  if (Globals::my_rank == 0) {
    printf("Rho_Floor0   : %f\n", rho_floor0);
    printf("Alpha        : %f\n", alpha);
    printf("Cooling Time : %f\n", Tc);
    printf("Inclination  : %f\n", inc);
  }

  return;
}

/*
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
  AllocateRealUserMeshBlockDataField(4);
  ruser_meshblock_data[0].NewAthenaArray(nx3, nx2, nx1);
  ruser_meshblock_data[1].NewAthenaArray(nx3, nx2, nx1);
  ruser_meshblock_data[2].NewAthenaArray(nx3, nx2, nx1);
  ruser_meshblock_data[3].NewAthenaArray(nx3, nx2, nx1);

  // enroll user output variables
  AllocateUserOutputVariables(4);
}
*/

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
      GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to (tilted) cylindrical coordinates
      // compute initial density/velocity profile relative to the tilted coordinates
      phydro->u(IDN,k,j,i) = DenProfileCyl(rad,phi,z);
      VelProfileCyl(rad,phi,z,v1,v2,v3, phydro->u(IDN,k,j,i));

      phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
      phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
      phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
      if (NON_BAROTROPIC_EOS) {
        Real p_over_r = PoverR(rad,phi,z);
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
  // Compute coordinates for the tilted disk.
  // Assume the coordinates entering are in spherical-polar coordinates, i.e.
  // (r, theta, phi) = (pco->x1v(i), pco->x2v(j), pco->x3v(k))
  // Then, convert them to tilted form.

  // Calculation of disk inclination and the tilted coordinate theta'.
  Real sinth = sin(pco->x3v(k));
  Real costh = cos(pco->x3v(k));
  Real sinphi = sin(pco->x2v(j));
  Real cosphi = cos(pco->x2v(j));
  Real sininc = sin(inc);
  Real cosinc = cos(inc);

  Real tp = asin(sqrt( pow(sinth*cosphi,2)+
  pow(sinth*sinphi*cosinc,2)+
  pow(costh*sininc,2)+
  2*sinth*costh*sinphi*sininc*cosinc )); 

  // Write cylindrical coordinates relative to the new disk.
  rad=pco->x1v(i)*sin(tp);
  phi=atan2(sinth*sinphi*cosinc+costh*sininc, sinth*cosphi);
  z=pco->x1v(i)*cos(tp);
  // Set the calculated z value as negative if tp is greater than PI/2.
  if (pco->x2v(j) > PI/2-inc*sin(pco->x3v(k))){
    z *= -1;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates

static Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  // Analytical solution
  /*
  Real denmid = rho0*pow(rad/r0,dslope);
  Real dentem = denmid*exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;

  Real new_dfloor = std::max(rho_floor(rad, phi, z), dfloor);
  // std::cout<<rho_floor(rad, phi, z)<<std::endl;
  return std::max(den,new_dfloor);
  */

  // Zhaohuan's numerical solution
  Real r = std::max(rad, xcut);
  Real denmid = rho0*pow(r/r0, dslope);
  // Add a Gaussian dropoff to inner and outer edges
  if (rad < rin_mu) {
     denmid *= exp(-(rad-rin_mu)*(rad-rin_mu)/rin_sigma/rin_sigma);
  }

  if (rad > rout_mu) {
    denmid *= exp(-(rad-rout_mu)*(rad-rout_mu)/rout_sigma/rout_sigma);
  }


  Real zo = 0.0;
  Real zn = zo;
  den = denmid;
  // Variables for numerical integration.
  Real Ro, phio, tho, poverro;
  Real Rn, phin, thn, poverrn;
  Real coe, h, dz;

  // Integrate density from the midplane to the current z coordinate.
  while (zn <= fabs(z)) {
    coe = 1.0*0.5*(1./sqrt(r*r+zn*zn) - 1./sqrt(r*r+zo*zo));

    Ro = sqrt(r*r+zo*zo);
    tho = atan(r/zo);
    phio = 0.0;
    poverro = PoverRSph(Ro, tho, phio);

    // Calculate P/R at the midplane
    Real poverr_mid = p_over_r*pow(r/r0, pslope);
    h = sqrt(poverr_mid)/sqrt(1.0/r/r/r);
    dz = h/32;

    Rn = sqrt(r*r+zn*zn);
    thn = atan(r/zn);
    phin = 0.0;
    poverrn = PoverRSph(Rn, thn, phin);

    // Increase the density and increment the z value.
    den=den*(coe+poverro)/(poverrn-coe);
    zo=zn;
    zn=zo+dz;
  }

  return(std::max(den, rho_floor(rad, phi,z)));

}

//----------------------------------------------------------------------------------------
//! \f  computes density in spherical coordinates
//----------------------------------------------------------------------------------------
static Real DenProfileSph(const Real R, const Real th, const Real phi) {
  Real rad = std::max(R*sin(th), xcut);
  Real z = R*cos(th);
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);

  // Zhaohuan's numerical solution
  //Real r = std::max(rad, xcut);  // Could eliminate this line by changing rad to r?
  Real r = rad;
  Real denmid = rho0*pow(r/r0, dslope);
  // Add a Gaussian dropoff to inner and outer edges
  if (rad < rin_mu) {
    denmid *= exp(-(rad-rin_mu)*(rad-rin_mu)/rin_sigma/rin_sigma);
  }

  if (rad > rout_mu) {
    denmid *= exp(-(rad-rout_mu)*(rad-rout_mu)/rout_sigma/rout_sigma);
  }

  //Real zo = 0.0;
  //Real zn = zo;
  den = denmid;
  // Variables for numerical integration.
  Real zo, zn;
  Real Ro, phio, tho, poverro;
  Real Rn, phin, thn, poverrn;
  Real coe, h, dz;

  // Calculate P/R at the midplane
  Real poverr_mid = p_over_r*pow(r/r0,pslope);
  //h = sqrt(poverr_mid)/sqrt(gm0/r/r/r);
  h = sqrt(poverr_mid)/sqrt(1.0/r/r/r);
  // h = sqrt(poverr_mid)/sqrt(r0*r0*r0/r/r/r);

  dz = h/32;
  zo = 0.0;
  zn = zo+dz;

  Ro = sqrt(r*r+zo*zo);
  tho = atan(r/zo);
  phio = 0.0;
  poverro = PoverRSph(Ro, tho, phio);

  // Integrate density from the midplane to the current z coordinate.
  while (zn <= fabs(z)) {

    Rn = sqrt(r*r+zn*zn);
    thn = atan(r/zn);
    phin = 0.0;
    poverrn = PoverRSph(Rn, thn, phin);

    coe = 1.0*0.5*(1./sqrt(r*r+zn*zn) - 1./sqrt(r*r+zo*zo));

    // Increase the density and increment the z value.
    den=den*(coe+poverro)/(poverrn-coe);
    zo=zn;
    zn=zo+dz;

    // Copy values of R, th, phi=0, and PoverRSph over to "old" variables as zo->zn.
    Ro = Rn;
    tho = thn;
    poverro = poverrn;
  }

  return(std::max(den, rho_floor(R*sin(th), phi,z)));

}


//----------------------------------------------------------------------------------------
//! \f  computes pressure/density in cylindrical coordinates

static Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*pow(std::sqrt(rad*rad+z*z)/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes pressure/density in spherical coordinates

static Real PoverRSph(const Real R, const Real th, const Real phi) {
  Real poverr;
  poverr = p0_over_r0*pow(R/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates

static void VelProfileCyl(const Real rad, const Real phi, const Real z,
                          Real &v1, Real &v2, Real &v3, Real den) {

  // Defining a new velocity which accounts for pressure gradients and rotational forces.

  // Real r = std::max(rad, xcut);	// Maximum of cylindrical radius and the minimum radial mesh
  Real R = std::sqrt(rad*rad+z*z);  // Spherical radius
  Real th = acos(z/R);          // Spherical polar angle
  Real dR = 0.01*R;
  Real dth = 0.01*PI;

  // Trig functions of th, phi, and inc
  Real sinth = sin(th);
  Real costh = cos(th);
  Real sinphi = sin(phi);
  Real cosphi = cos(phi);
  Real sininc = sin(inc);
  Real cosinc = cos(inc);
 
  // Calculate pressure gradient and velocity 
  Real dpdR = (PoverRSph(R+dR,th,phi)*DenProfileSph(R+dR,th,phi) - PoverRSph(R-dR,th,phi)*DenProfileSph(R-dR,th,phi))/2./dR*sinth
    + (PoverRSph(R,th+dth,phi)*DenProfileSph(R,th+dth,phi) - PoverRSph(R,th-dth,phi)*DenProfileSph(R,th-dth,phi))/2./dth*costh/R;

  Real vel = std::sqrt(std::max(1.0*rad*rad/R/R/R + rad/DenProfileSph(R,th,phi)*dpdR, 0.0));

  // Lower the velocity in regions close to the pole of rotation.
  if (den <=(1.0+amp)*rho_floor(rad, phi, z)) {
    vel = std::sqrt(1.0*rad*rad/R/R/R);
  }



  // Calculate the original values of phi and th before the transformation.
  Real sin_phi0 = sinth*sinphi*cosinc - costh*sininc;
  Real cos_phi0 = sinth*cosphi;
  Real sin_th0 = sqrt( pow(sinth,2)*pow(cosphi,2)+
  pow(sinth,2)*pow(sinphi,2)*pow(cosinc,2)+
  pow(costh,2)*pow(sininc,2)-
  2*sinth*costh*sinphi*sininc*cosinc );

  // Write the R, theta, and phi components of the velocity (in the tilted disk plane).
  v1 = 0.0;
  v2 = -vel*cos(phi)*sin(inc)/sin_th0;
  v3 = vel*(sin(phi)*sin_phi0 + cos(phi)*cos_phi0*cos(inc))/sin_th0;

  return;
}

//----------------------------------------------------------------------------------------
//! \f: Function for alpha-viscosity. It is only enrolled and used if alpha is positive.
void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
  Coordinates *pco = pmb->pcoord;
  for (int k=ks; k<=ke; ++k) {
    Real phi = pco->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real th = pco->x2v(j);
      Real sinth = sin(th);
      for (int i=is; i<ie; ++i) {
        Real R = pco->x1v(i);
        Real Rcyl = R*sinth;
        Real Fcutoff = 1.0;

        // Add an exponential cutoff for regions close to the inner boundary to increase timestep.
        // Since viscous dt has cylindrical symmetry, cutoff is based on Rcyl.
        if (Rcyl < xcut)
        {
          Fcutoff = exp((Rcyl-xcut)/0.075);
        }

        // Set viscosity
        // Use spherical symmetry, remove double division for orbital frequency
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = Fcutoff*alpha*PoverRSph(R, th, phi)*sqrt(R*R*R/1.0);
        //phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha;

      }
    }
  }

}


//----------------------------------------------------------------------------------------
//!\f: Spherical density floor
//
static Real rho_floor(const Real rad, const Real phi, const Real z)
{
  // Simple density floor
  Real rhofloor=rho_floor0*pow(std::sqrt(rad*rad+z*z)/r0, rho_floor_slope);
  return std::max(rhofloor, dfloor);
}


//----------------------------------------------------------------------------------------
//!\f: User-defined function for source terms.
//
void UserSourceTerms(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) {

  Real src[NHYDRO];   // Array for gas density/energy changes.
  Real rad, phi, z;   // Cylindrical coordinates

  // Iterate over the entire grid.
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real th = pmb->pcoord->x3v(k);
    Real sinth = sin(th);
    Real costh = cos(th);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real phi = pmb->pcoord->x2v(j);
      Real sinphi = sin(phi);
      Real cosphi = cos(phi);
      for (int i=pmb->is; i<=pmb->ie; ++i) {     

        Real dmin = SIZE_MAX;

        /*================ Binary Gravity ====================*/
        Real R = pmb->pcoord->x1v(i);
        // Real dR = R/10000.;

        // Calculate cartesian coordinates from spherical coordinates.
        Real x = R*sinth*cosphi;
        Real y = R*sinth*sinphi;
        Real z = R*costh;

        // Initialize the forces from the binary star system.
        Real ax = 0.0, ay = 0.0, az = 0.0;        // Cartesian
        Real ar = 0.0, ath = 0.0, aph = 0.0;      // Spherical

        // Calcucate the total force from the particles (in Cartesian).
        for (int p=0; p<2; p++) {
          Particle pn = ParticleList[p];
          Real dx = x-pn.x;
          Real dy = y-pn.y;
          Real dz = z-pn.z;
          Real d = std::sqrt(dx*dx + dy*dy + dz*dz);

          // Record minimum stellar distance
          dmin = std::min(dmin,d);

          // Add accelerations
          Real acc = -1*pn.M/d/d;
          ax += acc*dx/d;
          ay += acc*dy/d;
          az += acc*dz/d;
        }

        // Convert the forces back into spherical coordinates.
        ar = ax*sinth*cosphi + ay*sinth*sinphi + az*costh;
        ath = ax*costh*cosphi + ay*costh*sinphi - az*sinth;
        aph = -ax*sinphi + ay*cosphi;

  
        // Remove the central force, so the gravitational force is only from the binary
        ar += gm0/R/R;

        // Update the gas density
        src[IM1] = dt*prim(IDN,k,j,i)*ar;
        src[IM2] = dt*prim(IDN,k,j,i)*ath;
        src[IM3] = dt*prim(IDN,k,j,i)*aph;

        cons(IM1,k,j,i) += src[IM1];
        cons(IM2,k,j,i) += src[IM2];
        cons(IM3,k,j,i) += src[IM3];

        // Update the gas energy
        if (NON_BAROTROPIC_EOS) {
          src[IEN] = src[IM1]*prim(IM1,k,j,i) + src[IM2]*prim(IM2,k,j,i) + src[IM3]*prim(IM3,k,j,i);
          cons(IEN,k,j,i) += src[IEN];
        }

 
        /*=============== Disk Cooling ===================*/
        // Calculate the cylindrical coordinates and PoverR for each grid
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        Real p_over_r = PoverR(rad,phi,z);

        // Instant Cooling
        /*
            pmb->phydro->u(IEN,k,j,i) = p_over_r*pmb->phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
            pmb->phydro->u(IEN,k,j,i) += 0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i)) + SQR(pmb->phydro->u(IM3,k,j,i)))/pmb->phydro->u(IDN,k,j,i);
        */

        // Cooling the gas with a cooling parameter Tc
        // Calculate the internal energy of the gas E_int = Etot - KE
        Real eint = cons(IEN,k,j,i) - 0.5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i))
            + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
            
        // Calculate the energy difference dE and the fractional timestep dtr for cooling relaxation.
        // Real pres_over_r = E_int*(gamma_gas-1.0)/cons(IDN,k,j,i);
        //Real dtr = std::max(Tc*2.0*PI/std::sqrt(1.0/rad/rad/rad), dt);   // Using cylindrical r
        Real dtr = std::max(Tc*2.0*PI/std::sqrt(1.0/dmin/dmin/dmin), dt);  // Using min(r1, r2)
        Real dE = eint - p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
        
        // Update the gas energy by a fraction of dE, determined by the fraction dt/dtr.
        cons(IEN,k,j,i) -= (dt/dtr)*dE;


      }
    }
  }

}


//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
//


// Inner & Outer X1 Boundaries: One-way outflow:
// Mass can flow out of the simulation, but not back in
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          // One-Way Outflow
              prim(n,k,j,is-i) = prim(n,k,j,is);
              if (n == IVX && prim(n,k,j,is-i) > 0.0)
                prim(n,k,j,is-i) = 0.0;

        }
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
       Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real rad,phi,z;
  Real v1, v2, v3;
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          // One-Way outflow
              prim(n,k,j,ie+i) = prim(n,k,j,ie);
              if (n == IVX && prim(n,k,j,ie+i) < 0.0)
                prim(n,k,j,ie+i) = 0.0;
        }
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
        VelProfileCyl(rad,phi,z,v1,v2,v3, prim(IDN,k,js-j,i));
        prim(IM1,k,js-j,i) = v1;
        prim(IM2,k,js-j,i) = v2;
        prim(IM3,k,js-j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,js-j,i) = PoverR(rad, phi, z)*prim(IDN,k,js-j,i);
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
        VelProfileCyl(rad,phi,z,v1,v2,v3, prim(IDN,k,je+j,i));
        prim(IM1,k,je+j,i) = v1;
        prim(IM2,k,je+j,i) = v2;
        prim(IM3,k,je+j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,je+j,i) = PoverR(rad, phi, z)*prim(IDN,k,je+j,i);
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
        VelProfileCyl(rad,phi,z,v1,v2,v3, prim(IDN,ks-k,j,i));
        prim(IM1,ks-k,j,i) = v1;
        prim(IM2,ks-k,j,i) = v2;
        prim(IM3,ks-k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,ks-k,j,i) = PoverR(rad, phi, z)*prim(IDN,ks-k,j,i);
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
        VelProfileCyl(rad,phi,z,v1,v2,v3, prim(IDN,ke+k,j,i));
        prim(IM1,ke+k,j,i) = v1;
        prim(IM2,ke+k,j,i) = v2;
        prim(IM3,ke+k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,ke+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ke+k,j,i);
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//!\f: Mesh->UserWorkInLoop: User-defined tasks for the Mesh, called once per cycle
//
void Mesh::UserWorkInLoop(void)
{
  // Integrate the gravitational bodies. Call this only once per cycle.
  Particle_Leapfrog_Subcycle(ParticleList, 2, time, dt, dt_sub);
}


//----------------------------------------------------------------------------------------
//!\f: UserWorkInLoop: User-defined tasks to be completed each cycle
//

void MeshBlock::UserWorkInLoop(void)
{
  Real rad, phi, z;
  for(int k=ks; k<=ke; ++k){
     for (int j=js; j<=je; ++j) {
       for (int i=is; i<=ie; ++i) {
         // Set a minimum level for the density floor.
         GetCylCoord(pcoord, rad, phi, z, i, j, k);
         phydro->u(IDN,k,j,i) = std::max(phydro->u(IDN,k,j,i), rho_floor(rad, phi, z));

          // Debug check for crashes
          if (std::isnan(phydro->u(IM1,k,j,i)) && !HasCrashed)
          {
            HasCrashed = true;
            printf("NAN at (%f, %f, %f), Index [%d, %d, %d] \n", rad, phi, z, i, j, k);
          }
      }
    }
  }

  // Output coordinates if a time orbit_dt has elapsed.
  if (Globals::my_rank == 0) {

    if (pmy_mesh->time-orbit_t >= orbit_dt) {
      for (int i=0; i<2; i++)
      {
        Particle P = ParticleList[i];
        printf("particle%c %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",
          65+i, pmy_mesh->time, P.x, P.y, P.z, P.vx, P.vy, P.vz);
      }
      orbit_t += orbit_dt;
    }
  }

}


//----------------------------------------------------------------------------------------
//!\f: UserWorkAfterLoop: User-defined tasks to be completed after the simulation
//

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  // Writes to an output file once at the end of the simulation.
  // Records the final position of the particles in a .rst file.
  if(Globals::my_rank == 0) {
        std::cout << "Creating particle restart file" << std::endl;
        std::ofstream opfile ("particle.rst");
        for (int i = 0; i < 2; i++) {
          Particle P = ParticleList[i];
          opfile << P.x << ' ' << P.y << ' ' << P.z << ' ' << P.vx << ' ' << P.vy << ' ' << P.vz << ' ' << P.M << std::endl;
          // DEBUG: Print particle info
          printf("Particle %d:, %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
            i, P.x, P.y, P.z, P.vx, P.vy, P.vz, P.M);
        }
        opfile.close();
  }
}

//----------------------------------------------------------------------------------------
//!\f: User-defined timestep
//
Real UserTimestep(MeshBlock *pmb)
{
  return 7e-4;    // Lower dt to set a constant timestep for binary
}

/*
//----------------------------------------------------------------------------------------
//!\f: UserWorkBeforeOutput: User-defined tasks before output is written
//
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // USER DATA DEBUG
  // Record user data to output file
  int il = is - NGHOST, iu = ie + NGHOST;
  int jl = js - NGHOST, ju = je + NGHOST;
  int kl = ks - NGHOST, ku = ke + NGHOST;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for(int n=0; n<4; ++n) {
          user_out_var(n,k,j,i) = ruser_meshblock_data[n](k,j,i);
        }
      }
    }
  }
}
*/
