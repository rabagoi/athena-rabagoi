#include<iostream>
#include<valarray>
#include<cmath>
#include<fstream>


#define N_PARTICLES 3
#define N_EQ 6
#define G 1.0

// ============================================================================================================================
// Struct for a gravitationally interacting particle.
// ============================================================================================================================
struct Particle
{
        // Array of positional elements for integration.
        std::valarray<double> elements = std::valarray<double> (N_EQ);

        // Aliases of the elements using their familiar names.
        double &x = elements[0], &y = elements[2], &z = elements[4];
        double &vx = elements[1], &vy = elements[3], &vz = elements[5];
        double M;       // Particle mass
    
        Particle(double xo=0., double yo=0., double zo=0., double vxo=0., double vyo=0., double vzo=0., double Mo=0.)
        {
                x = xo, y = yo, z = zo;
                vx = vxo, vy = vyo, vz = vzo;
                M = Mo;
        }

        void print_elements();

};


// Print the position and velocity elements of a particle.
void Particle::print_elements()
{
    //std::cout << x << ' ' << y << ' ' << z << std::endl;
    //std::cout << vx << ' ' << vy << ' ' << vz << std::endl;
    //std::cout << M << std::endl;
    //std::cout << "Elements: ";
    //for (int i=0; i<N_EQ;i++)
    //{
    //    std::cout << elements[i] << ' ';
    //}

    printf("Pos: %8f %8f %8f\n", x, y, z);
    printf("Vel: %8f, %8f %8f\n", vx, vy, vz);
    printf("Mass: %8f\n", M); 
    std::cout << std::endl;
}


// Particle integration functions
void Particle_RK4(Particle[], int, double, double);
void Particle_Leapfrog(Particle[], int, double, double);
void Particle_Leapfrog_Subcycle(Particle[], int, double, double, double);
std::valarray<double> Particle_F(double, Particle, Particle, std::valarray<double>);

// Analytic orbit functions
void orbit(Particle[], int, double);

// Misc. functions
double Particle_TotalEnergy(Particle[], int);
void move_to_com(Particle P[], int);
void Particle_L(Particle[], double*);

// Rotation functions
void Rx(Particle[], double);
void Rz(Particle[], double);

// ============================================================================================================================
// Integrators for the Particle class.
// ============================================================================================================================


// Calculates the gravitational force between two Particle objects.
// Includes a parameter for k values in RK4 integrator, which defaults to zero if not specified.
std::valarray<double> Particle_F(double t, Particle P1, Particle P2, std::valarray<double> k = std::valarray<double> (0.0, N_EQ))
{
    std::valarray<double> f (N_EQ);
    std::valarray<double> Pk = P1.elements+k;

    // x, y, z, and R between the two particles
    double Px = Pk[0]-P2.x;
    double Py = Pk[2]-P2.y;
    double Pz = Pk[4]-P2.z;
    double R = sqrt( pow(Px, 2) + pow(Py, 2) + pow(Pz, 2) );

    f[0] = Pk[1];
    f[1] = -G*P2.M/(pow(R,2))*Px/R;
    f[2] = Pk[3];
    f[3] = -G*P2.M/(pow(R,2))*Py/R;
    f[4] = Pk[5];
    f[5] = -G*P2.M/(pow(R,2))*Pz/R;

    return f;
}

// Performs an RK4 integration step for all particles in a particle array.
void Particle_RK4(Particle p[], int n, double t, double h)
{
    std::valarray<std::valarray<double>> push (n);

    // Iterate over all particles in the simulation.
    for (int i=0; i<n; i++)
    {
        std::valarray<double> dp_total(N_EQ);     // The total step each particle must take after the integration.
        
        // Calculate the steps from each particle and sum them into one large step.
        for (int j=0; j<n; j++)
        {
            // Ignore forces from the particle on itself.
            if (i == j)
            {
                continue;
            }

            // Set up k1-k4.
            std::valarray<double> k1(N_EQ);
            std::valarray<double> k2(N_EQ);
            std::valarray<double> k3(N_EQ);
            std::valarray<double> k4(N_EQ);
            std::valarray<double> dy(N_EQ);

            // Calculate the four slopes for the step
            k1 = h*Particle_F(t, p[i], p[j]);
            k2 = h*Particle_F(t+0.5*h, p[i], p[j], 0.5*k1);
            k3 = h*Particle_F(t+0.5*h, p[i], p[j], 0.5*k2);
            k4 = h*Particle_F(t+h, p[i], p[j], k3);

            dp_total += (k1+2.0*k2+2.0*k3+k4)/6.0;
        }
        push[i] = dp_total;
    }

    // Once all forces for all particles are calculated, update the positions of the particles.
    for(int i=0; i<n; i++)
    {
        p[i].elements += push[i];
    }
}


// Performs an Leapfrog integration step for all particles in a particle array.
void Particle_Leapfrog(Particle p[], int n, double t, double h)
{


    // "Drift" step. Update particle positions using their velocities.
    for (int i=0; i<n; i++)
    {
        p[i].x += 0.5*h*p[i].vx;
        p[i].y += 0.5*h*p[i].vy;
        p[i].z += 0.5*h*p[i].vz;
    }

    // "Kick" step. Calculate forces over all particles in the simulation.
    std::valarray<std::valarray<double>> drifts (n);
    for (int i=0; i<n; i++)
    {
        std::valarray<double> dv_total(N_EQ);     // The total step each particle must take after the integration.
        
        // Calculate the steps from each particle and sum them into one large step.
        for (int j=0; j<n; j++)
        {
            // Ignore forces from the particle on itself.
            if (i == j)
            {
                continue;
            }

            dv_total += 1.0*h*Particle_F(t, p[i], p[j]);
        }
        drifts[i] = dv_total;
    }

    // Once all forces are calculated, update the velocities of the particles.
    for(int i=0; i<n; i++)
    {
        p[i].vx += drifts[i][1];
        p[i].vy += drifts[i][3];
        p[i].vz += drifts[i][5];

        // Second "Drift" step. Update particle positions using their (updated) velocities.
        p[i].x += 0.5*h*p[i].vx;
        p[i].y += 0.5*h*p[i].vy;
        p[i].z += 0.5*h*p[i].vz;
    }
}


 
// Performs a subcycled Leapfrog integration step for all particles in a particle array.
void Particle_Leapfrog_Subcycle(Particle p[], int n, double t, double dt, double dt_sub)
{

    // Start subcycling of the particle integrator.
    double t_sub = 0.0;
    while (t_sub < dt) {
        // Integrate exactly to dt if the final time goes past dt.
        dt_sub = (t_sub + dt_sub > dt) ? dt - t_sub : dt_sub;

        // "Drift" step. Update particle positions using their velocities.
        for (int i=0; i<n; i++)
        {
            p[i].x += 0.5*dt_sub*p[i].vx;
            p[i].y += 0.5*dt_sub*p[i].vy;
            p[i].z += 0.5*dt_sub*p[i].vz;
        }

        // "Kick" step. Calculate forces over all particles in the simulation.
        std::valarray<std::valarray<double>> drifts (n);
        for (int i=0; i<n; i++)
        {
            std::valarray<double> dv_total(N_EQ);     // The total step each particle must take after the integration.
            
            // Calculate the steps from each particle and sum them into one large step.
            for (int j=0; j<n; j++)
            {
                // Ignore forces from the particle on itself.
                if (i == j)
                {
                    continue;
                }

                dv_total += 1.0*dt_sub*Particle_F(t, p[i], p[j]);
            }
            drifts[i] = dv_total;
        }

        // Once all forces are calculated, update the velocities of the particles.
        for(int i=0; i<n; i++)
        {
            p[i].vx += drifts[i][1];
            p[i].vy += drifts[i][3];
            p[i].vz += drifts[i][5];

            // Second "Drift" step. Update particle positions using their (updated) velocities.
            p[i].x += 0.5*dt_sub*p[i].vx;
            p[i].y += 0.5*dt_sub*p[i].vy;
            p[i].z += 0.5*dt_sub*p[i].vz;
        }

        t_sub += dt_sub;
    }
}


// ============================================================================================================================
// Analytic Solutions for Circular and Binary Orbits
// ============================================================================================================================

// Analytic solution for a planet in a circular orbit.
void orbit(Particle p[], int n, double time)
{
  int i;
  for (i=0; i<n; ++i)
  {
    Particle P = p[i];
    // Set particle masses
    // Current mass function: m=0 for 20 orbits, then m grows by the sin^2 function for 20 more orbits,
    // reaching a max of 1 Jupiter masses (m=0.001)
    /*    
    double t_acc = 20.*2.*PI;
    double t_planet = 40.*2.*PI;
    double max_mass = 0.004;    // Set to mratio???
    if (i > 0)    // Only change the masses to the planets
    {
      
      if (time < t_acc) {
        P.M = 0.0;
      }
      else if (time > t_acc && time < t_planet) {
        P.M = max_mass*sin((time-t_acc)/20.0/4.0)*sin((time-t_acc)/20.0/4.0);
      }
      else
      {
	P.M = max_mass;
      }
      
      P.M = std::min(max_mass*time/t_acc, max_mass);
    }
    */    
    // Set particle positions
    // For the star (Particle 0), the position is fixed to the origin.
    double dis = sqrt(P.x*P.x+P.y*P.y);
    if (dis == 0.0)
    {
      P.x = 0;
      P.y = 0;
    }
    else
    {
      double ome=sqrt((G+P.M)/dis/dis/dis);
      double ang=acos(P.x/dis);
      ang = ome*time;
      P.x = dis*cos(ang);
      P.y = dis*sin(ang);
    }
  }
  return;
}


// ============================================================================================================================
// Miscellaneous status functions
// ============================================================================================================================

// Calculate the total energy of the system.
double Particle_TotalEnergy(Particle p[], int n)
{
    double E = 0.0;
    for(int i=0; i<n; i++)
    {
        E += 0.5*p[i].M*(p[i].vx*p[i].vx + p[i].vy*p[i].vy) - G*p[i].M/sqrt(p[i].x*p[i].x + p[i].y*p[i].y);
    }
    return E;
}


// Move the system to a center-of-mass frame of reference.
// This zeroes out the COM and COV for the entire system.
void move_to_com(Particle P[], int n = -1)
{
    // Number of particles to iterate over.  Defaults to the entire system (N_PARTICLES).
    int np = (n > 0) ? n : N_PARTICLES;

    // Calculate center of mass and center of velocity
    double Mtot = 0.0;
    double cmx = 0.0, cmy = 0.0, cmz = 0.0;
    double cvx = 0.0, cvy = 0.0, cvz = 0.0;
    for (int i=0; i<np; i++)
    {
        Mtot += P[i].M;
        cmx += P[i].M*P[i].x;
        cmy += P[i].M*P[i].y;
        cmz += P[i].M*P[i].z;
        cvx += P[i].M*P[i].vx;
        cvy += P[i].M*P[i].vy;
        cvz += P[i].M*P[i].vz;
    }

    //std::cout << "Center of Mass: " << cmx << ' ' << cmy << std::endl;
    //std::cout << "Center of Velocity: " << cvx << ' ' << cvy << std::endl;
    // Move each particle in the system.
    for (int i=0; i<np; i++)
    {
        P[i].x -= cmx/Mtot;
        P[i].y -= cmy/Mtot;
        P[i].z -= cmz/Mtot;
        P[i].vx -= cvx/Mtot;
        P[i].vy -= cvy/Mtot;
        P[i].vz -= cvz/Mtot;
    }
}

// Calculate the angualr momentum of the system and save the result in the array Ltot
void Particle_L(Particle Plist[], double* Ltot)
{
    double Ltotx = 0.0, Ltoty = 0.0, Ltotz = 0.0;
    for (int i=0; i<N_PARTICLES; ++i)
    {
        double Lx, Ly, Lz;

        Particle P = Plist[i];
        Lx = P.M * (P.y*P.vz - P.z*P.vy);
        Ly = -1.0 * P.M * (P.x*P.vz - P.z*P.vx);
        Lz = P.M * (P.x*P.vy - P.y*P.vx);

        Ltotx += Lx;
        Ltoty += Ly;
        Ltotz += Lz;
    }

    Ltot[0] = Ltotx;
    Ltot[1] = Ltoty;
    Ltot[2] = Ltotz;
}

// ------------------------------------------------------------------------------------------------
// Rotation Functions
// ------------------------------------------------------------------------------------------------
void Rx(Particle p[], double ang)
{
    double py, pz, pvy, pvz;
    for (int i=0; i<N_PARTICLES; i++)
    {
        Particle P = p[i];
        py = P.y;
        pz = P.z;
        pvy = P.vy;
        pvz = P.vz;

        P.y = py*cos(ang) - pz*sin(ang);
        P.z = py*sin(ang) + pz*cos(ang);
        P.vy = pvy*cos(ang) - pvz*sin(ang);
        P.vz = pvy*sin(ang) + pvz*cos(ang);
    }
}

void Rz(Particle p[], double ang)
{
    double px, py, pvx, pvy;
    for (int i=0; i<N_PARTICLES; i++)
    {
        Particle P = p[i];
        px = P.x;
        py = P.y;
        pvx = P.vx;
        pvy = P.vy;

        P.x = px*cos(ang) - py*sin(ang);
        P.y = px*sin(ang) + py*cos(ang);
        P.vx = pvx*cos(ang) - pvy*sin(ang);
        P.vy = pvx*sin(ang) + pvy*cos(ang);
    }
}