#include "GSimulation.hpp"
#include "cpu_time.hpp"

GSimulation::GSimulation()
{
    std::cout << "===============================" << std::endl;
    std::cout << " Initialize Gravity Simulation" << std::endl;
    set_npart(1000);
    set_nsteps(10);
    set_simtime(1);
}

void GSimulation::set_number_of_particles(int N)
{
    set_npart(N);
}

void GSimulation::set_number_of_steps(int N)
{
    set_nsteps(N);
}

void GSimulation::init_pos()
{
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<real_type> unif_d(0, 1.0);

    for (int i = 0; i < get_npart(); ++i)
    {
        particles[i].pos[0] = unif_d(gen);
        particles[i].pos[1] = unif_d(gen);
        particles[i].pos[2] = unif_d(gen);
    }
}

void GSimulation::init_vel()
{
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<real_type> unif_d(-1.0, 1.0);

    for (int i = 0; i < get_npart(); ++i)
    {
        particles[i].vel[0] = unif_d(gen) * 1.0e-1f;
        particles[i].vel[1] = unif_d(gen) * 1.0e-1f;
        particles[i].vel[2] = unif_d(gen) * 1.0e-1f;
    }
}

void GSimulation::init_acc()
{
    for (int i = 0; i < get_npart(); ++i)
    {
        particles[i].acc[0] = 0.f;
        particles[i].acc[1] = 0.f;
        particles[i].acc[2] = 0.f;
    }
}

void GSimulation::init_mass()
{
    real_type n = static_cast<real_type>(get_npart());
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<real_type> unif_d(0.0, 1.0);

    for (int i = 0; i < get_npart(); ++i)
    {
        particles[i].mass = n * unif_d(gen);
    }
}

// prevents explosion in the case the particles are really close to each other
static constexpr double softeningSquared = 1e-9;
static constexpr double G = 6.67259e-11;

//#pragma omp declare simd
void sum_with_correction(double& sum, double& value_to_add, double& correction)
{
    double corrected = value_to_add - correction;
    double new_sum = sum + corrected;
    correction = (new_sum - sum) - corrected;
    sum = new_sum;
}

void compute_impulse(Particle* particles, int n, double sum_impulse[])
{
    double correction[] = { 0, 0, 0 };
    int i = 0;
    //#pragma omp parallel for reduction(+ : sum_impulse[i])
    for (i = 0; i < n; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double curr_impulse = particles[i].mass * particles[i].vel[j];
            sum_with_correction(sum_impulse[j], curr_impulse, correction[j]);
        }
    }
}

double compute_k_energy(Particle* particles, int num_parts)
{
    double energy = 0.;
    double correction = 0.;
//#pragma omp parallel for reduction(+ : energy)
    for (int i = 0; i < num_parts; ++i)
    {
        double curr_energy = particles[i].mass * (particles[i].vel[0] * particles[i].vel[0] +
            particles[i].vel[1] * particles[i].vel[1] +
            particles[i].vel[2] * particles[i].vel[2]);
        sum_with_correction(energy, curr_energy, correction);
    }
    return energy / 2;
}

double compute_p_energy(Particle* particles, int num_parts)
{
    double p_energy = 0.;
    double correction = 0.;

    #pragma omp parallel for reduction(+ : p_energy)
    for (unsigned short i = 0; i < num_parts; ++i)
    {
        #pragma omp simd
    //#pragma vector aligned
    //#pragma vector always
        for (unsigned short j = 0; j < num_parts; ++j)
        {
            if (i == j)
                continue;

            double dx = particles[j].pos[0] - particles[i].pos[0];
            double dy = particles[j].pos[1] - particles[i].pos[1];
            double dz = particles[j].pos[2] - particles[i].pos[2];

            double distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared;
            double distanceInv = 1.0 / sqrt(distanceSqr);
            double curr_energy = -G * particles[j].mass * distanceInv * particles[i].mass;
            sum_with_correction(p_energy, curr_energy, correction);
        }
    }
    return p_energy / 2;
}

void computeAcc(Particle* src, Particle* temp, real_type step, int part_num) // part_num - число частиц
{
#pragma omp parallel for
    for (unsigned short i = 0; i < part_num; i++) // update acceleration
    {
        src[i].acc[0] = 0.;
        src[i].acc[1] = 0.;
        src[i].acc[2] = 0.; 

        #pragma omp simd
#pragma vector aligned
#pragma vector always
        for (unsigned short j = 0; j < part_num; j++)
        {
            if (i != j)
            {
                real_type dx, dy, dz;
                real_type distanceSqr = 0.0;
                real_type distanceInv = 0.0;

                dx = src[j].pos[0] - src[i].pos[0];
                dy = src[j].pos[1] - src[i].pos[1];
                dz = src[j].pos[2] - src[i].pos[2];

                distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared; 
                distanceInv = 1.0 / sqrt(distanceSqr);                        

                temp[i].acc[1] = src[i].acc[1] + dy * G * src[j].mass * distanceInv * distanceInv * distanceInv;//distanceInv * distanceInv * distanceInv;//pow(distanceInv, 3);
                temp[i].acc[0] = src[i].acc[0] + dx * G * src[j].mass * distanceInv * distanceInv * distanceInv;//distanceInv * distanceInv * distanceInv;//pow(distanceInv, 3);
                temp[i].acc[2] = src[i].acc[2] + dz * G * src[j].mass * distanceInv * distanceInv * distanceInv;//distanceInv * distanceInv * distanceInv;//pow(distanceInv, 3);
            }
        }
    }
}

void updatePos(Particle* dist, Particle* source, Particle* newparts, real_type step, int part_num)
{
#pragma omp parallel for
    for (int i = 0; i < part_num; ++i) // update position and velocity
    {

        newparts[i].vel[0] += newparts[i].acc[0] * step;
        newparts[i].vel[1] += newparts[i].acc[1] * step;
        newparts[i].vel[2] += newparts[i].acc[2] * step;

        dist[i].pos[0] = source[i].pos[0] + newparts[i].vel[0] * step;
        dist[i].pos[1] = source[i].pos[1] + newparts[i].vel[1] * step;
        dist[i].pos[2] = source[i].pos[2] + newparts[i].vel[2] * step;

        dist[i].vel[0] = source[i].vel[0] + newparts[i].acc[0] * step;
        dist[i].vel[1] = source[i].vel[1] + newparts[i].acc[1] * step;
        dist[i].vel[2] = source[i].vel[2] + newparts[i].acc[2] * step;
    }
}

void GSimulation::start()
{
    init_tstep();
    real_type energy_k, energy_p;
    real_type dt = get_tstep();
    int n = get_npart();

    particles = new Particle[n];

    init_pos();
    init_vel();
    init_acc();
    init_mass();

    energy_k = compute_k_energy(particles, n);
    energy_p = compute_p_energy(particles, n);
    _energy = energy_k + energy_p;
    double impulse[] = { 0, 0, 0 };
    compute_impulse(particles, n, impulse);
    _impulse = sqrt(pow(impulse[0], 2) + pow(impulse[1], 2) + pow(impulse[2], 2));

    std::cout << "Initial system energy k: " << energy_k << " p:" << energy_p << " Sum: " << _energy << " Impulse: " << _impulse << std::endl;

    print_header();

    double _totTime = 0.;

    CPUTime time;
    double ts0 = 0;
    double ts1 = 0;

    const double t0 = time.start();
    for (int s = 1; s <= get_nsteps(); ++s)
    {
        ts0 += time.start();

        // 
        Particle* k1 = new Particle[n];
        computeAcc(particles, k1, dt * 0, n);
        updatePos(particles, particles, k1, dt * 0, n);

        Particle* k2 = new Particle[n];
        computeAcc(particles, k2, dt / 2, n);
        updatePos(particles, particles, k2, dt / 2, n);

        Particle* k3 = new Particle[n];
        computeAcc(particles, k3, dt * 2 / 3, n);
        updatePos(particles, particles, k3, dt * 2 / 3, n);

        updatePos(particles, particles, k1, dt / 4, n);
        updatePos(particles, particles, k2, dt * 0, n);
        updatePos(particles, particles, k3, dt * 3 / 4, n);

        delete[] k1;
        delete[] k2;
        delete[] k3;

        energy_k = compute_k_energy(particles, n);
        energy_p = compute_p_energy(particles, n);

        double curr_energy = energy_k + energy_p;
        impulse[0] = impulse[1] = impulse[2] = 0;
        compute_impulse(particles, n, impulse);

        double curr_impulse = sqrt(impulse[0] * impulse[0] + impulse[1] * impulse[1] + impulse[2] * impulse[2]);

        ts1 += time.stop();
        // if (!(s % get_sfreq()))
        {
            std::cout << " "
                << std::left << std::setw(8) << s
                << std::left << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(9) << std::setw(16) << fabs(100 * (curr_energy - _energy) / _energy)
                << std::left << std::setprecision(9) << std::setw(16) << fabs(100 * (curr_impulse - _impulse) / _impulse)
                << std::left << std::setprecision(5) << std::setw(16) << (ts1 - ts0)
                << std::endl;
            ts0 = 0;
            ts1 = 0;
        }
        // _energy = curr_energy;
        // _impulse = curr_impulse;
    } // end of the time step loop

    const double t1 = time.stop();
    _totTime = (t1 - t0);

    delete[] particles;

    std::cout << std::endl;
    std::cout << "# Total Time (s)     : " << _totTime << std::endl;
    std::cout << "===============================" << std::endl;
}

void GSimulation::print_header()
{

    std::cout << " nPart = " << get_npart() << "; "
        << "nSteps = " << get_nsteps() << "; "
        << "dt = " << get_tstep() << std::endl;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << " "
        << std::left << std::setw(8) << "s"
        << std::left << std::setw(8) << "dt"
        << std::left << std::setw(16) << "s_energy"
        << std::left << std::setw(16) << "impulse"
        << std::left << std::setw(16) << "time (s)"
        << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

GSimulation ::~GSimulation()
{
}