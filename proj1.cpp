#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <dirent.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

// print debug messages?
#ifndef DEBUG
#define DEBUG false
#endif

// set number of threads
#ifndef NUMT
#define NUMT 1
#endif

// set number of trials in the simulation
#ifndef NUMTRIALS
#define NUMTRIALS 50000 // for probability
#endif

// how many tries to discover the maximum performance
#ifndef NUMTRIES
#define NUMTRIES 5000  // for performance
#endif

// ranges for random numbers:
const float GMIN = 20.0;    // ground distance in meters
const float GMAX = 30.0;    // ground distance in meters
const float HMIN = 10.0;    // cliff height in meters
const float HMAX = 20.0;    // cliff height in meters
const float DMIN = 10.0;    // distance to castle in meters
const float DMAX = 20.0;    // distance to castle in meters
const float VMIN = 10.0;    // initial cannonball velocity in meters/second
const float VMAX = 30.0;    // initial cannonball velocity in meters/second
const float THMIN = 30.0;   // cannonball launch angle in degrees
const float THMAX = 70.0;   // cannonball launch angle in degrees
const float GRAVITY = -9.8; // acceleration due to gravity in meters / second^2
const float TOL = 5.0;      // tolerance in cannonball hitting the castle in meters; castle is destroyed if cannonball lands between d-TOL and d+TOL

// PROTOTYPES
float Ranf(float, float);
int Ranf(int, int);
void TimeOfDaySeed();

// degrees to radians
inline float Radians(float d)
{
    return (M_PI / 180.f) * d;
}

// FUNCTIONS
float Ranf(float low, float high)
{
    float r = (float) rand();
    float t = r / (float) RAND_MAX;

    return low + t * (high - low);
}

int Ranf(int ilow, int ihigh)
{
    float low = (float)ilow;
    float high = ceil((float)ihigh);
    return (int) Ranf(low,high);
}

void TimeOfDaySeed()
{
    struct tm y2k = {0};
    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;

    time_t timer;
    time(&timer);
    double seconds = difftime(timer, mktime(&y2k));
    unsigned int seed = (unsigned int)(1000.*seconds);
    srand(seed);
}

// MAIN
int main(int argc, char *argv[])
{
#ifndef _OPENMP
    fprintf(stderr, "No OpenMP Support!\n");
    return 1;
#endif

    TimeOfDaySeed(); // seed the random number generator

    omp_set_num_threads(NUMT); // set the number of threads to use in parallelizing the for-loop

    // better to define these here so that the rand() calls don't get into the thread timing
    float *vs = new float[NUMTRIALS];
    float *ths = new float[NUMTRIALS];
    float *gs = new float[NUMTRIALS];
    float *hs = new float[NUMTRIALS];
    float *ds = new float[NUMTRIALS];

    // fill the random-value arrays
    for (int n = 0; n < NUMTRIALS; n++)
    {
        vs[n] = Ranf(VMIN, VMAX);
        ths[n] = Ranf(THMIN, THMAX);
        gs[n] = Ranf(GMIN, GMAX);
        hs[n] = Ranf(HMIN, HMAX);
        ds[n] = Ranf(DMIN, DMAX);
    }

    // get ready to record the maximum performance and the probability
    double maxPerformance = 0.; // must be declared outside the NUMTRIES loop
    int numHits;

    // looking for maximum performance
    for (int tries = 0; tries < NUMTRIES; tries++)
    {
        double time0 = omp_get_wtime();
        numHits = 0;

        #pragma omp parallel for shared(vs, ths, gs, hs, ds), reduction(+:numHits)
        for (int n = 0; n < NUMTRIALS; n++)
        {
            // randomize everything
            float v = vs[n];
            float thr = Radians(ths[n]);
            float vx = v * cos(thr);
            float vy = v * sin(thr);
            float g = gs[n];
            float h = hs[n];
            float d = ds[n];

            // see if the ball doesn't even reach the cliff:
            float t = ((-vy - (sqrtf(vy*vy - (19.6f*h))))/GRAVITY);
            float x = vx * t;
            if (x <= g)
            {
                if (DEBUG) fprintf(stderr, "Ball didn't reach the cliff\n");
            }
            else{
                // see if the ball hits the vertical cliff face
                t = g / vx;
                float y = vy*t + 0.5*GRAVITY*(t*t);
                if (y <= h)
                {
                    if (DEBUG) fprintf(stderr, "Ball hit the cliff face\n");
                }
                else{
                    // ball hits the upper deck
                    // the time solution for this is a quadratic equation of the form:
                    // at^2 + bt + c = 0
                    float a = -4.9;
                    float b = vy;
                    float c = -h;
                    float disc = b * b - 4.f*a*c;
                    
                    // ball doesn't go as high as the upper deck (this shouldn't happen)
                    if (disc < 0.)
                    {
                        if (DEBUG) fprintf(stderr, "Ball didn't reach the upper deck\n");
                        exit(1);
                    }
                    // successfully hits the ground above the cliffs
                    // get the intersection
                    disc = sqrtf(disc);
                    float t1 = (-b + disc) / (2.f*a); // time to intersect high ground
                    float t2 = (-b - disc) / (2.f*a); // time to intersect high ground

                    // only care about second intersection
                    float tmax = t1;
                    if (t2 > t1)
                    {
                        tmax = t2;
                    }

                    // how far does the ball land horizontally from the edge of the cliff? 
                    float upperDist = vx * tmax - g;

                    // see if the ball hits the castle
                    if (fabs(upperDist - d) > TOL)
                    {
                        if (DEBUG) fprintf(stderr, "Missed the castle at upperDist = %8.3f\n", upperDist);
                    }
                    else
                    {
                        if (DEBUG) fprintf(stderr, "Hits the castle at upperDist = %8.3f\n", upperDist);
                        numHits++;
                    }
                } // if ball clears the cliff face
            } // if ball gets as far as the cliff face
        } // for (# of monte carlo trials)

        double time1 = omp_get_wtime();
        double megaTrialsPerSecond = (double)NUMTRIALS / (time1 - time0) / 1000000.;
        if (megaTrialsPerSecond > maxPerformance)
        {
            maxPerformance = megaTrialsPerSecond;
        }
    } // for (# of timing tries)

    float probability = (float)numHits / float(NUMTRIALS); // just get for the last NUMTRIES run
    fprintf(stderr, "%2d threads: %8d trials; probability = %6.2f%% ; megatrials/sec = %6.2f\n", NUMT, NUMTRIALS, 100.*probability, maxPerformance);

    // print to csv
    int fd;
    char filePath[] = "./A1results.csv";
    fd = open(filePath, O_WRONLY | O_CREAT | O_APPEND, 0640);
    char buff[15];
    sprintf(buff, "%f,", maxPerformance);
    write(fd, buff, strlen(buff));
    close(fd);


    return 0;
}