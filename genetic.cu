/*
Compile with: 

    gcc genetic.c -o genetic -O2 -lm -std=c99 

    -O2      Optimization
    -lm      Link to math lib
    -std=c99 Use of for(;;;) with declaration among other things

Usage (3D viewer):

    ./genetic > data && ./geneticViewer data
    
Usage (debug):

    ./genetic

Jan Mas Rovira
Andrés Mingorance López
Albert Puente Encinas
*/

#include <stdio.h>  // e.g. printf
#include <stdlib.h> // e.g. malloc, RAND_MAX, exit
#include <math.h>   // e.g. sin, abs
#include <sys/time.h>

// Genetic algorithm parameters
#define N 1024
#define N_POINTS 64
#define ITERATION_LIMIT 1000
#define GOAL_SCORE -1.0
#define POINT_SET_MUTATION_PROB 0.5
#define POINT_MUTATION_PROB 0.01
#define N_SURVIVORS N/4
#define POINT_RADIUS 0.25
#define OBSTACLE_RADIUS 2.0
#define MAX_DELTA 2
#define MAX_TRIES 1e3   // max amount of times we try to find a position for a point

// Obstacles
#define CHECK_OBSTACLES true
#define CHECK_COLLISIONS true

// Deterministic algorithm (testing purposes)
#define SEED 27
# define RAND01 ((float)rand()/(float)(RAND_MAX))

// c++ style
typedef int bool;
#define true 1
#define false 0

// Timers
unsigned long long mutationTime;
unsigned long long reproductionTime;
unsigned long long sortingTime;
unsigned long long evaluationTime;
unsigned long long initialGenTime;
unsigned long long totalTime;

inline void tic(unsigned long long* time) {
    struct timeval t;
    gettimeofday(&t, NULL);
    *time = t.tv_sec*1000000 + t.tv_usec - *time;
    
}
#define toc tic

// Output toggles
bool DUMP;

typedef struct {
    float x, y, z; // Position
} Point;

typedef struct {
    Point points[N_POINTS];
    float score;
} PointSet;

typedef struct {
    PointSet pointSets[N];
    float maxScore;
} Population;

typedef struct {
    Point centre;
    float radius;
} Obstacle;

#define N_OBSTACLES 27
Obstacle obstacles[N_OBSTACLES];

inline bool randomChoice(float probability) {
    if ((float)rand()/(float)(RAND_MAX) <= probability) return true;
    else return false;
}

inline float dist(Point* a, Point* b) {
    return sqrt(pow(a->x - b->x, 2)+pow(a->y - b->y, 2)+pow(a->z - b->z, 2));
}


Point destination;

// checks whether the point p collides with any of the points in between
// PS[from] and PS[to], 'to' not included.
bool collides(Point* p, PointSet* PS, int from, int to) {
    if (CHECK_COLLISIONS)
        for (int i = from; i < to; ++i) {
            if (dist(p, &PS->points[i]) < POINT_RADIUS*2) {
                return true;
            }
        }
    if (CHECK_OBSTACLES)
        for (int i = 0; i < N_OBSTACLES; ++i) {
            Obstacle o = obstacles[i];
            if (dist(p, &o.centre) < POINT_RADIUS + o.radius) {
                return true;
            }
        }
    return false;
}

// compute collisions only up to QS[ixp], since the rest
// haven't mutated yet
// also compute collisions from PS[ixp+1] onwards to avoid inconsistencies
// like a point being trapped by the ones that have moved before
/*
bool collides(PointSet* PS, PointSet* QS, int ixq, int ixp) {
    for (int i = 0; i < ixq; ++i) {
        if (dist(&PS->points[ixp], &QS->points[i]) < COLLISION_DISTANCE) {
            return true;
        }
    }
    for (int i = ixq + 1; i < ixp; ++i) {
        if (dist(&PS->points[ixp], &PS->points[i]) < COLLISION_DISTANCE) {
            return true;
        }
    }
    return false;
}
*/

void generateInitialPopulation(Population* P) {
    tic(&initialGenTime);
    
    float range = POINT_RADIUS * pow((float)N_POINTS, 1.0/3.0) * 10;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N_POINTS; ++j) {
            PointSet* PS = &(P->pointSets[i]);
            Point* p = &(PS->points[j]); // p is passed to 'collides' via PS
            p->x = (float)rand()/(float)(RAND_MAX/range) + 12.5; //kappa
            p->y = (float)rand()/(float)(RAND_MAX/range) + 12.5;
            p->z = (float)rand()/(float)(RAND_MAX/range) + 12.5;
            
            int try = 0;
            while (try < MAX_TRIES && collides(p, PS, 0, j)) {
                p->x = (float)rand()/(float)(RAND_MAX/range) + 12.5;
                p->y = (float)rand()/(float)(RAND_MAX/range) + 12.5;
                p->z = (float)rand()/(float)(RAND_MAX/5.0) + 12.5;
                ++try;
            }
            if (try == MAX_TRIES) {
                printf("Error during the generation of the initial population\n");
                exit(1);
            }
            //P->pointSets[i].points[j] = p;
        }
    }
    
    toc(&initialGenTime);
}

inline float heur_1(Point* P) {
    return fabs(P->y - 3.0*sin(P->x/2.0)) + fabs(P->z - 3.0*cos(P->x/2.0));
}


inline float heur_2(Point* P) {
    return dist(P, &destination);
}

void evaluate(Population* P) {
    tic(&evaluationTime);
    
    for (int i = 0; i < N; ++i) {
        PointSet* C = &P->pointSets[i];
        C->score = 0;
        for (int j = 0; j < N_POINTS; j++) {
            Point* E = &C->points[j];
            C->score += heur_2(E);
        } 
    }    
    
    toc(&evaluationTime);
}

int cmpScores(const void* a, const void* b) { 
    PointSet* c1 = (PointSet*) a;
    PointSet* c2 = (PointSet*) b;
    if (c1->score < c2->score) return -1;
    else return (c1->score > c2->score);
}

void sort(Population* P) {
    tic(&sortingTime);
    
    qsort(P->pointSets, N, sizeof(PointSet), cmpScores);
    P->maxScore = P->pointSets[0].score;
    
    toc(&sortingTime);
}

void mix(PointSet* AP, PointSet* AQ) {

    for (int i = 0; i < N_POINTS; ++i) {
        
        if (!randomChoice(POINT_MUTATION_PROB)) {
            AQ->points[i] = AP->points[i];
            continue;
        }           
   
        int try = 0;
        Point p;
        while (try < MAX_TRIES) {            
            // Choose a reference point
            int j = rand()%N_POINTS;
            
            // Calculate the direction from AP[i] to AP[j]
            float dx =  AP->points[j].x - AP->points[i].x;
            float dy =  AP->points[j].y - AP->points[i].y;
            float dz =  AP->points[j].z - AP->points[i].z;
            // "Normalization" ||direction|| = 0.5
            float norm = sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));
            norm *= (1.0/MAX_DELTA);
            norm /= RAND01; // move a random portion of MAX_DELTA
            if (norm < 1e-4 && norm > -1e-4) {
                dx = 0;
                dy = 0;
                dz = 0;
            }
            else {
                dx /= norm;
                dy /= norm;  
                dz /= norm;   
            }       
            
            // 50% of getting closer, 50% of getting further away from the ref point
            if (randomChoice(0.5)) {
                p.x = AP->points[i].x + dx;
                p.y = AP->points[i].y + dy;
                p.z = AP->points[i].z + dz;
            }
            else {
                p.x = AP->points[i].x - dx;
                p.y = AP->points[i].y - dy;
                p.z = AP->points[i].z - dz;
            }
            // if the point doesn't collide with a point that has already moved
            if (!collides(&p, AQ, 0, i) &&
                // and it doesn't collide with a point that has yet to be moved
                // (this 2nd check prevents inconsistencies like a point being unable to move at all)
                !collides(&p, AP, i + 1, N_POINTS))
                    break;
            ++try;
        }
        if (try == MAX_TRIES) {
            //printf("Error during the mix() of points\n");
            //exit(1);
            p = AP->points[i];
        }
        AQ->points[i] = p;
    }
}

void randomMove(PointSet* AP, PointSet* AQ) {
    for (int i = 0; i < N_POINTS; ++i) {
        
        if (!randomChoice(POINT_MUTATION_PROB)) {
            AQ->points[i] = AP->points[i];
            continue;
        }
        int try = 0;
        Point p;
        while (try < MAX_TRIES) {
            p.x = AP->points[i].x + (RAND01-0.5)*2*MAX_DELTA;
            p.y = AP->points[i].y + (RAND01-0.5)*2*MAX_DELTA;
            p.z = AP->points[i].z + (RAND01-0.5)*2*MAX_DELTA;
            // if the point doesn't collide with a point that has already moved
            if (!collides(&p, AQ, 0, i) &&
                // and it doesn't collide with a point that has yet to be moved
                // (this 2nd check prevents inconsistencies like a point being unable to move at all)
                !collides(&p, AP, i + 1, N_POINTS))
                    break;
            ++try;
        }
        if (try == MAX_TRIES) {
            //printf("Error during the mix() of points\n");
            //exit(1);
            p = AP->points[i];
        }
        AQ->points[i] = p;
    } 
}

// Q = mutation of the X% best portion of P
// llegeix de P, escriu a Q
void mutate(Population* P, Population* Q) {
    tic(&mutationTime);
    
    for (int i = 0; i < N; ++i) {
        PointSet* AP = &P->pointSets[i];    // original points
        PointSet* AQ = &Q->pointSets[i];    // mutated points
        if (randomChoice(POINT_SET_MUTATION_PROB)) { // Mutate
            int type = rand()%2;
            if (type == 0) { // Mix of two
                mix(AP, AQ);
            }
            else if (type == 1) {
                randomMove(AP, AQ);;
            }
        }        
        else { // Copy
            *AQ = *AP;
        }
    }
    
    toc(&mutationTime);
}

void dump(PointSet* C) {
    for (int i = 0; i < N_POINTS; ++i) {
        printf("%f %f %f\n", C->points[i].x, C->points[i].y, C->points[i].z);
    }
}

void pork(PointSet* p1, PointSet* p2, PointSet* child) {
    for (int i = 0; i < N_POINTS; ++i) {
        Point* p = &child->points[i];        
        if (randomChoice(0.5)) {
            *p = p1->points[i];
        }
        else {
            *p = p2->points[i];
        }
    }
}

void reproduce(Population* P, Population* Q) {
    tic(&reproductionTime);
    
    int i;
    for (i = 0; i < N_SURVIVORS; ++i) {
        Q->pointSets[i] = P->pointSets[i];
    }
    
    for (; i < N; ++i) {
        PointSet* p1 = &P->pointSets[rand()%N_SURVIVORS];
        PointSet* p2 = &P->pointSets[rand()%N_SURVIVORS];        
        PointSet* child = &Q->pointSets[i];
        pork(p1, p2, child);
    }
    
    toc(&reproductionTime);
}

void progressAnim(int it) {
    int i = 0;
    for (; i < (it*40)/ITERATION_LIMIT; ++i) printf("|");
    for (; i < 40; ++i) printf("·");
    printf(" %i%%\n", it*100/ITERATION_LIMIT);
}

void DUMPInitialParams() {
    printf("%i\n", N_OBSTACLES);
    for (int i = 0; i < N_OBSTACLES; ++i) {
        Obstacle o = obstacles[i];
        printf("%f %f %f %f\n", o.centre.x, o.centre.y, o.centre.z, o.radius); 
    }
    printf("%i %i\n", N_POINTS, ITERATION_LIMIT);
    
}

void initTimes() {
    initialGenTime = mutationTime = evaluationTime = sortingTime = reproductionTime = 0;
}

void printTimes() {
    printf("Sequential genetic algorithm has finished:\n");
    printf("    Init gen:     %f s.\n", (double)initialGenTime/1000000);
    printf("    Mutations:    %f s.\n", (double)mutationTime/1000000);
    printf("    Evaluations:  %f s.\n", (double)evaluationTime/1000000);
    printf("    Sorting:      %f s.\n", (double)sortingTime/1000000);
    printf("    Reproduction: %f s.\n", (double)reproductionTime/1000000);
    printf("    Total time:   %f s.\n", (double)totalTime/1000000);
}

void sequentialGenetic() {
    tic(&totalTime);
    
    srand(SEED);
    
    destination.x = destination.y = destination.z = 0.0;    
    
    Population* P = malloc(sizeof(Population));
    Population* Q = malloc(sizeof(Population));
    
    if (P == NULL || Q == NULL) {
        printf("ERROR: Failed to allocate %i KB.\n", 2*sizeof(Population)/1024);
        exit(EXIT_FAILURE);
    }
    if (DUMP) DUMPInitialParams();
    else initTimes();
    
    generateInitialPopulation(P);
    
    int it = 0;
    while (true) {
        mutate(P, Q);
        evaluate(Q);
        sort(Q);
        
        
        if (DUMP) dump(&Q->pointSets[0]);
        else {            
            printf("\nIt: %i/%i Score: %f -> %f\n", 
                   it, ITERATION_LIMIT, Q->maxScore, GOAL_SCORE);
            progressAnim(it);
        }
        
        if (it >= ITERATION_LIMIT || Q->maxScore <= GOAL_SCORE) 
            break;
        
        // reproduce replaces the worst candidates with combinations
        // of better ones. from Q to P, so the population ends up in P
        // prepared for the next iteration
        reproduce(Q, P);
        it++;
    }
    
    toc(&totalTime);
    
    if (!DUMP) printTimes();
}

void initObstacles() {
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                Point origin;
                origin.x = 3*i;
                origin.y = 3*j;
                origin.z = 3*k;

                obstacles[i*9 + j*3 + k].centre = origin;
                obstacles[i*9 + j*3 + k].radius = 1.0;
            }
        }
    }    
}

int main(int argc, char** argv) {
    DUMP = (argc == 1);
    initObstacles();
    sequentialGenetic();
    return 0;
}

