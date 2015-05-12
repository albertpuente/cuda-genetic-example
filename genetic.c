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

// Genetic algorithm parameters
#define N 512
#define N_POINTS 400
#define ITERATION_LIMIT 1000
#define GOAL_SCORE -1.0
#define POINT_SET_MUTATION_PROB 0.3
#define POINT_MUTATION_PROB 0.1
#define N_SURVIVORS N/4
#define COLLISION_DISTANCE 0.2
#define OBSTACLE_SIZE 2.0
#define MAX_DELTA 2
#define MAX_TRIES 1e4   // max amount of times we try to find a position for a point

// Deterministic algorithm (testing purposes)
#define SEED 27
# define RAND01 ((float)rand()/(float)(RAND_MAX))

// c++ style
typedef int bool;
#define true 1
#define false 0

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

inline bool randomChoice(float probability) {
    if ((float)rand()/(float)(RAND_MAX) <= probability) return true;
    else return false;
}

inline float dist(Point* a, Point* b) {
    return sqrt(pow(a->x - b->x, 2)+pow(a->y - b->y, 2)+pow(a->z - b->z, 2));
}


Point destination;
Point obstacle;

// checks whether the point p collides with any of the points in between
// PS[from] and PS[to], 'to' not included.
bool collides(Point* p, PointSet* PS, int from, int to) {
    for (int i = from; i < to; ++i) {
        if (dist(p, &PS->points[i]) < COLLISION_DISTANCE) {
            return true;
        }
    }
    if (dist(p, &obstacle) < OBSTACLE_SIZE) {
        return true;
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
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N_POINTS; ++j) {
            PointSet* PS = &(P->pointSets[i]);
            Point* p = &(PS->points[j]); // p is passed to 'collides' via PS
            p->x = (float)rand()/(float)(RAND_MAX/5.0) + 12.5;
            p->y = (float)rand()/(float)(RAND_MAX/5.0) + 12.5;
            p->z = (float)rand()/(float)(RAND_MAX/5.0) + 12.5;
            
            int try = 0;
            while (try < MAX_TRIES && collides(p, PS, 0, j)) {
                p->x = (float)rand()/(float)(RAND_MAX/5.0) + 12.5;
                p->y = (float)rand()/(float)(RAND_MAX/5.0) + 12.5;
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
}

inline float heur_1(Point* P) {
    return fabs(P->y - 3.0*sin(P->x/2.0)) + fabs(P->z - 3.0*cos(P->x/2.0));
}


inline float heur_2(Point* P) {
    float s = dist(P, &destination);
    if (s < OBSTACLE_SIZE * 1.1) s = 0.0;
    return s;
}

void evaluate(Population* P) {
    for (int i = 0; i < N; ++i) {
        PointSet* C = &P->pointSets[i];
        C->score = 0;
        for (int j = 0; j < N_POINTS; j++) {
            Point* E = &C->points[j];
            C->score += heur_2(E);
        } 
    }    
}

int cmpScores(const void* a, const void* b) { 
    PointSet* c1 = (PointSet*) a;
    PointSet* c2 = (PointSet*) b;
    if (c1->score < c2->score) return -1;
    else return (c1->score > c2->score);
}

void sort(Population* P) {
    qsort(P->pointSets, N, sizeof(PointSet), cmpScores);
    P->maxScore = P->pointSets[0].score;
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
}

void swap(Population** a, Population** b) {
   Population* c = *a;
   *a = *b;
   *b = c;  
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
}

void progressAnim(int it) {
    int i = 0;
    for (; i < (it*40)/ITERATION_LIMIT; ++i) printf("|");
    for (; i < 40; ++i) printf("·");
    printf(" %i%%\n", it*100/ITERATION_LIMIT);
}

void DUMPInitialParams() {
    printf("%i %i\n", N_POINTS, ITERATION_LIMIT);
}

void sequentialGenetic() {
    
    srand(SEED);
    
    destination.x = destination.y = destination.z = 0.0;
    obstacle = destination;
    
    
    Population* P = malloc(sizeof(Population));
    Population* Q = malloc(sizeof(Population));
    
    if (P == NULL || Q == NULL) {
        printf("ERROR: Failed to allocate %i KB.\n", 2*sizeof(Population)/1024);
        exit(EXIT_FAILURE);
    }
    if (DUMP) DUMPInitialParams();
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
    
    if (!DUMP) printf("Sequential genetic algorithm has finished.\n");
}

// CUDA

void cudaGenetic() {
    
    Population* P = malloc(sizeof(Population));
    
    if (P == NULL) {
        printf("ERROR: Failed to allocate %i KB (HOST).\n", sizeof(Population)/1024);
        exit(EXIT_FAILURE);
    }
    if (DUMP) DUMPInitialParams();
    
    generateInitialPopulation(P);
    
    // STEPS (idea)
    // 1. Copy the initial population to the device...
    //          1 thread for each entity => N threads 
    //          N/4 threads for each GPU (Multi GPU)
    // 2. After each copy, its "mutate" kernel can be called (STREAMS)
    // 3. Each "mutate" kernel creates his own "evaluate" kernel 
    //    after mutating (Dynamic parallelism)
    // 4. Launch cuda quicksort over the population
    // 5. Copy the maximum score to the CPU (if OK or it > ITERATION_LIMIT then GOTO )
    // 6. CPU calls copy CPU calls "reproduce" kernels at random elements.
    // ...
    // 
}

//

int main(int argc, char** argv) {
    
    DUMP = (argc == 1);
    
    sequentialGenetic();
    cudaGenetic();
    return 0;
}

