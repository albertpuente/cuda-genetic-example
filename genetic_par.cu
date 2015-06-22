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
#include <cuda.h>
#include <curand_kernel.h>
#define CURAND curand_uniform(&localState)

// Genetic algorithm parameters
#define N 1024*10
#define N_POINTS 1024*10
#define ITERATION_LIMIT 200
#define GOAL_SCORE -1.0
#define POINT_SET_MUTATION_PROB 0.5
#define POINT_MUTATION_PROB 0.01
#define N_SURVIVORS N/4
#define POINT_RADIUS 0.25
#define OBSTACLE_RADIUS 2.0
#define MAX_DELTA 2

// Deterministic algorithm (testing purposes)
#define SEED 27
#define RAND01 ((float)rand()/(float)(RAND_MAX))

// c++ style
#define bool int
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
//inline void toc(unsigned long long* time) { tic(time); }

// Output toggles
bool DUMP;

typedef struct {
    float x, y, z; // Position
    float score;
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

// Obstacles
#define CHECK_OBSTACLES true
#define N_OBSTACLES 27
Obstacle cpu_obstacles[N_OBSTACLES];

// target position our points try to get to
Point destination;

// CUDA Variables
unsigned int nThreads = 1024;
unsigned int nBlocks = N/nThreads;  // N multiple de nThreads
    
// GPU Pointers
Obstacle* gpu_obstacles;
Point* gpu_destination;
curandState *devStates;


// returns true with probability=probability(0,1), false otherwise
__device__ inline bool cuda_randomChoice(float probability, curandState* localState) {
    if (curand_uniform(localState) <= probability) return true;
    else return false;    
}

// check if an error occured, print msg
void checkCudaError(char msg[]) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error) {
        printf("Error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

// squared dist, faster than euclidean dist and serves the same purpose
__device__ inline float cuda_squared_dist(Point* a, Point* b) {
    return (float) (a->x-b->x)*(a->x-b->x)+(a->y-b->y)*(a->y-b->y)+(a->z-b->z)*(a->z-b->z);
}

// check if point collides with any of the obstacles
__device__ bool collidesWithObstacles(Point* p, Obstacle* obstacles) {
    if (!CHECK_OBSTACLES) return false;
    for (int i = 0; i < N_OBSTACLES; ++i) {
        Obstacle o = obstacles[i];
        //                                    mult. by itself since we compare to squared dist
        if (cuda_squared_dist(p, &o.centre) < (POINT_RADIUS + o.radius)*(POINT_RADIUS + o.radius)) {
            return true;
        }
    }
    return false;
}

__global__ void kernel_generatePointSetMembers(PointSet* PS, float range, Obstacle* obstacles, curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[id]; // al threads share this atm
    
    Point* p = &(PS->points[id]);
    p->x = CURAND * range + 12.5;
    p->y = CURAND * range + 12.5;
    p->z = CURAND * range + 12.5;
    
    // prolly inefficient af
    while (collidesWithObstacles(p, obstacles)) {
        p->x = CURAND * range + 12.5;
        p->y = CURAND * range + 12.5;
        p->z = CURAND * range + 12.5;
    }
}

// initial population generation kernel
__global__ void kernel_generateInitialPopulation(Population* P, 
                    Obstacle* obstacles, int* idxs, curandState* state) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    PointSet* PS = &(P->pointSets[id]);
    curandState* localState = &(state[id*N_POINTS]);
    // Indexs initialization
    idxs[id] = id;

    int n_threads = 128; //?
    int n_blocks = N_POINTS / n_threads;
    float range = POINT_RADIUS * pow((float)N_POINTS, 1.0f/3.0f) * 10;   
    
    kernel_generatePointSetMembers<<<n_threads, n_blocks>>>(PS, range, obstacles, localState);
}

__global__ void kernel_randSetup(curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init ( 1234, id, 0, &state[id] );
}

void generateInitialPopulation(Population* gpu_P, int* gpu_idxs) {
    tic(&initialGenTime);
    
    //RANDOM SETUP
    cudaMalloc((void **)&devStates, N * N_POINTS * sizeof(curandState));
    kernel_randSetup<<<nBlocks, nThreads>>>(devStates);
    checkCudaError((char *) "generateInitialPopulation: random setup kernel");    
    //RANDOM END
    
    // kernel 
    kernel_generateInitialPopulation<<<nBlocks, nThreads>>>(gpu_P, gpu_obstacles, 
                                                            gpu_idxs, devStates);
    checkCudaError((char *) "kernel call in generateInitialPopulation");
    
    // wait
    cudaDeviceSynchronize();
    toc(&initialGenTime);
}

__device__ inline float heur_1(Point* P) {
    return fabs(P->y - 3.0*sin(P->x/2.0)) + fabs(P->z - 3.0*cos(P->x/2.0));
}

__device__ inline float heur_2(Point* P, Point* destination) {
    return cuda_squared_dist(P, destination);
}


/* ---------------------USING THESE IS SLOW ---------------------*/
__global__ void kernel_evaluatePointSet(PointSet* PS, Point* destination) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    Point* P = &(PS->points[id]);
    P->score = heur_2(P, destination);
    atomicAdd(&(PS->score), P->score);
}

__global__ void kernel_evaluatePopulation(Population* P, Point* destination) {    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    PointSet* PS = &P->pointSets[id];
    int n_threads = 1024; //?
    int n_blocks = N_POINTS / n_threads;
    PS->score = 0;
    kernel_evaluatePointSet<<<n_threads, n_blocks>>>(PS, destination);
    
    /* alternative to atomicAdd in kernel_evaluatePointSet, just as slow
     * reduction would be another alternative, probably too slow too.
    cudaDeviceSynchronize();
    
    int score = 0;
    for (int j = 0; j < N_POINTS; j++) {
       score += PS[j].score;
    }
    PS->score = score;
    */
}
/* --------------------------------------------------------------- */

__global__ void kernel_evaluatePopulationFlat(Population* P, Point* destination) {    
    int ps_id = blockIdx.x * blockDim.x + threadIdx.x;
    int point_id = blockIdx.y * blockDim.y + threadIdx.y;
    
    PointSet* PS = &P->pointSets[ps_id];
    Point* p = &PS->points[point_id];
    
    if (point_id == 0) {
        PS->score = 0;        
    }
    __syncthreads();
    
    float point_heur = heur_2(p, destination);
    atomicAdd(&PS->score, point_heur);
}

void evaluate(Population* gpu_P) {
    tic(&evaluationTime);
    
    bool flat = false;
    if (flat) {
        int n_threads = 32; //sqrt(1024)
        int width = N / n_threads;
        int height = N_POINTS / n_threads;
        // kernel
        dim3 gridSize(width, height, 1);
        dim3 blockSize(n_threads, n_threads, 1);
        kernel_evaluatePopulationFlat<<<gridSize, blockSize>>>(gpu_P, gpu_destination);
        checkCudaError((char *) "kernel call in generateInitialPopulation");
    }
    else {
        // kernel 
        kernel_evaluatePopulation<<<nBlocks, nThreads>>>(gpu_P, gpu_destination);
        checkCudaError((char *) "kernel call in generateInitialPopulation");
    }
    
    // wait
    cudaDeviceSynchronize();
    toc(&evaluationTime);
}

//////////////////////////////////////////
//////////////  CUDA QUICK SORT
//////         

#define MAX_DEPTH       16
#define INSERTION_SORT  32

// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ void selection_sort(Population* P, int* idxs, int left, int right ) {
    for (int i = left ; i <= right ; ++i) {
        float min_score = P->pointSets[ idxs[i] ].score;
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1 ; j <= right ; ++j) {
            float score_j = P->pointSets[ idxs[j] ].score;
            if (score_j < min_score) {
                min_idx = j;
                min_score = score_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            int aux = idxs[i];
            idxs[i] = idxs[min_idx];
            idxs[min_idx] = aux;
        }
    }
}

__global__ void dynamic_quicksort(Population* P, int* idxs, int left, int right, int depth) {
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(P, idxs, left, right);
        return;
    }
    
    int lindex = left;
    int rindex = right;
    float pscore = P->pointSets[ idxs[(left+right)/2] ].score; // Pivot

    // Do the partitioning.
    while (lindex <= rindex) {
        // Find the next left- and right-hand values to swap
        float lscore = P->pointSets[ idxs[lindex] ].score; 
        float rscore = P->pointSets[ idxs[rindex] ].score;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lscore < pscore) {
            lindex++;
            lscore = P->pointSets[ idxs[lindex] ].score; 
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rscore > pscore) {
            rindex--;
            rscore = P->pointSets[ idxs[rindex] ].score;
        }

        // If the swap points are valid, do the swap!
        if (lindex <= rindex) {
            
            int aux = idxs[lindex];
            idxs[lindex] = idxs[rindex];
            idxs[rindex] = aux;
            
            lindex++;
            rindex--;
        }
    }

    // Now the recursive part
    // Launch a new block to sort the left part.
    if (left < rindex) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        dynamic_quicksort<<< 1, 1, 0, s >>>(P, idxs, left, rindex, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if (lindex < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        dynamic_quicksort<<< 1, 1, 0, s1 >>>(P, idxs, lindex, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

__global__ void copyBestPointSet(Population* P, int* idxs, PointSet* best) {
    *best = P->pointSets[ idxs[0] ];
}

__global__ void checkSort(Population* P, int* idxs) {
    for (int i = 1; i < N; ++i) {
       float a = P->pointSets[ idxs[i - 1] ].score;
       float b = P->pointSets[ idxs[i] ].score;
       if (a > b) printf("SORT IS NOT WORKING\n");
    }
}


void sort(Population* gpu_P, int* gpu_idxs, PointSet* best) {
    tic(&sortingTime);
    
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    
    dynamic_quicksort<<<1, 1>>>(gpu_P, gpu_idxs, 0, N-1, 0);
    checkCudaError((char *) "kernel call in sort");
    cudaDeviceSynchronize();
   
    //checkSort<<<1, 1>>>(gpu_P, gpu_idxs);
    //cudaDeviceSynchronize();
    //checkCudaError((char *) "check sort");
    
    PointSet* gpu_best;
    cudaMalloc(&gpu_best, sizeof(PointSet));
    checkCudaError((char *) "cudaMalloc bestPointSet");
    
    copyBestPointSet<<<1, 1>>>(gpu_P, gpu_idxs, gpu_best);
    checkCudaError((char *) "kernel copy best pointset");
    cudaDeviceSynchronize();
    
    cudaMemcpy(best, gpu_best, sizeof(PointSet), cudaMemcpyDeviceToHost);
    checkCudaError((char *) "copy of best point set");
    cudaDeviceSynchronize();
    
    toc(&sortingTime);
}

__device__ void mix(PointSet* AP, PointSet* AQ, Obstacle* obstacles, 
                                                    curandState* localState) {

    for (int i = 0; i < N_POINTS; ++i) {
        
        if (!cuda_randomChoice(POINT_MUTATION_PROB, localState)) {
            AQ->points[i] = AP->points[i];
            continue;
        }           
   
        int tries = 0;
        int MAX_TRIES = 100;
        Point p;
        while (tries < MAX_TRIES) {            
            // Choose a reference point
            int j = curand_uniform(localState)*(N_POINTS-1);
            
            // Calculate the direction from AP[i] to AP[j]
            float dx =  AP->points[j].x - AP->points[i].x;
            float dy =  AP->points[j].y - AP->points[i].y;
            float dz =  AP->points[j].z - AP->points[i].z;
            // "Normalization" ||direction|| = 0.5
            float norm = sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));
            norm *= (1.0/MAX_DELTA);
            norm /= curand_uniform(localState); // move a random portion of MAX_DELTA
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
            if (cuda_randomChoice(0.5f, localState)) {
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
            if (!collidesWithObstacles(&p, obstacles))
                break;
            ++tries;
        }
        if (tries == MAX_TRIES) {
            //printf("Error during the mix() of points\n");
            //exit(1);
            p = AP->points[i];
        }
        AQ->points[i] = p;
    }
}

__device__ void randomMove(PointSet* AP, PointSet* AQ, Obstacle* obstacles, 
                                                curandState* localState) {
    for (int i = 0; i < N_POINTS; ++i) {
        
        if (!cuda_randomChoice(POINT_MUTATION_PROB, localState)) {
            AQ->points[i] = AP->points[i];
            continue;
        }
        int tries = 0;
        int MAX_TRIES = 100;
        Point p;
        while (tries < MAX_TRIES) {
            p.x = AP->points[i].x + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
            p.y = AP->points[i].y + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
            p.z = AP->points[i].z + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
            // if the point doesn't collide with a point that has already moved
            if (!collidesWithObstacles(&p, obstacles))
                break;
            ++tries;
        }
        if (tries == MAX_TRIES) {
            //printf("Error during the mix() of points\n");
            //exit(1);
            p = AP->points[i];
        }

        AQ->points[i] = p;
    } 
}

__global__ void kernel_mutate(Population* P, Population* Q, Obstacle* obstacles, 
                                                             curandState* state) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState localState = state[id];
    
    PointSet* AP = &P->pointSets[id];    // original points
    PointSet* AQ = &Q->pointSets[id];    // mutated points    
    
    if (cuda_randomChoice(POINT_SET_MUTATION_PROB, &localState)) { // Mutate
        if (cuda_randomChoice(0.5f, &localState)) {
            mix(AP, AQ, obstacles, &localState);
        }
        else {
            randomMove(AP, AQ, obstacles, &localState);
        }
    }        
    else { // Copy
        *AQ = *AP;
    }
}
__device__ void mixFlat(Point* P, PointSet* AP, Point* Q, Obstacle* obstacles, curandState* localState) {

    Point p;
    do {            
        // Choose a reference point
        int j = curand_uniform(localState)*(N_POINTS-1);
        
        // Calculate the direction from AP[i] to AP[j]
        float dx =  P->x - AP->points[j].x;
        float dy =  P->y - AP->points[j].y;
        float dz =  P->z - AP->points[j].z;
        // pseudonormalization
        float norm = sqrt(dx*dx + dy*dy + dz*dz);
        norm *= 1.0/(MAX_DELTA);
        norm /= curand_uniform(localState); // move a random portion of MAX_DELTA
        float val = 1.0 / norm;
        if (norm < 1e-4 && norm > -1e-4) val = 0;
        dx *= val;
        dy *= val;  
        dz *= val;
            
        // 50% of getting closer, 50% of getting further away from the ref point
        int closer = 1;
        if (cuda_randomChoice(0.5f, localState)) closer = -1;
        p.x = P->x + dx*closer;
        p.y = P->y + dy*closer;
        p.z = P->z + dz*closer;
        // if the point doesn't collide with a point that has already moved
    } while (collidesWithObstacles(&p, obstacles));
    
    *Q = p;
}

__device__ void randomMoveFlat(Point* P, Point* Q, Obstacle* obstacles, curandState* localState) {
    Point p;
    p.x = P->x + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
    p.y = P->y + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
    p.z = P->z + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
    // if the point doesn't collide with a point that has already moved
    while (collidesWithObstacles(&p, obstacles)) {
        p.x = P->x + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
        p.y = P->y + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
        p.z = P->z + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
    }
    *Q = *P;
}

__global__ void kernel_mutateFlat(Population* P, Population* Q, Obstacle* obstacles, curandState* state) {    
    
    int ps_id = blockIdx.x * blockDim.x + threadIdx.x;
    int point_id = blockIdx.y * blockDim.y + threadIdx.y;
    
    // decide, for each pointset, its mutation type
    // 0 = no mutation, 1 = mix, 2 = randomMove
    __shared__ int mutate_pointset[N]; // store if pointsets need to be mutated
    if (point_id == 0) {
        if (cuda_randomChoice(POINT_SET_MUTATION_PROB, &state[ps_id])) {
            if (cuda_randomChoice(0.5, &state[ps_id])) mutate_pointset[ps_id] = 1;
            else mutate_pointset[ps_id] = 2;
        }
        else mutate_pointset[ps_id] = 0;
    }
    __syncthreads();
    
    // now treat points    
    PointSet* AP = &P->pointSets[ps_id];    // original points
    PointSet* AQ = &Q->pointSets[ps_id];    // mutated points    
    
    Point* p = &(AP->points[point_id]);    // original
    Point* q = &(AQ->points[point_id]);    // mutated point
    
    curandState localState = state[ps_id*N_POINTS + point_id];
    
    // no mutation
    if (mutate_pointset[ps_id] == 0 || cuda_randomChoice(1 - POINT_MUTATION_PROB, &localState)) {
        *p = *q;        
    }
    // mix
    else if (mutate_pointset[ps_id] == 1) {
        mixFlat(p, AP, q, obstacles, &localState);
    }
    // randomMove
    else {
        randomMoveFlat(p, q, obstacles, &localState);
    }
}

// Q = mutation of the X% best portion of P
// llegeix de P, escriu a Q
void mutate(Population* gpu_P, Population* gpu_Q) {
    tic(&mutationTime);
    
    bool flat = false;
    if (flat) {
        int n_threads = 32; //sqrt(1024)
        int width = N / n_threads;
        int height = N_POINTS / n_threads;
        // kernel
        dim3 gridSize(width, height, 1);
        dim3 blockSize(n_threads, n_threads, 1);
        kernel_mutateFlat<<<gridSize, blockSize>>>(gpu_P, gpu_Q, gpu_obstacles, devStates);
        checkCudaError((char *) "kernel call in mutate");
        cudaDeviceSynchronize();
    }
    else {
        // kernel 
        kernel_mutate<<<nBlocks, nThreads>>>(gpu_P, gpu_Q, gpu_obstacles, devStates);
        checkCudaError((char *) "kernel call in mutate");
        cudaDeviceSynchronize();
    }
    toc(&mutationTime);
}

void dump(PointSet* C) {
    for (int i = 0; i < N_POINTS; ++i) {
        printf("%f %f %f\n", C->points[i].x, C->points[i].y, C->points[i].z);
    }
}

__device__ void pork(PointSet* p1, PointSet* p2, PointSet* child, curandState* localState) {
    for (int i = 0; i < N_POINTS; ++i) {
        Point* p = &child->points[i];        
        if (cuda_randomChoice(0.5, localState)) {
            *p = p1->points[i];
        }
        else {
            *p = p2->points[i];
        }
    }
}

__global__ void kernel_reproduce(Population* P, Population* Q, int* idxs, 
                                                        curandState* state) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idxs[id] < N_SURVIVORS) {
        Q->pointSets[id] = P->pointSets[id];
    }
    else {
        curandState localState = state[id];
        int ip1 = idxs[(unsigned int) (CURAND*(N_SURVIVORS-1))];
        PointSet* p1 = &P->pointSets[ip1];
        int ip2 = idxs[(unsigned int) (CURAND*(N_SURVIVORS-1))];
        PointSet* p2 = &P->pointSets[ip2];        
        PointSet* child = &Q->pointSets[id];
        pork(p1, p2, child, &localState);
    }
}

__global__ void kernel_reproduceFlat(Population* P, Population* Q, int* idxs, 
                                                        curandState* state) {
    
    int ps_id = blockIdx.x * blockDim.x + threadIdx.x;
    int point_id = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ int parents[(N - N_SURVIVORS) * 2];
    
    PointSet* AP = &P->pointSets[ps_id];    // original points
    PointSet* AQ = &Q->pointSets[ps_id];    // mutated points    
    
    Point* p = &(AP->points[point_id]);    // original
    Point* q = &(AQ->points[point_id]);    // mutated point
    
    
    int idxid = idxs[ps_id];
    if (idxid < N_SURVIVORS) {
        *q = *p;
    }
    else {
        curandState localState = state[ps_id*N_POINTS + point_id];
        if (point_id == 0) {
            parents[(idxid - N_SURVIVORS) * 2] = idxs[(unsigned int) (CURAND*(N_SURVIVORS-1))];
            parents[(idxid - N_SURVIVORS) * 2 + 1] = idxs[(unsigned int) (CURAND*(N_SURVIVORS-1))];
        }
        __syncthreads();
        int parent_idx = (idxid - N_SURVIVORS) * 2;
        
        if (cuda_randomChoice(0.5, &localState)) {
            parent_idx += 1;            
        }
//         int parent_ps = parents[parent_idx];
//         if (parent_idx > (N-N_SURVIVORS)*2) {
//             printf("parent index out of range\n");
//         }
//         if (parent_ps > N) {
//             printf("pointset index out of range!\n");
//         }
//         if (point_id > N_POINTS) {
//         
//             printf("point index out of range");
//         }
//         PointSet* test = &P->pointSets[parent_ps];
//         Point* p = &test->points[point_id];
//         *q = *p;
//         *q = P->pointSets[parent_ps].points[point_id];
        *q = P->pointSets[parents[parent_idx]].points[point_id];
    }
}

void reproduce(Population* gpu_P, Population* gpu_Q, int* gpu_idxs) {
    tic(&reproductionTime);
    
    bool flat = false;
    if (flat) {
        int n_threads = 32; //sqrt(1024)
        int width = N / n_threads;
        int height = N_POINTS / n_threads;
        // kernel
        dim3 gridSize(width, height, 1);
        dim3 blockSize(n_threads, n_threads, 1);
        kernel_reproduceFlat<<<gridSize, blockSize>>>(gpu_P, gpu_Q, gpu_idxs, devStates);
        checkCudaError((char *) "kernel call in reproduce");   
    }
    else {
        kernel_reproduce<<<nBlocks, nThreads>>>(gpu_P, gpu_Q, gpu_idxs, devStates);
        checkCudaError((char *) "kernel call in reproduce");   
    }
    
    cudaDeviceSynchronize();
    
    toc(&reproductionTime);
}

void DUMPInitialParams() {
    printf("%i\n", N_OBSTACLES);
    for (int i = 0; i < N_OBSTACLES; ++i) {
        Obstacle o = cpu_obstacles[i];
        printf("%f %f %f %f\n", o.centre.x, o.centre.y, o.centre.z, o.radius); 
    }
    printf("%i %i\n", N_POINTS, ITERATION_LIMIT);
    
}

void initTimes() {
    initialGenTime = mutationTime = evaluationTime = sortingTime = reproductionTime = 0;
}

void printTimes() {
    printf("CUDA genetic algorithm has finished:\n");
    printf("    Init gen:     %f s.\n", (double)initialGenTime/1000000);
    printf("    Mutations:    %f s.\n", (double)mutationTime/1000000);
    printf("    Evaluations:  %f s.\n", (double)evaluationTime/1000000);
    printf("    Sorting:      %f s.\n", (double)sortingTime/1000000);
    printf("    Reproduction: %f s.\n", (double)reproductionTime/1000000);
    printf("    Total time:   %f s.\n", (double)totalTime/1000000);
}

void initObstacles() {    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                Point origin;
                
                origin.x = 3*i;
                origin.y = 3*j;
                origin.z = 3*k;

                cpu_obstacles[i*9 + j*3 + k].centre = origin;
                cpu_obstacles[i*9 + j*3 + k].radius = 1.0;
            }
        }
    }

    cudaMalloc(&gpu_obstacles, sizeof(Obstacle)*N_OBSTACLES);
    //cudaMemcpy(gpu_obstacles, cpu_obstacles, sizeof(Obstacle)*N_OBSTACLES, cudaMemcpyHostToDevice);
    checkCudaError((char *) "host -> gpu obstacles");
    
    cudaDeviceSynchronize();
}

void initDestinationPoint() {
    
    destination.x = destination.y = destination.z = 0.0;   
    
    // malloc
    cudaMalloc(&gpu_destination, sizeof(Point));
    checkCudaError((char *) "cudaMalloc in initDestinationPoint");
    
    // copy    
    cudaMemcpy(gpu_destination, &destination, sizeof(Point), cudaMemcpyHostToDevice);
    checkCudaError((char *) "host -> gpu in initDestinationPoint");
}

void cudaGenetic() {
    srand(SEED);  
    
    initObstacles(); 
    return;
    initDestinationPoint();
    
    tic(&totalTime);   
    
    // Malloc
    Population* gpu_P;
    Population* gpu_Q;
    cudaMalloc((void **) &gpu_P, sizeof(Population));
    checkCudaError((char *) "cudaMalloc of P");
    cudaMalloc((void **) &gpu_Q, sizeof(Population));
    checkCudaError((char *) "cudaMalloc of Q");    
    
    int* gpu_idxs;
    cudaMalloc((void **) &gpu_idxs, sizeof(int)*N);
    checkCudaError((char *) "cudaMalloc of idxs"); 
    
    PointSet* bestPointSet = (PointSet*) malloc(sizeof(PointSet));

    if (DUMP) DUMPInitialParams();
    else initTimes();
    
    generateInitialPopulation(gpu_P, gpu_idxs);
    
    int it = 0;
    while (true) {
        mutate(gpu_P, gpu_Q);
        
        evaluate(gpu_Q);
        
        sort(gpu_Q, gpu_idxs, bestPointSet);
        
        if (DUMP) dump(bestPointSet);
        else {            
            printf("\nIt: %i/%i Score: %f -> %f\n", 
                   it, ITERATION_LIMIT, bestPointSet->score, GOAL_SCORE);
        }
        
        if (it >= ITERATION_LIMIT || bestPointSet->score <= GOAL_SCORE) 
            break;
        
        // reproduce replaces the worst candidates with combinations
        // of better ones. from Q to P, so the population ends up in P
        // prepared for the next iteration
        reproduce(gpu_Q, gpu_P, gpu_idxs);
        it++;
    }
    // The result is bestPointSet
    
    toc(&totalTime);
    
    if (!DUMP) printTimes();
}

int main(int argc, char** argv) {
    DUMP = (argc == 1);
    cudaGenetic();
    return 0;
}

