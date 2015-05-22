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
#define N 2048
#define N_POINTS 128
#define ITERATION_LIMIT 50
#define GOAL_SCORE -1.0
#define POINT_SET_MUTATION_PROB 0.5
#define POINT_MUTATION_PROB 0.01
#define N_SURVIVORS N/4
#define POINT_RADIUS 0.25
#define OBSTACLE_RADIUS 2.0
#define MAX_DELTA 2
#define MAX_TRIES 1e3   // max amount of times we tries to find a position for a point

// Obstacles
#define CHECK_OBSTACLES true
#define CHECK_COLLISIONS true

// Deterministic algorithm (testing purposes)
#define SEED 27
# define RAND01 ((float)rand()/(float)(RAND_MAX))

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
Point destination;

// CUDA Variables
unsigned int nThreads = 1024;
unsigned int nBlocks = N/nThreads;  // N multiple de nThreads
    
// GPU Pointers
Obstacle* gpu_obstacles;
Point* gpu_destination;

__device__ inline bool cuda_randomChoice(float probability, curandState* localState) {
    if (curand_uniform(localState) <= probability) return true;
    else return false;    
}

void checkCudaError(char msg[]) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error) {
        printf("Error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

__device__ inline float cuda_dist(Point* a, Point* b) {
    return sqrt(pow(a->x - b->x, 2)+pow(a->y - b->y, 2)+pow(a->z - b->z, 2));
}

__device__ bool cuda_collides(Point* p, PointSet* PS, int from, int to, Obstacle* obstacles) {
    if (CHECK_COLLISIONS)
        for (int i = from; i < to; ++i) {
            if (cuda_dist(p, &PS->points[i]) < POINT_RADIUS*2) {
                return true;
            }
        }
    if (CHECK_OBSTACLES)
        for (int i = 0; i < N_OBSTACLES; ++i) {
            Obstacle o = obstacles[i];
            if (cuda_dist(p, &o.centre) < POINT_RADIUS + o.radius) {
                return true;
            }
        }
    return false;
}

__global__ void kernel_generateInitialPopulation(Population* P, 
                    Obstacle* obstacles, curandState* state) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState localState = state[id];
    
    float range = POINT_RADIUS * pow((float)N_POINTS, 1.0f/3.0f) * 10;    
    
    /*
    float r1 = curand_uniform(&localState)/(float)(RAND_MAX);
    float r2 = curand_uniform(&localState);///(float)(RAND_MAX);
    float r3 = curand_uniform(&localState);///(float)(RAND_MAX);
    
    printf("%f %f %f\n", r1, r2, r3);
    */
    
    for (int j = 0; j < N_POINTS; ++j) {
        PointSet* PS = &(P->pointSets[id]);
        Point* p = &(PS->points[j]); // p is passed to 'collides' via PS
        p->x = CURAND * range + 12.5;
        p->y = CURAND * range + 12.5;
        p->z = CURAND * range + 12.5;
        
        int tries = 0;
        while (tries < MAX_TRIES && cuda_collides(p, PS, 0, j, obstacles)) {
            p->x = CURAND * range + 12.5;
            p->y = CURAND * range + 12.5;
            p->z = CURAND * 5.0 + 12.5;
            ++tries;
        }
        if (tries == MAX_TRIES) {
            printf("Error during the generation of the initial population\n");
            //exit(1);
        }
    }
}

__global__ void setup_kernel(curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init ( 1234, id, 0, &state[id] );
}

void generateInitialPopulation(Population* gpu_P) {
    tic(&initialGenTime);
    
    //RANDOM SETUP
    curandState *devStates;
    cudaMalloc((void **)&devStates, N * sizeof(curandState));
    setup_kernel<<<nBlocks, nThreads>>>(devStates);
    checkCudaError((char *) "setup random kernel");    
    //RANDOM END
    
    // kernel 
    kernel_generateInitialPopulation<<<nBlocks, nThreads>>>(gpu_P, gpu_obstacles, devStates);
    checkCudaError((char *) "kernel call in generateInitialPopulation");
    
    // wait
    cudaDeviceSynchronize();
    toc(&initialGenTime);
}

__device__ inline float heur_1(Point* P) {
    return fabs(P->y - 3.0*sin(P->x/2.0)) + fabs(P->z - 3.0*cos(P->x/2.0));
}


__device__ inline float heur_2(Point* P, Point* destination) {
    return cuda_dist(P, destination);
}

__global__ void kernel_evaluate(Population* P, Point* destination) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    PointSet* C = &P->pointSets[id];
    C->score = 0;
    for (int j = 0; j < N_POINTS; j++) {
        Point* E = &C->points[j];
        C->score += heur_2(E, destination);
    }
}

void evaluate(Population* gpu_P) {
    tic(&evaluationTime);
    
    // kernel 
    kernel_evaluate<<<nBlocks, nThreads>>>(gpu_P, gpu_destination);
    checkCudaError((char *) "kernel call in generateInitialPopulation");
    
    // wait
    cudaDeviceSynchronize();
    toc(&evaluationTime);
}

//////////////////////////////////////////
//////////////  CUDA QUICK SORT
//////         

#define MAX_DEPTH       16
#define INSERTION_SORT  32

__device__ void swap_pointSets(PointSet* a, PointSet* b) {
    PointSet aux = *a;
    *a = *b;
    *b = aux;
}


// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ void selection_sort(Population* P, int left, int right ) {
    for (int i = left ; i <= right ; ++i) {
        float min_score = P->pointSets[i].score;
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1 ; j <= right ; ++j) {
            float score_j = P->pointSets[j].score;
            if (score_j < min_score) {
                min_idx = j;
                min_score = score_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            swap_pointSets(&P->pointSets[i], &P->pointSets[min_idx]);
        }
    }
}


__device__ inline PointSet* candidate(Population* p, int ix) {
    return &p->pointSets[ix];
}

__device__ inline void swapCandidates(Population* p, int i, int j) {
    swap_pointSets(candidate(p, i), candidate(p, j));
}

// Partition algorithm (from wikipedia)
//////////////////////////////////////////////////////////////////////////////
//  // lo is the index of the leftmost element of the subarray              //
//  // hi is the index of the rightmost element of the subarray (inclusive) //
//  partition(A, lo, hi)                                                    //
//     pivotIndex := choosePivot(A, lo, hi)                                 //
//     pivotValue := A[pivotIndex]                                          //
//     // put the chosen pivot at A[hi]                                     //
//     swap A[pivotIndex] and A[hi]                                         //
//     storeIndex := lo                                                     //
//     // Compare remaining array elements against pivotValue = A[hi]       //
//     for i from lo to hi−1, inclusive                                     //
//         if A[i] <= pivotValue                                            //
//             swap A[i] and A[storeIndex]                                  //
//             storeIndex := storeIndex + 1                                 //
//     swap A[storeIndex] and A[hi]  // Move pivot to its final place       //
//     return storeIndex                                                    //
//////////////////////////////////////////////////////////////////////////////

__device__ int partition(Population* P, int lo, int hi) {
    int pivIx = (lo + hi)/2;
    float pivScore = candidate(P, pivIx)->score;
    swapCandidates(P, hi, pivIx);
    int stIx = lo;
    for (int i = lo; i < hi; ++i) {
        if (candidate(P, i)->score <= pivScore) {
            swapCandidates(P, i, stIx);
            ++stIx;
        }
    }
    swapCandidates(P, stIx, hi);
    return stIx;
}

// quicksort algorithm (from wikipedia)
/////////////////////////////////////
// quicksort(A, lo, hi):           //
//     if lo < hi:                 //
//       p := partition(A, lo, hi) //
//         quicksort(A, lo, p - 1) //
//         quicksort(A, p + 1, hi) //
/////////////////////////////////////
__global__ void dynamic_quicksort_jan(Population* P, int left, int right, int depth) {
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(P, left, right);
        return;
    }
        int piv = partition(P, left, right);
        
        // left
        if (left < piv - 1) {
            cudaStream_t ls;
            cudaStreamCreateWithFlags(&ls, cudaStreamNonBlocking);
            dynamic_quicksort_jan<<< 1, 1, 0, ls >>>(P, left, piv - 1, depth + 1);
            cudaStreamDestroy(ls);
        }

        //right
        if (piv + 1 < right) { 
            cudaStream_t rs;
            cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking);
            dynamic_quicksort_jan<<< 1, 1, 0, rs >>>(P, piv + 1, right, depth + 1);
            cudaStreamDestroy(rs);
        }    
}

__global__ void dynamic_quicksort(Population* P, int left, int right, int depth) {
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(P, left, right);
        return;
    }
    
    int lindex = left;
    int rindex = right;
    float pscore = P->pointSets[(left+right)/2].score; // Pivot

    // Do the partitioning.
    while (lindex <= rindex) {
        // Find the next left- and right-hand values to swap
        float lscore = P->pointSets[lindex].score; 
        float rscore = P->pointSets[rindex].score;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        // TODO: dicotomic search
        while (lscore < pscore) {
            lindex++;
            lscore = P->pointSets[lindex].score; 
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        // TODO: dicotomic search
        while (rscore > pscore) {
            rindex--;
            rscore = P->pointSets[rindex].score;
        }

        // If the swap points are valid, do the swap!
        if (lindex <= rindex) {
            
            // TODO: This needs to be improved, we can sort a vector
            // of indices instead of copying the whole pointSets.
            swap_pointSets(&P->pointSets[lindex], &P->pointSets[rindex]);
            
            lindex++;
            rindex--;
        }
    }

    // Now the recursive part
    // Launch a new block to sort the left part.
    if (left < rindex) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        dynamic_quicksort<<< 1, 1, 0, s >>>(P, left, rindex, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if (lindex < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        dynamic_quicksort<<< 1, 1, 0, s1 >>>(P, lindex, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}


void sort(Population* gpu_P) {
    tic(&sortingTime);
    
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    
    dynamic_quicksort<<<1, 1>>>(gpu_P, 0, N-1, 0);
    checkCudaError((char *) "kernel call in mutate");
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
            if (!cuda_collides(&p, AQ, 0, i, obstacles) &&
                // and it doesn't collide with a point that has yet to be moved
                // (this 2nd check prevents inconsistencies like a point being unable to move at all)
                !cuda_collides(&p, AP, i + 1, N_POINTS, obstacles))
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
        Point p;
        while (tries < MAX_TRIES) {
            p.x = AP->points[i].x + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
            p.y = AP->points[i].y + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
            p.z = AP->points[i].z + (curand_uniform(localState)-0.5)*2*MAX_DELTA;
            // if the point doesn't collide with a point that has already moved
            if (!cuda_collides(&p, AQ, 0, i, obstacles) &&
                // and it doesn't collide with a point that has yet to be moved
                // (this 2nd check prevents inconsistencies like a point being unable to move at all)
                !cuda_collides(&p, AP, i + 1, N_POINTS, obstacles))
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

// Q = mutation of the X% best portion of P
// llegeix de P, escriu a Q
void mutate(Population* gpu_P, Population* gpu_Q) {
    tic(&mutationTime);
    
     //RANDOM SETUP
    curandState *devStates;
    cudaMalloc((void **)&devStates, N * sizeof(curandState));
    setup_kernel<<<nBlocks, nThreads>>>(devStates);
    checkCudaError((char *) "setup random kernel");    
    //RANDOM END
    
    // kernel 
    kernel_mutate<<<nBlocks, nThreads>>>(gpu_P, gpu_Q, gpu_obstacles, devStates);
    checkCudaError((char *) "kernel call in mutate");
    cudaDeviceSynchronize();
    
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

__global__ void kernel_reproduce(Population* P, Population* Q, curandState* state) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < N_SURVIVORS) {
        Q->pointSets[id] = P->pointSets[id];
    }
    else {
        curandState localState = state[id];
        PointSet* p1 = &P->pointSets[(unsigned int) (CURAND*(N_SURVIVORS-1))];
        PointSet* p2 = &P->pointSets[(unsigned int) (CURAND*(N_SURVIVORS-1))];        
        PointSet* child = &Q->pointSets[id];
        pork(p1, p2, child, &localState);
    }
}

void reproduce(Population* gpu_P, Population* gpu_Q) {
    tic(&reproductionTime);
    
    //RANDOM SETUP
    curandState *devStates;
    cudaMalloc((void **)&devStates, N * sizeof(curandState));
    setup_kernel<<<nBlocks, nThreads>>>(devStates);
    checkCudaError((char *) "setup random kernel");    
    //RANDOM END
    
    // kernel 
    kernel_reproduce<<<nBlocks, nThreads>>>(gpu_P, gpu_Q, devStates);
    checkCudaError((char *) "kernel call in mutate");   
    cudaDeviceSynchronize();
    
    toc(&reproductionTime);
}

void getBestFromGPU(Population* gpu_P, PointSet* best) {
    
    // ?????????????????????????????????????????????????????????????? 
    cudaMemcpy(best, gpu_P, sizeof(PointSet), cudaMemcpyDeviceToHost);
    checkCudaError((char *) "gpu -> host getBestFromGPU");    
    cudaDeviceSynchronize();
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

                obstacles[i*9 + j*3 + k].centre = origin;
                obstacles[i*9 + j*3 + k].radius = 1.0;
            }
        }
    }
    
    cudaMalloc(&gpu_obstacles, sizeof(Obstacle)*N_OBSTACLES);
    cudaMemcpy(gpu_obstacles, obstacles, sizeof(Obstacle)*N_OBSTACLES, cudaMemcpyHostToDevice);
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
    initDestinationPoint();
    
    tic(&totalTime);   
    
    // Malloc
    Population* gpu_P;
    Population* gpu_Q;
    cudaMalloc((void **) &gpu_P, sizeof(Population));
    checkCudaError((char *) "cudaMalloc of P");
    cudaMalloc((void **) &gpu_Q, sizeof(Population));
    checkCudaError((char *) "cudaMalloc of Q");    
    
    PointSet* bestPointSet = (PointSet*) malloc(sizeof(PointSet));

    if (DUMP) DUMPInitialParams();
    else initTimes();
    
    generateInitialPopulation(gpu_P);
    
    int it = 0;
    while (true) {
        mutate(gpu_P, gpu_Q);
        evaluate(gpu_Q);
        sort(gpu_Q);        
        getBestFromGPU(gpu_Q, bestPointSet);
        
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
        reproduce(gpu_Q, gpu_P);
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

