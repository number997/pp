#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

void input_m(double *m, int dim) {
    printf("Matrix input.\n");
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            printf("Input element %d, %d: ", i, j);
            scanf("%lf", (m+i*dim+j));
        }
    }
}

void print_m(double *m, int dim) {
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            printf("%5.5f \t", *(m+i*dim+j));
        }
        printf("\n");
    }
    printf("\n");
}

void print_v(double *v, int rows) {
    for(int i = 0; i < rows; i++) {
        printf("%5.5f \n", *(v+i));
    }
    printf("\n");
}

bool is_diag_dominant(double *m, int dim, int threads) {
    bool is_diag = true;
    #pragma omp parallel for num_threads(threads) reduction(&&:is_diag)
    for(int i = 0; i < dim; i++) {
        double col = 0.0;
        for(int j = 0; j < dim; j++) {
            if(i!=j) {
                col+=abs(*(m+i*dim+j));
            }
        }
        if(abs(*(m+i*dim+i)) < col) {
            is_diag = is_diag && false;
        }
    }

    return is_diag;
}

bool is_null_matrix(double *m, int dim, int threads) {
    int zeros = 0;
    #pragma omp parallel for num_threads(threads) reduction(+:zeros)
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            if(*(m+i*dim+j) == 0.0) {
                zeros += 1;
            }
        }
    }
    if(zeros == dim*dim) {
        return true;
    }
    return false;
}

//to run program use following argument list
// ./program threads matrix_dimension random_matrix maximum_number_of_iterations

int main(int argc, char** argv) {
    int threads, dim, steps, random, i, j, max_iter;
    //m=matrix v=vector so that mX=v, s=solutionVector(x_{n+1}) h=previousSolutionVector(x_{n})
    double *m, *v, *s, *h;

    if(argc != 5) {
        printf("Wrong number of arguments");
        return -1;
    }

    threads = strtol(argv[1], NULL, 10);
    if(threads <= 0) {
        printf("Wrong number of threads.");
        return -1;
    }

    dim = strtol(argv[2], NULL, 10);
    if(dim <= 0 || dim%threads != 0) {
        printf("Wrong dimension.");
        return -1;
    }

    random = strtol(argv[3], NULL, 10);
    if(random != 0 && random != 1) {
        printf("3rd argument should be 1 or 0 (true, false).");
        return -1;
    }

    max_iter = strtol(argv[4], NULL, 10);
    if(max_iter <= 0) {
        printf("Wrong number of maximum Jacobi itterations.");
        return -1;
    }

    //set matrices and vectors to null matrices and null vectors
    m = calloc(dim*dim, sizeof(double));
    v = calloc(dim, sizeof(double));
    s = calloc(dim, sizeof(double));
    h = calloc(dim, sizeof(double));

    steps = dim/threads;

    //to prevent while loop to print string all the time it generates matrix
    bool info_printed = false;

    //as long as matrix is null matrix or matrix is not diagonally dominant
    while(is_null_matrix(m, dim, threads) || !is_diag_dominant(m, dim, threads)) {
        if(random) {
            if(!info_printed) {
                printf("Generating diagonally dominant matrix (system of linear equations).\n");
            }
            info_printed = true;
            //generate matrix
            #pragma omp parallel num_threads(threads) private(i, j)
            {
                int thread = omp_get_thread_num();
                srand(time(NULL) + thread);
                for(i = thread*steps; i < thread*steps + steps; i++) {
                    for(j = 0; j < dim; j++) {
                        *(m+i*dim+j) = rand()%50+1;
                    }
                }
            }
        }
        else {
            printf("Input diagonally dominant matrix.");
            input_m(m, dim);
        }
    }

    #pragma omp parallel num_threads(threads) private(i)
    {
        int thread = omp_get_thread_num();
        srand(time(NULL)+thread+rand());
        for(i = steps*thread; i < steps*thread+steps; i++) {
            *(v+i) = rand()%50+1;
        }
    }

    //jacobi method on given number of iterations
    for(int iter = 0; iter < max_iter; iter++) {
        #pragma omp parallel num_threads(threads) private(i, j)
        {
            int thread = omp_get_thread_num();
            for(i = thread*steps; i < thread*steps+steps; i++) {
                double temp = 0;
                for(j = 0; j < dim; j++) {
                    if(i != j) {
                        temp += *(m+i*dim+j)**(h+j);
                    }
                }
                *(s+i) = (*(v+i)-temp)/(*(m+i*dim+i));
            }
            for(int k = 0; k < dim; k++) {
                *(h+k) = *(s+k);
            }
        }
    }

    printf("Matrix.\n");
    print_m(m, dim);
    printf("Vector. \n");
    print_v(v, dim);
    printf("Solution. \n");
    print_v(s, dim);

    free(m);
    free(v);
    free(s);
    free(h);

    return 0;
}