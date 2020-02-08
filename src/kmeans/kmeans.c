#include <err.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>
#include <unistd.h>

/* read from file (binary format) */
float *loadData(char *fileName, unsigned nbVec, unsigned dim)
{
    int fd = open(fileName, O_RDONLY);
    if (fd == -1)
        err(1, "Error while openning %s", fileName);

    struct stat st;
    if (fstat(fd, &st) != -1 && nbVec * dim * sizeof(float) > (size_t)st.st_size)
        errx(1, "Error in parameters");

    void *tab = mmap(NULL, nbVec * dim * sizeof(float), PROT_READ,
                     MAP_SHARED, fd, 0);
    if (tab == MAP_FAILED)
        err(1, "Error while mmap");
    close(fd);

    return tab;
}

/* write in file (binary format) */
void writeClassinFloatFormat(unsigned char *data, unsigned nbelt, char *fileName)
{
    FILE *fp = fopen(fileName, "w");
    if (!fp)
        err(1, "Cannot create File: %s\n", fileName);

    for(unsigned i = 0; i < nbelt; ++i)
    {
        float f = data[i];
        fwrite(&f, sizeof(float), 1, fp);
    }

    fclose(fp);
}

/* compute euclidian distance between two points */
double distance(float *vec1, float *vec2, unsigned dim)
{
    double dist = 0;

    # pragma omp parallel for simd reduction(+:dist)
    for(unsigned i = 0; i < dim; ++i)
    {
        double d = *vec1 - *vec2;
        dist += d * d;
        vec1++;
        vec2++;
    }

    return sqrt(dist);
}

/* vec: one vector (want to find its nearest cluster) */
/* means: list of vector (clusters) */
/* dim: dimension of each vector (2351 in our case) */
/* K: number of cluster */
/* e: error computed (min dist between our point and the clusters) */
/* -> basically distance between vec and its nearest cluster */
unsigned char classify(float *vec, float *means, unsigned dim,
                       unsigned char K, double *e)
{
    unsigned char min = 0;
    float dist, distMin = FLT_MAX;

    for(unsigned i = 0; i < K; ++i)
    {
        dist = distance(vec, means + i * dim, dim);
        if(dist < distMin)
        {
            distMin = dist;
            min = i;
        }
    }

    *e = distMin;
    return min;
}

/* iter: number of iteration done by the kmeans */
/* time: execution time of the kmeans  */
/* err: error computed by the kmeans */
static inline void print_result(int iter, double time, float err)
{
    if (getenv("TEST") != NULL)
        printf("{\"iteration\": \"%d\", \"time\": \"%lf\", \"error\": \"%f\"}\n", iter, time, err);
    else
        printf("Iteration: %d, Time: %lf, Error: %f\n", iter, time, err);
}

/* implem of the kmeans algo */
/* data: vector with all the data */
/* nbVec: number of vector in the dataset */
/* dim: dimension of each vector (nbr of element) */
/* K: number of cluster */
/* minErr: min error => stop the kmeans when this rate is reach */
/* => sum of the min distance divided by the number of vector */
/* maxIter: max iteration for the kmeans algo */
unsigned char *Kmeans(float *data, unsigned nbVec, unsigned dim,
                      unsigned char K, double minErr, unsigned maxIter)
{
    unsigned iter = 0;
    double e = 0.;
    double diffErr = DBL_MAX;
    double err = DBL_MAX;

    /* vector of clusters */
    float *means = malloc(sizeof(float) * dim * K);

    /* nb of point for each cluster */
    unsigned *card = malloc(sizeof(unsigned) * K);

    /* vector of positions */
    /* position of the nearest cluster (in means) for each elm */
    unsigned char* c = malloc(sizeof(unsigned char) * nbVec);

    /* zeros init of means */
    # pragma omp parallel for
    for (unsigned i = 0; i < dim * K; ++i)
        means[i] = 0.;

    /* zeros init of card */
    for (unsigned i = 0; i < K; ++i)
        card[i] = 0.;

    /* Random init of c */
    # pragma omp parallel for simd
    for(unsigned i = 0; i < nbVec; ++i)
        c[i] = rand() / (RAND_MAX + 1.) * K;

    for(unsigned i = 0; i < nbVec; ++i)
    {
        /* init the cluster */
        for(unsigned j = 0; j < dim; ++j)
            means[c[i] * dim + j] += data[i * dim  + j];
        /* increase the number of point assiociated to the cluster */
        ++card[c[i]];
    }

    /* divide each feature of each cluster by the cardinal of each cluster */
    for(unsigned i = 0; i < K; ++i)
    {
        for(unsigned j = 0; j < dim; ++j)
            means[i * dim + j] /= card[i];
    }

    while ((iter < maxIter) && (diffErr > minErr))
    {
        double t1 = omp_get_wtime();
        diffErr = err;
        // Classify data
        err = 0.;
        /* each vector is associated to its nearest cluster */
        # pragma omp parallel for reduction(+:err)
        for(unsigned i = 0; i < nbVec; ++i)
        {
            c[i] = classify(data + i * dim, means, dim, K, &e);
            err += e;
        }

        // update Mean
        # pragma omp parallel for
        for(unsigned i = 0; i < dim * K; ++i)
            means[i] = 0.;

        for(unsigned i = 0; i < K; ++i)
            card[i] = 0.;

        /* update of means (clusters) */
        for(unsigned i = 0; i < nbVec; ++i)
        {
                for(unsigned j = 0; j < dim; ++j)
                    means[c[i] * dim + j] += data[i * dim  + j];
            ++card[c[i]];
        }

        for(unsigned i = 0; i < K; ++i)
        {
            for(unsigned j = 0; j < dim; ++j)
                means[i * dim + j] /= card[i];
        }

        ++iter;
        err /= nbVec;
        double t2 = omp_get_wtime();
        diffErr = fabs(diffErr - err);

        print_result(iter, t2 - t1, err);
    }

    free(means);
    free(card);

    return c;
}

int main(int ac, char *av[])
{
    if (ac != 8)
        errx(1, "Usage :\n\t%s <K: int> <maxIter: int> <minErr: float> <dim: int> <nbvec:int> <datafile> <outputClassFile>\n", av[0]);

    unsigned maxIter = atoi(av[2]);
    double minErr = atof(av[3]);
    unsigned K = atoi(av[1]);
    unsigned dim = atoi(av[4]);
    unsigned nbVec = atoi(av[5]);

    /* set number of thread */
    omp_set_num_threads(8);

    printf("Start Kmeans on %s datafile [K = %d, dim = %d, nbVec = %d]\n", av[6], K, dim, nbVec);

    float *tab = loadData(av[6], nbVec, dim);

    /* tab: vector with all the data */
    /* nbVec: number of vector in the dataset (nbr of elm) */
    /* dim: nbr of features of each vector (dimension of each vector) */
    /* K: nbr of cluster (3 in our case) */
    /* minErr: min Error => stop when you reach this error */
    /* maxIter: max iteration to execute for the kmeans algorithm */
    unsigned char * classif = Kmeans(tab, nbVec, dim, K, minErr, maxIter);

    writeClassinFloatFormat(classif, nbVec, av[7]);

    munmap(tab, nbVec * dim * sizeof(float));
    free(classif);

    return 0;
}
