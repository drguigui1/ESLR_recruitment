#include <err.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NBVEC 900000
#define NEWVEC 600000
#define DIM 2351


/* read from file (binary format) */
float *loadData(char *fileName)
{
    int fd = open(fileName, O_RDONLY);
    if (fd == -1)
        err(1, "Error while openning %s", fileName);

    void *tab = mmap(NULL, NBVEC * DIM * sizeof(float), PROT_READ,
                     MAP_SHARED, fd, 0);
    if (tab == MAP_FAILED)
        err(1, "Error while mmap");
    close(fd);

    return tab;
}

static void writeOneVect(float *data, unsigned pos, FILE *fp)
{
    for (unsigned i = 0; i < DIM; ++i)
    {
        float f = data[pos + i];
        fwrite(&f, sizeof(float), 1, fp);
    }
}

/* write in file (binary format) */
/* write only 1 or 0 labels from the dataset */
void write_proper_data(float *data, float *labels, char *fileName1, char *fileName2)
{
    FILE *fp1 = fopen(fileName1, "w");
    FILE *fp2 = fopen(fileName2, "w");
    if (!fp1)
        err(1, "Cannot create File: %s\n", fileName1);
    if (!fp2)
        err(1, "Cannot create File: %s\n", fileName2);

    unsigned pos_data = 0;
    for(unsigned i = 0; i < NBVEC; ++i)
    {
        int lab = labels[i];
        if (lab != -1)
        {
            /* write on vector in the file */
            writeOneVect(data, pos_data, fp1);

            /* write the label in the file */
            float l = labels[i];
            fwrite(&l, sizeof(float), 1, fp2);
        }

        /* go to the next element */
        pos_data += 2351;
    }

    fclose(fp1);
    fclose(fp2);
}

int main(int argc, char **argv)
{
    /* argv[1] => input file */
    /* argv[2] => out file */
    if (argc != 5)
        errx(1, "Usage: \n\t <argv[1]: input file1> <argv[2]: input (data) file2> <argv[3]: output (data) file1> <argv[4]: output (labels) file2>\n");
    float *data = loadData(argv[1]);
    float *labels = loadData(argv[2]);

    write_proper_data(data, labels, argv[3], argv[4]);

    return 0;
}
