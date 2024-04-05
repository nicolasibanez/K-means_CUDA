////////////////////////////////////////////////////////////////////////////////
// k-means on CPU & GPU
// S. Vialle March 2022 (with the help of G. He)
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <omp.h>

#include "main.h"
#include "init.h"


/*------------------------------------------------------------------------------------------------*/
/* Input dataset                                                                                  */
/*------------------------------------------------------------------------------------------------*/
void InputData()
{
    FILE  *fp = NULL;

    // Global array allocations
    instance = (T_real *) malloc(NB_INSTANCES*NB_DIMS*sizeof(T_real));   // Instance matrix
    instance_T = (T_real *) malloc(NB_DIMS*NB_INSTANCES*sizeof(T_real)); // Transposed instance matrix
    label = (int *) malloc(NB_INSTANCES*sizeof(int));                    // Labels of the instances
    
    centroid = (T_real *) malloc(NB_CLUSTERS*NB_DIMS*sizeof(T_real));
    count = (long long unsigned *) malloc(NB_CLUSTERS*sizeof(unsigned long long int)); // Nb of pts/cluster
    
    // Build the data file name (with complete path) 
    int len = strlen(FileDirectory) + strlen(DATA_FILE_NAME);
    char *DataFileName = (char *) malloc(sizeof(char)*(len+2));
    strcpy(DataFileName,FileDirectory);
    if (FileDirectory[strlen(FileDirectory)-1] != '/')
      strcat(DataFileName,"/");
    strcat(DataFileName,DATA_FILE_NAME);
    
    // Open data file and check if it has been successfully opened 
    fp = fopen(DataFileName, "r");  
    if(fp == NULL){
        printf("Fail to open file '%s'!\n",DataFileName);
        exit(0);
    }

    // Load data points one by one
    int count1 = 0;
    int count2 = 0;
    T_real value;
    for (int i = 0; i < NB_INSTANCES; i++){
      for (int j = 0; j < NB_DIMS; j++){
        count1 += fscanf(fp, T_REAL_PRINT, &value); 
        instance[i*NB_DIMS + j] = value;
        instance_T[j*NB_INSTANCES + i] = value;
      }
      count2 += fscanf(fp, "\n");
    }

    // Check if the input data has been successfully loaded
    if (count1 == NB_INSTANCES*NB_DIMS)
      printf("The data instances have been successfully loaded!\n");
    else
      printf("Failed to load data instances!\n");
 
    // Close input file and free input file name buffer
    fclose(fp); 
    free(DataFileName);
}


/*------------------------------------------------------------------------------------------------*/
/* Output result                                                                                  */
/*------------------------------------------------------------------------------------------------*/
void OutputResult()
{
    FILE  *fp = NULL;
    char *LabelFileName = NULL;
    char *CentroidFileName = NULL;

    // Build the label file name with complete path 
    int lenLabel = strlen(FileDirectory) + strlen(LABELFILEROOT) + strlen(DATA_FILE_NAME) + 2;
    LabelFileName = (char *) malloc(sizeof(char)*lenLabel);
    strcpy(LabelFileName,FileDirectory);
    if (FileDirectory[strlen(FileDirectory)-1] != '/')
      strcat(LabelFileName,"/");
    strcat(LabelFileName,LABELFILEROOT);
    strcat(LabelFileName,DATA_FILE_NAME);

    // Open Label File and Write the clustering result into the output file
    fp = fopen(LabelFileName, "w");
    if (fp == NULL) {
        printf("Fail to create file '%s'!\n",LabelFileName);
        exit(0);
    }
    for (int i = 0; i < NB_INSTANCES; i++) {
        fprintf(fp, "%d\n", label[i]);
    }

    // Close Label File and free Label File Name buffer
    fclose(fp); 
    free(LabelFileName);

    // Write the final centroids into the output file
    int lenCentroid = strlen(FileDirectory) + strlen(CENTROIDFILEROOT) + strlen(DATA_FILE_NAME) + 2;
    CentroidFileName = (char *) malloc(sizeof(char)*lenCentroid);
    strcpy(CentroidFileName,FileDirectory);
    if (FileDirectory[strlen(FileDirectory)-1] != '/')
      strcat(CentroidFileName,"/");
    strcat(CentroidFileName,CENTROIDFILEROOT);
    strcat(CentroidFileName,DATA_FILE_NAME);

    // Open Centroid file and Write the final centroids into the output file
    fp = fopen(CentroidFileName, "w");
    if (fp == NULL) {
        printf("Fail to open file '%s'!\n",CentroidFileName);
        exit(0);
    }
    for (int i = 0; i < NB_CLUSTERS; i++) {
      for (int j = 0; j < NB_DIMS; j++)
         fprintf(fp, "%f\t", centroid[i*NB_DIMS + j]);
      fprintf(fp, "\n");
    }
    // Close Centroid file and free Centroid file name buffer
    fclose(fp);
    free(CentroidFileName);

    printf("The clustering results have been stored into files!\n");

    // Free memory
    free(instance);
    free(instance_T);
    free(label);
    free(centroid);
    free(count);
}

/*------------------------------------------------------------------------------------------------*/
/* Command Line parsing.                                                                          */
/*------------------------------------------------------------------------------------------------*/
void usage(int ExitCode, FILE *std)
{
    fprintf(std,"K-means Clustering usage: \n");
    fprintf(std,"\t [-h]: print this help\n");
    fprintf(std,"\t [-cpu-nt <nb of CPU threads (default: 1)>]\n");
    fprintf(std,"\t [-d <path to the file directory (default: ./)>]\n");
    fprintf(std,"\t [-t <GPU(default)|CPU>]: run computations on target GPU or on target CPU\n");

    exit(ExitCode);
}


void CommandLineParsing(int argc, char *argv[])
{
    // Default init
    NbThreads = DEFAULT_NB_THREADS;
    OnGPUFlag = DEFAULT_ONGPUFLAG;
    FileDirectory = DEFAULT_FILE_DIRECTORY;;

    // Init from the command line
    argc--; argv++;
    while (argc > 0) {
      if (strcmp(argv[0],"-t") == 0) {
        argc--; argv++;
        if (argc > 0) {
          if (strcmp(argv[0],"GPU") == 0) {
             OnGPUFlag = 1;
             argc--; argv++;
           } else if (strcmp(argv[0],"CPU") == 0) {
             OnGPUFlag = 0;
             argc--; argv++;
           } else {
             fprintf(stderr,"Error: unknown computation target '%s'!\n",argv[0]);
             exit(EXIT_FAILURE);
           }
        } else {
           usage(EXIT_FAILURE, stderr);
        }

      } else if (strcmp(argv[0],"-d") == 0) {
	 argc--; argv++;
         if (argc > 0) {
           FileDirectory = argv[0];
	   argc--; argv++;
         } else {
           usage(EXIT_FAILURE, stderr);
         }

      } else if (strcmp(argv[0],"-cpu-nt") == 0) {
         argc--; argv++;
         if (argc > 0) {
           NbThreads = atoi(argv[0]);
	   if (NbThreads <= 0)
	     usage(EXIT_FAILURE, stderr);
           argc--; argv++;
         } else {
           usage(EXIT_FAILURE, stderr);
         }

      } else if (strcmp(argv[0],"-h") == 0) {
         usage(EXIT_SUCCESS, stdout);

      } else {
         usage(EXIT_FAILURE, stderr);
      }
    }
}

