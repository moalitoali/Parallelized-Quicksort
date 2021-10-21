/**********************************************************************
 * Parallel Quick sort
 * Usage: ./a.out sequence length strategy
 *
 **********************************************************************/
#define PI 3.14159265358979323846
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int partition(double *data, int left, int right, int pivotIndex);
void quicksort(double *data, int left, int right);
double *merge(double *v1, int n1, double *v2, int n2);
void printList(double* data, int len);
void p_quicksort(double *local_data, int len, MPI_Comm comm, int strategy, int world_rank);
void writeOutput(char* filename, double* data, int n);

int main(int argc, char *argv[]) {
    double start_time, execution_time, max_time, *data, *local_data, pivot, *temp_data;
    MPI_Status status;
    MPI_Request request;
    int size, rank, len, seq, strategy, num_splits, chunk, last_chunk = 0, uneven_distribution = 0;
    
    if(argc != 4){
        printf("ERROR! Expected input: quicksort sequence length strategy\n");
    }
    seq=atoi(argv[1]);
    len=atoi(argv[2]);
    strategy=atoi(argv[3]);

    MPI_Init(&argc, &argv);               // Initialize MPI 
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get my number

    // Check assumptions
    if((seq < 0) || (seq > 3)){
        if(rank == 0) printf("ERROR! Invalid sequence selection. Use 0, 1 or 2!\n");
        exit(0);
    }
    if((strategy < 0) || (strategy > 2)){
        if(rank == 0) printf("ERROR! Invalid strategy selection. Use 0, 1, 2 or 3!\n");
        exit(0);
    }
    if(len <= 0){
        if(rank == 0) printf("ERROR! Invalid length selection. Please use a positive integer!\n");
        exit(0);
    }
    num_splits = log(size) / log(2);
    if(pow(2, num_splits) != size){
        if(rank == 0) printf("ERROR! Invalid number of processes.\n");
        exit(0);
    }

    // Fill list
    if(rank==0){
        data = (double *)malloc(len*sizeof(double));
        if(seq == 0){ // Uniform random numbers
            for (int i = 0; i < len; i++)
                data[i] = drand48();
        }else if(seq == 1){ // Exponential distribution
            double lambda = 10;
            for (int i = 0; i < len; i++)
                data[i] =- lambda*log(1-drand48());
        }else if(seq == 2){ // Normal distribution
            double x, y;
            for (int i = 0;i < len; i++){
                x = drand48(); y = drand48();
                data[i] = sqrt(-2*log(x)) * cos(2*PI*y);
            }
        }else if(seq == 3){
            for(int i = 0; i < len; i++)
                data[i] = len - i;
        }
        // Print initial list
        /*
        printf("Initial list\n");
        printList(data, len);
        */
    }

    // Calculate local_data size and allocate memory
    chunk = len/size;
    if(len % size != 0){
        uneven_distribution = 1;
        if(rank==0){
            last_chunk = len % size;
        }
    }
    local_data = (double *)malloc((chunk+last_chunk)*sizeof(double));
    
    // Start timer
	start_time = MPI_Wtime();

    // Scatter data to local_data arrays
    MPI_Scatter(&data[0], chunk, MPI_DOUBLE, &local_data[0], chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(uneven_distribution == 1){ // add rest to rank 0
        if(rank == 0){
            int counter = 0;
            for(int i = size*chunk; i < size*chunk+last_chunk; i++){
                local_data[chunk+counter] = data[i];
                counter++;
            }
        }
    }

    // Sort local lists
    quicksort(local_data, 0, chunk+last_chunk-1);
    
    // Call parallel quicksort
    p_quicksort(local_data, chunk+last_chunk, MPI_COMM_WORLD, strategy, rank);

    // Receive data
    if(rank == 0){
        int* recv_sizes = (int*)malloc(size*sizeof(int));
        int position = 0;
        for(int i = 0; i < size; i++){
            MPI_Probe(i, 200+i, MPI_COMM_WORLD, &status); // fetch status thing for communication
            MPI_Get_count(&status, MPI_DOUBLE, &recv_sizes[i]); // find number of elements to receive

            MPI_Irecv(&data[position], recv_sizes[i], MPI_DOUBLE, i, 200+i, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);

            position += recv_sizes[i];
        }
        free(recv_sizes);
    }
    // Print final list
    /*
    if(rank == 0) {
        printf("FINAL LIST \n");
        printList(data, len);
    }
    */

    // Compute time
	execution_time = MPI_Wtime()-start_time; // stop timer
	MPI_Reduce(&execution_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
    
	// Display timing results
	if(rank==0){ 
		printf("%f\n", max_time);
	}

    // Check results
    if(rank == 0){
        int OK = 1;
        for (int i = 0; i < len-1; i++) {
            if(data[i] > data[i+1]) {
                printf("Wrong result: data[%d] = %f, data[%d] = %f\n", i, data[i], i+1, data[i+1]);
                OK = 0;
            }
        }
        if (OK) printf("Data sorted correctly!\n");
    }

    // Clean up
    if (rank == 0){
        free(data);
    }
    MPI_Finalize();
    return 0;
}

void p_quicksort(double *local_data, int len, MPI_Comm comm, int strategy, int world_rank){
    int local_size, local_rank;
    double pivot;
    MPI_Request request, request_send, request_recv;
    MPI_Status status;

    MPI_Comm_size(comm, &local_size);
    MPI_Comm_rank(comm, &local_rank);

    // Basecase
    if(local_size == 1){
        MPI_Isend(&local_data[0], len, MPI_DOUBLE, 0, 200+world_rank, MPI_COMM_WORLD, &request);
        return;
    }

    // Select pivot element according to chosen strategy 
    if(strategy == 0){ // median in one process
        if(local_rank == 0){
            pivot = local_data[len/2];
        } 
        MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, comm);
    } else if(strategy == 1){ // median of all medians
        double *medians;
        int *lengths;

        // Find how many local lists that are NOT empty
        if(local_rank == 0){
            lengths = (int*)malloc(local_size*sizeof(int));
        }
        MPI_Gather(&len, 1, MPI_INT, &lengths[0], 1, MPI_INT, 0, comm);

        if(local_rank == 0){
            int num_medians = 0;
            for(int i = 0; i < local_size; i++){
                if(lengths[i]>0){
                    num_medians++;
                }
            }
            medians = (double *)malloc(local_size*sizeof(double));
        }

        // If local list not empty: send median
        if(len > 0){
            MPI_Isend(&local_data[len/2], 1, MPI_DOUBLE, 0, 500+local_rank, comm, &request);
        }

        // Receive medians
        if(local_rank == 0){
            int counter = 0;
            for(int i = 0; i < local_size; i++){
                if(lengths[i] > 0){
                    MPI_Irecv(&medians[counter], 1, MPI_DOUBLE, i, 500+i, comm, &request);
                    MPI_Wait(&request, &status);
                    counter++;
                }
            }

            // Compute medians of medians
            quicksort(medians, 0, counter-1);
            pivot = medians[counter/2];
            free(medians);
            free(lengths);
        }
        MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, comm);

    } else if (strategy == 2){ // mean of all medians
        double median = local_data[len/2];
        MPI_Reduce(&median, &pivot, 1, MPI_DOUBLE, MPI_SUM, 0, comm); 
        pivot = pivot/local_size;
        MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, comm);
    }

    // Find split in local list
    int num_smaller_elements = 0;
    for(int i = 0; i < len; i++){
        if(local_data[i] <= pivot){
            num_smaller_elements++;
        } else {
            break;
        }
    }
    int num_larger_elements = len - num_smaller_elements;

    // EXCHANGE DATA PAIRWISE
    // Split processes in two groups
    int rank_friend; // rank to exchange with
    int group;
    if(local_rank < local_size/2) {
        group = 0; // left part of list
        rank_friend = local_rank + local_size/2;
    } else {
        group = 1; // right part of list
        rank_friend = local_rank - local_size/2;
    }

    // Send elements
    if(group == 0){ // left group
        MPI_Isend(&local_data[num_smaller_elements], num_larger_elements, MPI_DOUBLE, rank_friend, 100+rank_friend, comm, &request_send);
    } else { // right group
        MPI_Isend(&local_data[0], num_smaller_elements, MPI_DOUBLE, rank_friend, 100+rank_friend, comm, &request_send);
    }

    MPI_Probe(rank_friend, 100+local_rank, comm, &status); // fetch status thing for communication
    int recv_size;
    MPI_Get_count(&status, MPI_DOUBLE, &recv_size); // find number of elements to receive

    double *temp_data, *old_data;
    int new_len;
    if(group == 0){
        temp_data = (double*)malloc(recv_size*sizeof(double));
        MPI_Irecv(&temp_data[0], recv_size, MPI_DOUBLE, rank_friend, 100+local_rank, comm, &request_recv);
        MPI_Wait(&request_recv, &status);

        old_data = (double*)malloc(num_smaller_elements*sizeof(double));
        for(int i = 0; i < num_smaller_elements; i++){
            old_data[i] = local_data[i];
        }

        // Merge
        MPI_Wait(&request_send, &status);
        free(local_data);
        local_data = merge(temp_data, recv_size, old_data, num_smaller_elements);
        new_len = recv_size+num_smaller_elements;
    } else {
        temp_data = (double*)malloc(recv_size*sizeof(double));
        MPI_Irecv(&temp_data[0], recv_size, MPI_DOUBLE, rank_friend, 100+local_rank, comm, &request_recv);
        MPI_Wait(&request_recv, &status);

        old_data = (double*)malloc(num_larger_elements*sizeof(double));

        int counter = 0;
        for(int i = num_smaller_elements; i < len; i++){
            old_data[counter] = local_data[i];
            counter++;
        }
        // Merge
        MPI_Wait(&request_send, &status);
        free(local_data);
        local_data = merge(temp_data, recv_size, old_data, num_larger_elements);
        new_len = recv_size+num_larger_elements;
    }

    // Split comm into two parts
    MPI_Comm comm_new;
    MPI_Comm_split(comm, group, local_rank, &comm_new); 

    // Local clean up
    free(temp_data);
    free(old_data);

    // Recursive call
    p_quicksort(local_data, new_len, comm_new, strategy, world_rank);
}

int partition(double *data, int left, int right, int pivotIndex){
    double pivotValue, temp;
    int storeIndex, i;
    pivotValue = data[pivotIndex];
    temp = data[pivotIndex]; 
    data[pivotIndex] = data[right]; 
    data[right] = temp;
    storeIndex = left;
    for (i = left; i < right; i++) 
    if (data[i] <= pivotValue){
        temp = data[i];
        data[i] = data[storeIndex];
        data[storeIndex] = temp;

        storeIndex = storeIndex + 1;
    }
    temp = data[storeIndex];
    data[storeIndex] = data[right]; 
    data[right] = temp;
    return storeIndex;
}

void quicksort(double *data, int left, int right){
    int pivotIndex, pivotNewIndex;
    if (right > left){ 
        pivotIndex = left+(right-left)/2;
        pivotNewIndex = partition(data, left, right, pivotIndex); 
        quicksort(data, left, pivotNewIndex - 1); 
        quicksort(data, pivotNewIndex + 1, right); 
    }
}

double *merge(double *v1, int n1, double *v2, int n2){
    int i,j,k;
    double *result;
    
    result = (double *)malloc((n1+n2)*sizeof(double));
    
    i=0; j=0; k=0;
    while(i<n1 && j<n2)
        if(v1[i]<v2[j])
        {
            result[k] = v1[i];
            i++; k++;
        }
        else
        {
            result[k] = v2[j];
            j++; k++;
        }
    if(i==n1)
        while(j<n2)
        {
            result[k] = v2[j];
            j++; k++;
        }
    else
        while(i<n1)
        {
            result[k] = v1[i];
            i++; k++;
        }
    return result;
}

void printList(double* data, int len){
    for(int i = 0; i < len; i++){
        printf("%10f ", data[i]);
    }
    printf("\n");
}