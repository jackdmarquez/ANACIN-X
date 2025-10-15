#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int neighbor_rank(int rank, int size, int offset) {
    int target = (rank + offset) % size;
    if (target < 0) {
        target += size;
    }
    return target;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank = -1;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "This demo requires at least two MPI ranks.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (world_rank == 0) {
        printf("[collective_comm_demo] world_size=%d\n", world_size);
    }

    // Phase 1: synchronize all ranks with a collective barrier.
    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: non-blocking sends/receives around the ring, completed with MPI_Waitall.
    int send_value = world_rank;
    int recv_value = -1;
    MPI_Request ring_requests[2];

    int right = neighbor_rank(world_rank, world_size, +1);
    int left = neighbor_rank(world_rank, world_size, -1);

    MPI_Irecv(&recv_value, 1, MPI_INT, left, 100, MPI_COMM_WORLD, &ring_requests[0]);
    MPI_Isend(&send_value, 1, MPI_INT, right, 100, MPI_COMM_WORLD, &ring_requests[1]);
    MPI_Waitall(2, ring_requests, MPI_STATUSES_IGNORE);

    // Phase 3: demonstrate MPI_Ibarrier paired with MPI_Wait.
    MPI_Request ibarrier_request;
    MPI_Ibarrier(MPI_COMM_WORLD, &ibarrier_request);
    MPI_Wait(&ibarrier_request, MPI_STATUS_IGNORE);

    // Phase 4: create multiple outstanding receives and drive them to completion with MPI_Waitany.
    const int message_count = 3;
    int waitany_values[message_count];
    MPI_Request waitany_requests[message_count];
    MPI_Request send_requests[message_count];

    for (int i = 0; i < message_count; ++i) {
        int src = neighbor_rank(world_rank, world_size, -(i + 1));
        MPI_Irecv(&waitany_values[i], 1, MPI_INT, src, 200 + i, MPI_COMM_WORLD, &waitany_requests[i]);
    }
    for (int i = 0; i < message_count; ++i) {
        int dst = neighbor_rank(world_rank, world_size, +(i + 1));
        int payload = world_rank * 10 + i;
        MPI_Isend(&payload, 1, MPI_INT, dst, 200 + i, MPI_COMM_WORLD, &send_requests[i]);
    }

    int completed = 0;
    while (completed < message_count) {
        int index = MPI_UNDEFINED;
        MPI_Waitany(message_count, waitany_requests, &index, MPI_STATUS_IGNORE);
        if (index != MPI_UNDEFINED) {
            ++completed;
        }
    }
    MPI_Waitall(message_count, send_requests, MPI_STATUSES_IGNORE);

    // Phase 5: aggregate the received values with an all-reduce to create another collective.
    int local_sum = send_value;
    int global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("[collective_comm_demo] global sum=%d\n", global_sum);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
