/*
 * reduce_kchain.c: k-chain algorithms for MPI_Reduce.
 *
 * (C) Mikhail Kurnosov, 2016 <mkurnosov@gmail.com>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

enum {
    MPI_REDUCE_TAG = 1
};

/*
 * MPI_Reduce_linear: Linear/flat tree O(p) algorithm for reduce operation.
 *                    Result is combined in reverse order: a * (b * (c * (d * e))).
 *                    For commutative and non-commutative operations.
 *
 * Processes: 0 1 2 3 4
 * Data:      a b c d e
 * Steps at root 0:
 *   - recv data from process 4 (e)
 *   - recv data from process 3 (d)
 *   - local reduce: d * e
 *   - recv data from process 2 (c)
 *   - local reduce: c * (d * e)
 *   - recv data from process 1 (b)
 *   - local reduce: b * (c * (d * e))
 *   - local reduce: a * (b * (c * (d * e)))
 */
int MPI_Reduce_linear(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int rc = MPI_SUCCESS;
    int rank, commsize;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    if (root < 0 || root >= commsize)
        return MPI_ERR_ROOT;

    if (count == 0)
        return MPI_SUCCESS;

    if (rank != root) {
        MPI_Send(sendbuf, count, datatype, root, MPI_REDUCE_TAG, comm);
        return rc;
    }

    void *inplace_rbuf = NULL;
    void *tempbuf = NULL;
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    size_t size = count * extent;

    if (sendbuf == MPI_IN_PLACE) {
        sendbuf = recvbuf;
        inplace_rbuf = malloc(size);
        recvbuf = inplace_rbuf;
    }

    if (root == commsize - 1) {
        memcpy(recvbuf, sendbuf, size);
    } else {
        MPI_Recv(recvbuf, count, datatype, commsize - 1, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
    }
    // Assert: recvbuf contains data from process commsize - 1

    if (commsize > 1)
        tempbuf = malloc(size);

    for (int i = commsize - 2; i >= 0; i--) {
        void *lhs_buf = sendbuf;
        if (i != root) {
            MPI_Recv(tempbuf, count, datatype, i, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            lhs_buf = tempbuf;
        }
        // Local reduce: recvbuf[i] = lhs_buf[i] op recvbuf[i]
        MPI_Reduce_local(lhs_buf, recvbuf, count, datatype, op);
    }

    // Copy result to recv buffer
    if (inplace_rbuf)
        memcpy(sendbuf, recvbuf, size);

    if (tempbuf)
        free(tempbuf);
    if (inplace_rbuf)
        free(inplace_rbuf);

    return MPI_SUCCESS;
}

/*
 * MPI_Reduce_pipeline_commutative: Pipeline algorithm for reduce operation.
 *                                  For commutative operations only.
 *
 * Communication pattern: 0 <-- 1 <-- 2 <-- ... <-- p-1.
 * Ranks are shifted -- root always has shifted rank 0.
 *
 * Example: p = 5, root = 0.
 * Processes: 0 1 2 3 4
 * Data:      a b c d e
 * Result at root 0: a * (b * (c * (d * e)))
 * Result at root 0 (for MPI_IN_PLACE): (b * (c * (d * e))) * a
 *
 */
int MPI_Reduce_pipeline_commutative(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int rank, commsize;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    if (root < 0 || root >= commsize)
        return MPI_ERR_ROOT;

    if (count == 0)
        return MPI_SUCCESS;

    void *tempbuf = NULL;
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    size_t size = count * extent;

    int srank = (rank - root + commsize) % commsize;
    if (srank != commsize - 1)
        tempbuf = malloc(size);

    if (commsize == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);
        return MPI_SUCCESS;
    }

    if (srank == 0) {
        int recvfrom = (srank + 1 + root) % commsize;
        if (sendbuf == MPI_IN_PLACE) {
            MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tempbuf, recvbuf, count, datatype, op);
        } else {
            MPI_Recv(recvbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(sendbuf, recvbuf, count, datatype, op);
        }
    } else if (srank == commsize - 1) {
        int sendto = (srank - 1 + root + commsize) % commsize;
        MPI_Send(sendbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    } else {
        int recvfrom = (srank + 1 + root) % commsize;
        MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Reduce_local(sendbuf, tempbuf, count, datatype, op);
        int sendto = (srank - 1 + root + commsize) % commsize;
        MPI_Send(tempbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    }
    if (tempbuf)
        free(tempbuf);
    return MPI_SUCCESS;
}

/*
 * MPI_Reduce_kchain: k-chain algorithm for reduce operation (long chains, short chains).
 *                    For commutative operations only.
 *
 * Example: nchains = 4, p = 15
 * Root 0
 * Chain 1: 1  <--  2 <--  3 <-- 4    chain result: r1 = 1 * (2 * (3 * 4))
 * Chain 2: 5  <--  6 <--  7 <-- 8    chain result: r2 = 5 * (6 * (7 * 8))
 * Chain 3: 9  <-- 10 <-- 11          chain result: r3 = 9 * (10 * 11)
 * Chain 4: 12 <-- 13 <-- 14          chain result: r4 = 12 * (13 * 14)
 *
 * Result at root 0: r4 * (r3 * (r2 * (r1 * 0)))
 */
int MPI_Reduce_kchain(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, int nchains)
{
    int rank, commsize;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    //int nchains = 4;
    if (commsize - 1 < nchains)
        return MPI_ERR_COMM;

    if (root < 0 || root >= commsize)
        return MPI_ERR_ROOT;

    if (count == 0)
        return MPI_SUCCESS;

    void *tempbuf = NULL;
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    size_t size = count * extent;
    tempbuf = malloc(size);

    int srank = (rank - root + commsize) % commsize;
    if (commsize == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);
        free(tempbuf);
        return MPI_SUCCESS;
    }

    // Distribute processes between chains
    int short_chain_size = (commsize - 1) / nchains;
    int long_chain_size = (commsize - 1) / nchains;
    if ((commsize - 1) % nchains > 0)
        long_chain_size++;
    int long_chains = (commsize - 1) % nchains;
    int short_chains = nchains - long_chains;
    // Now we have nchains chains: long_chains contain long_chain_size processes and short_chains contain short_chain_size processes.

    if (srank == 0) {
        // Root process
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);

        for (int i = 0; i < long_chains; i++) {
            // Recv from head of each long chain
            int recvfrom = (i * long_chain_size + 1 + root) % commsize;
            MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tempbuf, recvbuf, count, datatype, op);
        }
        for (int i = 0; i < short_chains; i++) {
            // Recv from head of each short chain
            int recvfrom = (i * short_chain_size + long_chains * long_chain_size + 1 + root) % commsize;
            MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tempbuf, recvbuf, count, datatype, op);
        }
        free(tempbuf);
        return MPI_SUCCESS;
    }

    int rem = srank % long_chain_size;
    int proc_chain_size = long_chain_size;
    if (srank > long_chains * long_chain_size) {
        // Process in short chain
        rem = (srank - long_chains * long_chain_size) % short_chain_size;
        proc_chain_size = short_chain_size;
    }

    if (rem == 0) {
        // Process is leaf (last in chain)
        int sendto = (srank - 1 + root + commsize) % commsize;
        if (proc_chain_size == 1)
            sendto = root;
        MPI_Send(sendbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    } else {
        // Process is internal node or head of chain
        int recvfrom = (srank + 1 + root) % commsize;
        MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Reduce_local(sendbuf, tempbuf, count, datatype, op);
        int sendto = (rem != 1) ? (srank - 1 + root + commsize) % commsize : root;
        MPI_Send(tempbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    }

    free(tempbuf);
    return MPI_SUCCESS;
}

/*
 * MPI_Reduce_kchain_opt: k-chain algorithm for reduce operation (short chains, long chains).
 *                        For commutative operations only.
 *
 * Example: nchains = 4, p = 15
 * Root 0
 * Chain 1: 1  <--  2 <--  3          chain result: r1 = 1 * (2 * 3)
 * Chain 2: 4  <--  5 <--  6          chain result: r2 = 4 * (5 * 6)
 * Chain 3: 7  <--  8 <--  9 <-- 10   chain result: r3 = 7 * (8 * (9 * 10))
 * Chain 4: 11 <-- 12 <-- 13 <-- 14   chain result: r4 = 11 * (12 * (13 * 14))
 *
 * Result at root 0: r4 * (r3 * (r2 * (r1 * 0)))
 */
int MPI_Reduce_kchain_opt(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int rank, commsize;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    int nchains = 4;
    if (commsize - 1 < nchains)
        return MPI_ERR_COMM;

    if (root < 0 || root >= commsize)
        return MPI_ERR_ROOT;

    if (count == 0)
        return MPI_SUCCESS;

    void *tempbuf = NULL;
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    size_t size = count * extent;
    tempbuf = malloc(size);

    int srank = (rank - root + commsize) % commsize;
    if (commsize == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);
        free(tempbuf);
        return MPI_SUCCESS;
    }

    // Distribute processes between chains
    int short_chain_size = (commsize - 1) / nchains;
    int long_chain_size = (commsize - 1) / nchains;
    if ((commsize - 1) % nchains > 0)
        long_chain_size++;
    int long_chains = (commsize - 1) % nchains;
    int short_chains = nchains - long_chains;
    // Now we have nchains chains: long_chains contain long_chain_size processes and short_chains contain short_chain_size processes.

    if (srank == 0) {
        // Root process
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);

        for (int i = 0; i < short_chains; i++) {
            // Recv from head of each short chain
            int recvfrom = (i * short_chain_size + 1 + root) % commsize;
            MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tempbuf, recvbuf, count, datatype, op);
        }
        for (int i = 0; i < long_chains; i++) {
            // Recv from head of each long chain
            int recvfrom = (i * long_chain_size + short_chains * short_chain_size + 1 + root) % commsize;
            MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tempbuf, recvbuf, count, datatype, op);
        }
        free(tempbuf);
        return MPI_SUCCESS;
    }

    int rem = srank % short_chain_size;
    int proc_chain_size = short_chain_size;
    if (srank > short_chains * short_chain_size) {
        // Process in long chain
        rem = (srank - short_chains * short_chain_size) % long_chain_size;
        proc_chain_size = long_chain_size;
    }
    //printf("proc %d chain_size %d, rem %d\n", srank, proc_chain_size, rem);

    if (rem == 0) {
        // Process is leaf (last in chain)
        int sendto = (srank - 1 + root + commsize) % commsize;
        if (proc_chain_size == 1)
            sendto = root;
        MPI_Send(sendbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    } else {
        // Process is internal node or head of chain
        int recvfrom = (srank + 1 + root) % commsize;
        MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Reduce_local(sendbuf, tempbuf, count, datatype, op);
        int sendto = (rem != 1) ? (srank - 1 + root + commsize) % commsize : root;
        MPI_Send(tempbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    }

    free(tempbuf);
    return MPI_SUCCESS;
}

/*
 * MPI_Reduce_adaptive_kchain:
 */
int MPI_Reduce_adaptive_kchain(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int rank, commsize;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    if (commsize < 2)
        return MPI_ERR_COMM;

    if (root < 0 || root >= commsize)
        return MPI_ERR_ROOT;

    if (count == 0)
        return MPI_SUCCESS;

    void *tempbuf = NULL;
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    size_t size = count * extent;
    tempbuf = malloc(size);

    int srank = (rank - root + commsize) % commsize;
    if (commsize == 1) {
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);
        free(tempbuf);
        return MPI_SUCCESS;
    }

    if (srank == 0) {
        // Root process
        if (sendbuf != MPI_IN_PLACE)
            memcpy(recvbuf, sendbuf, size);

        // Recv from head of each chain
        int nchains = ceil((sqrt(8 * (commsize - 1) + 1) - 1.0) / 2.0);
        int head = 1;
        for (int i = 0; i < nchains; i++) {
            MPI_Recv(tempbuf, count, datatype, head, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tempbuf, recvbuf, count, datatype, op);
            head += i + 1;
        }
        free(tempbuf);
        return MPI_SUCCESS;
    }
    int chain = ceil((sqrt(8 * srank + 1) - 1.0) / 2.0);
    int tail = (chain + chain * chain) / 2;
    int head = tail - chain + 1;
    if (tail >= commsize)
        tail = commsize - 1;

    int prev = (srank < tail) ? srank + 1 : MPI_PROC_NULL;
    int next = (srank > head) ? srank - 1 : 0;
    //printf("proc %d chain %d head %d tail %d : next %d prev %d\n", srank, chain, head, tail, next, prev);

    if (srank == tail) {
        // Process is leaf (last in chain)
        int sendto = (next + root) % commsize;
        MPI_Send(sendbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    } else {
        // Process is internal node or head of chain
        int recvfrom = (prev + root) % commsize;
        MPI_Recv(tempbuf, count, datatype, recvfrom, MPI_REDUCE_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Reduce_local(sendbuf, tempbuf, count, datatype, op);
        int sendto = (next + root) % commsize;
        MPI_Send(tempbuf, count, datatype, sendto, MPI_REDUCE_TAG, comm);
    }

    free(tempbuf);
    return MPI_SUCCESS;
}

double measure_clock_offset_adaptive(MPI_Comm comm, int root, int peer)
{
    int rank, commsize, rttmin_notchanged = 0;
    double starttime, stoptime, peertime, rtt, rttmin = 1E12,
           invalidtime = -1.0, offset;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &commsize);

    offset = 0.0;
    for (;;) {
        if (rank != root) {
            /* Peer process */
            starttime = MPI_Wtime();
            MPI_Send(&starttime, 1, MPI_DOUBLE, root, 0, comm);
            MPI_Recv(&peertime, 1, MPI_DOUBLE, root, 0, comm, MPI_STATUS_IGNORE);
            stoptime = MPI_Wtime();
            rtt = stoptime - starttime;

            if (rtt < rttmin) {
                rttmin = rtt;
                rttmin_notchanged = 0;
                offset = peertime - rtt / 2.0 - starttime;
            } else {
                if (++rttmin_notchanged == 100) {
                    MPI_Send(&invalidtime, 1, MPI_DOUBLE, root, 0, comm);
                    break;
                }
            }
        } else {
            /* Root process */
            MPI_Recv(&starttime, 1, MPI_DOUBLE, peer, 0, comm, MPI_STATUS_IGNORE);
            peertime = MPI_Wtime();
            if (starttime < 0.0)
                break;
            MPI_Send(&peertime, 1, MPI_DOUBLE, peer, 0, comm);
        }
    } /* for */
    return offset;
}

double sync_clock_linear(MPI_Comm comm, int root)
{
    int rank, commsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &commsize);
        
    if (commsize < 2)
        return 0.0;

    double local_offset = 0.0;
    for (int i = 1; i < commsize; i++) {
        MPI_Barrier(comm);
        if (rank == root || rank == i) {
            local_offset = measure_clock_offset_adaptive(comm, root, i);
        }
    }
    return local_offset;
}

double measure_bcast_double(MPI_Comm comm)
{
    double buf, totaltime = 0.0, optime, maxtime = 0.0;
    int i, nreps = 3;

    /* Warmup call */
    MPI_Bcast(&buf, 1, MPI_DOUBLE, 0, comm);
    /* Measures (upper bound) */
    for (i = 0; i < nreps; i++) {
        MPI_Barrier(comm);
        optime = MPI_Wtime();
        MPI_Bcast(&buf, 1, MPI_DOUBLE, 0, comm);
        optime = MPI_Wtime() - optime;
        MPI_Reduce(&optime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        totaltime = totaltime > maxtime ? totaltime : maxtime;
    }
    return totaltime;
}

int main(int argc, char **argv)
{
    int rank, commsize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);    

    int kchains = (argc > 1) ? atoi(argv[1]) : 4;    

    // Warmup
    double *rbuf = malloc(sizeof(*rbuf) * 1024);
    double *sbuf = malloc(sizeof(*sbuf) * 1024);
    for (int i = 0; i < 1024; i++)
        sbuf[i] = rank;
    MPI_Reduce_kchain(sbuf, rbuf, 1024, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, kchains);
    
    // Synchronize clocks
    double clock_offset = sync_clock_linear(MPI_COMM_WORLD, 0);

    // Measure bcast time
    double bcast_time = measure_bcast_double(MPI_COMM_WORLD);    
    MPI_Barrier(MPI_COMM_WORLD);

    int is_late = 1;
    double barrier_time, tlocal = 0.0;
    
    for (double delta = 10.0; is_late; delta *= 2.0) {
        // Broadcast barrier time
        if (rank == 0)
            barrier_time = MPI_Wtime() + bcast_time * delta;
        MPI_Bcast(&barrier_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Translate global time to local
        int is_local_late = 0;
        barrier_time -= clock_offset;    
            
        // Wait for barrier time
        volatile double t = MPI_Wtime();
        if (t > barrier_time)
            is_local_late = 1;
        else {        
            while ((t = MPI_Wtime()) < barrier_time) 
                /* Wait */;
        }
        
        // Measure function   
        tlocal = MPI_Wtime();
        MPI_Reduce_kchain(sbuf, rbuf, 1024, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, kchains);
        tlocal = MPI_Wtime() - tlocal;

        // Check results
        MPI_Allreduce(&is_local_late, &is_late, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);                
        /*
        if (!is_late) {
            if (rank == 0) {
                printf("Sync: %.2f\n", delta);
            }
        }
        */
    }    
                
    double tmax = 0.0;
    MPI_Reduce(&tlocal, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%6d %4d %12.6f\n", commsize, kchains, tmax);
    }    

    MPI_Finalize();
    return 0;
}
