# Collective Communication Demo

This mini application exercises the collective and completion routines that ANACIN-X now records explicitly (`MPI_Barrier`,
`MPI_Ibarrier`/`MPI_Wait`, `MPI_Waitall`, `MPI_Waitany`, and `MPI_Allreduce`).
Use it to validate that your instrumentation stack is capturing the new
collective metadata and that the event-graph ingestion classifies the
corresponding vertices with the `collective_type` and `is_collective` attributes.

## Build Instructions

The demo is a single C source file that depends only on an MPI implementation.
You can build it either with CMake or directly with your MPI compiler wrapper.

### Option A: CMake

```bash
cmake -S apps/collective_comm_demo -B apps/collective_comm_demo/build
cmake --build apps/collective_comm_demo/build
```

The resulting executable is
`apps/collective_comm_demo/build/collective_comm_demo`.

### Option B: MPI compiler wrapper

```bash
mpicc -std=c99 -O2 \
  apps/collective_comm_demo/src/collective_comm_demo.c \
  -o apps/collective_comm_demo/collective_comm_demo
```

## Running the Demo

Execute the program with two or more ranks:

```bash
mpirun -n 4 apps/collective_comm_demo/build/collective_comm_demo
```

On launch the ranks synchronize with a barrier, exchange ring traffic that is
completed via `MPI_Waitall`, wait on an `MPI_Ibarrier`, drive an additional set of
receives with `MPI_Waitany`, and finalize with an `MPI_Allreduce` so the event
trace contains a rich mixture of collective and point-to-point activity.

## Tracing and Event-Graph Construction

1. **Collect traces** – run the executable under PnMPI with the ANACIN-X modules
enabled. Substitute the paths that correspond to your build of PnMPI and the
configuration that loads `sst-dumpi`, `Pluto`, and `CSMPI`:

   ```bash
   LD_PRELOAD=/path/to/libpnmpi.so \
   PNMPI_LIB_PATH=/path/to/pnmpi/modules \
   PNMPI_CONF=/path/to/pnmpi/configs/collective_demo.conf \
   mpirun -n 4 apps/collective_comm_demo/build/collective_comm_demo
   ```

   Place the output from each run in its own run directory just as you would for
   a production workload so the downstream tooling can iterate over the sample.

2. **Convert to event graphs** – invoke `dumpi_to_graph` on each run directory as
   outlined in the project README. The resulting GraphML will include
   vertices whose `event_type` is set to values such as `barrier`, `waitall`,
   `waitany`, and `wait` with `collective_type` mirroring the normalized MPI call.

3. **Analyze** – slice and analyze the graphs with the updated ANACIN-X
   pipeline. Collective synchronization vertices remain eligible for slicing and
   for any kernel policies that reference the `collective_type` attribute.

Running the demo through this workflow provides a compact regression test for
collective tracing end-to-end, from execution through visualization.
