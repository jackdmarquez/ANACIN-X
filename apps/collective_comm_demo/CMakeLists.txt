cmake_minimum_required(VERSION 3.12)
project(collective_comm_demo C)

find_package(MPI REQUIRED)

add_executable(collective_comm_demo src/collective_comm_demo.c)
target_link_libraries(collective_comm_demo PRIVATE MPI::MPI_C)

set_target_properties(collective_comm_demo PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
)
