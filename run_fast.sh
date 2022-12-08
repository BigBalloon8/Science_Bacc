
# $MPI_NET_INT is the interface name for the network that is used for MPI communication
python3 main.py --experiment_type=H

tc qdisc add dev $MPI_NET_INT root tbf rate 1gbit

JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl ^openib --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=C

JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl ^openib --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=E3

tc qdisc del dev $MPI_NET_INT root