
#$HOSTS in format 192.0.0.2:n_1,192.0.0.3:n_2,192.0.0.3:n_3,... (where n_i is the number of gpus on the i-th host)

#mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl tcp,sm,self --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=

python3 main.py --experiment_type=H

JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl ^openib --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=C
tc qdisc add dev $MPI_NET_INT root tbf rate 1gbit
JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl ^openib --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=E1
JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl ^openib --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=E2
JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun --np $NUM_GPUS --H $HOSTS --map-by slot --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca pml ob1 --mca btl ^openib --mca btl_tcp_if_include $MPI_NET_INT python3 main.py --experiment_type=E3
tc qdisc del dev $MPI_NET_INT root