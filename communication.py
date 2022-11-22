import jax
import jax.numpy as jnp
import mpi4jax

@jax.jit
def ring_all_reduce(comm, grads, compression=True): 
    # Copied from a personal project of mine DEICNT https://github.com/mayfieldmobster
    rank = comm.Get_rank()
    size = comm.Get_size()
    if not compression:
        flat_grads = jax.tree_flatten(grads)
        grads = flat_grads[0]

    grads = jnp.array_split(grads, size)
    grads = jnp.concatenate((grads[:(rank + 1)] + grads[(rank + 1):]))

    token = mpi4jax.barrier(comm=comm)

    for i, j in enumerate(grads): 

        if comm.Get_rank() == 0:
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)
            new_chunk, token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)
            grads[(i + 1) % size] = jax.vmap(jax.lax.add)(grads[(i + 1) % size], new_chunk)

        else:
            new_chunk, token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)
            grads[(i + 1) % size] = jax.vmap(jax.lax.add)( grads[(i + 1) % size], new_chunk) 
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)

    grads = jnp.concatenate((grads[1:] + grads[:1]))  # set completed arr to index 0

    for i, j in enumerate(grads):
        if comm.Get_rank() == 0:
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)
            grads[(i + 1) % size], token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)

        else:
            grads[(i + 1) % size], token = mpi4jax.recv(j, source=((rank - 1) % size), comm=comm, token=token)
            token = mpi4jax.send(j, dest=((rank + 1) % size), comm=comm, token=token)

    grads = jnp.concatenate((grads[:(rank - 2) % size] + grads[(rank - 2) % size:]))  # arrange back to norm
    grads = jax.vmap(jax.lax.div)(grads, size)

    if not compression:
        return jax.tree_unflatten(flat_grads[1], grads)
    return grads