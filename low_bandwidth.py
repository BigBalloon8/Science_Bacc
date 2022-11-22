import jax
import jax.numpy as jnp
import train_model


def get_temp_grads(params):

    params, flattened =  jax.tree_flatten(params)
    flat_temp_params = []
    for i in params:
      flat_temp_params.append(jnp.zeros_like(i))
    params = jax.tree_unflatten(flattened, flat_temp_params)
    return params


@jax.jit
def gradient_sparcification(grads, temp_grads, threshold):
    flat_grads = jax.tree_util.tree_flatten(grads)
    flat_temp_grads = jax.tree_util.tree_flatten(temp_grads)
    pre_grads = jax.tree_map(jax.vmap(lambda x,y: x+y), flat_temp_grads[0], flat_grads[0])
    grads =  [jnp.where(jax.lax.gt(jnp.abs(x), jnp.float32(threshold)), x, jnp.float32(0.0)) for x in pre_grads]
    grads = jax.tree_util.tree_unflatten(flat_grads[1], grads)
    flat_grads = jax.tree_util.tree_flatten(grads)
    temp_grads = jax.tree_util.tree_unflatten(flat_temp_grads[1],jax.tree_map(jax.vmap(lambda x,y:x-y), pre_grads, flat_grads[0]))
    return grads, temp_grads


@jax.jit
def compress(grads):
    flat_grads = jax.tree_util.tree_flatten(grads)
    compressed_grads = [jax.tree_map(jax.vmap(lambda x : jnp.float16(x)), i) for i in flat_grads[0]]
    return compressed_grads

@jax.jit
def decompress(compressed_grads, grads):
    flat_grads = jax.tree_util.tree_flatten(grads)
    inter_flat_grads = [jax.tree_map(lambda x : jnp.float32(x), i) for i in compressed_grads]#TODO find a way to vmap
    grads = jax.tree_util.tree_unflatten(flat_grads[1],inter_flat_grads)
    return grads


