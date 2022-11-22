import tensorflow_datasets as tfds
import jax.numpy as jnp

def get_data(cifar_data_set, comm):
    ds_builder = tfds.builder(f'cifar{cifar_data_set}')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    train_ds = shard_data(train_ds, comm)
    return train_ds, test_ds

def shard_data(train_ds, comm):
    my_rank = comm.Get_rank()
    num_workers = comm.Get_size()
    len_per_worker = len(train_ds['image']) // num_workers
    train_ds["image"] = train_ds["image"][len_per_worker*my_rank:len_per_worker*(my_rank+1)]
    train_ds["label"] = train_ds["label"][len_per_worker*my_rank:len_per_worker*(my_rank+1)]
    
    return train_ds

