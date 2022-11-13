import jax
import jax.numpy as jnp
import train_model
import low_bandwidth
import optax
import numpy as np
import data_set
from mpi4py import MPI
import communication
import result_collecting
import time

def compute_metrics(*, logits, labels):
    loss = train_model.get_loss(logits=logits, labels=labels, num_classes=num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy,}
    return metrics

@jax.jit
def train_step(params, opt_state, temp_grads, batch_image, batch_label, comm):
    @jax.jit
    def forward(params):
        resnet = train_model.get_resnet(no_params=True, num_classes=num_classes)
        logits = resnet.apply(params, batch_image)
        loss = train_model.get_loss(logits=logits, labels=batch_label, num_classes=num_classes)
        return loss, logits
    
    grad_fn = jax.value_and_grad(forward, has_aux=True)
    (_, logits), grads = grad_fn(params)

    grads, temp_grads = low_bandwidth.gradient_accum(grads, temp_grads, 0.000001)
    compressed_grads = low_bandwidth.compress(grads)
    grads = communication.ring_all_reduce(comm, grads, compression=True)
    grads = low_bandwidth.decompress(compressed_grads, grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)  # optimizer defined in global name space
    params = optax.apply_updates(params, updates)
    
    metrics = compute_metrics(logits=logits, labels=batch_label)
    return params, opt_state, temp_grads, metrics


def train_epoch(params, opt_state, train_ds, temp_grads, batch_size, epoch, rng, comm):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    permed_data = jax.random.permutation(rng, train_ds_size)
    permed_data = permed_data[:steps_per_epoch * batch_size]
    permed_data = permed_data.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    if not temp_grads:
        temp_batch = {k: v[permed_data[0], ...] for k, v in train_ds.items()}
        temp_grads = low_bandwidth.get_temp_grads(params, temp_batch["image"], temp_batch["label"], num_classes=num_classes)
    start = time.perf_counter()
    for batch in permed_data:
        batch = {k: v[batch, ...] for k, v in train_ds.items()}
        #print(jax.make_jaxpr(train_step)(state,batch,temp_grads))
        params, opt_state, temp_grads, metrics = train_step(params, opt_state, temp_grads, batch["image"], batch["label"], comm)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {i: np.mean([metrics[i] for metrics in batch_metrics_np])for i in batch_metrics_np[0]}
    print(f"\nepoch: {epoch}  -  loss: {epoch_metrics_np['loss']}  -  accuracy: {epoch_metrics_np['accuracy'] * 100}")
    result_collecting.save_as_json(experiment_type=experiment_type, epoch=epoch, loss=epoch_metrics_np['loss'], accuracy=epoch_metrics_np['accuracy'], time_for_epoch=(time.perf_counter-start))
    return params, opt_state, temp_grads


def eval_step(params, batch):
    logits = train_model.get_resnet(no_params=True).apply(params, batch['image'], num_classes=num_classes)
    return compute_metrics(logits=logits, labels=batch['label'])


def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


def train(optimizer, train_ds, test_ds):
    #print(train_ds)

    num_epochs = 50
    batch_size = 32
    comm = MPI.COMM_WORLD
    print(train_ds["image"][0].shape)

    params, _ = train_model.get_resnet(num_classes=num_classes)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    temp_grads = False
    #print(params,temp_grads)
    opt_state = optimizer.init(params)
    
    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        #print(temp_grads)
        params, opt_state, temp_grads = train_epoch(params, opt_state, train_ds, temp_grads, batch_size, epoch, rng, comm)

    test_loss, test_accuracy = eval_model(params, test_ds)

    print(f"model_loss: {test_loss}  -  model_accuracy: {test_accuracy * 100}%")



if __name__ == "__main__":
    num_classes = 10
    experiment_type = "control_results"
    train_ds, test_ds = data_set.get_data(cifar_data_set=num_classes)
    optimizer = optax.sgd(learning_rate = 0.001)
    train(optimizer, train_ds, test_ds)
    
