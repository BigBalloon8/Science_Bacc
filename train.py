import jax
import jax.numpy as jnp
import train_model
import low_bandwidth
import optax
import numpy as np
import data_set
import communication
import result_collecting
import time

def compute_metrics(*, logits, labels, num_classes):
    loss = train_model.get_loss(logits=logits, labels=labels, num_classes=num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy,}
    return metrics

@jax.jit
def train_step(params, opt_state, temp_grads, batch_image, batch_label, comm, model, optimizer, num_classes, compression=False, gradient_spar=False):

    @jax.jit
    def forward(params, model, batch_image, batch_label, loss_func):
        logits = model.apply(params, batch_image)
        loss = loss(logits=logits, labels=batch_label, num_classes=num_classes)
        return loss, logits
    
    grad_fn = jax.value_and_grad(forward, has_aux=True)
    if type(model).__name__ == "inception_v4":
        loss_func = train_model.get_loss
    else:
        loss_func = train_model.l2_loss
    (_, logits), grads = grad_fn(params, model, batch_image, batch_label, loss_func)
    
    if compression and gradient_spar:
        grads, temp_grads = low_bandwidth.gradient_sparcification(grads, temp_grads, 0.000001)
        compressed_grads = low_bandwidth.compress(grads)
        grads = communication.ring_all_reduce(comm, grads, compression=True)
        grads = low_bandwidth.decompress(compressed_grads, grads)
    elif not compression and gradient_spar:
        grads, temp_grads = low_bandwidth.gradient_sparcification(grads, temp_grads, 0.000001)
        grads = communication.ring_all_reduce(comm, grads, compression=False)
    elif compression and not gradient_spar:
        compressed_grads = low_bandwidth.compress(grads)
        grads = communication.ring_all_reduce(comm, grads, compression=True)
        grads = low_bandwidth.decompress(compressed_grads, grads)


    updates, opt_state = optimizer.update(grads, opt_state, params)  # optimizer defined in global name space
    params = optax.apply_updates(params, updates)
    
    metrics = compute_metrics(logits=logits, labels=batch_label)
    return params, opt_state, temp_grads, metrics


@jax.jit
def hypo_train_step(params, opt_state,  batch_image, batch_label, model, optimizer, num_classes):

    @jax.jit
    def forward(params, model, batch_image, batch_label):
        logits = model.apply(params, batch_image)
        loss = train_model.get_loss(logits=logits, labels=batch_label, num_classes=num_classes)
        return loss, logits

    grad_fn = jax.value_and_grad(forward, has_aux=True)
    (_, logits), grads = grad_fn(params, model, batch_image, batch_label)

    updates, opt_state = optimizer.update(grads, opt_state, params)  # optimizer defined in global name space
    params = optax.apply_updates(params, updates)
    
    metrics = compute_metrics(logits=logits, labels=batch_label)
    return params, opt_state, metrics



def train_epoch(params, opt_state, train_ds, temp_grads, batch_size:int, 
                epoch:int, rng, comm, model, optimizer, experiment_type, 
                input_shape, num_classes, compression:bool, gradient_spar:bool, hypothesis:bool):

    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    permed_data = jax.random.permutation(rng, train_ds_size)  # shuffle the data
    permed_data = permed_data[:steps_per_epoch * batch_size]  # remove the last batch if it is not full
    permed_data = permed_data.reshape((steps_per_epoch, batch_size))  # split the data into batches

    batch_metrics = []

    start = time.perf_counter()
    for batch in permed_data:
        batch = {k: v[batch, ...] for k, v in train_ds.items()}
        if hypothesis:
            params, opt_state, metrics = hypo_train_step(params, opt_state,  batch['image'], batch['label'], model, optimizer, num_classes)
        else:
            params, opt_state, temp_grads, metrics = train_step(params, opt_state, temp_grads, jax.image.resize(image=batch["image"],shape=(batch_size,input_shape,input_shape,3)), batch["label"], comm, model, optimizer, num_classes, compression, gradient_spar, hypothesis)
        batch_metrics.append(metrics)
    
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {i: np.mean([metrics[i] for metrics in batch_metrics_np])for i in batch_metrics_np[0]}
    print(f"\nepoch: {epoch}  -  loss: {epoch_metrics_np['loss']}  -  accuracy: {epoch_metrics_np['accuracy'] * 100}")
    if not hypothesis:
        if comm.Get_rank() == 0:
            result_collecting.save_as_json(experiment_type=experiment_type, epoch=epoch, loss=epoch_metrics_np['loss'], accuracy=epoch_metrics_np['accuracy'], time_for_epoch=(time.perf_counter-start))
    else:
        result_collecting.save_as_json(experiment_type=experiment_type, epoch=epoch, loss=epoch_metrics_np['loss'], accuracy=epoch_metrics_np['accuracy'], time_for_epoch=(time.perf_counter-start))

    return params, opt_state, temp_grads



def train(optimizer, train_ds, comm,experiment_type, num_classes, get_model_func, compression, gradient_spar, input_shape, hypothesis = False):
    #print(train_ds)

    num_epochs = 20
    batch_size = 32

    params, model = get_model_func(num_classes=num_classes)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    temp_grads = low_bandwidth.get_temp_grads(params)
    #print(params,temp_grads)
    opt_state = optimizer.init(params)
    
    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        #print(temp_grads)
        params, opt_state, temp_grads = train_epoch(params=params, opt_state=opt_state, train_ds=train_ds, temp_grads=temp_grads, batch_size=batch_size, epoch=epoch, rng=rng, comm=comm, model=model, optimizer=optimizer, experiment_type=experiment_type, num_classes=num_classes, input_shape=input_shape, compression=compression, gradient_spar=gradient_spar, hypothesis = hypothesis)
        return params


