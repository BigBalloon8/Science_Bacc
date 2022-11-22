import jax
import train
import result_collecting


def eval_step(params, batch, get_model_func, num_classes):
    logits = get_model_func(no_params=True).apply(params, batch['image'], num_classes=num_classes)
    return train.compute_metrics(logits=logits, labels=batch['label'])


def eval_model(test_ds, params, experiment_type):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    result_collecting.save_as_json(experiment_type=experiment_type, epoch='eval', loss=summary['loss'], accuracy=summary['accuracy'], time_for_epoch=0)
    return summary['loss'], summary['accuracy']
