import train
import train_model
import data_set
import optax
import argparse
import eval
from mpi4py import MPI




def main():
    """Experiments:

        Models: VGG16, ResNet50, InceptionV4

        Hypothetical - Single Node (used to calculate theoretical speedup)
    
        Control - No compression, no gradient accumulation

        Experiment 1 - Gradient sparcification, no compression
        
        Experiment 2 - No Gradient sparcification, compression

        Experiment 3 - gradient sparcification, compression
        
    """
    models = [("VGG16", train_model.VGG16, "244"), ("ResNet50", train_model.ResNet50, "244"), ("InceptionV4", train_model.InceptionV4, "299")]
    optimizer = optax.adam(learning_rate=0.001)
    num_classes = 100
    comm = MPI.COMM_WORLD
    train_ds, test_ds = data_set.get_data(num_classes, comm)
    kwargs = {"optimizer": optimizer, "num_classes": num_classes, "train_ds": train_ds, "test_ds": test_ds, "comm": comm}

    # parse arguments for experiment type
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, default="C", help="Type of experiment to run")

    #Hypothetical
    if parser.parse_args().experiment_type == "H":
        for model_name, model_func, input_shape in models:
            experiment_type = ("hypothetical", model_name)
            params = train.train(model_func=model_func, experiment_type=experiment_type, 
                        input_shape=input_shape,hypothesis=True, **kwargs)
            eval.eval_model(test_ds=test_ds, params=params,experiment_type=experiment_type)
            
    #Control
    elif parser.parse_args().experiment_type == "C":
        for model_name, model_func, input_shape in models:
            experiment_type = ("control", model_name)
            params = train.train(model_func=model_func, experiment_type=experiment_type, 
                        input_shape=input_shape, hypothesis=False, **kwargs)
            eval.eval_model(test_ds=test_ds, params=params,experiment_type=experiment_type)
    #Experiment 1
    elif parser.parse_args().experiment_type == "E1":
        for model_name, model_func, input_shape in models:
            experiment_type = ("experiment_1", model_name)
            params = train.train(model_func=model_func, experiment_type=experiment_type, 
                        input_shape=input_shape, hypothesis=False, gradient_spar=True, **kwargs)
            eval.eval_model(test_ds=test_ds, params=params,experiment_type=experiment_type)
    #Experiment 2
    elif parser.parse_args().experiment_type == "E2":
        for model_name, model_func, input_shape in models:
            experiment_type = ("experiment_2", model_name)
            params = train.train(model_func=model_func, experiment_type=experiment_type, 
                        input_shape=input_shape, hypothesis=False, compression=True, **kwargs)
            eval.eval_model(test_ds=test_ds, params=params,experiment_type=experiment_type)
    #Experiment 3
    elif parser.parse_args().experiment_type == "E3":
        for model_name, model_func, input_shape in models:
            experiment_type = ("experiment_3", model_name)
            params = train.train(model_func=model_func, experiment_type=experiment_type, 
                        input_shape=input_shape, hypothesis=False, gradient_spar=True, compression=True, **kwargs)
            eval.eval_model(test_ds=test_ds, params=params,experiment_type=experiment_type)

    #Run with pytohn main.py --experiment_type=H
if __name__ == "__main__":
    main()
