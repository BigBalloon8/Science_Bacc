import json
import os

def save_as_json(experiment_type, epoch, loss, accuracy, time_for_epoch):
    with open(f"{os.path.dirname(__file__)}/results.json", "r") as file:
        results = json.load(file)

    if experiment_type not in results:
        raise KeyError(f"experiment_type incorrectly named, Possible names {results.keys}")

    results[experiment_type[0]][experiment_type[1]][epoch] = {}
    results[experiment_type[0]][experiment_type[1]][epoch]["loss"] = loss
    results[experiment_type[0]][experiment_type[1]][epoch]["accuracy"] = accuracy
    results[experiment_type[0]][experiment_type[1]][epoch]["time_for_epoch"] = time_for_epoch

    with open(f"{os.path.dirname(__file__)}/results.json", "w") as file:
        results = json.dump(results, file)

