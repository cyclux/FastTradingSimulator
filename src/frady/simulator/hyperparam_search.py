"""_summary_

"""
from joblib import parallel_backend
import optuna
from frady import simulator
from frady.simulator.utils import to_numba_dict


def determine_param_type(param_vals):
    is_float = any(isinstance(val, float) for val in param_vals.values())
    param_type = "float" if is_float else "int"
    return param_type


optuna_samplers = {
    "RandomSampler": optuna.samplers.RandomSampler,
    "TPESampler": optuna.samplers.TPESampler,
    "BruteForceSampler": optuna.samplers.BruteForceSampler,
}

optuna_pruners = {
    "HyperbandPruner": optuna.pruners.HyperbandPruner,
    "MedianPruner": optuna.pruners.MedianPruner,
}

file_path = "./optuna_hyperparameter_search.log"
lock_obj = optuna.storages.JournalFileOpenLock(file_path)

optuna_storages = {
    "JournalStorage": optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
}


def run(fts, optuna_config):
    bfx_history, history_buy_factors = simulator.prepare_sim(fts)
    config = optuna_config["config"]
    params = optuna_config["params"]

    def objective(trial):
        sim_params = fts.config.to_dict()
        for param, val in params.items():
            param_type = determine_param_type(val)
            if param_type == "int":
                sim_params[param] = trial.suggest_int(param, val["min"], val["max"], step=val["step"])
            elif param_type == "float":
                sim_params[param] = trial.suggest_float(param, val["min"], val["max"], step=val["step"])

        sim_params_numba = to_numba_dict(sim_params)
        sim_result = simulator.run(fts, bfx_history, history_buy_factors, sim_params_numba)
        return sim_result["profit"]

    study = optuna.create_study(
        study_name=config["study_name"],
        direction=config["direction"],
        storage=optuna_storages.get(config["storage"], None),
        load_if_exists=config["load_if_exists"],
        sampler=optuna_samplers.get(config["sampler"], None)(),
        pruner=optuna_pruners.get(config["pruner"], None)(),
    )
    with parallel_backend("multiprocessing", n_jobs=config["n_jobs"]):
        study.optimize(objective, n_trials=config["n_trials"], n_jobs=1)
    return study
