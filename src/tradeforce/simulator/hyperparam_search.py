"""_summary_

"""
import optuna
from tradeforce import simulator
from tradeforce.simulator.utils import to_numba_dict


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
    "HyperbandPruner": optuna.pruners.HyperbandPruner,  # type: ignore
    "MedianPruner": optuna.pruners.MedianPruner,  # type: ignore
}


def get_optuna_storage(config_storage):
    if "sql:" not in str(config_storage):
        file_path = "./optuna_hyperparameter_search.log"
        lock_obj = optuna.storages.JournalFileOpenLock(file_path)
        optuna_storages = {
            "JournalStorage": optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
            )
        }
        config_storage = optuna_storages.get(config_storage, None)
    return config_storage


def run(root, optuna_config):
    # FIXME: prepare_sim does not exist anymore
    asset_prices, history_buy_factors = simulator.prepare_sim(root)
    config = optuna_config["config"]
    params = optuna_config["params"]

    def objective(trial):
        sim_params = root.config.to_dict()
        for param, val in params.items():
            param_type = determine_param_type(val)
            if param_type == "int":
                sim_params[param] = trial.suggest_int(param, val["min"], val["max"], step=val["step"])
            elif param_type == "float":
                sim_params[param] = trial.suggest_float(param, val["min"], val["max"], step=val["step"])

        sim_params_numba = to_numba_dict(sim_params)
        sim_result = simulator.run(root, asset_prices, history_buy_factors, sim_params_numba)
        return sim_result["profit"]

    study = optuna.create_study(
        study_name=config["study_name"],
        direction=config["direction"],
        storage=get_optuna_storage(config["storage"]),
        load_if_exists=config["load_if_exists"],
        sampler=optuna_samplers.get(config["sampler"], None)(),
        pruner=optuna_pruners.get(config["pruner"], None)(),
    )
    study.optimize(objective, n_trials=config["n_trials"], n_jobs=config["n_jobs"])
    return study
