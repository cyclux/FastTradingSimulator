"""_summary_
"""

import numpy as np
import numba as nb
import numba.typed as nb_types
from numba.core import types

NB_PARALLEL = False
NB_CACHE = False


def numba_dict_defaults(sim_params):
    set_default_params = [
        "buy_opportunity_factor",
        "buy_opportunity_boundary",
        "buy_opportunity_factor_min",
        "buy_opportunity_factor_max",
    ]
    for default_param in set_default_params:
        sim_params[default_param] = sim_params.get(default_param, 999)
    return sim_params


def to_numba_dict(sim_params):
    sim_params = numba_dict_defaults(sim_params)
    if sim_params["prefer_performance"] == "positive":
        sim_params["prefer_performance"] = 1
    if sim_params["prefer_performance"] == "negative":
        sim_params["prefer_performance"] = -1
    if sim_params["prefer_performance"] == "center":
        sim_params["prefer_performance"] = 0

    sim_params_numba = nb_types.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for key, val in sim_params.items():
        sim_params_numba[key] = np.float64(val)
    return sim_params_numba


@nb.njit(cache=NB_CACHE, parallel=False)
def sanitize_snapshot_params(snapshot_size, snapshot_amount, snapshot_idx_boundary):
    if snapshot_size <= 0:
        snapshot_size = -1
    if snapshot_amount <= 0:
        snapshot_amount = 1
    if snapshot_amount == 1 and snapshot_size == -1:
        snapshot_size = snapshot_idx_boundary
    if snapshot_amount > 1 and snapshot_size == -1:
        snapshot_size = snapshot_idx_boundary // snapshot_amount
    return snapshot_size, snapshot_amount


@nb.njit(cache=NB_CACHE, parallel=False)
def array_diff(arr1, arr2):
    diff_list = nb_types.List(set(arr1) - set(arr2))
    diff_array = np.array([x for x in diff_list])
    return diff_array


@nb.njit(cache=NB_CACHE, parallel=False)  # parallel not checked
def fill_nan(nd_array):
    shape = nd_array.shape
    nd_array = nd_array.ravel()
    nd_array[np.isnan(nd_array)] = 0
    nd_array = nd_array.reshape(shape)
    return nd_array


@nb.njit(cache=NB_CACHE, parallel=False)
def calc_fee(volume_crypto, maker_fee, taker_fee, price_current, order_type):
    volume_crypto = abs(volume_crypto)
    exchange_fee = taker_fee if order_type == "buy" else maker_fee
    amount_fee_crypto = (volume_crypto / 100) * exchange_fee
    volume_crypto_incl_fee = volume_crypto - amount_fee_crypto
    amount_fee_fiat = np.round(amount_fee_crypto * price_current, 3)
    return volume_crypto_incl_fee, amount_fee_crypto, amount_fee_fiat


@nb.njit(cache=NB_CACHE, parallel=False)
def get_snapshot_indices(snapshot_idx_boundary, snapshot_amount=10, snapshot_size=10000):
    snapshot_idx_boundary = np.int64(snapshot_idx_boundary - snapshot_size)
    snapshot_idxs = np.linspace(0, snapshot_idx_boundary, snapshot_amount).astype(np.int64)
    return snapshot_idxs


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def calc_metrics(soldbag):
    total_profit = soldbag[:, 14:15].sum()
    return np.int64(total_profit)
