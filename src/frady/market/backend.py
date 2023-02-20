"""_summary_

Returns:
    _type_: _description_
"""

import sys
from urllib.parse import quote_plus
import numpy as np
import pandas as pd

from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure, CollectionInvalid, ConfigurationError, OperationFailure

from frady.utils import get_df_datetime_index

##################
# DB interaction #
##################


def drop_dict_na_values(record):
    return {key: record[key] for key in record if not pd.isna(record[key])}


class Backend:
    """Fetch market history from disk and store in DataFrame

    Returns:
        _type_: _description_
    """

    def __init__(self, root):

        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)
        self.sync_check_needed = False
        self.is_collection_new = True
        self.is_filled_na = False
        self.backend_client = self.connect_backend()

    #################
    # Connecting DB #
    #################

    def construct_uri(self):
        if self.config.backend_user and self.config.backend_password:
            backend_uri = (
                f"mongodb://{quote_plus(self.config.backend_user)}"
                + f":{quote_plus(self.config.backend_password)}"
                + f"@{self.config.backend_host}"
            )
        else:
            backend_uri = f"mongodb://{self.config.backend_host}"
        return backend_uri

    def connect_backend(self):
        backend_client = None
        if self.config.backend == "mongodb":
            backend_uri = self.construct_uri()
            try:
                backend_client = MongoClient(backend_uri)
                self.backend_db = backend_client[self.config.mongo_exchange_db]
                self.mongo_exchange_coll = self.backend_db[self.config.mongo_collection]
                backend_client.start_session()
            except (AutoReconnect, ConnectionFailure) as exc:
                self.log.error(
                    "Failed connecting to MongoDB (%s). Is the MongoDB instance running and reachable?",
                    self.config.backend_host,
                )
                self.log.exception(exc)
            else:
                self.log.info("Successfully connected to MongoDB backend (%s)", self.config.backend_host)

            try:
                self.backend_db.validate_collection(self.mongo_exchange_coll)
            except (CollectionInvalid, OperationFailure):
                self.is_collection_new = True
            else:
                self.is_collection_new = False
                if self.config.load_history_via != "mongodb":
                    if self.config.load_history_via == "api":
                        sys.exit(
                            f"[ERROR] MongoDB history '{self.config.mongo_collection}' already exists. "
                            + "Cannot load history via API. Choose different history DB name or loading method."
                        )
                    self.sync_check_needed = True

        return backend_client

    #############
    # DB checks #
    #############

    def db_sync_check(self):
        internal_db_index = np.array(self.get_internal_db_index())

        external_db_index = self.mongo_exchange_coll.find({}, projection={"_id": False, "t": True}).sort("t", 1)
        external_db_index = np.array([index_dict["t"] for index_dict in list(external_db_index)])

        only_exist_in_external_db = np.setdiff1d(external_db_index, internal_db_index)
        only_exist_in_internal_db = np.setdiff1d(internal_db_index, external_db_index)

        sync_from_external_db_needed = len(only_exist_in_external_db) > 0
        sync_from_internal_db_needed = len(only_exist_in_internal_db) > 0

        if sync_from_external_db_needed:
            external_db_entries_to_sync = self.mongo_exchange_coll.find(
                {"t": {"$in": only_exist_in_external_db.tolist()}}, projection={"_id": False}
            )
            df_external_db_entries = pd.DataFrame(list(external_db_entries_to_sync)).set_index("t")
            self.root.market_history.df_market_history = pd.concat(
                [self.root.market_history.df_market_history, df_external_db_entries]
            ).sort_values("t")
            self.log.info(
                "%s candles synced from external to internal DB (%s to %s)",
                len(only_exist_in_external_db),
                min(only_exist_in_external_db),
                max(only_exist_in_external_db),
            )

        if sync_from_internal_db_needed:
            internal_db_entries = self.root.market_history.get_market_history(from_list=only_exist_in_internal_db)
            internal_db_entries_to_sync = [
                drop_dict_na_values(record) for record in internal_db_entries.reset_index(drop=False).to_dict("records")
            ]
            self.mongodb_insert(internal_db_entries_to_sync)
            self.log.info(
                "%s candles synced from internal to external DB (%s to %s)",
                len(only_exist_in_internal_db),
                min(only_exist_in_internal_db),
                max(only_exist_in_internal_db),
            )

        self.log.info("Internal and external DB are synced.")
        if self.config.load_history_via == "feather" and sync_from_external_db_needed:
            self.root.market_history.dump_to_feather()

    def check_db_consistency(self):
        is_consistent = False
        index = self.get_internal_db_index()
        timeframe = {
            "start_datetime": pd.Timestamp(index[0], unit="ms", tz="UTC"),
            "end_datetime": pd.Timestamp(index[-1], unit="ms", tz="UTC"),
        }
        real_index = get_df_datetime_index(timeframe, freq=self.config.candle_interval)["t"].to_list()
        current_index = index.to_list()
        index_diff = np.setdiff1d(real_index, current_index)
        if len(index_diff) > 0:
            self.log.warning("Inconsistent asset history. Missing candle timestamps: %s", str(index_diff))
            # TODO: fetch index_diff from remote
        else:
            is_consistent = True
            self.log.info("Consistency check of DB history successful!")
        return is_consistent

    ################
    # DB functions #
    ################

    def update_status(self, status_updates):
        db_acknowledged = False
        for status, value in status_updates.items():
            setattr(self.root.trader, status, value)
        if self.config.use_backend:
            db_acknowledged = (
                self.backend_db["trader_status"]
                .update_one({"trader_id": self.config.trader_id}, {"$set": status_updates})
                .acknowledged
            )
        return db_acknowledged

    def db_add_history(self, df_history_update):
        self.root.market_history.df_market_history = pd.concat(
            [self.root.market_history.df_market_history, df_history_update], axis=0
        ).sort_index()
        if self.config.backend == "mongodb":
            db_result = False
            df_history_update.sort_index(inplace=True)
            payload_update = [
                drop_dict_na_values(record) for record in df_history_update.reset_index(drop=False).to_dict("records")
            ]
            if len(payload_update) <= 1:
                db_result = self.mongodb_update(payload_update[0], upsert=True)
            else:
                db_result = self.mongodb_insert(payload_update)
            if db_result and self.is_collection_new:
                self.mongo_exchange_coll.create_index("t", unique=True)
                self.is_collection_new = False

    def db_sync_trader_state(self):
        trader_id = self.config.trader_id
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_response = list(
                self.backend_db["trader_status"].find({"trader_id": trader_id}, projection={"_id": False})
            )
            if len(db_response) > 0 and db_response[0]["trader_id"] == trader_id:
                trader_status = db_response[0]
                self.root.trader.gid = trader_status["gid"]
                if self.config.budget == 0:
                    self.config.budget = trader_status["budget"]
                # TODO: Save remaining vals to DB
            else:
                trader_status = {
                    "trader_id": trader_id,
                    "window": self.config.window,
                    "budget": self.config.budget,
                    "buy_opportunity_factor": self.config.buy_opportunity_factor,
                    "buy_opportunity_boundary": self.config.buy_opportunity_boundary,
                    "profit_factor": self.config.profit_factor,
                    "amount_invest_fiat": self.config.amount_invest_fiat,
                    "maker_fee": self.config.maker_fee,
                    "taker_fee": self.config.taker_fee,
                    "gid": self.root.trader.gid,
                }

                db_acknowledged = self.backend_db["trader_status"].insert_one(trader_status).acknowledged

            self.root.trader.open_orders = list(
                self.backend_db["open_orders"].find({"trader_id": trader_id}, projection={"_id": False})
            )
            self.root.trader.closed_orders = list(
                self.backend_db["closed_orders"].find({"trader_id": trader_id}, projection={"_id": False})
            )
        return db_acknowledged

    def get_internal_db_index(self):
        return self.root.market_history.df_market_history.sort_index().index

    ######################
    # MongoDB operations #
    ######################

    def mongodb_insert(self, payload_insert, filter_nan=False):
        if filter_nan:
            for entry in payload_insert:
                payload_insert = {items[0]: items[1] for items in entry.items() if pd.notna(items[1])}
        try:
            insert_result = self.mongo_exchange_coll.insert_many(payload_insert)
            insert_result = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_result = False
            self.log.error("Insert into DB failed!")
        return insert_result

    def mongodb_update(self, payload_update, upsert=False, filter_nan=False):
        payload_update_copy = payload_update.copy()
        t_index = payload_update_copy["t"]
        del payload_update_copy["t"]
        if filter_nan:
            payload_update_copy = {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}
        try:
            update_result = self.mongo_exchange_coll.update_one(
                {"t": t_index}, {"$set": payload_update_copy}, upsert=upsert
            )
            update_result = update_result.acknowledged
        except (TypeError, ValueError):
            self.log.error("Update into DB failed!")
            update_result = False
        return update_result

    ####################
    # Order operations #
    ####################

    def order_new(self, order, order_type):
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_acknowledged = self.backend_db[order_type].insert_one(order).acknowledged
        return db_acknowledged

    def order_edit(self, order, order_type):
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_acknowledged = (
                self.backend_db[order_type]
                .update_one({"buy_order_id": order["buy_order_id"]}, {"$set": order})
                .acknowledged
            )
        return db_acknowledged

    def order_del(self, order, order_type):
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_acknowledged = (
                self.backend_db[order_type]
                .delete_one({"asset": order["asset"], "buy_order_id": order["buy_order_id"]})
                .acknowledged
            )
        return db_acknowledged
