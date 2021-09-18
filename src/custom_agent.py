import gym
import random
import numpy as np
from gym.spaces.discrete import Discrete
import tensorflow.keras as keras
import pickle
import common
from tensorflow import saved_model


class CustomAgent:
    def __init__(
        self, env: gym.Env, config: dict = {}, env_config: dict = {}, worker_id=0
    ):
        self.env = env(env_config)
        self.train_max_steps = self.env.train_max_steps
        self.validate_max_steps = self.env.validate_max_steps
        self.max_quotes = self.env.max_quotes
        self.hist_model = self.create_hist_model()
        self.rt_model = self.create_rt_model()
        self.avg_reward = 0
        self.avg_total = 0
        self.max_total = None
        self.explore = 1
        self.fitted = False
        self.best_score = None
        self.model_dir = "../data"
        self.model_changed = False
        self.worker_id = worker_id

    def create_hist_model(self):
        inputs = keras.layers.Input(shape=(self.hist_max_quotes,5))
        l = inputs
        l = keras.layers.Reshape((self.hist_max_quotes, 1))(l)
        l = keras.layers.LSTM(self.hist_max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(1, activation="sigmoid")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="mean_squared_logarithmic_error",
        )
        return model

    def create_rt_model(self):
        inputs = keras.layers.Input(shape=(self.rt_max_quotes,))
        l = inputs
        l = keras.layers.Reshape((self.rt_max_quotes, 1))(l)
        l = keras.layers.LSTM(self.rt_max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(2, activation="sigmoid")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="mean_squared_logarithmic_error",
        )
        return model

    def predict_action(self, x):
        if self.env.last_event_type == self.env.HIST_EVENT:
            confidence = self.hist_model.predict_on_batch(np.array(x))[0]
            action = [confidence, 0.5, 0.5]
        else:
            buy_price, sell_price = self.rt_model.predict_on_batch(np.array(x))[0]
            action = [0, buy_price, sell_price]
        if type(self.env.action_space) == Discrete:
            if np.shape(action) == (1,):
                action = action[0]
            action = round(action)
            action = max(0, action)
            action = min(self.env.action_space.n - 1, action)
        return action

    def transform_x(self, state):
        return (
            state
            if np.isscalar(state)
            else [y for x in state for y in np.array(x).flatten()]
        )

    def transform_y(self, action):
        return action[0] if np.shape(action) == (1,) else action

    def run_episode(self, train=True, live=False):
        stage = (
            self.env.TRAINING
            if train
            else (self.env.VALIDATION if not live else self.env.PREDICTION)
        )
        self.env.stage = stage
        state = self.env.reset()
        hist_set = {
            "train_x": [],
            "train_y": [],
            "hist_x": [],
            "hist_y": [],
        }
        rt_set = {
            "train_x": [],
            "train_y": [],
            "hist_x": [],
            "hist_y": [],
        }
        total = 0
        for _ in range(self.max_steps + 1):
            self.niter += 1
            x = self.transform_x(state)
            if self.env.last_event_type == self.env.HIST_EVENT:
                trainset = hist_set
                model = self.hist_model
            else:
                trainset = rt_set
                model = self.rt_model
            if (train and random.random() < self.explore) or not self.fitted:
                action = self.env.action_space.sample()
            else:
                action = self.predict_action([x], stage)
            state, r, d, info = self.env.step(action)
            total += r
            if train:
                y = self.transform_y(action)
                if r > self.avg_reward + 3 * abs(self.avg_reward):
                    trainset["train_x"].append(x)
                    trainset["train_y"].append(y)
                trainset["hist_x"].append(x)
                trainset["hist_y"].append(y)
            self.avg_reward = self.avg_reward * 0.99 + r * 0.01
            if d:
                break
        if train and total > self.avg_total + 3 * abs(self.avg_total):
            trainset["train_x"].extend(trainset["hist_x"])
            trainset["train_y"].extend(trainset["hist_y"])
        self.avg_total = self.avg_total * 0.9 + total * 0.1
        if self.max_total is None or total >= self.max_total:
            self.max_total = total
        if trainset["train_x"]:
            common.log("Fit")
            nit = max(
                1,
                round(1 * (r - self.avg_total + 1) / (abs(self.avg_total) + 1) - 1000),
            )
            for _ in range(nit):
                model.fit(
                    np.array(trainset["train_x"]),
                    np.array(trainset["train_y"]),
                    epochs=10,
                    verbose=0,
                )
            self.fitted = True
            self.explore = max(0.3, self.explore * 0.9999)
        return total

    def train(self):
        self.niter = 0
        self.max_total = None
        for _ in range(1000):
            self.run_episode()
            if self.niter > 100000:
                break
        common.log(
            "avg r:",
            self.avg_reward,
            ", avg total:",
            self.avg_total,
            "max:",
            self.max_total,
            "explore:",
            self.explore,
        )

    def evaluate(self, quick=False):
        if quick:
            total = self.max_total
        else:
            total = self.run_episode(train=False)
        if self.best_score is None or total >= self.best_score:
            self.best_score = total
            self.save_checkpoint(self.model_dir)
            return total
        return None

    def __getstate__(self) -> dict:
        keras.models.save_model(
            self.model, f"{self.model_dir}/model-{self.worker_id}.h5", save_format="h5"
        )
        return {
            "best_score": self.best_score,
            "explore": self.explore,
            "fitted": self.fitted,
        }

    def __setstate__(self, state: dict):
        self.model = keras.models.load_model(f"{self.model_dir}/model.h5")
        self.best_score = state["best_score"]
        self.explore = state["explore"]
        self.fitted = state["fitted"]

    def save_checkpoint(self, checkpoint_dir: str = None) -> str:
        if checkpoint_dir is None:
            checkpoint_dir = self.model_dir
        checkpoint_path = f"{checkpoint_dir}/agent-{self.worker_id}.dat"
        pickle.dump(self.__getstate__(), open(checkpoint_path, "wb"))
        self.model_changed = True
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        self.__setstate__(extra_data)

    def save_model(self, path):
        saved_model.save(self.model, path)
