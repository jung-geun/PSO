import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

# import cupy as cp

from tqdm import tqdm
from datetime import datetime
import json
import gc

from pso.particle import Particle


class Optimizer:
    def __init__(
        self,
        model: keras.models,
        loss = "mse",
        n_particles: int = 10,
        c0=0.5,
        c1=1.5,
        w_min=0.5,
        w_max=1.5,
    ):
        self.model = model  # 모델 구조
        self.loss = loss  # 손실함수
        self.n_particles = n_particles  # 파티클 개수
        self.particles = [None] * n_particles  # 파티클 리스트
        self.c0 = c0  # local rate - 지역 최적값 관성 수치
        self.c1 = c1  # global rate - 전역 최적값 관성 수치
        self.w_min = w_min  # 최소 관성 수치
        self.w_max = w_max  # 최대 관성 수치

        self.g_best_score = 0  # 최고 점수 - 시작은 0으로 초기화
        self.g_best = None  # 최고 점수를 받은 가중치
        self.g_best_ = None  # 최고 점수를 받은 가중치 - 값의 분산을 위한 변수

        for i in tqdm(range(self.n_particles), desc="Initializing Particles"):
            m = keras.models.model_from_json(model.to_json())
            m.compile(loss=self.loss, optimizer="sgd", metrics=["accuracy"])

            self.particles[i] = Particle(m, loss)

    """
    Returns:
        (cupy array) : 가중치 - 1차원으로 풀어서 반환
        (list) : 가중치의 원본 shape
        (list) : 가중치의 원본 shape의 길이
    """

    def _encode(self, weights):
        # w_gpu = cp.array([])
        w_gpu = np.array([])
        lenght = []
        shape = []
        for layer in weights:
            shape.append(layer.shape)
            w_ = layer.reshape(-1)
            lenght.append(len(w_))
            # w_gpu = cp.append(w_gpu, w_)
            w_gpu = np.append(w_gpu, w_)

        return w_gpu, shape, lenght

    """
    Returns:
        (list) : 가중치 원본 shape으로 복원
    """

    def _decode(self, weight, shape, lenght):
        weights = []
        start = 0
        for i in range(len(shape)):
            end = start + lenght[i]
            w_ = weight[start:end]
            # w_ = weight[start:end].get()
            w_ = np.reshape(w_, shape[i])
            # w_ = w_.reshape(shape[i])
            weights.append(w_)
            start = end
        del weight
        del shape
        del lenght
        gc.collect()

        return weights

    def f(self, x, y, weights):
        self.model.set_weights(weights)
        self.model.compile(loss=self.loss, optimizer="sgd", metrics=["accuracy"])
        score = self.model.evaluate(x, y, verbose=0)[1]
        if score > 0:
            return 1 / (1 + score)
        else:
            return 1 + np.abs(score)

    """
    parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    epochs : int
    save : bool
    save_path : str ex) "./result"
    renewal : str ex) "acc" or "loss"
    """

    """
    parameters
    fit(
        x_test : numpy.ndarray,
        y_test : numpy.ndarray,
        epochs : int,
        save : bool - True : save, False : not save
        save_path : str ex) "./result",
        renewal : str ex) "acc" or "loss",
        empirical_balance : bool - True : empirical balance, False : no balance
        Dispersion : bool - True : random search, False : PSO
    """
    def fit(
        self,
        x,
        y,
        epochs: int = 100,
        save: bool = False,
        save_path: str = "./result",
        renewal: str = "acc",
        empirical_balance: bool = False,
        Dispersion: bool = False,
        check_point: int = None,
    ):
        self.renewal = renewal
        if renewal == "acc":
            self.g_best_score = 0
        elif renewal == "loss":
            self.g_best_score = np.inf

        if save:
            if save_path is None:
                raise ValueError("save_path is None")
            else:
                self.save_path = save_path
                os.makedirs(save_path, exist_ok=True)
                self.day = datetime.now().strftime("%m-%d-%H-%M")

        for i, p in enumerate(self.particles):
            local_score = p.get_score(x, y, renewal=renewal)

            if renewal == "acc":
                if local_score[1] > self.g_best_score:
                    self.g_best_score = local_score[1]
                    self.g_best = p.get_best_weights()
                    self.g_best_ = p.get_best_weights()
            elif renewal == "loss":
                if local_score[0] < self.g_best_score:
                    self.g_best_score = local_score[0]
                    self.g_best = p.get_best_weights()
                    self.g_best_ = p.get_best_weights()
        
        print(f"initial g_best_score : {self.g_best_score}")
        
        for _ in range(epochs):
            acc = 0
            loss = 0
            min_score = np.inf
            max_score = 0
            min_loss = np.inf
            max_loss = 0

            # for i in tqdm(range(len(self.particles)), desc=f"epoch {_ + 1}/{epochs}", ascii=True):
            for i in range(len(self.particles)):
                w = self.w_min + (self.w_max - self.w_min) * _ / epochs

                if Dispersion:
                    g_best = self.g_best_
                else:
                    g_best = self.g_best

                if empirical_balance:
                    if np.random.rand() < np.exp(-(_) / epochs):
                        w_p_ = self.f(x, y, self.particles[i].get_best_weights())
                        w_g_ = self.f(x, y, self.g_best)
                        w_p = w_p_ / (w_p_ + w_g_)
                        w_g = w_p_ / (w_p_ + w_g_)

                    else:
                        p = 1 / (self.n_particles * np.linalg.norm(self.c1 - self.c0))
                        p = np.exp(-p)
                        w_p = p
                        w_g = 1 - p

                    score = self.particles[i].step_w(
                        x, y, self.c0, self.c1, w, g_best, w_p, w_g, renewal=renewal
                    )

                else:
                    score = self.particles[i].step(
                        x, y, self.c0, self.c1, w, g_best, renewal=renewal
                    )

                if renewal == "acc":
                    if score[1] >= self.g_best_score:
                        self.g_best_score = score[1]
                        self.g_best = self.particles[i].get_best_weights()
                elif renewal == "loss":
                    if score[0] <= self.g_best_score:
                        self.g_best_score = score[0]
                        self.g_best = self.particles[i].get_best_weights()

                loss += score[0]
                acc += score[1]
                if score[0] < min_loss:
                    min_loss = score[0]
                if score[0] > max_loss:
                    max_loss = score[0]

                if score[1] < min_score:
                    min_score = score[1]
                if score[1] > max_score:
                    max_score = score[1]

                if save:
                    with open(
                        f"./{save_path}/{self.day}_{self.n_particles}_{epochs}_{self.c0}_{self.c1}_{self.w_min}_{renewal}.csv",
                        "a",
                    ) as f:
                        f.write(f"{score[0]}, {score[1]}")
                        if i != self.n_particles - 1:
                            f.write(", ")

            TS = self.c0 + np.random.rand() * (self.c1 - self.c0)
            g_, g_sh, g_len = self._encode(self.g_best)
            decrement = (epochs - (_) + 1) / epochs
            g_ = (1 - decrement) * g_ + decrement * TS
            self.g_best_ = self._decode(g_, g_sh, g_len)

            if save:
                with open(
                    f"./{save_path}/{self.day}_{self.n_particles}_{epochs}_{self.c0}_{self.c1}_{self.w_min}_{renewal}.csv",
                    "a",
                ) as f:
                    f.write("\n")

            print(f"epoch {_ + 1}/{epochs} finished")
            # print(f"loss min : {min_loss} | loss max : {max_loss} | acc min : {min_score} | acc max : {max_score}")
            # print(f"loss avg : {loss/self.n_particles} | acc avg : {acc/self.n_particles} | Best {renewal} : {self.g_best_score}")
            print(
                f"loss min : {min_loss} | acc avg : {max_score} | Best {renewal} : {self.g_best_score}"
            )

            gc.collect()
            
            if check_point is not None:
                if _ % check_point == 0:
                    self._check_point_save(f"./{save_path}/{self.day}/check_point_{_}.h5")

        return self.g_best, self.g_best_score

    def get_best_model(self):
        model = keras.models.model_from_json(self.model.to_json())
        model.set_weights(self.g_best)
        model.compile(loss=self.loss, optimizer="sgd", metrics=["accuracy"])
        return model

    def get_best_score(self):
        return self.g_best_score

    def get_best_weights(self):
        return self.g_best

    def save_info(self, path: str = "./result"):
        json_save = {
            "name": f"{self.day}_{self.n_particles}_{self.c0}_{self.c1}_{self.w_min}.h5",
            "n_particles": self.n_particles,
            "score": self.g_best_score,
            "c0": self.c0,
            "c1": self.c1,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "loss_method": self.loss,
            "renewal": self.renewal,
        }

        with open(
            f"./{path}/{self.day}_{self.loss}_{self.n_particles}_{self.g_best_score}.json",
            "w",
        ) as f:
            json.dump(json_save, f, indent=4)
            
    def _check_point_save(self, save_path: str = f"./result/check_point"):
        model = self.get_best_model()
        model.save(save_path)
            
    def model_save(self, save_path: str = "./result/model"):
        model = self.get_best_model()
        model.save(
            f"./{save_path}/{self.day}/{self.n_particles}_{self.c0}_{self.c1}_{self.w_min}.h5"
        )
        return model
