import os
import sys

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
    """
    Args:
        model (keras.models): 모델 구조
        loss (str): 손실함수
        n_particles (int): 파티클 개수
        c0 (float): local rate - 지역 최적값 관성 수치
        c1 (float): global rate - 전역 최적값 관성 수치
        w_min (float): 최소 관성 수치
        w_max (float): 최대 관성 수치
        random (float): 랜덤 파티클 비율 - 0 ~ 1 사이의 값
    """
    def __init__(
        self,
        model: keras.models,
        loss = "mse",
        n_particles: int = 10,
        c0=0.5,
        c1=1.5,
        w_min=0.5,
        w_max=1.5,
        random:float = 0,
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
        self.avg_score = 0  # 평균 점수

        for i in tqdm(range(self.n_particles), desc="Initializing Particles"):
            m = keras.models.model_from_json(model.to_json())
            init_weights = m.get_weights()
            w_, sh_, len_ = self._encode(init_weights)
            w_ = np.random.uniform(-1.5, 1.5, len(w_))
            m.set_weights(self._decode(w_, sh_, len_))
            m.compile(loss=self.loss, optimizer="sgd", metrics=["accuracy"])
            if i < random * self.n_particles:
                self.particles[i] = Particle(m, loss, random=True)
            else:
                self.particles[i] = Particle(m, loss, random=False)

    """
    Args:
        weights (list) : keras model의 가중치
    Returns:
        (numpy array) : 가중치 - 1차원으로 풀어서 반환
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
    Args:
        weight (numpy array) : 가중치 - 1차원으로 풀어진 상태
        shape (list) : 가중치의 원본 shape
        lenght (list) : 가중치의 원본 shape의 길이
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
    Args:
        x_test : numpy.ndarray,
        y_test : numpy.ndarray,
        epochs : int,
        save : bool - True : save, False : not save
        save_path : str ex) "./result",
        renewal : str ex) "acc" or "loss",
        empirical_balance : bool - True : 
        Dispersion : bool - True : g_best 의 값을 분산시켜 전역해를 찾음, False : g_best 의 값만 사용
        check_point : int - 저장할 위치 - None : 저장 안함
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
        self.save_path = save_path
        
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

        # for i, p in enumerate(self.particles):
        for i in tqdm(range(self.n_particles), desc="Initializing Particles"):
            p = self.particles[i]
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
        
        try:
            for _ in range(epochs):
                print(f"epoch {_ + 1}/{epochs}")
                acc = 0
                loss = 0
                min_score = np.inf
                max_score = 0
                min_loss = np.inf
                max_loss = 0

                # for i in tqdm(range(len(self.particles)), desc=f"epoch {_ + 1}/{epochs}", ascii=True):
                for i in range(len(self.particles)):
                    w = self.w_max - (self.w_max - self.w_min) * _ / epochs

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
                            del w_p_
                            del w_g_

                        else:
                            p_b = self.particles[i].get_best_score()
                            g_a = self.avg_score
                            l_b = p_b - g_a
                            l_b = np.sqrt(np.power(l_b, 2))
                            p_ = 1 / (self.n_particles * np.linalg.norm(self.c1 - self.c0)) * l_b
                            p_ = np.exp(-1 * p_)
                            w_p = p_
                            w_g = 1 - p_
                            
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

                # print(f"loss min : {min_loss} | loss max : {max_loss} | acc min : {min_score} | acc max : {max_score}")
                # print(f"loss avg : {loss/self.n_particles} | acc avg : {acc/self.n_particles} | Best {renewal} : {self.g_best_score}")
                print(
                    f"loss min : {round(min_loss, 4)} | acc max : {round(max_score, 4)} | Best {renewal} : {self.g_best_score}"
                )

                gc.collect()
                
                if check_point is not None:
                    if _ % check_point == 0:
                        os.makedirs(f"./{save_path}/{self.day}", exist_ok=True)
                        self._check_point_save(f"./{save_path}/{self.day}/ckpt-{_}")
                self.avg_score = acc/self.n_particles
        except KeyboardInterrupt:
            print("Ctrl + C : Stop Training")
        except MemoryError:
            print("Memory Error : Stop Training")
        except Exception as e:
            print(e)
        finally:
            self.model_save(save_path)
            print("model save")
            self.save_info(save_path)
            print("save info")
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
            f"./{path}/{self.day}/{self.loss}_{self.n_particles}.json",
            "a",
        ) as f:
            json.dump(json_save, f, indent=4)
            f.write(",\n")
            
    def _check_point_save(self, save_path: str = f"./result/check_point"):
        model = self.get_best_model()
        model.save_weights(save_path)
            
    def model_save(self, save_path: str = "./result"):
        model = self.get_best_model()
        model.save(
            f"./{save_path}/{self.day}/{self.n_particles}_{self.c0}_{self.c1}_{self.w_min}.h5"
        )
        return model
