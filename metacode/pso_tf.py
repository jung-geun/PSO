import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import datetime
import gc
import cupy as cp



class PSO(object):
    """
    Class implementing PSO algorithm
    """

    def __init__(self, model: keras.models, loss_method=keras.losses.MeanSquaredError(), n_particles: int = 5):
        """
        Initialize the key variables.

        Args:
            model : 학습할 모델 객체 (Sequential)
            loss_method : 손실 함수
            n_particles(int) : 파티클의 개수
        """
        self.model = model                                       # 모델
        self.n_particles = n_particles                           # 파티클의 개수
        self.loss_method = loss_method                           # 손실 함수
        model_structure = self.model.to_json()              # 모델의 구조 정보
        self.init_weights = self.model.get_weights()             # 검색할 차원
        self.particle_depth = len(self.model.get_weights())      # 검색할 차원의 깊이
        self.particles_weights = [None] * n_particles            # 파티클의 위치
        for _ in tqdm(range(self.n_particles), desc="init particles position"):
            m = keras.models.model_from_json(model_structure)
            m.compile(loss=self.loss_method,
                      optimizer="adam", metrics=["accuracy"])
            self.particles_weights[_] = m.get_weights()     

        # 입력받은 파티클의 개수 * 검색할 차원의 크기 만큼의 균등한 위치를 생성
        self.velocities = [
            [0 for i in range(self.particle_depth)] for n in range(n_particles)]
        for i in tqdm(range(n_particles), desc="init velocities"):
            
            self.init_weights = self.model.get_weights()
            w_,s_,l_ = self._encode(self.init_weights)
            w_ = np.random.rand(len(w_)) / 5 - 0.10
            self.velocities[i] = self._decode(w_,s_,l_)
            # for index, layer in enumerate(self.init_weights):
            #     self.velocities[i][index] = np.random.rand(
            #         *layer.shape) / 5 - 0.10
        

        # 입력받은 파티클의 개수 * 검색할 차원의 크기 만큼의 속도를 무작위로 초기화
        # 최대 사이즈로 전역 최적갑 저장 - global best
        self.p_best = self.particles_weights            # 각 파티클의 최적값(최적의 가중치)
        self.g_best=self.model.get_weights()            # 전역 최적값(최적의 가중치) | 초기값은 모델의 가중치

        # 각 파티클의 최적값의 점수
        self.p_best_score = [0 for i in range(n_particles)]

        # 전역 최적값의 점수(초기화 - 0)
        self.g_best_score = 0
        
    def __del__(self):
        del self.model
        del self.n_particles
        del self.loss_method
        del self.init_weights
        del self.particles_weights
        del self.velocities
        del self.p_best
        del self.g_best
        del self.p_best_score
        del self.g_best_score

    def _encode(self,weights: list):
        # w_gpu = cp.array([])
        w_gpu = np.array([])
        lenght = []
        shape = []
        for layer in weights:
            shape.append(layer.shape)
            w_ = layer.reshape(-1)
            lenght.append(len(w_))
            w_gpu = np.append(w_gpu, w_)
            # w_gpu = cp.append(w_gpu, w_)
        
        return w_gpu, shape, lenght

    def _decode(self,weight, shape, lenght):
        weights = []
        start = 0
        for i in range(len(shape)):
            end = start + lenght[i]
            # print(f"{start} ~ {end}")
            # print(f"{shape[i]}")
            w_ = weight[start:end]
            w_ = np.reshape(w_, shape[i])
            # w_ = w_.reshape(shape[i])
            weights.append(w_)
            start = end

        return weights
    
    def _update_weights(self, weights, v):
        """
        Update particle position

        Args:
            weights (array-like) : 파티클의 현재 가중치
            v (array-like) : 가중치의 속도

        Returns:
            (array-like) : 파티클의 새로운 가중치(위치)
        """
        # w = np.array(w)  # 각 파티클의 위치
        # v = np.array(v)  # 각 파티클의 속도(방향과 속력을 가짐)
        # new_weights = [0 for i in range(len(weights))]
        # print(f"weights : {weights}")
        encode_w, w_sh, w_len = self._encode(weights = weights)
        encode_v, _, _ = self._encode(weights = v)
        new_w = encode_w + encode_v
        new_weights = self._decode(new_w, w_sh, w_len)
        
        # for i in range(len(weights)):
            # new_weights[i] = tf.add(weights[i], v[i])
        # new_w = tf.add(w, v)   # 각 파티클을 랜덤한 속도만큼 진행
        return new_weights    # 진행한 파티클들의 위치를 반환

    def _update_velocity(self, weights, v, p_best, c0=0.5, c1=1.5, w=0.75):
        """
            Update particle velocity

            Args:
                weights (array-like) : 파티클의 현재 가중치
                v (array-like) : 속도
                p_best(array-like) : 각 파티클의 최적의 위치 (최적의 가중치)
                c0 (float) : 인지 스케일링 상수 (가중치의 중요도 - 지역) - 지역 관성
                c1 (float) : 사회 스케일링 상수 (가중치의 중요도 - 전역) - 전역 관성
                w (float) : 관성 상수 (현재 속도의 중요도)

            Returns:
                (array-like) : 각 파티클의 새로운 속도
        """
        # x = np.array(x)
        # v = np.array(v)
        # assert np.shape(weights) == np.shape(v), "Position and velocity must have same shape."
        # 두 데이터의 shape 이 같지 않으면 오류 출력
        # 0에서 1사이의 숫자를 랜덤 생성
        r0 = np.random.rand()
        r1 = np.random.rand()
        # p_best = np.array(p_best)
        # g_best = np.array(g_best)

        # 가중치(상수)*속도 + \
        # 스케일링 상수*랜덤 가중치*(나의 최적값 - 처음 위치) + \
        # 전역 스케일링 상수*랜덤 가중치*(전체 최적값 - 처음 위치)
        
        encode_w, w_sh, w_len = self._encode(weights = weights)
        encode_v, _, _ = self._encode(weights = v)
        encode_p, _, _ = self._encode(weights = p_best)
        encode_g, _, _ = self._encode(weights = self.g_best)
        
        new_v = encode_w * encode_v + c0*r0*(encode_p - encode_w) + c1*r1*(encode_g - encode_w)
        new_velocity = self._decode(new_v, w_sh, w_len)
        # new_velocity = [None] * len(weights)
        # for i, layer in enumerate(weights):

            # new_v = w*v[i]
            # new_v = new_v + c0*r0*(p_best[i] - layer)
            # new_v = new_v + c1*r1*(self.g_best[i] - layer)
            # new_velocity[i] = new_v

        # new_v = w*v + c0*r0*(p_best - weights) + c1*r1*(g_best - weights)
        return new_velocity

    def _get_score(self, x, y):
        """
        Compute the score of the current position of the particles.

        Args:
            x (array-like): The current position of the particles
            y (array-like): The current position of the particles
        Returns:
            (array-like) : 추론에 대한 점수
        """
        score = self.model.evaluate(x, y, verbose=0)

        return score

    def optimize(self, x_, y_, maxiter=10, c0=0.5, c1=1.5, w=0.75, save=False, save_path="./result/history"):
        """
        Run the PSO optimization process utill the stoping critera is met.
        Cas for minization. The aim is to minimize the cost function

        Args:
            maxiter (int): the maximum number of iterations before stopping the optimization
            파티클의 최종 위치를 위한 반복 횟수
        Returns:
            The best solution found (array-like)
        """
        if save:
            os.makedirs(save_path, exist_ok=True)
            day = datetime.datetime.now().strftime('%m-%d-%H-%M')
        
        for _ in range(maxiter):

            for i in tqdm(range(self.n_particles), desc=f"Iter {_}/{maxiter} ", ascii=True):
                weights = self.particles_weights[i]  # 각 파티클 추출
                v = self.velocities[i]    # 각 파티클의 다음 속도 추출
                p_best = self.p_best[i]   # 결과치 저장할 변수 지정
                # 2. 속도 계산
                self.velocities[i] = self._update_velocity(
                    weights, v, p_best, c0, c1, w)
                # 다음에 움직일 속도 = 최초 위치, 현재 속도, 현재 위치, 최종 위치
                # 3. 위치 업데이트
                self.particles_weights[i] = self._update_weights(weights, v)
                # 현재 위치 = 이전 위치 + 현재 속도
                # 내 현재 위치가 내 위치의 최소치보다 작으면 갱신
                self.model.set_weights(self.particles_weights[i])
                # self.particles_weights[i] = self.model.get_weights()
                # 4. 평가
                self.model.compile(loss=self.loss_method,
                                   optimizer='sgd', metrics=['accuracy'])
                score = self._get_score(x_, y_)

                if score[1] > self.p_best_score[i]:
                    self.p_best_score[i] = score[1]
                    self.p_best[i] = self.particles_weights[i]
                    if score[1] > self.g_best_score:
                        self.g_best_score = score[1]
                        self.g_best = self.particles_weights[i]
                
                if save:
                    with open(f"{save_path}/{day}_{self.n_particles}_{maxiter}_{c0}_{c1}_{w}.csv",'a')as f:
                        f.write(f"{score[0]}, {score[1]}")
                        if i != self.n_particles - 1:
                            f.write(",")
                            
            if save:   
                with open(f"{save_path}/{day}_{self.n_particles}_{maxiter}_{c0}_{c1}_{w}.csv",'a')as f:
                    f.write("\n")
            print(
                f"loss avg : {score[0]/self.n_particles} | acc avg : {score[1]/self.n_particles} | best score : {self.g_best_score}")
            gc.collect()

        # 전체 최소 위치, 전체 최소 벡터
        return self.g_best, self._get_score(x_, y_)

    """
    Returns:
        최종 가중치
    """

    def best_weights(self):
        return self.g_best

    """
    Returns:
        최종 가중치의 스코어
    """

    def best_score(self):
        return self.g_best_score