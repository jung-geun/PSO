
import tensorflow as tf
from tensorflow import keras

# import cupy as cp
import numpy as np

class Particle:
    def __init__(self, model:keras.models, loss, random:bool = False):
        self.model = model
        self.loss = loss
        self.init_weights = self.model.get_weights()
        i_w_,s_,l_ = self._encode(self.init_weights)
        i_w_ = np.random.rand(len(i_w_)) / 5 - 0.10
        self.velocities = self._decode(i_w_,s_,l_)
        self.random = random
        self.best_score = 0
        self.best_weights = self.init_weights

    """
    Returns:
        (cupy array) : 가중치 - 1차원으로 풀어서 반환
        (list) : 가중치의 원본 shape
        (list) : 가중치의 원본 shape의 길이
    """
    def _encode(self, weights:list):
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

    def _decode(self, weight:list, shape, lenght):
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

        return weights

    def get_score(self, x, y, renewal:str = "acc"):
        self.model.compile(loss=self.loss, optimizer="sgd", metrics=["accuracy"])
        score = self.model.evaluate(x, y, verbose=0)
        # print(score)
        if renewal == "acc":
            if score[1] > self.best_score:
                self.best_score = score[1]
                self.best_weights = self.model.get_weights()
        elif renewal == "loss":
            if score[0] < self.best_score:
                self.best_score = score[0]
                self.best_weights = self.model.get_weights()
            
        return score
    def _update_velocity(self, local_rate, global_rate, w, g_best):
        encode_w, w_sh, w_len = self._encode(weights = self.model.get_weights())
        encode_v, _, _ = self._encode(weights = self.velocities)
        encode_p, _, _ = self._encode(weights = self.best_weights)
        encode_g, _, _ = self._encode(weights = g_best)
        r0 = np.random.rand()
        r1 = np.random.rand()
        new_v = w * encode_v + local_rate * r0 * (encode_p - encode_w) + global_rate * r1 * (encode_g - encode_w)
        self.velocities = self._decode(new_v, w_sh, w_len)
    
    def _update_velocity_w(self, local_rate, global_rate, w, w_p, w_g, g_best):
        encode_w, w_sh, w_len = self._encode(weights = self.model.get_weights())
        encode_v, _, _ = self._encode(weights = self.velocities)
        encode_p, _, _ = self._encode(weights = self.best_weights)
        encode_g, _, _ = self._encode(weights = g_best)
        r0 = np.random.rand()
        r1 = np.random.rand()
        new_v = w * encode_v + local_rate * r0 * (w_p * encode_p - encode_w) + global_rate * r1 * (w_g * encode_g - encode_w)
        self.velocities = self._decode(new_v, w_sh, w_len)
    
    def _update_weights(self):
        encode_w, w_sh, w_len = self._encode(weights = self.model.get_weights())
        encode_v, _, _ = self._encode(weights = self.velocities)
        if self.random:
            encode_v = -0.5 * encode_v
        new_w = encode_w + encode_v
        self.model.set_weights(self._decode(new_w, w_sh, w_len))

    def f(self, x, y, weights):
        self.model.set_weights(weights)
        score = self.model.evaluate(x, y, verbose = 0)[1]
        if score > 0:
            return 1 / (1 + score)
        else:
            return 1 + np.abs(score)

    def step(self, x, y, local_rate, global_rate, w, g_best, renewal:str = "acc"):
        self._update_velocity(local_rate, global_rate, w, g_best)
        self._update_weights()
        return self.get_score(x, y, renewal)
    
    def step_w(self, x, y, local_rate, global_rate, w, g_best, w_p, w_g, renewal:str = "acc"):     
        self._update_velocity_w(local_rate, global_rate, w, w_p, w_g, g_best)
        self._update_weights()
        return self.get_score(x, y, renewal)
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_weights(self):
        return self.best_weights