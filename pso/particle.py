import gc

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Particle:
    """
    Particle Swarm Optimization의 Particle을 구현한 클래스
    한 파티클의 life cycle은 다음과 같다.
    1. 초기화
    2. 손실 함수 계산
    3. 속도 업데이트
    4. 가중치 업데이트
    5. 2번으로 돌아가서 반복
    """

    def __init__(
        self, model: keras.models, loss, negative: bool = False, mutation: float = 0
    ):
        """
        Args:
            model (keras.models): 학습 및 검증을 위한 모델
            loss (str|): 손실 함수
            negative (bool, optional): 음의 가중치 사용 여부 - 전역 탐색 용도(조기 수렴 방지). Defaults to False.
        """
        self.model = model
        self.loss = loss
        init_weights = self.model.get_weights()
        i_w_, s_, l_ = self._encode(init_weights)
        i_w_ = np.random.uniform(-0.5, 0.5, len(i_w_))
        self.velocities = self._decode(i_w_, s_, l_)
        self.negative = negative
        self.mutation = mutation
        self.best_score = 0
        self.best_weights = init_weights

        del i_w_, s_, l_
        del init_weights

    def __del__(self):
        del self.model
        del self.loss
        del self.velocities
        del self.negative
        del self.best_score
        del self.best_weights

    def _encode(self, weights: list):
        """
        가중치를 1차원으로 풀어서 반환

        Args:
            weights (list) : keras model의 가중치
        Returns:
            (numpy array) : 가중치 - 1차원으로 풀어서 반환
            (list) : 가중치의 원본 shape
            (list) : 가중치의 원본 shape의 길이
        """
        w_gpu = np.array([])
        length = []
        shape = []
        for layer in weights:
            shape.append(layer.shape)
            w_ = layer.reshape(-1)
            length.append(len(w_))
            w_gpu = np.append(w_gpu, w_)

        return w_gpu, shape, length

    def _decode(self, weight: list, shape, length):
        """
        _encode 로 인코딩된 가중치를 원본 shape으로 복원
        파라미터는 encode의 리턴값을 그대로 사용을 권장

        Args:
            weight (numpy array): 가중치 - 1차원으로 풀어서 반환
            shape (list): 가중치의 원본 shape
            length (list): 가중치의 원본 shape의 길이
        Returns:
            (list) : 가중치 원본 shape으로 복원
        """
        weights = []
        start = 0
        for i in range(len(shape)):
            end = start + length[i]
            w_ = weight[start:end]
            w_ = np.reshape(w_, shape[i])
            weights.append(w_)
            start = end
        del start, end, w_
        del shape, length
        del weight

        return weights

    def get_score(self, x, y, renewal: str = "acc"):
        """
        모델의 성능을 평가하여 점수를 반환

        Args:
            x (list): 입력 데이터
            y (list): 출력 데이터
            renewal (str, optional): 점수 갱신 방식. Defaults to "acc" | "acc" or "loss".

        Returns:
            (float): 점수
        """
        # self.model.compile(loss=self.loss, optimizer="sgd", metrics=["accuracy"])
        score = self.model.evaluate(x, y, verbose=0, use_multiprocessing=True)
        if renewal == "acc":
            if score[1] > self.best_score:
                self.best_score = score[1]
                self.best_weights = self.model.get_weights()
        elif renewal == "loss":
            if score[0] == "nan":
                score[0] = np.inf
            if score[0] < self.best_score:
                self.best_score = score[0]
                self.best_weights = self.model.get_weights()

        return score

    def _update_velocity(self, local_rate, global_rate, w, g_best):
        """
        현재 속도 업데이트

        Args:
            local_rate (float): 지역 최적해의 영향력
            global_rate (float): 전역 최적해의 영향력
            w (float): 현재 속도의 영향력 - 관성 | 0.9 ~ 0.4 이 적당
            g_best (list): 전역 최적해
        """
        encode_w, w_sh, w_len = self._encode(weights=self.model.get_weights())
        encode_v, v_sh, v_len = self._encode(weights=self.velocities)
        encode_p, p_sh, p_len = self._encode(weights=self.best_weights)
        encode_g, g_sh, g_len = self._encode(weights=g_best)
        r0 = np.random.rand()
        r1 = np.random.rand()
        if self.negative:
            new_v = (
                w * encode_v
                + -1 * local_rate * r0 * (encode_p - encode_w)
                + -1 * global_rate * r1 * (encode_g - encode_w)
            )
        else:
            new_v = (
                w * encode_v
                + local_rate * r0 * (encode_p - encode_w)
                + global_rate * r1 * (encode_g - encode_w)
            )

        if np.random.rand() < self.mutation:
            m_v = np.random.uniform(-0.2, 0.2, len(encode_v))
            new_v = m_v

        self.velocities = self._decode(new_v, w_sh, w_len)

        del encode_w, w_sh, w_len
        del encode_v, v_sh, v_len
        del encode_p, p_sh, p_len
        del encode_g, g_sh, g_len
        del r0, r1

    def _update_velocity_w(self, local_rate, global_rate, w, w_p, w_g, g_best):
        """
        현재 속도 업데이트
        기본 업데이트의 변형으로 지역 최적해와 전역 최적해를 분산시켜 조기 수렴을 방지

        Args:
            local_rate (float): 지역 최적해의 영향력
            global_rate (float): 전역 최적해의 영향력
            w (float): 현재 속도의 영향력 - 관성 | 0.9 ~ 0.4 이 적당
            w_p (float): 지역 최적해의 분산 정도
            w_g (float): 전역 최적해의 분산 정도
            g_best (list):  전역 최적해
        """
        encode_w, w_sh, w_len = self._encode(weights=self.model.get_weights())
        encode_v, v_sh, v_len = self._encode(weights=self.velocities)
        encode_p, p_sh, p_len = self._encode(weights=self.best_weights)
        encode_g, g_sh, g_len = self._encode(weights=g_best)
        r0 = np.random.rand()
        r1 = np.random.rand()

        if self.negative:
            new_v = (
                w * encode_v
                + -1 * local_rate * r0 * (w_p * encode_p - encode_w)
                + -1 * global_rate * r1 * (w_g * encode_g - encode_w)
            )
        else:
            new_v = (
                w * encode_v
                + local_rate * r0 * (w_p * encode_p - encode_w)
                + global_rate * r1 * (w_g * encode_g - encode_w)
            )

        if np.random.rand() < self.mutation:
            m_v = np.random.uniform(-0.1, 0.1, len(encode_v))
            new_v = m_v

        self.velocities = self._decode(new_v, w_sh, w_len)

        del encode_w, w_sh, w_len
        del encode_v, v_sh, v_len
        del encode_p, p_sh, p_len
        del encode_g, g_sh, g_len
        del r0, r1

    def _update_weights(self):
        """
        가중치 업데이트
        """
        encode_w, w_sh, w_len = self._encode(weights=self.model.get_weights())
        encode_v, v_sh, v_len = self._encode(weights=self.velocities)
        new_w = encode_w + encode_v
        self.model.set_weights(self._decode(new_w, w_sh, w_len))

        del encode_w, w_sh, w_len
        del encode_v, v_sh, v_len

    def step(self, x, y, local_rate, global_rate, w, g_best, renewal: str = "acc"):
        """
        파티클의 한 스텝을 진행합니다.

        Args:
            x (list): 입력 데이터
            y (list): 출력 데이터
            local_rate (float): 지역최적해의 영향력
            global_rate (float): 전역최적해의 영향력
            w (float): 관성
            g_best (list): 전역최적해
            renewal (str, optional): 최고점수 갱신 방식. Defaults to "acc" | "acc" or "loss"

        Returns:
            list: 현재 파티클의 점수
        """
        self._update_velocity(local_rate, global_rate, w, g_best)
        self._update_weights()

        return self.get_score(x, y, renewal)

    def step_w(
        self, x, y, local_rate, global_rate, w, g_best, w_p, w_g, renewal: str = "acc"
    ):
        """
        파티클의 한 스텝을 진행합니다.
        기본 스텝의 변형으로, 지역최적해와 전역최적해의 분산 정도를 조정할 수 있습니다

        Args:
            x (list): 입력 데이터
            y (list): 출력 데이터
            local_rate (float): 지역 최적해의 영향력
            global_rate (float): 전역 최적해의 영향력
            w (float): 관성
            g_best (list): 전역 최적해
            w_p (float): 지역 최적해의 분산 정도
            w_g (float): 전역 최적해의 분산 정도
            renewal (str, optional): 최고점수 갱신 방식. Defaults to "acc" | "acc" or "loss"

        Returns:
            float: 현재 파티클의 점수
        """
        self._update_velocity_w(local_rate, global_rate, w, w_p, w_g, g_best)
        self._update_weights()

        return self.get_score(x, y, renewal)

    def get_best_score(self):
        """
        파티클의 최고점수를 반환합니다.

        Returns:
            float: 최고점수
        """
        return self.best_score

    def get_best_weights(self):
        """
        파티클의 최고점수를 받은 가중치를 반환합니다

        Returns:
            list: 가중치 리스트
        """
        return self.best_weights
