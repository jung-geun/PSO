from typing import Any

import numpy as np
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

    g_best_score = [np.inf, 0, np.inf]
    g_best_weights = None
    count = 0

    MODEL_IS_NONE = "model is None"

    def __init__(
        self,
        model: keras.Model,
        loss: Any = None,
        negative: bool = False,
        mutation: float = 0,
        converge_reset: bool = False,
        converge_reset_patience: int = 10,
        converge_reset_monitor: str = "loss",
        converge_reset_min_delta: float = 0.0001,
    ):
        """
        Args:
            model (keras.models): 학습 및 검증을 위한 모델
            loss (str|): 손실 함수
            negative (bool, optional): 음의 가중치 사용 여부 - 전역 탐색 용도(조기 수렴 방지). Defaults to False.
            mutation (float, optional): 돌연변이 확률. Defaults to 0.
            converge_reset (bool, optional): 조기 종료 사용 여부. Defaults to False.
            converge_reset_patience (int, optional): 조기 종료를 위한 기다리는 횟수. Defaults to 10.
        """
        self.set_model(model)
        self.weights = self._encode(model.get_weights())
        self.loss = loss

        try:
            if converge_reset and converge_reset_monitor not in [
                "acc",
                "accuracy",
                "loss",
                "mse",
            ]:
                raise ValueError(
                    "converge_reset_monitor must be 'acc' or 'accuracy' or 'loss'"
                )
            if converge_reset and converge_reset_min_delta < 0:
                raise ValueError("converge_reset_min_delta must be positive")
            if converge_reset and converge_reset_patience < 0:
                raise ValueError("converge_reset_patience must be positive")
        except ValueError as e:
            print(e)
            exit(1)

        self.velocities = np.zeros(len(self.weights))
        self.__reset_particle()
        self.best_weights = self.weights
        self.negative = negative
        self.mutation = mutation
        self.local_best_score = [np.inf, 0, np.inf]
        self.score_history = []
        self.converge_reset = converge_reset
        self.converge_reset_patience = converge_reset_patience
        self.converge_reset_monitor = converge_reset_monitor
        self.converge_reset_min_delta = converge_reset_min_delta
        Particle.count += 1

    def __del__(self):
        del self.model
        del self.loss
        del self.velocities
        del self.negative
        del self.local_best_score
        del self.best_weights
        Particle.count -= 1

    def set_shape(self, weights: list):
        """
        가중치의 shape을 설정

        Args:
            weights (list): keras model의 가중치
        """
        self.shape = [layer.shape for layer in weights]

    def get_shape(self):
        return self.shape

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
        for layer in weights:
            w_tmp = layer.reshape(-1)
            w_gpu = np.append(w_gpu, w_tmp)

        return w_gpu

    def _decode(self, weight: np.ndarray):
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
        for i in range(len(self.shape)):
            end = start + np.prod(self.shape[i])
            w_ = weight[start:end]
            w_ = np.reshape(w_, self.shape[i])
            weights.append(w_)
            start = end

        del start, end, w_
        del weight

        return weights

    def get_model(self):
        if self.model is None:
            raise ValueError(self.MODEL_IS_NONE)

        return self.model

    def set_model(self, model: keras.Model):
        self.model = model
        self.set_shape(self.model.get_weights())

    def compile(self):
        if self.model is None:
            raise ValueError(self.MODEL_IS_NONE)

        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=["accuracy", "mse"],
        )

    def get_weights(self):
        weights = self._decode(self.weights)

        return weights

    def evaluate(self, x, y):
        if self.model is None:
            raise ValueError(self.MODEL_IS_NONE)

        return self.model.evaluate(x, y, verbose=0)  # type: ignore

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

        score = self.evaluate(x, y)
        if renewal == "loss":
            if score[0] < self.local_best_score[0]:
                self.local_best_score = score
                self.best_weights = self.weights
        elif renewal == "acc":
            if score[1] > self.local_best_score[1]:
                self.local_best_score = score
                self.best_weights = self.weights
        elif renewal == "mse":
            if score[2] < self.local_best_score[2]:
                self.local_best_score = score
                self.best_weights = self.weights
        else:
            raise ValueError("renewal must be 'acc' or 'loss' or 'mse'")

        return score

    def __check_converge_reset(
        self,
        score,
        monitor: str = "auto",
        patience: int = 10,
        min_delta: float = 0.0001,
    ):
        """
        early stop을 구현한 함수

        Args:
            score (float): 현재 점수 [0] - loss, [1] - acc
            monitor (str, optional): 감시할 점수. Defaults to acc. | "acc" or "loss" or "mse"
            patience (int, optional): early stop을 위한 기다리는 횟수. Defaults to 10.
            min_delta (float, optional): early stop을 위한 최소 변화량. Defaults to 0.0001.
        """
        if monitor == "auto":
            monitor = "acc"
        if monitor in ["loss"]:
            self.score_history.append(score[0])
        elif monitor in ["acc", "accuracy"]:
            self.score_history.append(score[1])
        elif monitor in ["mse"]:
            self.score_history.append(score[2])
        else:
            raise ValueError("monitor must be 'acc' or 'accuracy' or 'loss' or 'mse'")

        if len(self.score_history) > patience:
            last_scores = self.score_history[-patience:]
            if max(last_scores) - min(last_scores) < min_delta:
                return True
        return False

    def __reset_particle(self):

        self.model = keras.models.model_from_json(self.model.to_json())
        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=["accuracy", "mse"],
        )
        self.weights = self._encode(self.model.get_weights())
        rng = np.random.default_rng()
        self.velocities = rng.uniform(-0.2, 0.2, len(self.weights))

        self.score_history = []

    def _velocity_calculation(self, local_rate, global_rate, w):
        """
        현재 속도 업데이트

        Args:
            local_rate (float): 지역 최적해의 영향력
            global_rate (float): 전역 최적해의 영향력
            w (float): 현재 속도의 영향력 - 관성 | 0.9 ~ 0.4 이 적당
        """
        # 0회차 전역 최적해가 없을 경우 현재 파티클의 최적해로 설정 - 전역최적해의 방향을 0으로 만들기 위함
        best_particle_weights = (
            self.best_weights
            if Particle.g_best_weights is None
            else Particle.g_best_weights
        )

        rng = np.random.default_rng(seed=42)
        r_0 = rng.random()
        r_1 = rng.random()

        if self.negative:
            # 지역 최적해와 전역 최적해를 음수로 사용하여 전역 탐색을 유도
            new_v = (
                w * self.velocities
                + local_rate * r_0 * (self.best_weights - self.weights)
                - global_rate * r_1 * (best_particle_weights - self.weights)
            )
            if (
                len(self.score_history) > 10
                and max(self.score_history[-10:]) - min(self.score_history[-10:]) < 0.01
            ):
                self.__reset_particle()

        else:
            # 전역 최적해의 acc 가 높을수록 더 빠르게 수렴
            # 하지만 loss 가 커진 상태에서는 전역 최적해의 영향이
            new_v = (
                w * self.velocities
                + local_rate
                * self.local_best_score[1]
                * r_0
                * (self.best_weights - self.weights)
                + global_rate
                * Particle.g_best_score[1]
                * r_1
                * (best_particle_weights - self.weights)
            )

        if self.mutation != 0.0 and rng.random() < self.mutation:
            m_v = rng.uniform(-0.2, 0.2, len(self.velocities))
            new_v = m_v

        self.velocities = new_v

        del r_0, r_1

    def _position_update(self):
        """
        가중치 업데이트
        """
        new_w = np.add(self.weights, self.velocities)

        self.model.set_weights(self._decode(new_w))

    def step(self, x, y, local_rate, global_rate, w, renewal: str = "acc"):
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
        self._velocity_calculation(local_rate, global_rate, w)
        self._position_update()

        score = self.get_score(x, y, renewal)

        if self.converge_reset and self.__check_converge_reset(
            score,
            self.converge_reset_monitor,
            self.converge_reset_patience,
            self.converge_reset_min_delta,
        ):
            self.__reset_particle()
            score = self.get_score(x, y, renewal)

        while (
            np.isnan(score[0])
            or np.isnan(score[1])
            or np.isnan(score[2])
            or score[0] == 0
            or score[1] == 0
            or score[2] == 0
            or np.isinf(score[0])
            or np.isinf(score[1])
            or np.isinf(score[2])
            or score[0] > 1000
            or score[1] > 1
            or score[2] > 1000
        ):
            self.__reset_particle()
            score = self.get_score(x, y, renewal)

        return score

    def get_best_score(self):
        """
        파티클의 최고점수를 반환합니다.

        Returns:
            float: 최고점수
        """
        return self.local_best_score

    def get_best_weights(self):
        """
        파티클의 최고점수를 받은 가중치를 반환합니다

        Returns:
            list: 가중치 리스트
        """
        return self._decode(self.best_weights)

    def set_global_score(self):
        """전역 최고점수를 현재 파티클의 최고점수로 설정합니다"""
        Particle.g_best_score = self.local_best_score

    def set_global_weights(self):
        """전역 최고점수를 받은 가중치를 현재 파티클의 최고점수를 받은 가중치로 설정합니다"""
        Particle.g_best_weights = self.best_weights

    def update_global_best(self):
        """현재 파티클의 점수와 가중치를 전역 최고점수와 가중치로 설정합니다"""
        self.set_global_score()
        self.set_global_weights()

    def check_global_best(self, renewal: str = "loss"):
        if (
            (renewal == "loss" and self.local_best_score[0] < Particle.g_best_score[0])
            or (
                renewal == "acc" and self.local_best_score[1] > Particle.g_best_score[1]
            )
            or (
                renewal == "mse" and self.local_best_score[2] < Particle.g_best_score[2]
            )
        ):
            self.update_global_best()


# 끝
