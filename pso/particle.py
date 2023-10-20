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

    def __init__(
        self,
        model: keras.models,
        loss,
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
        self.model = model
        self.loss = loss

        try:
            if converge_reset and converge_reset_monitor not in ["acc", "accuracy", "loss", "mse"]:
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

        self.__reset_particle__()
        self.best_weights = self.model.get_weights()
        self.before_best = self.model.get_weights()
        self.negative = negative
        self.mutation = mutation
        self.best_score = [np.inf, 0, np.inf]
        self.before_w = 0
        self.score_history = []
        self.converge_reset = converge_reset
        self.converge_reset_patience = converge_reset_patience
        self.converge_reset_monitor = converge_reset_monitor
        self.converge_reset_min_delta = converge_reset_min_delta

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
            w_tmp = layer.reshape(-1)
            length.append(len(w_tmp))
            w_gpu = np.append(w_gpu, w_tmp)

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
        score = self.model.evaluate(x, y, verbose=0)
        if renewal == "loss":
            if score[0] < self.best_score[0]:
                self.best_score[0] = score[0]
                self.best_weights = self.model.get_weights()
        elif renewal == "acc":
            if score[1] > self.best_score[1]:
                self.best_score[1] = score[1]
                self.best_weights = self.model.get_weights()
        elif renewal == "mse":
            if score[2] < self.best_score[2]:
                self.best_score[2] = score[2]
                self.best_weights = self.model.get_weights()
        else:
            raise ValueError("renewal must be 'acc' or 'loss' or 'mse'")

        return score

    def __check_converge_reset__(self, score, monitor="loss", patience: int = 10, min_delta: float = 0.0001):
        """
        early stop을 구현한 함수

        Args:
            score (float): 현재 점수 [0] - loss, [1] - acc
            monitor (str, optional): 감시할 점수. Defaults to "loss".
            patience (int, optional): early stop을 위한 기다리는 횟수. Defaults to 10.
            min_delta (float, optional): early stop을 위한 최소 변화량. Defaults to 0.0001.
        """
        if monitor in ["acc", "accuracy"]:
            self.score_history.append(score[1])
        elif monitor in ["loss"]:
            self.score_history.append(score[0])
        elif monitor in ["mse"]:
            self.score_history.append(score[2])
        else:
            raise ValueError(
                "monitor must be 'acc' or 'accuracy' or 'loss' or 'mse'")

        if len(self.score_history) > patience:
            last_scores = self.score_history[-patience:]
            if max(last_scores) - min(last_scores) < min_delta:
                return True
        return False

    def __reset_particle__(self):
        self.model = keras.models.model_from_json(self.model.to_json())
        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=["accuracy", "mse"]
        )
        i_w_, i_s, i_l = self._encode(self.model.get_weights())
        i_w_ = np.random.uniform(-0.05, 0.05, len(i_w_))
        self.velocities = self._decode(i_w_, i_s, i_l)

        del i_w_, i_s, i_l
        self.score_history = []

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
        encode_before, before_sh, before_len = self._encode(
            weights=self.before_best
        )
        r_0 = np.random.rand()
        r_1 = np.random.rand()

        if not np.array_equal(encode_before, encode_g, equal_nan=True):
            self.before_w = w * 0.5
            w = w + self.before_w
        else:
            self.before_w *= 0.75
            w = w + self.before_w

        if self.negative:
            new_v = (
                w * encode_v
                + local_rate * r_0 * (encode_p - encode_w)
                + -1 * global_rate * r_1 * (encode_g - encode_w)
            )
            if len(self.score_history) > 10 and max(self.score_history[-10:]) - min(self.score_history[-10:]) < 0.01:
                self.__reset_particle__()

        else:
            new_v = (
                w * encode_v
                + local_rate * r_0 * (encode_p - encode_w)
                + global_rate * r_1 * (encode_g - encode_w)
            )

        if np.random.rand() < self.mutation:
            m_v = np.random.uniform(-0.1, 0.1, len(encode_v))
            new_v = m_v

        self.velocities = self._decode(new_v, w_sh, w_len)

        del encode_w, w_sh, w_len
        del encode_v, v_sh, v_len
        del encode_p, p_sh, p_len
        del encode_g, g_sh, g_len
        del encode_before, before_sh, before_len
        del r_0, r_1

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
        encode_before, before_sh, before_len = self._encode(
            weights=self.before_best
        )
        r_0 = np.random.rand()
        r_1 = np.random.rand()

        if not np.array_equal(encode_before, encode_g, equal_nan=True):
            self.before_w = w * 0.5
            w = w + self.before_w
        else:
            self.before_w *= 0.75
            w = w + self.before_w
        if self.negative:
            new_v = (
                w * encode_v
                + local_rate * r_0 * (w_p * encode_p - encode_w)
                + -1 * global_rate * r_1 * (w_g * encode_g - encode_w)
            )
        else:
            new_v = (
                w * encode_v
                + local_rate * r_0 * (w_p * encode_p - encode_w)
                + global_rate * r_1 * (w_g * encode_g - encode_w)
            )

        if np.random.rand() < self.mutation:
            m_v = np.random.uniform(-0.1, 0.1, len(encode_v))
            new_v = m_v

        self.velocities = self._decode(new_v, w_sh, w_len)

        del encode_w, w_sh, w_len
        del encode_v, v_sh, v_len
        del encode_p, p_sh, p_len
        del encode_g, g_sh, g_len
        del encode_before, before_sh, before_len
        del r_0, r_1

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

        score = self.get_score(x, y, renewal)

        if self.converge_reset and self.__check_converge_reset__(
                score, self.converge_reset_monitor, self.converge_reset_patience, self.converge_reset_min_delta):
            self.__reset_particle__()

        return score

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
