import gc
import json
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from .particle import Particle

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as r:
        print(r)


class Optimizer:
    """
    particle swarm optimization
    PSO 실행을 위한 클래스
    """

    def __init__(
        self,
        model: keras.models,
        loss: any = None,
        n_particles: int = None,
        c0: float = 0.5,
        c1: float = 1.5,
        w_min: float = 0.5,
        w_max: float = 1.5,
        negative_swarm: float = 0,
        mutation_swarm: float = 0,
        np_seed: int = None,
        tf_seed: int = None,
        random_state: tuple = None,
        convergence_reset: bool = False,
        convergence_reset_patience: int = 10,
        convergence_reset_min_delta: float = 0.0001,
        convergence_reset_monitor: str = "mse",
    ):
        """
        particle swarm optimization

        Args:
            model (keras.models): 모델 구조 - keras.models.model_from_json 을 이용하여 생성
            loss (str): 손실함수 - keras.losses 에서 제공하는 손실함수 사용
            n_particles (int): 파티클 개수
            c0 (float): local rate - 지역 최적값 관성 수치
            c1 (float): global rate - 전역 최적값 관성 수치
            w_min (float): 최소 관성 수치
            w_max (float): 최대 관성 수치
            negative_swarm (float): 최적해와 반대로 이동할 파티클 비율 - 0 ~ 1 사이의 값
            mutation_swarm (float): 돌연변이가 일어날 확률
            np_seed (int, optional): numpy seed. Defaults to None.
            tf_seed (int, optional): tensorflow seed. Defaults to None.
            convergence_reset (bool, optional): early stopping 사용 여부. Defaults to False.
            convergence_reset_patience (int, optional): early stopping 사용시 얼마나 기다릴지. Defaults to 10.
            convergence_reset_min_delta (float, optional): early stopping 사용시 얼마나 기다릴지. Defaults to 0.0001.
            convergence_reset_monitor (str, optional): early stopping 사용시 어떤 값을 기준으로 할지. Defaults to "loss". - "loss" or "acc" or "mse"
        """

        try:
            if model is None:
                raise ValueError("model is None")
            if model is not None and not isinstance(model, keras.models.Model):
                raise ValueError("model is not keras.models.Model")

            if loss is None:
                raise ValueError("loss is None")

            if n_particles is None:
                raise ValueError("n_particles is None")
            if n_particles < 1:
                raise ValueError("n_particles < 1")

            if c0 < 0 or c1 < 0:
                raise ValueError("c0 or c1 < 0")

            if np_seed is not None:
                np.random.seed(np_seed)
            if tf_seed is not None:
                tf.random.set_seed(tf_seed)

            self.random_state = np.random.get_state()

            if random_state is not None:
                np.random.set_state(random_state)

            model.compile(
                loss=loss,
                optimizer="adam",
                metrics=["accuracy", "mse"]
            )
            self.model = model  # 모델 구조
            self.loss = loss  # 손실함수
            self.n_particles = n_particles  # 파티클 개수
            self.particles = [None] * n_particles  # 파티클 리스트
            self.c0 = c0  # local rate - 지역 최적값 관성 수치
            self.c1 = c1  # global rate - 전역 최적값 관성 수치
            self.w_min = w_min  # 최소 관성 수치
            self.w_max = w_max  # 최대 관성 수치
            self.negative_swarm = negative_swarm  # 최적해와 반대로 이동할 파티클 비율 - 0 ~ 1 사이의 값
            self.mutation_swarm = mutation_swarm  # 관성을 추가로 사용할 파티클 비율 - 0 ~ 1 사이의 값
            self.g_best_score = [np.inf, 0, np.inf]  # 최고 점수 - 시작은 0으로 초기화
            self.g_best = model.get_weights()  # 최고 점수를 받은 가중치
            self.avg_score = 0  # 평균 점수
            # self.sigma = 1.0

            self.save_path = None  # 저장 위치
            self.renewal = "acc"
            self.dispersion = False
            self.day = datetime.now().strftime("%Y%m%d-%H%M%S")

            self.empirical_balance = False

            negative_count = 0

            self.train_summary_writer = [None] * self.n_particles

            print(f"start running time : {self.day}")
            for i in tqdm(range(self.n_particles), desc="Initializing Particles"):

                self.particles[i] = Particle(
                    model,
                    self.loss,
                    negative=True if i < self.negative_swarm * self.n_particles else False,
                    mutation=self.mutation_swarm,
                    converge_reset=convergence_reset,
                    converge_reset_patience=convergence_reset_patience,
                    converge_reset_monitor=convergence_reset_monitor,
                    converge_reset_min_delta=convergence_reset_min_delta,
                )

                if i < self.negative_swarm * self.n_particles:
                    negative_count += 1
                # del m, init_weights, w_, sh_, len_
                gc.collect()
                tf.keras.backend.reset_uids()
                tf.keras.backend.clear_session()

                # del model_

            print(f"negative swarm : {negative_count} / {n_particles}")
            print(f"mutation swarm : {mutation_swarm * 100}%")

            gc.collect()
            tf.keras.backend.reset_uids()
            tf.keras.backend.clear_session()
        except KeyboardInterrupt:
            sys.exit("Ctrl + C : Stop Training")
        except MemoryError:
            sys.exit("Memory Error : Stop Training")
        except ValueError as ve:
            sys.exit(ve)
        except Exception as e:
            sys.exit(e)

    def __del__(self):
        del self.model
        del self.loss
        del self.n_particles
        del self.particles
        del self.c0
        del self.c1
        del self.w_min
        del self.w_max
        del self.negative_swarm
        del self.g_best_score
        del self.g_best
        del self.avg_score

        gc.collect()
        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()

    def _encode_(self, weights):
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

        del weights

        return w_gpu, shape, length

    def _decode_(self, weight, shape, length):
        """
        _encode_ 로 인코딩된 가중치를 원본 shape으로 복원
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
            w_tmp = weight[start:end]
            w_tmp = np.reshape(w_tmp, shape[i])
            weights.append(w_tmp)
            start = end

        del weight, shape, length
        del start, end, w_tmp

        return weights

    def _f(self, x, y, weights):
        """
        EBPSO의 목적함수 (예상)

        Args:
            x (list): 입력 데이터
            y (list): 출력 데이터
            weights (list): 가중치

        Returns:
            (float): 목적 함수 값
        """
        self.model.set_weights(weights)
        score = self.model.evaluate(x, y, verbose=0)
        if self.renewal == "acc":
            score_ = score[1]
        else:
            score_ = score[0]

        if score_ > 0:
            return 1 / (1 + score_)
        else:
            return 1 + np.abs(score_)

    def __weight_range__(self):
        """
        가중치의 범위를 반환

        Returns:
            (float): 가중치의 최소값
            (float): 가중치의 최대값
        """
        w_, w_s, w_l = self._encode_(self.g_best)
        weight_min = np.min(w_)
        weight_max = np.max(w_)

        del w_, w_s, w_l

        return weight_min, weight_max

    class _batch_generator_:
        def __init__(self, x, y, batch_size: int = 32):
            self.batch_size = batch_size
            self.index = 0
            self.x = x
            self.y = y
            self.setBatchSize(batch_size)

        def next(self):
            self.index += 1
            if self.index >= self.max_index:
                self.index = 0
                self.__getBatchSlice__(self.batch_size)
            return self.dataset[self.index][0], self.dataset[self.index][1]

        def getMaxIndex(self):
            return self.max_index

        def getIndex(self):
            return self.index

        def setIndex(self, index):
            self.index = index

        def getBatchSize(self):
            return self.batch_size

        def setBatchSize(self, batch_size):
            self.batch_size = batch_size
            if self.batch_size > len(self.x):
                self.batch_size = len(self.x)
            print(f"batch size : {self.batch_size}")
            self.dataset = self.__getBatchSlice__(self.batch_size)
            self.max_index = len(self.dataset)

        def __getBatchSlice__(self, batch_size):
            return list(
                tf.data.Dataset.from_tensor_slices(
                    (self.x, self.y)).shuffle(len(self.x)).batch(batch_size)
            )

        def getDataset(self):
            return self.dataset

    def fit(
        self,
        x,
        y,
        epochs: int = 1,
        log: int = 0,
        log_name: str = None,
        save_info: bool = False,
        save_path: str = "./logs",
        renewal: str = "mse",
        empirical_balance: bool = False,
        dispersion: bool = False,
        check_point: int = None,
        batch_size: int = None,
        validate_data: any = None,
        back_propagation: bool = False,
    ):
        """
        # Args:
            x : numpy array,
            y : numpy array,
            epochs : int,
            log : int - 0 : log 기록 안함, 1 : csv, 2 : tensorboard,
            save_info : bool - 종료시 학습 정보 저장 여부 default : False,
            save_path : str - ex) "./result",
            renewal : str ex) "acc" or "loss" or "mse",
            empirical_balance : bool - True : EBPSO, False : PSO,
            dispersion : bool - True : g_best 의 값을 분산시켜 전역해를 찾음, False : g_best 의 값만 사용
            check_point : int - 저장할 위치 - None : 저장 안함
            batch_size : int - batch size default : None => len(x) // 10
                batch_size > len(x) : auto max batch size
        """
        try:
            if x.shape[0] != y.shape[0]:
                raise ValueError("x, y shape error")

            if log not in [0, 1, 2]:
                raise ValueError("log not in [0, 1, 2]")

            if save_info and save_path is None:
                raise ValueError("save_path is None")

            if renewal not in ["acc", "loss", "mse"]:
                raise ValueError("renewal not in ['acc', 'loss', 'mse']")

            if check_point is not None and save_path is None:
                raise ValueError("save_path is None")

        except ValueError as ve:
            sys.exit(ve)

        self.save_path = save_path
        self.empirical_balance = empirical_balance
        self.dispersion = dispersion

        if batch_size is None:
            batch_size = len(x) // 10

        self.renewal = renewal
        particle_sum = 0  # x_j
        try:
            train_log_dir = "logs/fit/" + self.day
            if log == 2:
                assert log_name is not None, "log_name is None"

                train_log_dir = f"logs/{log_name}/{self.day}/train"
                for i in range(self.n_particles):
                    self.train_summary_writer[i] = tf.summary.create_file_writer(
                        train_log_dir + f"/{i}"
                    )

            elif check_point is not None or log == 1:
                if save_path is None:
                    raise ValueError("save_path is None")
                else:
                    self.save_path = save_path
                    if not os.path.exists(f"{save_path}/{self.day}"):
                        os.makedirs(f"{save_path}/{self.day}", exist_ok=True)
        except ValueError as ve:
            sys.exit(ve)

        if back_propagation:
            model_ = keras.models.model_from_json(self.model.to_json())
            model_.compile(
                loss=self.loss,
                optimizer="adam",
                metrics=["accuracy", "mse"]
            )
            model_.fit(x, y, epochs=1, verbose=0)
            score = model_.evaluate(x, y, verbose=1)

            self.g_best_score = score

            self.g_best = model_.get_weights()

            del model_

        dataset = self._batch_generator_(x, y, batch_size=batch_size)

        try:
            epoch_sum = 0
            epochs_pbar = tqdm(
                range(epochs),
                desc=f"best - loss: {self.g_best_score[0]:.4f} - acc: {self.g_best_score[1]:.4f} - mse: {self.g_best_score[2]:.4f}",
                ascii=True,
                leave=True,
                position=0,
            )
            for epoch in epochs_pbar:
                # 이번 epoch의 평균 점수
                particle_avg = particle_sum / self.n_particles  # x_j
                particle_sum = 0
                # 각 최고 점수, 최저 loss, 최저 mse
                max_acc = 0
                min_loss = np.inf
                min_mse = np.inf
                # epoch_particle_sum = 0
                part_pbar = tqdm(
                    range(len(self.particles)),
                    desc=f"loss: {min_loss:.4f} acc: {max_acc:.4f} mse: {min_mse:.4f}",
                    ascii=True,
                    leave=False,
                    position=1,
                )
                w = self.w_max - (self.w_max - self.w_min) * epoch / epochs
                for i in part_pbar:
                    part_pbar.set_description(
                        f"loss: {min_loss:.4f} acc: {max_acc:.4f} mse: {min_mse:.4f}"
                    )

                    g_best = self.g_best

                    x_batch, y_batch = dataset.next()

                    weight_min, weight_max = self.__weight_range__()

                    if dispersion:
                        ts = weight_min + np.random.rand() * (weight_max - weight_min)

                        g_, g_sh, g_len = self._encode_(self.g_best)
                        decrement = (epochs - epoch + 1) / epochs
                        g_ = (1 - decrement) * g_ + decrement * ts
                        g_best = self._decode(g_, g_sh, g_len)

                    if empirical_balance:
                        if np.random.rand() < np.exp(-(epoch) / epochs):
                            w_p_ = self._f(
                                x, y, self.particles[i].get_best_weights()
                            )
                            w_g_ = self._f(x, y, g_best)

                            w_p = w_p_ / (w_p_ + w_g_)
                            w_g = w_p_ / (w_p_ + w_g_)

                            del w_p_
                            del w_g_

                        else:
                            p_b = self.particles[i].get_best_score()
                            g_a = self.avg_score
                            l_b = p_b - g_a
                            sigma_post = np.sqrt(np.power(l_b, 2))
                            sigma_pre = (
                                1
                                / (
                                    self.n_particles
                                    * np.linalg.norm(
                                        weight_min - weight_max
                                    )
                                )
                                * sigma_post
                            )
                            p_ = np.exp(-1 * sigma_pre * sigma_post)

                            # p_ = (
                            #     1
                            #     / (self.n_particles * np.linalg.norm(self.particle_max - self.particle_min))
                            #     * np.exp(
                            #         -np.power(l_b, 2) / (2 * np.power(self.sigma, 2))
                            #     )
                            # )
                            # g_ = (
                            #     1
                            #     / np.linalg.norm(self.c1 - self.c0)
                            #     * np.exp(
                            #         -np.power(l_b, 2) / (2 * np.power(self.sigma, 2))
                            #     )
                            # )
                            # w_p = p_ / (p_ + g_)
                            # w_g = g_ / (p_ + g_)
                            w_p = p_
                            w_g = 1 - p_

                            del p_b
                            del g_a
                            del l_b
                            del p_

                        score = self.particles[i].step_w(
                            x_batch,
                            y_batch,
                            self.c0,
                            self.c1,
                            w,
                            g_best,
                            w_p,
                            w_g,
                            renewal=renewal,
                        )
                        epoch_sum += np.power(score[1] - particle_avg, 2)

                    else:
                        score = self.particles[i].step(
                            x_batch, y_batch, self.c0, self.c1, w, g_best, renewal=renewal
                        )

                    if log == 2:
                        with self.train_summary_writer[i].as_default():
                            tf.summary.scalar("loss", score[0], step=epoch + 1)
                            tf.summary.scalar(
                                "accuracy", score[1], step=epoch + 1
                            )
                            tf.summary.scalar("mse", score[2], step=epoch + 1)

                    if renewal == "acc":
                        # 최고 점수 보다 높거나 같을 경우
                        if score[1] >= max_acc:
                            # 각 점수 갱신
                            min_loss, max_acc, min_mse = score
                        # 최고 점수 보다 같거나 높을 경우
                        if score[1] >= self.g_best_score[1]:
                            # 최고 점수 보다 높을 경우
                            if score[1] > self.g_best_score[1]:
                                # 최고 점수 갱신
                                self.g_best_score = score
                                # 최고 weight 갱신
                                self.g_best = self.particles[i].get_best_weights(
                                )
                            # 최고 점수 와 같을 경우
                            else:
                                # 최저 loss 보다 낮을 경우
                                if score[0] < self.g_best_score[0]:
                                    self.g_best_score[0] = score[0]
                                    self.g_best = self.particles[i].get_best_weights(
                                    )
                            epochs_pbar.set_description(
                                f"best - loss: {self.g_best_score[0]:.4f} - acc: {self.g_best_score[1]:.4f} - mse: {self.g_best_score[2]:.4f}"
                            )

                    elif renewal == "loss":
                        # 최저 loss 보다 작거나 같을 경우
                        if score[0] <= min_loss:
                            # 각 점수 갱신
                            min_loss, max_acc, min_mse = score

                        # 최저 loss 와 같거나 작을 경우
                        if score[0] <= self.g_best_score[0]:
                            # 최저 loss 보다 작을 경우
                            if score[0] < self.g_best_score[0]:
                                # 최고 점수 갱신
                                self.g_best_score = score
                                # 최고 weight 갱신
                                self.g_best = self.particles[i].get_best_weights(
                                )
                            # 최저 loss 와 같을 경우
                            else:
                                # 최고 acc 보다 높을 경우
                                if score[1] > self.g_best_score[1]:
                                    self.g_best_score[1] = score[1]
                                    self.g_best = self.particles[i].get_best_weights(
                                    )
                            epochs_pbar.set_description(
                                f"best - loss: {self.g_best_score[0]:.4f} - acc: {self.g_best_score[1]:.4f} - mse: {self.g_best_score[2]:.4f}"
                            )

                    elif renewal == "mse":
                        if score[2] <= min_mse:
                            min_loss, max_acc, min_mse = score

                        if score[2] <= self.g_best_score[2]:
                            if score[2] < self.g_best_score[2]:
                                self.g_best_score[0] = score[0]
                                self.g_best_score[1] = score[1]
                                self.g_best_score[2] = score[2]

                                self.g_best = self.particles[i].get_best_weights(
                                )

                            else:
                                if score[1] > self.g_best_score[1]:
                                    self.g_best_score[1] = score[1]
                                    self.g_best = self.particles[i].get_best_weights(
                                    )
                            epochs_pbar.set_description(
                                f"best - loss: {self.g_best_score[0]:.4f} - acc: {self.g_best_score[1]:.4f} - mse: {self.g_best_score[2]:.4f}"
                            )

                    particle_sum += score[1]

                    if log == 1:
                        with open(
                            f"./{save_path}/{self.day}/{self.n_particles}_{epochs}_{self.c0}_{self.c1}_{self.w_min}_{renewal}.csv",
                            "a",
                        ) as f:
                            f.write(f"{score[0]}, {score[1]}")
                            if i != self.n_particles - 1:
                                f.write(", ")
                            else:
                                f.write("\n")
                part_pbar.refresh()

                if check_point is not None:
                    if epoch % check_point == 0:
                        os.makedirs(f"./{save_path}/{self.day}", exist_ok=True)
                        self._check_point_save(
                            f"./{save_path}/{self.day}/ckpt-{epoch}")

                tf.keras.backend.reset_uids()
                tf.keras.backend.clear_session()
                gc.collect()

        except KeyboardInterrupt:
            print("Ctrl + C : Stop Training")

        except MemoryError:
            print("Memory Error : Stop Training")

        except Exception as e:
            print(e)

        finally:
            self.model_save(x, y, save_path)
            print("model save")
            if save_info:
                self.save_info(save_path)
                print("save info")

            return self.g_best_score

    def get_best_model(self):
        """
        최고 점수를 받은 모델을 반환

        Returns:
            (keras.models): 모델
        """
        model = keras.models.model_from_json(self.model.to_json())
        model.set_weights(self.g_best)
        model.compile(
            loss=self.loss,
            optimizer="adam",
            metrics=["accuracy", "mse"]
        )

        return model

    def get_best_score(self):
        """
        최고 점수를 반환

        Returns:
            (float): 점수
        """
        return self.g_best_score

    def get_best_weights(self):
        """
        최고 점수를 받은 가중치를 반환

        Returns:
            (float): 가중치
        """
        return self.g_best

    def save_info(self, path: str = "./result"):
        """
        학습 정보를 저장

        Args:
            path (str, optional): 저장 위치. Defaults to "./result".
        """
        json_save = {
            "name": f"{self.day}/{self.n_particles}_{self.c0}_{self.c1}_{self.w_min}.h5",
            "n_particles": self.n_particles,
            "score": self.g_best_score,
            "c0": self.c0,
            "c1": self.c1,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "loss_method": self.loss,
            "empirical_balance": self.empirical_balance,
            "dispersion": self.dispersion,
            "negative_swarm": self.negative_swarm,
            "mutation_swarm": self.mutation_swarm,
            "random_state_0": self.random_state[0],
            "random_state_1": self.random_state[1].tolist(),
            "random_state_2": self.random_state[2],
            "random_state_3": self.random_state[3],
            "random_state_4": self.random_state[4],
            "renewal": self.renewal,
        }

        with open(
            f"./{path}/{self.day}/{self.loss}_{self.g_best_score}.json",
            "a",
        ) as f:
            json.dump(json_save, f, indent=4)

    def _check_point_save(self, save_path: str = f"./result/check_point"):
        """
        중간 저장

        Args:
            save_path (str, optional): checkpoint 저장 위치 및 이름. Defaults to f"./result/check_point".
        """
        model = self.get_best_model()
        model.save_weights(save_path)

    def model_save(self, x, y, save_path: str = "./result"):
        """
        최고 점수를 받은 모델 저장

        Args:
            save_path (str, optional): 모델의 저장 위치. Defaults to "./result".

        Returns:
            (keras.models): 모델
        """
        model = self.get_best_model()
        score = model.evaluate(x, y, verbose=1)
        print(
            f"model score - loss: {score[0]} - acc: {score[1]} - mse: {score[2]}")
        model.save(
            f"./{save_path}/{self.day}/model_{score[0 if self.renewal == 'loss' else 1 if self.renewal == 'acc'  else 2 ]}.h5"
        )
        return model
