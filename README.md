[![Python Package Index publish](https://github.com/jung-geun/PSO/actions/workflows/pypi.yml/badge.svg?event=push)](https://github.com/jung-geun/PSO/actions/workflows/pypi.yml)
<a href="https://colab.research.google.com/github/jung-geun/PSO/blob/master/pso2keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PSO 알고리즘 구현 및 새로운 시도

pso 알고리즘을 사용하여 새로운 학습 방법을 찾는중 입니다
병렬처리로 사용하는 논문을 찾아보았지만 이보다 더 좋은 방법이 있을 것 같아서 찾아보고 있습니다 - \[1]

기본 pso 알고리즘의 수식은 다음과 같습니다

> $$V_{id(t+1)} = W_{V_id(t)} + c_1 * r_1 (p_{id(t)} - x_{id(t)}) + c_2 * r_2(p_{gd(t)} - x_{id(t)})$$

다음 속도을 구하는 수식입니다

> $$x_{id(t+1)} = x_{id(t)} + V_{id(t+1)}$$

다음 위치를 구하는 수식입니다

> $$
> p_{id(t+1)} =
> \begin{cases}
> x_{id(t+1)} & \text{if } f(x_{id(t+1)}) < f(p_{id(t)})\\
> p_{id(t)} & \text{otherwise}
> \end{cases}
> $$

# 초기 세팅

자동으로 conda 환경을 설정하기 위해서는 다음 명령어를 사용합니다

```shell
conda env create -f ./conda_env/environment.yaml
```

현재 python 3.9 버전, tensorflow 2.11 버전에서 테스트 되었습니다
</br>
직접 설치하여 사용할 경우 pso2keras 패키지를 pypi 에서 다운로드 받아서 사용하시기 바랍니다

```shell
pip install pso2keras==0.1.4
```

위의 패키지를 사용하기 위해서는 tensorflow 와 tensorboard 가 설치되어 있어야 합니다

python 패키지를 사용하기 위한 라이브러리는 아래 코드를 사용합니다

```python
from pso import Optimizer

pso_model = Optimizer(...)
pso_model.fit(...)
```

# 현재 진행 상황

## 1. PSO 알고리즘 구현

### 파일 구조

```plain
|-- /conda_env              # conda 환경 설정 파일
|  |-- environment.yaml     # conda 환경 설정 파일
|-- /metacode               # pso 기본 코드
|  |-- pso_bp.py            # 오차역전파 함수를 최적화하는 PSO 알고리즘 구현 - 성능이 99% 이상으로 나오나 목적과 다름
|  |-- pso_meta.py          # PSO 기본 알고리즘 구현
|  |-- pso_tf.py            # tensorflow 모델을 이용가능한 PSO 알고리즘 구현
|-- /pso                    # tensorflow 모델을 학습하기 위해 기본 pso 코드에서 수정 - (psokeras 코드 의 구조를 사용하여 만듬)
|  |-- __init__.py          # pso 모듈을 사용하기 위한 초기화 파일
|  |-- optimizer.py         # pso 알고리즘 이용을 위한 기본 코드
|  |-- particle.py          # 각 파티클의 정보 및 위치를 저장하는 코드
|-- xor.py                  # pso 를 이용한 xor 문제 풀이
|-- iris.py                 # pso 를 이용한 iris 문제 풀이
|-- iris_tf.py              # tensorflow 를 이용한 iris 문제 풀이
|-- mnist.py                # pso 를 이용한 mnist 문제 풀이
|-- mnist_tf.py             # tensorflow 를 이용한 mnist 문제 풀이
|-- plt.ipynb               # pyplot 으로 학습 결과를 그래프로 표현
|-- README.md               # 현재 파일
|-- requirements.txt        # pypi 에서 다운로드 받을 패키지 목록
```

pso 라이브러리는 tensorflow 모델을 학습하기 위해 기본 ./metacode/pso_meta.py 코드에서 수정하였습니다 [2]

## 2. PSO 알고리즘을 이용한 최적화 문제 풀이

pso 알고리즘을 이용하여 오차역전파 함수를 최적화 하는 방법을 찾는 중입니다

### 알고리즘 작동 방식

> 1. 파티클의 위치와 속도를 초기화 한다.
> 2. 각 파티클의 점수를 계산한다.
> 3. 각 파티클의 지역 최적해와 전역 최적해를 구한다.
> 4. 각 파티클의 속도를 업데이트 한다.

## 3. PSO 알고리즘을 이용하여 풀이한 문제들의 정확도

### 1. xor 문제

```python
loss = 'mean_squared_error'

pso_xor = Optimizer(
    model,
    loss=loss,
    n_particles=50,
    c0=0.35,
    c1=0.8,
    w_min=0.6,
    w_max=1.2,
    negative_swarm=0.1,
    mutation_swarm=0.2,
    particle_min=-3,
    particle_max=3,
)

best_score = pso_xor.fit(
    x_test,
    y_test,
    epochs=200,
    save=True,
    save_path="./result/xor",
    renewal="acc",
    empirical_balance=False,
    Dispersion=False,
    check_point=25,
)
```

위의 파라미터 기준 10 세대 근처부터 정확도가 100%가 나오는 것을 확인하였습니다
![xor](./history_plt/xor_2_10.png)

2. iris 문제

```python
loss = 'mean_squared_error'

pso_iris = Optimizer(
    model,
    loss=loss,
    n_particles=100,
    c0=0.35,
    c1=0.7,
    w_min=0.5,
    w_max=0.9,
    negative_swarm=0.1,
    mutation_swarm=0.2,
    particle_min=-3,
    particle_max=3,
)

best_score = pso_iris.fit(
    x_train,
    y_train,
    epochs=200,
    save=True,
    save_path="./result/iris",
    renewal="acc",
    empirical_balance=False,
    Dispersion=False,
    check_point=25
)
```

위의 파라미터 기준 7 세대에 97%, 35 세대에 99.16%의 정확도를 보였습니다
![iris](./history_plt/iris_99.17.png)

위의 그래프를 보면 epochs 이 늘어나도 정확도와 loss 가 수렴하지 않는것을 보면 파라미터의 이동 속도가 너무 빠르다고 생각합니다

3. mnist 문제

```python
loss = 'mean_squared_error'

pso_mnist = Optimizer(
    model,
    loss=loss,
    n_particles=100,
    c0=0.3,
    c1=0.5,
    w_min=0.4,
    w_max=0.7,
    negative_swarm=0.1,
    mutation_swarm=0.2,
    particle_min=-5,
    particle_max=5,
)

best_score = pso_mnist.fit(
    x_train,
    y_train,
    epochs=200,
    save_info=True,
    log=2,
    log_name="mnist",
    save_path="./result/mnist",
    renewal="acc",
    check_point=25,
)
```

위의 파라미터 기준 현재 정확도 51.84%를 보이고 있습니다
![mnist_acc](./history_plt/mnist_51.74_acc.png)
![mnist_loss](./history_plt/mnist_51.74_loss.png)

### Trouble Shooting

> 1. 딥러닝 알고리즘 특성상 weights는 처음 컴파일시 무작위하게 생성된다. weights의 각 지점의 중요도는 매번 무작위로 정해지기에 전역 최적값으로 찾아갈 때 값이 높은 loss를 향해서 상승하는 현상이 나타난다.<br>
>    따라서 weights의 이동 방법을 더 탐구하거나, weights를 초기화 할때 random 중요도를 좀더 노이즈가 적게 생성하는 방향을 모색해야할 것 같다.

-> 고르게 초기화 하기 위해 np.random.uniform 함수를 사용하였습니다

> 2. 지역최적값에 계속 머무르는 조기 수렴 현상이 나타난다. - 30% 정도의 정확도를 가진다

-> 지역최적값에 머무르는 것을 방지하기 위해 negative_swarm, mutation_swarm 파라미터를 추가하였습니다 - 현재 51% 정도의 정확도를 보이고 있습니다

> 3. 파티클의 수를 늘리면 전역 최적해에 좀더 가까워지는 현상을 발견하였다. 하지만 파티클의 수를 늘리면 메모리 사용량이 기하급수적으로 늘어난다.

-> keras 모델을 사용할때 predict, evaluate 함수를 사용하면 메모리 누수가 발생하는 문제를 찾았습니다. 해결방법을 추가로 찾아보는중 입니다.
-> 추가로 파티클의 수가 적을때에도 전역 최적해를 쉽게 찾는 방법을 찾는중 입니다

### 개인적인 생각

> 머신러닝 분류 방식에 존재하는 random forest 방식을 이용하여, 오차역전파 함수를 최적화 하는 방법이 있을것 같습니다
>
> > pso 와 random forest 방식이 매우 유사하다고 생각하여 학습할 때 뿐만 아니라 예측 할 때도 이러한 방식으로 사용할 수 있을 것 같습니다
>
> 각

# 참고 자료

[1]: [A partilce swarm optimization algorithm with empirical balance stategy](https://www.sciencedirect.com/science/article/pii/S2590054422000185#bib0005) </br>
[2]: [psokeras](https://github.com/mike-holcomb/PSOkeras) </br>
[3]: [PSO의 다양한 영역 탐색과 지역적 미니멈 인식을 위한 전략](https://koreascience.kr/article/JAKO200925836515680.pdf) </br>
[4]: [PC 클러스터 기반의 Multi-HPSO를 이용한 안전도 제약의 경제 급전](https://koreascience.kr/article/JAKO200932056732373.pdf) </br>
[5]: [Particle 2-Swarm Optimization for Robust Search](https://s-space.snu.ac.kr/bitstream/10371/29949/3/management_information_v18_01_p01.pdf) </br>
