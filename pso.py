import numpy as np

class PSO(object):
    """
    Class implementing PSO algorithm
    """

    def __init__(self, func, init_pos, n_particles):
        """
        Initialize the key variables.

        Args:
            fun (function): the fitness function to optimize
            init_pos(array_like):
            n_particles(int): the number of particles of the swarm.
        """
        self.func = func
        self.n_particles = n_particles
        self.init_pos = init_pos           # 곰샥헐 차원
        self.particle_dim = len(init_pos)  # 검색할 차원의 크기
        self.particles_pos = np.random.uniform(size=(n_particles, self.particle_dim)) \
            * self.init_pos
        # 입력받은 파티클의 개수 * 검색할 차원의 크기 만큼의 균등한 위치를 생성
        self.velocities = np.random.uniform(
            size=(n_particles, self.particle_dim))
        # 입력받은 파티클의 개수 * 검색할 차원의 크기 만큼의 속도를 무작위로 초기화
        self.g_best = init_pos             # 최대 사이즈로 전역 최적갑 저장 - global best
        self.p_best = self.particles_pos   # 모든 파티클의 위치 - particles best
        self.g_history = []
        self.history = []

    def update_position(self, x, v):
        """
        Update particle position

        Args:
            x (array-like): particle current position
            v (array-like): particle current velocity

        Returns:
            The updated position(array-like)
        """
        x = np.array(x)  # 각 파티클의 위치
        v = np.array(v)  # 각 파티클의 속도(방향과 속력을 가짐)
        new_x = x + v   # 각 파티클을 랜덤한 속도만큼 진행
        return new_x    # 진행한 파티클들의 위치를 반환

    def update_velocity(self, x, v, p_best, g_best, c0=0.5, c1=1.5, w=0.75):
        """
            Update particle velocity

            Args:
                x(array-like): particle current position
                v (array-like): particle current velocity
                p_best(array-like): the best position found so far for a particle
                g_best(array-like): the best position regarding all the particles found so far
                c0 (float): the congnitive scaling constant, 인지 스케일링 상수
                c1 (float): the social scaling constant
                w (float): the inertia weight, 관성 중량

            Returns:
                The updated velocity (array-like).
        """
        x = np.array(x)
        v = np.array(v)
        assert x.shape == v.shape, "Position and velocity must have same shape."
        # 두 데이터의 shape 이 같지 않으면 오류 출력
        # 0에서 1사이의 숫자를 랜덤 생성
        r = np.random.uniform()
        p_best = np.array(p_best)
        g_best = np.array(g_best)

        # 가중치(상수)*속도 + \
        # 스케일링 상수*랜덤 가중치*(나의 최적값 - 처음 위치) + \
        # 전역 스케일링 상수*랜덤 가중치*(전체 최적값 - 처음 위치)
        new_v = w*v + c0*r*(p_best - x) + c1*r*(g_best - x)
        return new_v

    def optimize(self, maxiter=200):
        """
        Run the PSO optimization process utill the stoping critera is met.
        Cas for minization. The aim is to minimize the cost function

        Args:
            maxiter (int): the maximum number of iterations before stopping the optimization
            파티클의 최종 위치를 위한 반복 횟수
        Returns:
            The best solution found (array-like)
        """
        for _ in range(maxiter):
            for i in range(self.n_particles):
                x = self.particles_pos[i]  # 각 파티클 추출
                v = self.velocities[i]    # 랜덤 생성한 속도 추출
                p_best = self.p_best[i]   # 결과치 저장할 변수 지정
                self.velocities[i] = self.update_velocity(
                    x, v, p_best, self.g_best)
                # 다음에 움직일 속도 = 최초 위치, 현재 속도, 현재 위치, 최종 위치
                self.particles_pos[i] = self.update_position(x, v)
                # 현재 위치 = 최초 위치 현재 속도
                # Update the besst position for particle i
                # 내 현재 위치가 내 위치의 최소치보다 작으면 갱신
                if self.func(self.particles_pos[i]) < self.func(p_best):
                    self.p_best[i] = self.particles_pos[i]
                # Update the best position overall
                # 내 현재 위치가 전체 위치 최소치보다 작으면 갱신
                if self.func(self.particles_pos[i]) < self.func(self.g_best):
                    self.g_best = self.particles_pos[i]
                    self.g_history.append(self.g_best)
                    
            self.history.append(self.particles_pos.copy())

        # 전체 최소 위치, 전체 최소 벡터
        return self.g_best, self.func(self.g_best)

    """
    Returns:
        현재 전체 위치
    """

    def position(self):
        return self.particles_pos.copy()

    """
    Returns:
        전체 위치 벡터 history
    """

    def position_history(self):
        return self.history

    """
    Returns:
        global best 의 갱신된 값의 변화를 반환
    """

    def global_history(self):
        return self.g_history.copy()
