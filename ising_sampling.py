import numpy as np

class IsingSampler:
    def __init__(self, L, T, nsteps):
        self.L = L
        self.N = L * L
        self.T = T
        self.nsteps = nsteps
        self.state = np.random.choice([1, -1], size=(L, L))
        self.energies = []
        self.magnetizations = []
        self.states = []

    def energy(self, state):
        E = 0
        for i in range(self.L):
            for j in range(self.L):
                S = state[i, j]
                nb = state[(i+1)%self.L, j] + state[i, (j+1)%self.L]
                E -= S * nb
        return E

    def magnetization(self, state):
        return np.sum(state)

    def mcmc_metropolis(self):
        state = self.state.copy()
        for step in range(self.nsteps):
            i, j = np.random.randint(0, self.L, size=2)
            S = state[i, j]
            nb = state[(i+1)%self.L, j] + state[(i-1)%self.L, j] + state[i, (j+1)%self.L] + state[i, (j-1)%self.L]
            delta_E = 2 * S * nb
            if np.random.rand() < np.exp(-delta_E / self.T):
                state[i, j] *= -1
            self.energies.append(self.energy(state))
            self.magnetizations.append(self.magnetization(state))
            self.states.append(state.copy())
        return np.array(self.states), np.array(self.energies), np.array(self.magnetizations)

    def random_sampling(self):
        # 不考慮能量，隨機產生配置
        for step in range(self.nsteps):
            state = np.random.choice([1, -1], size=(self.L, self.L))
            self.energies.append(self.energy(state))
            self.magnetizations.append(self.magnetization(state))
            self.states.append(state.copy())
        return np.array(self.states), np.array(self.energies), np.array(self.magnetizations)
    
    def wolff_cluster(self):
        state = self.state.copy()
        p_add = 1 - np.exp(-2.0 / self.T)
        for step in range(self.nsteps):
            visited = np.zeros_like(state, dtype=bool)
            # 隨機選一個種子
            i, j = np.random.randint(0, self.L, size=2)
            cluster_spin = state[i, j]
            stack = [(i, j)]
            cluster = []
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                visited[x, y] = True
                if state[x, y] == cluster_spin:
                    cluster.append((x, y))
                    # 檢查四個鄰居
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = (x+dx)%self.L, (y+dy)%self.L
                        if not visited[nx, ny] and state[nx, ny] == cluster_spin:
                            if np.random.rand() < p_add:
                                stack.append((nx, ny))
            # 翻轉整個 cluster
            for (x, y) in cluster:
                state[x, y] *= -1
            self.energies.append(self.energy(state))
            self.magnetizations.append(self.magnetization(state))
            self.states.append(state.copy())
        return np.array(self.states), np.array(self.energies), np.array(self.magnetizations)
    
    def mps_sampling(self, nsamples=None):
        import quimb as qu
        import quimb.tensor as qtn
        L = self.L
        if nsamples is None:
            nsamples = self.nsteps
        # 這裡僅示意：實際應用需根據物理模型構建正確的MPS
        # 這裡直接隨機產生樣本，展示接口
        samples = []
        for _ in range(nsamples):
            sample = np.random.choice([1, -1], size=(L, L))
            samples.append(sample)
        samples = np.array(samples)
        # 你可以進一步計算能量與磁化量
        energies = [self.energy(sample) for sample in samples]
        magnetizations = [self.magnetization(sample) for sample in samples]
        return samples, np.array(energies), np.array(magnetizations)
    
    def mps_sampling_1d(self, nsamples=None):
        """
        精確 1D Ising Boltzmann sampling (transfer matrix method)
        只適用於 1D Ising，產生 shape (nsamples, L) 的樣本
        """
        import numpy as np

        L = self.L
        J = 1.0
        h = 0.0
        T = self.T
        beta = 1.0 / T
        if nsamples is None:
            nsamples = self.nsteps

        # 定義 transfer matrix
        spins = np.array([-1, 1])
        Tmat = np.zeros((2, 2))
        for i, s in enumerate(spins):
            for j, sp in enumerate(spins):
                Tmat[i, j] = np.exp(beta * J * s * sp + 0.5 * beta * h * (s + sp))

        # 預先計算右向“消息” (backward propagation)
        # right_msg[n, i] = 從 site n+1 到 L 的配分函數，條件是 site n = spins[i]
        right_msg = np.ones((L+1, 2))
        for n in range(L-1, -1, -1):
            for i, s in enumerate(spins):
                right_msg[n, i] = np.sum(Tmat[i, :] * right_msg[n+1, :])

        # 開始逐步 sample
        samples = []
        for _ in range(nsamples):
            config = []
            # 第一個 site 的機率
            prob0 = right_msg[0, :] / np.sum(right_msg[0, :])
            s0 = np.random.choice(spins, p=prob0)
            config.append(s0)
            idx = np.where(spins == s0)[0][0]
            # 之後每個 site 的條件機率
            for n in range(1, L):
                cond_prob = Tmat[idx, :] * right_msg[n, :]
                cond_prob = cond_prob / np.sum(cond_prob)
                sn = np.random.choice(spins, p=cond_prob)
                config.append(sn)
                idx = np.where(spins == sn)[0][0]
            samples.append(config)
        samples = np.array(samples)  # shape (nsamples, L)

        # 能量與磁化量
        def energy_1d(state):
            # state: (L,)
            return -np.sum(state * np.roll(state, -1))

        energies = np.array([energy_1d(s) for s in samples])
        magnetizations = np.sum(samples, axis=1)
        return samples, energies, magnetizations