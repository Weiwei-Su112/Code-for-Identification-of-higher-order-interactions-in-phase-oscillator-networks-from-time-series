from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.interpolate import interp1d
from sklearn import linear_model
from tabulate import tabulate
import matplotlib as mpl
from scipy import stats
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import time
import csv
import os
from statsmodels.stats.proportion import proportions_ztest
import matplotlib


def Hamiltonian_path(adj, N):
    dp = [[False for i in range(1 << N)]
          for j in range(N)]

    # Set all dp[i][(1 << i)] to
    # true
    for i in range(N):
        dp[i][1 << i] = True

    # Iterate over each subset
    # of nodes
    for i in range(1 << N):
        for j in range(N):

            # If the jth nodes is included
            # in the current subset
            if (i & (1 << j)) != 0:

                # Find K, neighbour of j
                # also present in the
                # current subset
                for k in range(N):
                    if ((i & (1 << k)) != 0 and
                            adj[k][j] == 1 and
                            j != k and
                            dp[k][i ^ (1 << j)]):
                        # Update dp[j][i]
                        # to true
                        dp[j][i] = True
                        break

    # Traverse the vertices
    for i in range(N):
        # Hamiltonian Path exists
        if dp[i][(1 << N) - 1]:
            return True
    # Otherwise, return false
    return False


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time}")
        return result

    return wrapper


def gi_make_dirs():
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    tm_string = now.strftime("%H%M%S")
    file_path1 = "Fig3data/img/" + date + "/"
    file_path_time = "Fig3data/img/" + date + "/" + tm_string + "/"
    os.makedirs(file_path1, exist_ok=True)
    os.mkdir(file_path_time)

    return file_path_time


def reduce_conn_2(conn_2, cut_per=0.5, reduce_seed=None):
    conn_2copy = np.copy(conn_2)
    conn_2copy[conn_2copy != 0] = 1

    rng = np.random.default_rng(reduce_seed)

    for j in range(conn_2copy.shape[0]):
        for i in range(conn_2copy.shape[1]):
            if conn_2copy[j][i] == 1:
                indicator = rng.random()
                if indicator < cut_per:
                    conn_2copy[j][i] = 0

    return conn_2copy


def reduce_conn_3(conn_3, cut_per=0.5, reduce_seed=None):
    conn_3copy = np.copy(conn_3)
    conn_3copy[conn_3copy != 0] = 1

    rng = np.random.default_rng(reduce_seed)

    for i in range(conn_3copy.shape[0]):
        for j in range(conn_3copy.shape[1]):
            for k in range(conn_3copy.shape[1]):
                if j < k:
                    if conn_3copy[i][j][k] == 1:
                        indicator = rng.random()
                        if indicator < cut_per:
                            conn_3copy[i][j][k] = 0

    return conn_3copy


def get_nested_dict_values(d, avoid_keys=None):
    if avoid_keys is None:
        avoid_keys = []
    values = []
    for k_, v in d.items():
        if k_ in avoid_keys:
            continue
        elif isinstance(v, dict):
            values.extend(get_nested_dict_values(v, avoid_keys))
        else:
            values.append(v)
    return values


class GeneralInteraction:
    def __init__(self, dt=0.01, T=10, n_nodes=None, natfreqs=None, with_noise=True, noise_sth=1, random_sd=False,
                 pre_sd=None, conn=1, init_phase=None, coupling2=1, coupling3=1, random_coup2=False, random_coup3=False,
                 pre_coup2=None, pre_coup3=None, pre_conn2=None, pre_conn3=None, type2=1, type3=1, normalize=False,
                 regular2=False, regular3=False, conn_seed=None, coup_seed=None, noise_seed=None, all_connected=False,
                 starts_from=0.0, inf_last=1.0, conn2=0, conn3=0, r_ve=None, k_ve=None, seed_r_ve=None, seed_k_ve=None):
        """
        coupling: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        T: float
            Total time of simulated activity.
            From that the number of integration steps is T/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        """
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")

        # basis
        self.dt = dt
        self.T = T
        self.with_noise = with_noise
        self.coupling_coef2 = coupling2
        self.coupling_coef3 = coupling3
        self.type2 = type2  # has to be 0, 1, 2, or 3
        self.type3 = type3  # has to be 0, 1, 2, or 3
        if self.type2 == 3 and self.type3 != 3:
            raise ValueError("type3 must be 3 when type2 be 3")
        if self.type2 != 3 and self.type3 == 3:
            raise ValueError("type2 must be 3 when type3 be 3")
        self.conn_mat2 = None
        self.conn_mat3 = None
        self.coupling2 = None
        self.coupling3 = None
        self.starts_from = starts_from
        self.inf_last = inf_last
        assert starts_from + inf_last <= 1, "starts_from + inf_last need to be no larger than 1. "

        # natural frequency
        if natfreqs is not None:
            self.n_nodes = len(natfreqs)
            if self.type2 == 3:
                self.natfreqs = ["No omega"] * self.n_nodes
            else:
                self.natfreqs = natfreqs % (2 * np.pi)
        else:
            self.n_nodes = n_nodes
            if self.type2 == 3:
                self.natfreqs = ["No omega"] * self.n_nodes
            else:
                self.natfreqs = 2 * np.pi * np.random.normal(size=self.n_nodes) % (2 * np.pi)
                # self.natfreqs = 2 * np.pi * np.full(shape=self.n_nodes, fill_value=1) % (2 * np.pi)
        self.all_nodes = np.linspace(0, self.n_nodes - 1, self.n_nodes).astype(int)  # node number - 1

        # seeds
        self.conn_seed = None
        if pre_conn2 is None or pre_conn3 is None:
            if conn_seed is None:
                self.conn_seed = np.random.randint(low=0, high=1e9)
            elif conn_seed is not None:
                self.conn_seed = conn_seed

        self.coup_seed = None
        if pre_coup2 is None or pre_coup3 is None:
            if coup_seed is None:
                self.coup_seed = np.random.randint(low=0, high=1e9)
            elif coup_seed is not None:
                self.coup_seed = coup_seed

        if noise_seed is None and with_noise:
            self.noise_seed = np.random.randint(low=0, high=1e9)
        else:
            self.noise_seed = noise_seed

        if init_phase is not None:
            self.init_phase = init_phase
        else:
            rng = np.random.default_rng(self.noise_seed)
            self.init_phase = 2 * np.pi * rng.random(size=self.n_nodes)
            # print("The initial phases are ", self.init_phase)

        if pre_sd is None:
            if random_sd:
                self.sd = noise_sth * np.random.random(size=self.n_nodes)
            else:
                self.sd = np.full(shape=self.n_nodes, fill_value=noise_sth)
            # print("Preset standard deviations are", self.sd)
        else:
            self.sd = pre_sd

        conn_rng = np.random.default_rng(self.conn_seed)

        if conn2 == 0:
            conn2 = conn
            # print("conn2 changed.")
        if not all_connected:
            if pre_conn2 is None:
                conn2_seed = conn_rng.integers(0, 1e10)
                conn2_rng = np.random.default_rng(conn2_seed)
                if regular2:
                    conn_mat = np.zeros(shape=(self.n_nodes, self.n_nodes))
                    conn_per = round((self.n_nodes - 1) * conn2)
                    for i in range(self.n_nodes):
                        generator = conn_per
                        for j in range(self.n_nodes):
                            if generator > 0:
                                if i < j:
                                    indicator = conn2_rng.random()
                                    if indicator < conn2:
                                        conn_mat[j][i] = 1
                                        generator -= 1
                                    else:
                                        if generator == self.n_nodes - j:
                                            conn_mat[j][i] = 1
                                            generator -= 1
                                        else:
                                            conn_mat[j][i] = 0
                                elif i > j:
                                    indicator = conn2_rng.random()
                                    if indicator < conn2:
                                        conn_mat[j][i] = 1
                                        generator -= 1
                                    else:
                                        if generator == self.n_nodes - 1 - j:
                                            conn_mat[j][i] = 1
                                            generator -= 1
                                        else:
                                            conn_mat[j][i] = 0
                            else:
                                break
                else:
                    conn_mat = np.zeros(shape=(self.n_nodes, self.n_nodes))
                    for i in range(self.n_nodes):
                        for j in range(self.n_nodes):
                            if i != j:
                                indicator = conn2_rng.random()
                                if indicator < conn2:
                                    conn_mat[j][i] = 1
                                else:
                                    conn_mat[j][i] = 0
                self.conn_mat2 = conn_mat
            else:
                self.conn_mat2 = pre_conn2
        else:
            good_2_conn = False
            while not good_2_conn:
                if pre_conn2 is None:
                    conn2_seed = conn_rng.integers(0, 1e10)
                    conn2_rng = np.random.default_rng(conn2_seed)
                    if regular2:
                        conn_mat = np.zeros(shape=(self.n_nodes, self.n_nodes))
                        conn_per = round((self.n_nodes - 1) * conn2)
                        for i in range(self.n_nodes):
                            generator = conn_per
                            for j in range(self.n_nodes):
                                if generator > 0:
                                    if i < j:
                                        indicator = conn2_rng.random()
                                        if indicator < conn2:
                                            conn_mat[j][i] = 1
                                            generator -= 1
                                        else:
                                            if generator == self.n_nodes - j:
                                                conn_mat[j][i] = 1
                                                generator -= 1
                                            else:
                                                conn_mat[j][i] = 0
                                    elif i > j:
                                        indicator = conn2_rng.random()
                                        if indicator < conn2:
                                            conn_mat[j][i] = 1
                                            generator -= 1
                                        else:
                                            if generator == self.n_nodes - 1 - j:
                                                conn_mat[j][i] = 1
                                                generator -= 1
                                            else:
                                                conn_mat[j][i] = 0
                                else:
                                    break
                    else:
                        conn_mat = np.zeros(shape=(self.n_nodes, self.n_nodes))
                        for i in range(self.n_nodes):
                            for j in range(self.n_nodes):
                                if i != j:
                                    indicator = conn2_rng.random()
                                    if indicator < conn2:
                                        conn_mat[j][i] = 1
                                    else:
                                        conn_mat[j][i] = 0
                    self.conn_mat2 = conn_mat
                else:
                    self.conn_mat2 = pre_conn2

                leng = len(self.conn_mat2.T)
                if Hamiltonian_path(self.conn_mat2.T, leng):
                    good_2_conn = True
            self.conn_seed = conn2_seed
            conn_rng = np.random.default_rng(self.conn_seed)

        if conn3 == 0:
            conn3 = conn
            # print("conn3 changed.")
        if pre_conn3 is None:
            conn3_seed = conn_rng.integers(0, 1e10)
            conn3_rng = np.random.default_rng(conn3_seed)
            if regular3:
                conn_mat = np.zeros(shape=(self.n_nodes, self.n_nodes, self.n_nodes))
                conn_per = int((self.n_nodes - 1) * (self.n_nodes - 2) * 0.5 * conn3)
                for i in range(self.n_nodes):
                    generator = conn_per
                    for k in range(self.n_nodes):  # first iterate column, to ensure upper triangular matrix nonzero
                        for j in range(k):
                            if generator > 0:
                                if i != k:
                                    if i < j < k:
                                        indicator = conn3_rng.random()
                                        if indicator < conn3:
                                            conn_mat[i][j][k] = 1
                                            generator -= 1
                                        else:
                                            if generator == self.n_nodes - 1 - j:
                                                conn_mat[i][j][k] = 1
                                                generator -= 1
                                            else:
                                                conn_mat[i][j][k] = 0
                                    elif j < k and j < i:
                                        indicator = conn3_rng.random()
                                        if indicator < conn3:
                                            conn_mat[i][j][k] = 1
                                            generator -= 1
                                        else:
                                            if generator == self.n_nodes - 2 - j:
                                                conn_mat[i][j][k] = 1
                                                generator -= 1
                                            else:
                                                conn_mat[i][j][k] = 0
                            else:
                                break
            else:
                conn_mat = np.zeros(shape=(self.n_nodes, self.n_nodes, self.n_nodes))
                for i in range(self.n_nodes):
                    for k in range(self.n_nodes):  # first iterate column, to ensure upper triangular matrix nonzero
                        for j in range(k):
                            if i != j and i != k and j != k:
                                indicator = conn3_rng.random()
                                if indicator < conn3:
                                    conn_mat[i][j][k] = 1
                                else:
                                    conn_mat[i][j][k] = 0
            self.conn_mat3 = conn_mat
        else:
            self.conn_mat3 = pre_conn3

        # for non-zero connection generated
        if np.sum(self.conn_mat2) == 0:
            i_lst = random.sample(range(0, self.n_nodes), 2)
            self.conn_mat2[i_lst[0]][i_lst[1]] = 1
        if np.sum(self.conn_mat3) == 0:
            i_lst = random.sample(range(0, self.n_nodes), 3)
            if i_lst[1] < i_lst[2]:
                self.conn_mat3[i_lst[0]][i_lst[1]][i_lst[2]] = 1
            else:
                self.conn_mat3[i_lst[0]][i_lst[2]][i_lst[1]] = 1

        if normalize:
            # effective_conn2 = np.sum(self.conn_mat2) / self.n_nodes  # N^2/N = N
            # effective_conn3 = np.sum(self.conn_mat3) / self.n_nodes  # N^3/N = N^2
            effective_conn2 = conn2 * self.n_nodes  # N^2/N = N
            effective_conn3 = conn3 * self.n_nodes * self.n_nodes / 2  # N^3/N = N^2
        else:
            effective_conn2 = 1
            effective_conn3 = 1

        coup_rng = np.random.default_rng(self.coup_seed)

        if pre_coup2 is None:
            coup2_seed = coup_rng.integers(0, 1e10)
            coup2_rng = np.random.default_rng(coup2_seed)
            if random_coup2 is True:
                coup_mat = np.zeros(shape=(self.n_nodes, self.n_nodes))
                coupling = coupling2 / effective_conn2  # normalization
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i != j:
                            coup_mat[j][i] = coupling * coup2_rng.lognormal(mean=0.01, sigma=0.5)
            else:
                coup_mat = np.zeros(shape=(self.n_nodes, self.n_nodes))
                coupling = coupling2 / effective_conn2  # normalization
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i != j:
                            coup_mat[j][i] = coupling
            self.coupling2 = self.conn_mat2 * coup_mat
        elif pre_coup2 is not None and pre_conn2 is not None:
            self.coupling2 = pre_coup2 * pre_conn2
        else:
            raise ValueError("Preset coupling2 matrix needs to be paired with preset connectivity matrix. ")

        if pre_coup3 is None:
            coup3_seed = coup_rng.integers(0, 1e10)
            coup3_rng = np.random.default_rng(coup3_seed)
            if random_coup3 is True:
                coup_mat = np.zeros(shape=(self.n_nodes, self.n_nodes, self.n_nodes))
                coupling = coupling3 / effective_conn3  # normalization
                for i in range(self.n_nodes):
                    for k in range(self.n_nodes):
                        for j in range(k):
                            if i != j and i != k and j != k:
                                coup_mat[i][j][k] = coupling * coup3_rng.lognormal(mean=0.01, sigma=0.5)
            else:
                coup_mat = np.zeros(shape=(self.n_nodes, self.n_nodes, self.n_nodes))
                coupling = coupling3 / effective_conn3  # normalization
                for i in range(self.n_nodes):
                    for k in range(self.n_nodes):
                        for j in range(k):
                            if i != j and i != k and j != k:
                                coup_mat[i][j][k] = coupling
            self.coupling3 = self.conn_mat3 * coup_mat
        elif pre_coup3 is not None and pre_conn3 is not None:
            self.coupling3 = pre_coup3 * pre_conn3
        else:
            raise ValueError("Preset coupling3 matrix needs to be paired with preset connectivity matrix. ")

        if self.type2 == 3 and self.type3 == 3:
            if r_ve is not None:
                self.r_ve = r_ve
                self.r_ve_seed = None
            else:
                self.r_ve_seed = seed_r_ve
                self.r_ve = np.random.default_rng(seed_r_ve).random(size=self.n_nodes)
            if r_ve is not None:
                self.k_ve = k_ve
                self.k_ve_seed = None
            else:
                self.k_ve_seed = seed_k_ve
                self.k_ve = 10 + 90 * np.random.default_rng(seed_k_ve).random(size=self.n_nodes)

    def get_connectivity(self):
        test2 = np.copy(self.coupling2)
        test3 = np.copy(self.coupling3)

        test2[test2 != 0] = 1
        test3[test3 != 0] = 1

        p2 = np.sum(test2) / (self.n_nodes * (self.n_nodes - 1))
        p3 = np.sum(test3) / (self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2)
        return p2, p3

    def save_seed(self):
        save_path = gi_make_dirs()
        save_file = save_path + "seed.csv"

        with open(save_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([self.conn_seed, self.coup_seed, self.noise_seed])
            f.close()

    def load_seed(self, csv_file):
        seeds = np.genfromtxt(csv_file, delimiter=",", dtype=None)
        assert len(seeds) == 3, "Only 3 seeds needed. "

        if (self.conn_mat2 is None or self.conn_mat3 is None) and self.conn_seed is None:
            self.conn_seed = seeds[0]

        if (self.coupling2 is None or self.coupling3 is None) and self.coup_seed is None:
            self.coup_seed = seeds[1]

        if self.noise_seed is None:
            self.conn_seed = seeds[2]

    def derivative(self, phases_vec, t, coupling2, coupling3):
        """
        Compute derivative of all nodes for current state, defined as

        dx_i    natfreq_i + k  sum_j ( Aij* sin (phase_j - phase_i) )
        ---- =             ---
         dt                M_i

        t: for compatibility with scipy.odeint
        """
        assert len(phases_vec) == len(self.natfreqs), \
            'Input dimensions do not match, check lengths'

        if self.type2 == 3 or self.type3 == 3:
            x_i, x_j = np.meshgrid(phases_vec, phases_vec)
            interaction2 = coupling2 * x_i * x_j
            _ = interaction2.sum(axis=0)
            x_jj, x_ii, x_kk = np.meshgrid(phases_vec, phases_vec, phases_vec)
            # some = coupling3 * self.lv_dynamic_sum_3(phases_vec)
            interaction3 = coupling3 * x_ii * x_jj * x_kk
            __ = interaction3.sum(axis=1).sum(axis=1)
            ___ = self.lv_logistic_fn(phases_vec)
            dxdt = self.lv_logistic_fn(phases_vec) + interaction2.sum(axis=0) + \
                   interaction3.sum(axis=1).sum(axis=1)

        else:
            phases_jj, phases_ii, phases_kk = np.meshgrid(phases_vec, phases_vec, phases_vec)
            if self.type3 == 2:
                something = np.sin(phases_jj + phases_kk - 2 * phases_ii)
            elif self.type3 == 1:
                something = np.sin(2 * phases_jj - phases_kk - phases_ii) + \
                            np.sin(2 * phases_kk - phases_jj - phases_ii)
            else:
                raise ValueError("'self.type3' needs to be 1 or 2.")
            interactions3 = coupling3 * something  # Aijk * sin(j+k-2i)

            phases_i, phases_j = np.meshgrid(phases_vec, phases_vec)
            if self.type2 == 1:
                interactions2 = coupling2 * np.sin(phases_j - phases_i)  # Aji * sin(j-i)
            elif self.type2 == 2:
                interactions2 = coupling2 * np.sin(2 * (phases_j - phases_i))  # Aji * sin(j-i)
            else:
                raise ValueError("'self.type2' needs to be 1 or 2.")
            dxdt = self.natfreqs + interactions2.sum(axis=0) + interactions3.sum(axis=1).sum(
                axis=1)  # sum over incoming
            # interactions

        return dxdt

    def integrate(self, phases_vec):
        """Updates all states by integrating state of all nodes"""
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        coupling2 = self.coupling2  # normalize coupling by number of interactions
        coupling3 = self.coupling3  # normalize coupling by number of interactions

        t = np.linspace(0, self.T, int(self.T / self.dt))
        # timeseries = odeint(self.derivative, phases_vec, t, args=(coupling), rtol=0.000000001,
        #                     atol=0.000000001)
        timeseries = odeint(self.derivative, phases_vec, t, args=(coupling2, coupling3))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def sde_integrate(self, phases_vec, sim_stop=False):
        """Updates all states by integrating state of all nodes, but sde version"""
        coupling2 = self.coupling2  # normalize coupling by number of interactions
        coupling3 = self.coupling3  # normalize coupling by number of interactions

        dt = self.dt

        t = np.linspace(0, self.T, int(self.T / dt))
        ys = np.zeros((int(self.T / dt), self.n_nodes))
        ys[0] = phases_vec

        # Loop
        rng = np.random.default_rng(self.noise_seed)
        for i in range(1, t.size):
            y = ys[i - 1]
            dydt = self.derivative(y, t, coupling2, coupling3)
            dw_ = rng.normal(loc=0.0, scale=np.sqrt(dt), size=self.n_nodes)
            _ = self.sd * dw_
            __ = dydt * dt
            ys[i] = y + dydt * dt + self.sd * dw_

        return ys.T  # transpose for consistency (act_mat:node vs time)

    def show_real_coupling(self):
        demo_coup2 = self.coupling2
        demo_coup3 = self.coupling3
        for i in range(demo_coup2.shape[1]):
            demo_dict2 = {}
            for j in range(demo_coup2.shape[0]):
                if demo_coup2[j][i] != 0:
                    demo_dict2[f"k_{j + 1}{i + 1}"] = demo_coup2[j][i]
            print(f"The 2-coupling coefficients of node {str(i + 1)} is: ", demo_dict2)

        for i in range(demo_coup3.shape[0]):
            demo_dict3 = {}
            for j in range(demo_coup3.shape[1]):
                for k in range(demo_coup3.shape[2]):
                    if demo_coup3[i][j][k] != 0:
                        demo_dict3[f"k_{j + 1}{k + 1}{i + 1}"] = demo_coup3[i][j][k]
            print(f"The 3-coupling coefficients of node {str(i + 1)} is: ", demo_dict3)

    def run(self, stretch=True, sim_stop=False):
        """
        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        """
        phases_vec = self.init_phase

        if self.with_noise is False:
            result = self.integrate(phases_vec)
        else:
            result = self.sde_integrate(phases_vec)

        if result is None:
            raise Exception("Something goes wrong for the integration")

        if stretch:
            index_start = int(self.starts_from * result.shape[1])
            if self.inf_last != 1:
                index_end = index_start + int(self.inf_last * result.shape[1])
            else:
                index_end = int(result.shape[1])
            result = result[:, index_start: index_end]

        return result

    def mse_and_r2(self, result):
        # transfer matrix to connections
        demo_total2 = np.zeros(shape=(self.n_nodes, self.n_nodes - 1))
        demo_total3 = np.zeros(shape=(self.n_nodes, int((self.n_nodes - 1) * (self.n_nodes - 2) / 2)))
        for i in range(self.n_nodes):
            others = np.delete(self.all_nodes, i)
            more_others_lst = self.make_more_others_lst(others, i)

            # demo coupling
            demo_lst2 = np.array([])
            demo_lst3 = np.array([])
            for j in others:
                demo_lst2 = np.append(demo_lst2, self.coupling2[j][i])

                if j == others[-1]:
                    break
                elif j < i:
                    for k in more_others_lst[j]:
                        demo_lst3 = np.append(demo_lst3, self.coupling3[i][j][k])
                else:
                    for k in more_others_lst[j - 1]:
                        demo_lst3 = np.append(demo_lst3, self.coupling3[i][j][k])

            demo_total2[i, :] = demo_lst2
            demo_total3[i, :] = demo_lst3

        sum__2 = 0
        sum_2 = 0
        avg2 = np.average(demo_total2)

        sum__3 = 0
        sum_3 = 0
        avg3 = np.average(demo_total3)

        for i in range(self.n_nodes):
            se_2 = (demo_total2[i] - avg2) ** 2
            se__2 = (demo_total2[i] - result[i]["2"]) ** 2
            sum_2 += se_2.sum(axis=0)
            sum__2 += se__2.sum(axis=0)
        mse_2 = 1 / (self.n_nodes * (self.n_nodes - 1)) * sum__2
        var_2 = 1 / (self.n_nodes * (self.n_nodes - 1)) * sum_2
        R22 = 1 - mse_2 / var_2

        # print("The MSE of connectivities-2 is: ", mse_2)
        # print("The coefficient of determination-2 is: ", R22)

        for i in range(self.n_nodes):
            se_3 = (demo_total3[i] - avg3) ** 2
            se__3 = (demo_total3[i] - result[i]["3"]) ** 2
            sum_3 += se_3.sum(axis=0)
            sum__3 += se__3.sum(axis=0)
        mse_3 = 1 / (self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2) * sum__3
        var_3 = 1 / (self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2) * sum_3
        R23 = 1 - mse_3 / var_3
        # print("The MSE of connectivity-3 is: ", mse_3)
        # print("The coefficient of determination-3 is: ", R23)

        return mse_2, R22, mse_3, R23

    def mse_and_r2_combined(self, result):
        # transfer matrix to connections
        demo_total2 = np.zeros(shape=(self.n_nodes, self.n_nodes - 1))
        demo_total3 = np.zeros(shape=(self.n_nodes, int((self.n_nodes - 1) * (self.n_nodes - 2) / 2)))
        for i in range(self.n_nodes):
            others = np.delete(self.all_nodes, i)
            more_others_lst = self.make_more_others_lst(others, i)

            # demo coupling
            demo_lst2 = np.array([])
            demo_lst3 = np.array([])
            for j in others:
                demo_lst2 = np.append(demo_lst2, self.coupling2[j][i])

                if j == others[-1]:
                    break
                elif j < i:
                    for k in more_others_lst[j]:
                        demo_lst3 = np.append(demo_lst3, self.coupling3[i][j][k])
                else:
                    for k in more_others_lst[j - 1]:
                        demo_lst3 = np.append(demo_lst3, self.coupling3[i][j][k])

            demo_total2[i, :] = demo_lst2
            demo_total3[i, :] = demo_lst3

        demo_total = np.concatenate((demo_total2, demo_total3), axis=1)
        sum__ = 0
        sum_ = 0
        avg = np.average(demo_total)

        for i in range(self.n_nodes):
            infer_arr = np.concatenate((result[i]["2"], result[i]["3"]))
            se_ = (demo_total[i] - avg) ** 2
            se__ = (demo_total[i] - infer_arr) ** 2
            sum_ += se_.sum(axis=0)
            sum__ += se__.sum(axis=0)

        mse_ = 1 / \
               (self.n_nodes * (self.n_nodes - 1) + self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2) * sum__
        var_ = 1 / \
               (self.n_nodes * (self.n_nodes - 1) + self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2) * sum_
        R2 = 1 - mse_ / var_
        # print("The MSE of connectivity is: ", mse_)
        # print("The coefficient of determination is: ", R2)

        return mse_, R2

    def lv_logistic_fn(self, x_vec):
        return self.r_ve * x_vec * (1 - 1 / self.k_ve * x_vec)

    def lv_logistic_fn_one_node(self, x_arr, i):
        return self.r_ve[i] * x_arr * (1 - 1 / self.k_ve[i] * x_arr)

    def lv_dynamic_sum_3(self, x_vec):
        results_3 = np.zeros(shape=(self.n_nodes, self.n_nodes, self.n_nodes))
        for i in range(len(x_vec)):
            for j in range(len(x_vec)):
                for k in range(j + 1, len(x_vec)):
                    results_3[i][j][k] = x_vec[i] * x_vec[j] * x_vec[k]

        return results_3

    @staticmethod
    def phase_coherence(phases_vec):
        """
        Compute global order parameter R_t - mean length of resultant vector
        """
        suma = sum([(np.e ** (1j * i)) for i in phases_vec])
        return abs(suma / len(phases_vec))

    @staticmethod
    def phase_coherence_2(phases_vec):
        """
        Compute global order parameter R_t - mean length of resultant vector
        """
        suma = sum([(np.e ** (1j * i * 2)) for i in phases_vec])
        return abs(suma / len(phases_vec))

    def mean_frequency(self, act_mat):
        """
        Compute average frequency within the time window (self.T) for all nodes
        """
        # assert len(self.coupling) == act_mat.shape[0], 'coupling does not match act_mat'
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time_ in range(n_steps):
            dxdt[:, time_] = self.derivative(act_mat[:, time_], None, self.coupling2, self.coupling3, 0)

        # Integrate all nodes over the time window T
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.T
        return meanfreq

    @staticmethod
    def make_more_others_lst(others, j):
        more_others_lst = []
        for i in others:
            if j < i:
                more_others = others[i:]
            elif j > i:
                more_others = others[i + 1:]
            else:
                raise Exception("j shouldn't be i, something strange happened. ")
            if more_others.size == 0:
                break
            more_others_lst.append(more_others)

        return more_others_lst

    def prepare_diffs(self, results: np.array):
        """
        Testing.
        """
        # assert self.with_noise is True, "Need to infer with noise"

        outputs_all = []

        for j in range(results.shape[0]):
            N = results[j].size

            # create lst for every other node, and initialize the phase difference lst
            others = np.delete(self.all_nodes, j)
            phi_diff2 = np.zeros(shape=(self.n_nodes - 1, N - 1))

            # create lst for every other node, and initialize the phase difference lst
            # create j lst
            need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2)
            phi_diff3 = np.zeros(shape=(need_dim, N - 1))
            phi_diff3_sub = np.zeros(shape=(need_dim, N - 1))

            # lst of self phase
            self_phase = results[j]

            # lst of phase differences with respect to other nodes
            for i in others:
                if self.type2 == 1:
                    phi_diff_array = results[i] - self_phase
                elif self.type2 == 2:
                    phi_diff_array = 2 * (results[i] - self_phase)
                elif self.type2 == 0:
                    phi_diff_array = np.zeros(shape=results[i].size)
                else:
                    raise ValueError("'type' can only be 0, 1 or 2. ")
                if j > i:
                    phi_diff2[i] = phi_diff_array[:-1]
                elif j < i:
                    phi_diff2[i - 1] = phi_diff_array[:-1]

            # maybe better to create the dict from here
            # create k lst, and then calculate the phase difference lst
            counter = 0
            more_others_lst = []
            for i in others:
                if j < i:
                    more_others = others[i:]
                elif j > i:
                    more_others = others[i + 1:]
                else:
                    raise Exception("j shouldn't be i, something strange happened. ")
                if more_others.size == 0:
                    break
                more_others_lst.append(more_others)
                for kkk_ in more_others:
                    if self.type3 == 2:
                        phi_diff_array = results[i] + results[kkk_] - 2 * self_phase
                    elif self.type3 == 1:
                        phi_diff_array = 2 * results[i] - results[kkk_] - self_phase
                        phi_diff_array_sub = 2 * results[kkk_] - results[i] - self_phase
                    elif self.type3 == 0:
                        phi_diff_array = np.zeros(shape=results[i].size)
                    else:
                        raise ValueError("'type' can only be 0, 1 or 2. ")
                    phi_diff3[counter] = phi_diff_array[:-1]
                    phi_diff3_sub[counter] = phi_diff_array_sub[:-1]
                    counter += 1

            # lst of self phase differences
            self_diff = np.array([])
            for i_ in range(N - 1):
                self_diff = np.append(self_diff, self_phase[i_ + 1] - self_phase[i_])
            self_diff = self_diff / self.dt

            # sin the phase differences - 2
            two_lib = np.zeros(shape=(self.n_nodes - 1, N - 1))
            for i__ in range(phi_diff2.shape[0]):
                sin_diff2 = np.sin(phi_diff2[i__])
                two_lib[i__] = sin_diff2

            # sin the phase differences - 3
            three_lib = np.zeros(shape=(need_dim, N - 1))
            for i__ in range(need_dim):
                sin_diff3 = np.sin(phi_diff3[i__]) + np.sin(phi_diff3_sub[i__])
                three_lib[i__] = sin_diff3

            outputs = {"N": N, "self_phase": self_phase, "self_diff": self_diff, "two_lib": two_lib,
                       "three_lib": three_lib}

            outputs_all.append(outputs)

        return outputs_all

    def prepare_diffs_lv(self, results: np.array):
        """
        Testing.
        """
        # assert self.with_noise is True, "Need to infer with noise"
        assert self.type2 == 3 and self.type3 == 3, "This is Lotka-Volterra"

        outputs_all = []
        need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2)

        for i in range(results.shape[0]):
            N = results[i].size

            # create lst for every other node, and initialize the phase difference lst
            others = np.delete(self.all_nodes, i)
            element_2 = np.zeros(shape=(self.n_nodes - 1, N - 1))

            # create lst for every other node, and initialize the phase difference lst
            # create j lst

            element_3 = np.zeros(shape=(need_dim, N - 1))

            # lst of self phase
            self_phase = results[i]

            # lst of 2-elements
            for j in others:
                _ = results[j]
                element_arr = self_phase * results[j]
                if i > j:
                    element_2[j] = element_arr[:-1]
                elif i < j:
                    element_2[j - 1] = element_arr[:-1]

            # maybe better to create the dict from here
            # create k lst, and then calculate the 3-elements
            counter = 0
            more_others_lst = []
            for j in others:
                if i < j:
                    more_others = others[j:]
                elif i > j:
                    more_others = others[j + 1:]
                else:
                    raise Exception("j shouldn't be i, something strange happened. ")
                if more_others.size == 0:
                    break
                more_others_lst.append(more_others)
                for k in more_others:
                    element_arr = self_phase * results[j] * results[k]
                    element_3[counter] = element_arr[:-1]
                    counter += 1

            # lst of self phase differences
            self_diff = np.array([])
            for i_ in range(N - 1):
                self_diff = np.append(self_diff, self_phase[i_ + 1] - self_phase[i_])
            # logistic fns
            logi_arr = self.lv_logistic_fn_one_node(self_phase, i)
            logi_arr = logi_arr[:-1]
            self_diff = self_diff / self.dt
            self_diff = self_diff - logi_arr

            outputs = {"N": N, "self_phase": self_phase, "self_diff": self_diff, "two_lib": element_2,
                       "three_lib": element_3}

            outputs_all.append(outputs)

        return outputs_all

    def prepare_diffs_one_node(self, j, results: np.array):
        """
        Testing.
        """
        # assert self.with_noise is True, "Need to infer with noise"

        N = results[j].size

        # create lst for every other node, and initialize the phase difference lst
        others = np.delete(self.all_nodes, j)
        phi_diff2 = np.zeros(shape=(self.n_nodes - 1, N - 1))

        # create lst for every other node, and initialize the phase difference lst
        # create j lst
        need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2)
        phi_diff3 = np.zeros(shape=(need_dim, N - 1))
        phi_diff3_sub = np.zeros(shape=(need_dim, N - 1))

        # lst of self phase
        self_phase = results[j]

        # lst of phase differences with respect to other nodes
        for i in others:
            if self.type2 == 1:
                phi_diff_array = results[i] - self_phase
            elif self.type2 == 2:
                phi_diff_array = 2 * (results[i] - self_phase)
            elif self.type2 == 0:
                phi_diff_array = np.zeros(shape=results[i].size)
            else:
                raise ValueError("'type' can only be 0, 1 or 2. ")
            if j > i:
                phi_diff2[i] = phi_diff_array[:-1]
            elif j < i:
                phi_diff2[i - 1] = phi_diff_array[:-1]

        # may be better to create the dict from here
        # create k lst, and then calculate the phase difference lst
        counter = 0
        more_others_lst = []
        for i in others:
            if j < i:
                more_others = others[i:]
            elif j > i:
                more_others = others[i + 1:]
            else:
                raise Exception("j shouldn't be i, something strange happened. ")
            if more_others.size == 0:
                break
            more_others_lst.append(more_others)
            for kkk_ in more_others:
                if self.type3 == 2:
                    phi_diff_array = results[i] + results[kkk_] - 2 * self_phase
                elif self.type3 == 1:
                    phi_diff_array = 2 * results[i] - results[kkk_] - self_phase
                    phi_diff_array_sub = 2 * results[kkk_] - results[i] - self_phase
                elif self.type3 == 0:
                    phi_diff_array = np.zeros(shape=results[i].size)
                else:
                    raise ValueError("'type' can only be 0, 1 or 2. ")
                phi_diff3[counter] = phi_diff_array[:-1]
                phi_diff3_sub[counter] = phi_diff_array_sub[:-1]
                counter += 1

        # lst of self phase differences
        self_diff = np.array([])
        for i_ in range(N - 1):
            self_diff = np.append(self_diff, self_phase[i_ + 1] - self_phase[i_])

        # sin the phase differences - 2
        two_lib = np.zeros(shape=(self.n_nodes - 1, N - 1))
        for i__ in range(phi_diff2.shape[0]):
            sin_diff2 = np.sin(phi_diff2[i__])
            two_lib[i__] = sin_diff2

        # sin the phase differences - 3
        three_lib = np.zeros(shape=(need_dim, N - 1))
        for i__ in range(need_dim):
            sin_diff3 = np.sin(phi_diff3[i__]) + np.sin(phi_diff3_sub[i__])
            three_lib[i__] = sin_diff3

        return {"N": N, "self_phase": self_phase, "self_diff": self_diff, "two_lib": two_lib,
                "three_lib": three_lib}

    # # legacy code
    # def solve_mle(self, all_prepared):
    #     """
    #     Testing.
    #     :return: numpy.array of maximum likelihood estimators of nat_freq, coupling strength, and noise deviation
    #     """
    # #   assert self.with_noise is True, "Need to infer with noise"
    #     assert len(all_prepared) == self.n_nodes, "all_prepared BOOMS"
    #
    #     need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2)
    #     dt = self.dt
    #     dt_inverse = 1 / dt
    #
    #     output = []
    #
    #     # for information criterions (legacy)
    #     # log_l = 0
    #
    #     for i in range(self.n_nodes):
    #         N = all_prepared[i]["N"]
    #         self_phase = all_prepared[i]["self_phase"]
    #         self_diff = all_prepared[i]["self_diff"]
    #         two_lib = all_prepared[i]["two_lib"]
    #         three_lib = all_prepared[i]["three_lib"]
    #
    #         self_diff = self_diff * dt
    #
    #         # solve the log-likelihood equations
    #         A = np.zeros(shape=(self.n_nodes + need_dim, self.n_nodes + need_dim))
    #         B = np.zeros(shape=self.n_nodes + need_dim)
    #         A[0, 0] = N
    #         B[0] = (self_phase[N - 1] - self_phase[0]) * dt_inverse
    #         for iii_ in range(1, self.n_nodes + need_dim):
    #             if iii_ < self.n_nodes:
    #                 A[iii_, 0] = np.sum(two_lib[iii_ - 1])
    #                 A[0, iii_] = np.sum(two_lib[iii_ - 1])
    #                 B[iii_] = np.sum(self_diff * two_lib[iii_ - 1]) * dt_inverse
    #                 for jjj_ in range(1, self.n_nodes):
    #                     A[iii_, jjj_] = np.sum(two_lib[iii_ - 1] * two_lib[jjj_ - 1])
    #                 for jjj_ in range(self.n_nodes, self.n_nodes + need_dim):
    #                     A[iii_, jjj_] = np.sum(two_lib[iii_ - 1] * three_lib[jjj_ - self.n_nodes])
    #             else:
    #                 A[iii_, 0] = np.sum(three_lib[iii_ - self.n_nodes])
    #                 A[0, iii_] = np.sum(three_lib[iii_ - self.n_nodes])
    #                 B[iii_] = np.sum(self_diff * three_lib[iii_ - self.n_nodes]) * dt_inverse
    #                 for jjj_ in range(1, self.n_nodes):
    #                     A[iii_, jjj_] = np.sum(three_lib[iii_ - self.n_nodes] * two_lib[jjj_ - 1])
    #                 for jjj_ in range(self.n_nodes, self.n_nodes + need_dim):
    #                     A[iii_, jjj_] = np.sum(three_lib[iii_ - self.n_nodes] * three_lib[jjj_ - self.n_nodes])
    #         X = np.linalg.solve(A, B)
    #         omega = X[0] % (2 * np.pi)
    #         k2 = X[1:self.n_nodes]
    #         k3 = X[self.n_nodes:]
    #
    #         # sort terms needed for the standard deviation calc, and calc
    #         total_sm = np.zeros(N - 1)
    #         for kk_ in range(k2.size):
    #             something = k2[kk_] * two_lib[kk_]
    #             total_sm += something
    #         for kk_ in range(k3.size):
    #             something = k3[kk_] * three_lib[kk_]
    #             total_sm += something
    #         se = (self_diff - (omega + total_sm) * dt) ** 2
    #         variance = (1 / N * dt_inverse) * np.sum(se)
    #         sd = np.sqrt(variance)
    #
    #         # for information criterions (legacy)
    #         # log_l += - np.sum(se) / (2 * variance * dt) - N/2 * np.log(2 * np.pi * variance * dt)
    #
    #         output.append({"2": k2, "3": k3, "natfreq": omega, "standard deviation": sd})
    #
    #     # # derive information criterions (legacy)
    #     # k_ = (self.n_nodes + 1) * self.n_nodes
    #     # aic = 2 * k_ - 2 * log_l
    #     # bic = k_ * np.log((N - 1) * self.n_nodes) - 2 * log_l
    #     # output.append({"aic": aic, "bic": bic})
    #
    #     return output

    # # legacy code
    # def solve_mle_one_node(self, all_prepared):
    #     """
    #     Testing.
    #     :return: numpy.array of maximum likelihood estimators of nat_freq, coupling strength, and noise deviation
    #     """
    # #   assert self.with_noise is True, "Need to infer with noise"
    #
    #     need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2)
    #     dt = self.dt
    #     dt_inverse = 1 / dt
    #
    #     # for information criterions (legacy)
    #     # log_l = 0
    #
    #     N = all_prepared["N"]
    #     self_phase = all_prepared["self_phase"]
    #     self_diff = all_prepared["self_diff"]
    #     two_lib = all_prepared["two_lib"]
    #     three_lib = all_prepared["three_lib"]
    #
    #     self_diff = self_diff * dt
    #
    #     # solve the log-likelihood equations
    #     A = np.zeros(shape=(self.n_nodes + need_dim, self.n_nodes + need_dim))
    #     B = np.zeros(shape=self.n_nodes + need_dim)
    #     A[0, 0] = N
    #     B[0] = (self_phase[N - 1] - self_phase[0]) * dt_inverse
    #     for iii_ in range(1, self.n_nodes + need_dim):
    #         if iii_ < self.n_nodes:
    #             A[iii_, 0] = np.sum(two_lib[iii_ - 1])
    #             A[0, iii_] = np.sum(two_lib[iii_ - 1])
    #             B[iii_] = np.sum(self_diff * two_lib[iii_ - 1]) * dt_inverse
    #             for jjj_ in range(1, self.n_nodes):
    #                 A[iii_, jjj_] = np.sum(two_lib[iii_ - 1] * two_lib[jjj_ - 1])
    #             for jjj_ in range(self.n_nodes, self.n_nodes + need_dim):
    #                 A[iii_, jjj_] = np.sum(two_lib[iii_ - 1] * three_lib[jjj_ - self.n_nodes])
    #         else:
    #             A[iii_, 0] = np.sum(three_lib[iii_ - self.n_nodes])
    #             A[0, iii_] = np.sum(three_lib[iii_ - self.n_nodes])
    #             B[iii_] = np.sum(self_diff * three_lib[iii_ - self.n_nodes]) * dt_inverse
    #             for jjj_ in range(1, self.n_nodes):
    #                 A[iii_, jjj_] = np.sum(three_lib[iii_ - self.n_nodes] * two_lib[jjj_ - 1])
    #             for jjj_ in range(self.n_nodes, self.n_nodes + need_dim):
    #                 A[iii_, jjj_] = np.sum(three_lib[iii_ - self.n_nodes] * three_lib[jjj_ - self.n_nodes])
    #     X = np.linalg.solve(A, B)
    #     omega = X[0] % (2 * np.pi)
    #     k2 = X[1:self.n_nodes]
    #     k3 = X[self.n_nodes:]
    #
    #     # sort terms needed for the standard deviation calc, and calc
    #     total_sm = np.zeros(N - 1)
    #     for kk_ in range(k2.size):
    #         something = k2[kk_] * two_lib[kk_]
    #         total_sm += something
    #     for kk_ in range(k3.size):
    #         something = k3[kk_] * three_lib[kk_]
    #         total_sm += something
    #     se = (self_diff - (omega + total_sm) * dt) ** 2
    #     variance = (1 / N * dt_inverse) * np.sum(se)
    #     sd = np.sqrt(variance)
    #
    #     return {"2": k2, "3": k3, "natfreq": omega, "standard deviation": sd}

    def make_result_dict(self, k2, k3, i):
        others = np.delete(self.all_nodes, i)
        more_others_lst = self.make_more_others_lst(others, i)

        # make dict to sort the coupling terms
        k2_dict = {}
        for j in range(others.size):
            key = f"k_{str(others[j] + 1)}{str(i + 1)}"
            k2_dict[key] = k2[j]

        # make dict to sort the coupling terms
        k3_dict = {}
        counter = 0
        for j in range(others.size - 1):
            for kk__ in range(len(more_others_lst[i])):
                key = f"k_{str(others[j] + 1)}{str(more_others_lst[j][kk__] + 1)}{str(i + 1)}"
                k3_dict[key] = k3[counter]
                counter += 1

        return k2_dict, k3_dict

    def solve_ols(self, all_prepared):
        total_results = []

        # total_nonzero_inf = np.zeros(6)

        # for histogram
        # twos = np.array([])
        # threes = np.array([])

        for i in range(self.n_nodes):
            self_diff = all_prepared[i]["self_diff"]
            two_lib = all_prepared[i]["two_lib"]
            three_lib = all_prepared[i]["three_lib"]

            sin_diff_lib = np.concatenate((two_lib, three_lib), axis=0)

            transpose = sin_diff_lib.T
            if self.type2 == 3 and self.type3 == 3:
                reg = linear_model.LinearRegression(fit_intercept=False)
            elif self.type2 == 3 and self.type3 != 3:
                raise ValueError("type2 and type3 must be 3 simultaneously. ")
            elif self.type2 != 3 and self.type3 == 3:
                raise ValueError("type2 and type3 must be 3 simultaneously. ")
            else:
                reg = linear_model.LinearRegression()
            reg.fit(transpose, self_diff)
            two_conn_lst = reg.coef_[:self.n_nodes - 1]
            three_conn_lst = reg.coef_[self.n_nodes - 1:]

            results2 = two_conn_lst.tolist()
            results3 = three_conn_lst.tolist()

            if self.type2 == 3 and self.type3 == 3:
                omega = "No omega"
                error = self.dt / (len(self_diff) - len(two_conn_lst) - len(three_conn_lst) - 1) * \
                        np.sum((self_diff - np.dot(two_conn_lst, two_lib) - np.dot(three_conn_lst, three_lib)) ** 2)
            else:
                omega = reg.intercept_ % (2 * np.pi)
                error = self.dt / (len(self_diff) - len(two_conn_lst) - len(three_conn_lst) - 1) * \
                        np.sum((self_diff - omega
                                - np.dot(two_conn_lst, two_lib) - np.dot(three_conn_lst, three_lib)) ** 2)
            results2.insert(0, omega)
            results2.insert(0, "Estimates-2 - OLS")
            results3.insert(0, "Estimates-3 - OLS")

            sd = np.sqrt(error)

            results = {"2": two_conn_lst, "3": three_conn_lst, "natfreq": omega, "standard deviation": sd}
            total_results.append(results)

        return total_results

    def solve_lasso(self, all_prepared):
        total_results = []

        # total_nonzero_inf = np.zeros(6)

        # for histogram
        # twos = np.array([])
        # threes = np.array([])

        for i in range(self.n_nodes):
            self_diff = all_prepared[i]["self_diff"]
            two_lib = all_prepared[i]["two_lib"]
            three_lib = all_prepared[i]["three_lib"]

            sin_diff_lib = np.concatenate((two_lib, three_lib), axis=0)

            transpose = sin_diff_lib.T
            if self.type2 == 3 and self.type3 == 3:
                reg = linear_model.LassoLarsIC(criterion="bic", fit_intercept=False)
            elif self.type2 == 3 and self.type3 != 3:
                raise ValueError("type2 and type3 must be 3 simultaneously. ")
            elif self.type2 != 3 and self.type3 == 3:
                raise ValueError("type2 and type3 must be 3 simultaneously. ")
            else:
                reg = linear_model.LassoLarsIC(criterion="bic")
            reg.fit(transpose, self_diff)
            two_conn_lst = reg.coef_[:self.n_nodes - 1]
            three_conn_lst = reg.coef_[self.n_nodes - 1:]

            results2 = two_conn_lst.tolist()
            results3 = three_conn_lst.tolist()

            if self.type2 == 3 and self.type3 == 3:
                omega = "No omega"
            else:
                omega = reg.intercept_ % (2 * np.pi)
            results2.insert(0, omega)
            results2.insert(0, "Estimates-2 - LASSO")
            results3.insert(0, "Estimates-3 - LASSO")
            alpha = reg.alpha_
            if alpha < 0:
                print("The alpha is smaller than 0 for lasso! ")
            # print("The alpha is", alpha, "by bic. ")

            results = {"2": two_conn_lst, "3": three_conn_lst, "natfreq": omega, "alpha": alpha}
            total_results.append(results)

        return total_results

    def solve_lasso_one_node(self, all_prepared):
        self_diff = all_prepared["self_diff"]
        two_lib = all_prepared["two_lib"]
        three_lib = all_prepared["three_lib"]

        sin_diff_lib = np.concatenate((two_lib, three_lib), axis=0)

        transpose = sin_diff_lib.T
        if self.type2 == 3 and self.type3 == 3:
            reg = linear_model.LassoLarsIC(criterion="bic", fit_intercept=False)
        elif self.type2 == 3 and self.type3 != 3:
            raise ValueError("type2 and type3 must be 3 simultaneously. ")
        elif self.type2 != 3 and self.type3 == 3:
            raise ValueError("type2 and type3 must be 3 simultaneously. ")
        else:
            reg = linear_model.LassoLarsIC(criterion="bic")
        reg.fit(transpose, self_diff)
        two_conn_lst = reg.coef_[:self.n_nodes - 1]
        three_conn_lst = reg.coef_[self.n_nodes - 1:]

        results2 = two_conn_lst.tolist()
        results3 = three_conn_lst.tolist()

        if self.type2 == 3 and self.type3 == 3:
            omega = "No omega"
        else:
            omega = reg.intercept_ % (2 * np.pi)
        results2.insert(0, omega)
        results2.insert(0, "Estimates-2 - LASSO")
        results3.insert(0, "Estimates-3 - LASSO")
        alpha = reg.alpha_
        if alpha < 0:
            print("The alpha is smaller than 0 for lasso! ")
        # print("The alpha is", alpha, "by bic. ")

        return {"2": two_conn_lst, "3": three_conn_lst, "natfreq": omega, "alpha": alpha}

    def solve_ada_lasso(self, all_prepared, mle_or_ols_results):
        total_results = []

        for i in range(self.n_nodes):
            self_diff = all_prepared[i]["self_diff"]
            two_lib = all_prepared[i]["two_lib"]
            three_lib = all_prepared[i]["three_lib"]

            mle_2 = mle_or_ols_results[i]["2"]
            mle_3 = mle_or_ols_results[i]["3"]

            sin_diff_lib = np.concatenate((two_lib, three_lib), axis=0)

            new_X = np.zeros(shape=sin_diff_lib.shape)
            mle_2_array = abs(mle_2)
            mle_3_array = abs(mle_3)
            mle_array = np.append(mle_2_array, mle_3_array)
            assert new_X.shape[0] == len(mle_array), "something wrong"
            for iiii in range(len(mle_array)):
                new_X[iiii] = sin_diff_lib[iiii] * mle_array[iiii]
            new_X = new_X.T

            if self.type2 == 3 and self.type3 == 3:
                ada_reg = linear_model.LassoLarsIC(criterion="bic", fit_intercept=False)
            elif self.type2 == 3 and self.type3 != 3:
                raise ValueError("type2 and type3 must be 3 simultaneously. ")
            elif self.type2 != 3 and self.type3 == 3:
                raise ValueError("type2 and type3 must be 3 simultaneously. ")
            else:
                ada_reg = linear_model.LassoLarsIC(criterion="bic")
            ada_reg.fit(new_X, self_diff)
            two_conn_lst = ada_reg.coef_[:self.n_nodes - 1] * mle_2_array
            three_conn_lst = ada_reg.coef_[self.n_nodes - 1:] * mle_3_array

            if self.type2 == 3 and self.type3 == 3:
                omega = "No omega"
            else:
                omega = ada_reg.intercept_ % (2 * np.pi)
            alpha = ada_reg.alpha_
            if alpha < 0:
                print("The alpha is smaller than 0 for a. lasso! ")
            # print("The alpha is", alpha, "by bic. ")

            results = {"2": two_conn_lst, "3": three_conn_lst, "natfreq": omega, "alpha": alpha}
            total_results.append(results)

        return total_results

    def solve_ada_lasso_one_node(self, all_prepared, mle_or_ols_results_one_node):
        self_diff = all_prepared["self_diff"]
        two_lib = all_prepared["two_lib"]
        three_lib = all_prepared["three_lib"]

        mle_2 = mle_or_ols_results_one_node["2"]
        mle_3 = mle_or_ols_results_one_node["3"]

        sin_diff_lib = np.concatenate((two_lib, three_lib), axis=0)

        new_X = np.zeros(shape=sin_diff_lib.shape)
        mle_2_array = abs(mle_2)
        mle_3_array = abs(mle_3)
        mle_array = np.append(mle_2_array, mle_3_array)
        assert new_X.shape[0] == len(mle_array), "something wrong"
        for iiii in range(len(mle_array)):
            new_X[iiii] = sin_diff_lib[iiii] * mle_array[iiii]
        new_X = new_X.T

        if self.type2 == 3 and self.type3 == 3:
            ada_reg = linear_model.LassoLarsIC(criterion="bic", fit_intercept=False)
        elif self.type2 == 3 and self.type3 != 3:
            raise ValueError("type2 and type3 must be 3 simultaneously. ")
        elif self.type2 != 3 and self.type3 == 3:
            raise ValueError("type2 and type3 must be 3 simultaneously. ")
        else:
            ada_reg = linear_model.LassoLarsIC(criterion="bic")
        ada_reg.fit(new_X, self_diff)
        two_conn_lst = ada_reg.coef_[:self.n_nodes - 1] * mle_2_array
        three_conn_lst = ada_reg.coef_[self.n_nodes - 1:] * mle_3_array

        if self.type2 == 3 and self.type3 == 3:
            omega = "No omega"
        else:
            omega = ada_reg.intercept_ % (2 * np.pi)
        alpha = ada_reg.alpha_
        if alpha < 0:
            print("The alpha is smaller than 0 for a. lasso! ")
        # print("The alpha is", alpha, "by bic. ")

        return {"2": two_conn_lst, "3": three_conn_lst, "natfreq": omega, "alpha": alpha}

    @staticmethod
    def result_to_matrix(results):
        n_nodes = len(results)
        two_mat = np.zeros(shape=(n_nodes, n_nodes))
        three_mat = np.zeros(shape=(n_nodes, n_nodes, n_nodes))

        for i in range(n_nodes):
            two_conn_lst = results[i]["2"]
            three_conn_lst = results[i]["3"]

            j = 0
            jj = 0
            kk = 0
            while j < n_nodes:
                if i != j:
                    two_mat[j][i] = two_conn_lst[jj]
                    jj += 1

                    if i < j:
                        k = j + 1
                    else:
                        k = i + 1
                    while k < n_nodes:
                        three_mat[i][j][k] = three_conn_lst[kk]
                        k += 1
                        kk += 1

                j += 1

        return two_mat, three_mat

    def demo_results(self, mle_or_ols_result=None, lasso_result=None, ada_result=None):
        demo_coup2 = self.coupling2
        demo_coup3 = self.coupling3

        total_table_2 = []
        total_table_3 = []

        for i in range(self.n_nodes):
            others = np.delete(self.all_nodes, i)
            more_others_lst = self.make_more_others_lst(others, i)

            # demo coupling
            demo_lst2 = ["Real-2", self.natfreqs[i]]
            for j in others:
                demo_lst2.append(demo_coup2[j][i])

            demo_lst3 = ["Real-3"]
            for j in others:
                if j == others[-1]:
                    break
                elif j < i:
                    for k in more_others_lst[j]:
                        demo_lst3.append(demo_coup3[i][j][k])
                else:
                    for k in more_others_lst[j - 1]:
                        demo_lst3.append(demo_coup3[i][j][k])

            table2 = [demo_lst2]
            table3 = [demo_lst3]

            if ada_result is not None:
                two_conn_lst = ada_result[i]["2"]
                three_conn_lst = ada_result[i]["3"]
                omega = ada_result[i]["natfreq"]

                ada_results2 = two_conn_lst.tolist()
                ada_results3 = three_conn_lst.tolist()

                ada_results2.insert(0, omega)
                ada_results2.insert(0, "Estimates-2 - Ada. LASSO")
                ada_results3.insert(0, "Estimates-3 - Ada. LASSO")

                table2.append(ada_results2)
                table3.append(ada_results3)

            if lasso_result is not None:
                two_conn_lst = lasso_result[i]["2"]
                three_conn_lst = lasso_result[i]["3"]
                omega = lasso_result[i]["natfreq"]

                lasso_results2 = two_conn_lst.tolist()
                lasso_results3 = three_conn_lst.tolist()

                lasso_results2.insert(0, omega)
                lasso_results2.insert(0, "Estimates-2 - LASSO")
                lasso_results3.insert(0, "Estimates-3 - LASSO")

                table2.append(lasso_results2)
                table3.append(lasso_results3)

            if mle_or_ols_result is not None:
                mle_estimates2 = ["Estimates-2 - OLS", mle_or_ols_result[i]["natfreq"]] + mle_or_ols_result[i][
                    "2"].tolist()
                mle_estimates3 = ["Estimates-3 - OLS"] + mle_or_ols_result[i]["3"].tolist()

                table2.append(mle_estimates2)
                table3.append(mle_estimates3)

            col_names2 = ["Form: k_ji", "natural frequencies_" + str(i + 1)]
            counter_2 = 0
            for index in others:
                name = "k_" + str(index + 1) + str(i + 1)
                col_names2.append(name)
                counter_2 += 1
            print(tabulate(table2, headers=col_names2))
            print("/////////////////////////////////////")

            col_names3 = ["Form: k_jli"]
            for index in others:
                if index == others[-1]:
                    break
                elif index < i:
                    for inde2 in more_others_lst[index]:
                        name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                        col_names3.append(name)
                else:
                    for inde2 in more_others_lst[index - 1]:
                        name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                        col_names3.append(name)
            print(tabulate(table3, headers=col_names3))
            print("====================================================================")

            total_table_2.append(table2)
            total_table_3.append(table3)

        return total_table_2, total_table_3

    def demo_solve(self, mle=False, lasso=False, ada=False, print_result=False):
        act_mat = self.run()

        prepare_all = self.prepare_diffs(act_mat)
        if ada and not mle:
            raise Exception("Need 'mle' to be true if 'ada' is true. ")
        if mle:
            mle_or_ols_results = self.solve_ols(all_prepared=prepare_all)
            if ada:
                ada_results = self.solve_ada_lasso(all_prepared=prepare_all, mle_or_ols_results=mle_or_ols_results)
            else:
                ada_results = None
        else:
            mle_or_ols_results = None
        if lasso:
            lasso_results = self.solve_lasso(all_prepared=prepare_all)
        else:
            lasso_results = None

        if print_result:
            _, _ = self.demo_results(mle_or_ols_result=mle_or_ols_results, lasso_result=lasso_results,
                                     ada_result=ada_results)

        return mle_or_ols_results, lasso_results, ada_results

    def demo_solve_alt(self, mle=False, lasso=False, ada=False, print_result=False):
        act_mat = self.run()

        prepare_all = self.prepare_diffs(act_mat)
        if ada and not mle:
            raise Exception("Need 'mle' to be true if 'ada' is true. ")
        if mle:
            mle_or_ols_results = self.solve_mle(all_prepared=prepare_all)
            if ada:
                ada_results = self.solve_ada_lasso(all_prepared=prepare_all, mle_or_ols_results=mle_or_ols_results)
            else:
                ada_results = None
        else:
            mle_or_ols_results = None
        if lasso:
            lasso_results = self.solve_lasso(all_prepared=prepare_all)
        else:
            lasso_results = None

        if print_result:
            _, _ = self.demo_results(mle_or_ols_result=mle_or_ols_results, lasso_result=lasso_results,
                                     ada_result=ada_results)

        return mle_or_ols_results, lasso_results, ada_results

    # TODO: use pool to multithreading this fn,
    #  https://martinxpn.medium.com/thread-pools-and-process-pools-in-python-75-100-days-of-python-4101f10f64fc
    def demo_solve_one_node(self, mle=False, lasso=False, ada=False):
        # solve for all nodes but calling only one_node mode, suitable for mass computing
        act_mat = self.run()
        if mle:
            mle_or_ols_results = []
        else:
            mle_or_ols_results = None
        if lasso:
            lasso_results = []
        else:
            lasso_results = None
        if ada:
            ada_results = []
        else:
            ada_results = None

        for i in range(self.n_nodes):
            prepare_all = self.prepare_diffs_one_node(i, act_mat)
            if ada and not mle:
                raise Exception("Need 'mle' to be true if 'ada' is true. ")
            if mle:
                mle_or_ols_result = self.solve_mle_one_node(all_prepared=prepare_all)
                mle_or_ols_results.append(mle_or_ols_result)
                if ada:
                    ada_results.append(
                        self.solve_ada_lasso_one_node(all_prepared=prepare_all,
                                                      mle_or_ols_results_one_node=mle_or_ols_result))

            if lasso:
                lasso_results.append(self.solve_lasso_one_node(all_prepared=prepare_all))

        return mle_or_ols_results, lasso_results, ada_results

    def fdr_basic(self, inf_result_real, node_num):
        nats_inf = np.zeros(node_num)
        noiseinf = np.zeros(node_num)
        for _ in range(node_num):
            if self.type2 != 3:
                nats_inf[_] = inf_result_real[_]["natfreq"]
            noiseinf[_] = inf_result_real[_]["standard deviation"]
        # print("Noise is ", noiseinf)

        # ================================================================================
        # Step 1: Simulate the version with inferred natural frequencies and noise strengths but no coupling,
        # see the inferred result again
        model_non = GeneralInteraction(coupling2=0, coupling3=0, dt=self.dt, T=self.T, natfreqs=nats_inf,
                                       with_noise=True, pre_sd=noiseinf, noise_seed=self.noise_seed,
                                       starts_from=self.starts_from, inf_last=self.inf_last)
        ts_0 = model_non.run()
        prepare_0 = model_non.prepare_diffs(results=ts_0)
        inf_result_0 = model_non.solve_ols(all_prepared=prepare_0)

        # Step 2: collect all the coupling strength data and calc the sample mean and variance of it.
        css2_arr = np.array([])
        css3_arr = np.array([])
        for i in range(self.n_nodes):
            css2_arr = np.concatenate((css2_arr, inf_result_0[i]["2"]), axis=0)
            css3_arr = np.concatenate((css3_arr, inf_result_0[i]["3"]), axis=0)
        assert len(css2_arr) == self.n_nodes * (self.n_nodes - 1), "css2_arr BOOOM<"
        assert len(css3_arr) == self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2, "css3_arr BOOOM<"
        sp2_mean = np.mean(css2_arr)
        sp2_var = np.var(css2_arr, ddof=1)
        sp2_sd = np.sqrt(sp2_var)
        sp3_mean = np.mean(css3_arr)
        sp3_var = np.var(css3_arr, ddof=1)
        sp3_sd = np.sqrt(sp3_var)

        return [sp2_mean, sp2_var, sp2_sd, sp3_mean, sp3_var, sp3_sd]

    def fdr_control_for_mle(self, inf_result_real, sig_level=0.05):
        def double_tailed_p_value(sample_value, mean, variance):
            # Calculate the standard deviation
            std_dev = variance ** 0.5

            # Calculate the z-score
            z_score = (sample_value - mean) / std_dev

            # Calculate the p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            return p_value

        # Take freq and noise strength
        node_num = len(inf_result_real)
        resultss = self.fdr_basic(inf_result_real, node_num)
        sp2_mean = resultss[0]
        sp2_var = resultss[1]
        sp3_mean = resultss[3]
        sp3_var = resultss[4]

        # get the p-value arr, for both type of interactions
        p_value2 = np.zeros(shape=(node_num, node_num - 1))
        p_value3 = np.zeros(shape=(node_num, int((node_num - 1) * (node_num - 2) / 2)))
        for i in range(self.n_nodes):
            # 2 adds
            result__ = inf_result_real[self.all_nodes[i]]["2"].copy()
            p_s = double_tailed_p_value(result__, sp2_mean, sp2_var)
            p_value2[i] = p_s

            # 3 adds
            result___ = inf_result_real[self.all_nodes[i]]["3"].copy()
            p_s = double_tailed_p_value(result___, sp3_mean, sp3_var)
            p_value3[i] = p_s

        p2_adj = np.reshape(p_value2, p_value2.size)
        p3_adj = np.reshape(p_value3, p_value3.size)

        rejected2, test2, _, _ = multipletests(p2_adj, alpha=sig_level, method='fdr_bh')
        rejected3, test3, _, _ = multipletests(p3_adj, alpha=sig_level, method='fdr_bh')

        back2 = np.reshape(test2, p_value2.shape)
        back3 = np.reshape(test3, p_value3.shape)

        p_value2 = back2
        p_value3 = back3

        # Step 3: Determine the connectivity matrix
        conn_all = []
        for i in range(len(self.all_nodes)):
            conn2_ = np.zeros(shape=int(node_num - 1))
            conn3_ = np.zeros(shape=int((node_num - 1) * (node_num - 2) / 2))
            for j in range(conn2_.shape[0]):
                if p_value2[i][j] <= sig_level:
                    conn2_[j] = 1

            for j in range(conn3_.shape[0]):
                if p_value3[i][j] <= sig_level:
                    conn3_[j] = 1

            conn_all.append({"2": conn2_, "3": conn3_})
        return conn_all, resultss

    def mle_with_fdr(self, mle_or_ols_results, fdr_results, compact=False):
        if compact:
            coup2_ = np.zeros(shape=(self.n_nodes, self.n_nodes))
            coup3_ = np.zeros(shape=(self.n_nodes, self.n_nodes, self.n_nodes))
        else:
            coup = []

        for i in range(self.n_nodes):
            results2 = mle_or_ols_results[i]["2"] * fdr_results[i]["2"]
            results3 = mle_or_ols_results[i]["3"] * fdr_results[i]["3"]

            if compact:
                others = np.delete(self.all_nodes, i)
                more_others_lst = self.make_more_others_lst(others, i)

                for j in others:
                    coup2_[j][i] = results2[j]

                counter = 0
                for j in others:
                    if j == others[-1]:
                        assert counter == (self.n_nodes - 1) * (self.n_nodes - 2) / 2, "infer_coup_mle_with_fdr wrong. "
                        break
                    elif j < i:
                        for k in more_others_lst[j]:
                            coup3_[i][j][k] = results3[counter]
                            counter += 1
                    else:
                        for k in more_others_lst[j - 1]:
                            coup3_[i][j][k] = results3[counter]
                            counter += 1
            else:
                coup.append({"natfreq": mle_or_ols_results[i]["natfreq"], "2": results2, "3": results3,
                             "standard deviation": mle_or_ols_results[i]["standard deviation"]})

        if compact:
            return coup2_, coup3_
        else:
            return coup

    def conn_criteria_base(self, mle_or_ols_results=None, lasso_results=None, ada_results=None, sig_level=0.05,
                           mle_ori=False,
                           mle_threshold=0, conn_out=False, weak_strong=0):
        demo_coup2 = self.coupling2
        demo_coup3 = self.coupling3

        mle_ori_output = None
        mle_output = None
        if mle_or_ols_results is not None:
            assert len(mle_or_ols_results) == self.n_nodes, "mle_or_ols_results BOOM conn_criteria"
            mle_conn_lst_total, paramss = self.fdr_control_for_mle(inf_result_real=mle_or_ols_results,
                                                                   sig_level=sig_level)
            mle_output = {"A": 0, "B": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                          "TP_2": 0, "TN_2": 0, "FP_2": 0, "FN_2": 0,
                          "TP_3": 0, "TN_3": 0, "FP_3": 0, "FN_3": 0,
                          "TP_2_weak": 0, "TN_2_weak": 0, "FP_2_weak": 0, "FN_2_weak": 0,
                          "TP_3_weak": 0, "TN_3_weak": 0, "FP_3_weak": 0, "FN_3_weak": 0,
                          "TP_2_strong": 0, "TN_2_strong": 0, "FP_2_strong": 0, "FN_2_strong": 0,
                          "TP_3_strong": 0, "TN_3_strong": 0, "FP_3_strong": 0, "FN_3_strong": 0,
                          "TP_weak": 0, "TN_weak": 0, "FP_weak": 0, "FN_weak": 0,
                          "TP_strong": 0, "TN_strong": 0, "FP_strong": 0, "FN_strong": 0,
                          "right_num": 0, "total_num": 0}
            if mle_ori:
                mle_ori_output = {"A": 0, "B": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                                  "TP_2": 0, "TN_2": 0, "FP_2": 0, "FN_2": 0,
                                  "TP_3": 0, "TN_3": 0, "FP_3": 0, "FN_3": 0,
                                  "TP_2_weak": 0, "TN_2_weak": 0, "FP_2_weak": 0, "FN_2_weak": 0,
                                  "TP_3_weak": 0, "TN_3_weak": 0, "FP_3_weak": 0, "FN_3_weak": 0,
                                  "TP_2_strong": 0, "TN_2_strong": 0, "FP_2_strong": 0, "FN_2_strong": 0,
                                  "TP_3_strong": 0, "TN_3_strong": 0, "FP_3_strong": 0, "FN_3_strong": 0,
                                  "TP_weak": 0, "TN_weak": 0, "FP_weak": 0, "FN_weak": 0,
                                  "TP_strong": 0, "TN_strong": 0, "FP_strong": 0, "FN_strong": 0,
                                  "right_num": 0, "total_num": 0}
                # # get max coup
                # max_coup = 0
                # for i in range(self.n_nodes):
                #     max2_sub = max(mle_or_ols_results[i]["2"])
                #     max3_sub = max(mle_or_ols_results[i]["3"])
                #     max_coup = max(abs(max_coup), max2_sub)
                #     max_coup = max(abs(max_coup), max3_sub)
                # max_coup = abs(max_coup)
                # if mle_threshold == 0:
                #     mle_threshold = 0.1

                # # mean&variance method
                # total_poss = self.n_nodes * (self.n_nodes - 1) + \
                #              self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2
                # sum2 = 0
                # sum3 = 0
                # for i in range(self.n_nodes):
                #     sum2 += mle_or_ols_results[i]["2"].sum()
                #     sum3 += mle_or_ols_results[i]["3"].sum()
                # all_mean = (sum2 + sum3) / total_poss
                # square_sum = 0
                # for i in range(self.n_nodes):
                #     square_sum += ((mle_or_ols_results[i]["2"] - all_mean) ** 2).sum() + \
                #                   ((mle_or_ols_results[i]["3"] - all_mean) ** 2).sum()
                # all_var = np.sqrt(square_sum / total_poss)
                #
                # node_num = len(mle_or_ols_results)
                # nats_inf = np.zeros(node_num)
                # noiseinf = np.zeros(node_num)
                # for _ in range(node_num):
                #     nats_inf[_] = mle_or_ols_results[_]["natfreq"]
                #     noiseinf[_] = mle_or_ols_results[_]["standard deviation"]

                # 1210: update
                sp2_mean = paramss[0]
                sp2_sd = paramss[2]
                sp3_mean = paramss[3]
                sp3_sd = paramss[5]

        lasso_output = None
        if lasso_results is not None:
            assert len(lasso_results) == self.n_nodes, "lasso_results BOOM conn_criteria"
            lasso_output = {"A": 0, "B": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0, "TP_2": 0, "TN_2": 0, "FP_2": 0,
                            "FN_2": 0, "TP_3": 0, "TN_3": 0, "FP_3": 0, "FN_3": 0, "TP_2_weak": 0, "TN_2_weak": 0,
                            "FP_2_weak": 0,
                            "FN_2_weak": 0, "TP_3_weak": 0, "TN_3_weak": 0, "FP_3_weak": 0, "FN_3_weak": 0,
                            "TP_2_strong": 0, "TN_2_strong": 0, "FP_2_strong": 0,
                            "FN_2_strong": 0, "TP_3_strong": 0, "TN_3_strong": 0, "FP_3_strong": 0, "FN_3_strong": 0,
                            "TP_weak": 0, "TN_weak": 0,
                            "FP_weak": 0,
                            "FN_weak": 0,
                            "TP_strong": 0, "TN_strong": 0, "FP_strong": 0,
                            "FN_strong": 0,
                            "right_num": 0, "total_num": 0}
        ada_output = None
        if ada_results is not None:
            assert len(ada_results) == self.n_nodes, "ada_results BOOM conn_criteria"
            ada_output = {"A": 0, "B": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0, "TP_2": 0, "TN_2": 0, "FP_2": 0,
                          "FN_2": 0, "TP_3": 0, "TN_3": 0, "FP_3": 0, "FN_3": 0, "TP_2_weak": 0, "TN_2_weak": 0,
                          "FP_2_weak": 0,
                          "FN_2_weak": 0, "TP_3_weak": 0, "TN_3_weak": 0, "FP_3_weak": 0, "FN_3_weak": 0,
                          "TP_2_strong": 0, "TN_2_strong": 0, "FP_2_strong": 0,
                          "FN_2_strong": 0, "TP_3_strong": 0, "TN_3_strong": 0, "FP_3_strong": 0, "FN_3_strong": 0,
                          "TP_weak": 0, "TN_weak": 0,
                          "FP_weak": 0,
                          "FN_weak": 0,
                          "TP_strong": 0, "TN_strong": 0, "FP_strong": 0,
                          "FN_strong": 0,
                          "right_num": 0, "total_num": 0}

        for i in range(self.n_nodes):
            others = np.delete(self.all_nodes, i)
            more_others_lst = self.make_more_others_lst(others, i)

            # demo coupling
            demo_lst2 = []
            demo_lst3 = []
            for j in others:
                demo_lst2.append(demo_coup2[j][i])

                if j == others[-1]:
                    break
                elif j < i:
                    for k in more_others_lst[j]:
                        demo_lst3.append(demo_coup3[i][j][k])
                else:
                    for k in more_others_lst[j - 1]:
                        demo_lst3.append(demo_coup3[i][j][k])

            if weak_strong > 0:
                two_real_ws = np.array(demo_lst2)
                three_real_ws = np.array(demo_lst3)

            two_real = np.array(demo_lst2)
            two_real[two_real != 0] = 1
            three_real = np.array(demo_lst3)
            three_real[three_real != 0] = 1

            if lasso_results is not None:
                two_conn_lst_lasso = lasso_results[i]["2"].copy()
                three_conn_lst_lasso = lasso_results[i]["3"].copy()

                cc = np.count_nonzero(two_conn_lst_lasso)
                dd = np.count_nonzero(three_conn_lst_lasso)

                two_conn_lst_lasso[two_conn_lst_lasso != 0] = 1
                three_conn_lst_lasso[three_conn_lst_lasso != 0] = 1

                TP_values_lasso_2 = np.sum(np.logical_and(two_real == 1, two_conn_lst_lasso == 1))
                TN_values_lasso_2 = np.sum(np.logical_and(two_real == 0, two_conn_lst_lasso == 0))
                FP_values_lasso_2 = np.sum(np.logical_and(two_real == 0, two_conn_lst_lasso == 1))
                FN_values_lasso_2 = np.sum(np.logical_and(two_real == 1, two_conn_lst_lasso == 0))

                TP_values_lasso_3 = np.sum(np.logical_and(three_real == 1, three_conn_lst_lasso == 1))
                TN_values_lasso_3 = np.sum(np.logical_and(three_real == 0, three_conn_lst_lasso == 0))
                FP_values_lasso_3 = np.sum(np.logical_and(three_real == 0, three_conn_lst_lasso == 1))
                FN_values_lasso_3 = np.sum(np.logical_and(three_real == 1, three_conn_lst_lasso == 0))

                if weak_strong > 0:
                    TP_values_lasso_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                   two_conn_lst_lasso == 1))
                    TN_values_lasso_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_lasso == 0))
                    FP_values_lasso_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_lasso == 1))
                    FN_values_lasso_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                   two_conn_lst_lasso == 0))

                    TP_values_lasso_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                     two_conn_lst_lasso == 1))
                    TN_values_lasso_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_lasso == 0))
                    FP_values_lasso_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_lasso == 1))
                    FN_values_lasso_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                     two_conn_lst_lasso == 0))

                    TP_values_lasso_3_weak = np.sum(np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                                                   three_conn_lst_lasso == 1))
                    TN_values_lasso_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_lasso == 0))
                    FP_values_lasso_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_lasso == 1))
                    FN_values_lasso_3_weak = np.sum(np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                                                   three_conn_lst_lasso == 0))

                    TP_values_lasso_3_strong = np.sum(np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                                                     three_conn_lst_lasso == 1))
                    TN_values_lasso_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_lasso == 0))
                    FP_values_lasso_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_lasso == 1))
                    FN_values_lasso_3_strong = np.sum(np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                                                     three_conn_lst_lasso == 0))

                    TP_values_lasso_weak = TP_values_lasso_2_weak + TP_values_lasso_3_weak
                    TN_values_lasso_weak = TN_values_lasso_2_weak + TN_values_lasso_3_weak
                    FP_values_lasso_weak = FP_values_lasso_2_weak + FP_values_lasso_3_weak
                    FN_values_lasso_weak = FN_values_lasso_2_weak + FN_values_lasso_3_weak
                    TP_values_lasso_strong = TP_values_lasso_2_strong + TP_values_lasso_3_strong
                    TN_values_lasso_strong = TN_values_lasso_2_strong + TN_values_lasso_3_strong
                    FP_values_lasso_strong = FP_values_lasso_2_strong + FP_values_lasso_3_strong
                    FN_values_lasso_strong = FN_values_lasso_2_strong + FN_values_lasso_3_strong

                    lasso_output["TP_2_weak"] += TP_values_lasso_2_weak
                    lasso_output["TN_2_weak"] += TN_values_lasso_2_weak
                    lasso_output["FP_2_weak"] += FP_values_lasso_2_weak
                    lasso_output["FN_2_weak"] += FN_values_lasso_2_weak
                    lasso_output["TP_3_weak"] += TP_values_lasso_3_weak
                    lasso_output["TN_3_weak"] += TN_values_lasso_3_weak
                    lasso_output["FP_3_weak"] += FP_values_lasso_3_weak
                    lasso_output["FN_3_weak"] += FN_values_lasso_3_weak
                    lasso_output["TP_2_strong"] += TP_values_lasso_2_strong
                    lasso_output["TN_2_strong"] += TN_values_lasso_2_strong
                    lasso_output["FP_2_strong"] += FP_values_lasso_2_strong
                    lasso_output["FN_2_strong"] += FN_values_lasso_2_strong
                    lasso_output["TP_3_strong"] += TP_values_lasso_3_strong
                    lasso_output["TN_3_strong"] += TN_values_lasso_3_strong
                    lasso_output["FP_3_strong"] += FP_values_lasso_3_strong
                    lasso_output["FN_3_strong"] += FN_values_lasso_3_strong
                    lasso_output["TP_weak"] += TP_values_lasso_weak
                    lasso_output["TN_weak"] += TN_values_lasso_weak
                    lasso_output["FP_weak"] += FP_values_lasso_weak
                    lasso_output["FN_weak"] += FN_values_lasso_weak
                    lasso_output["TP_strong"] += TP_values_lasso_strong
                    lasso_output["TN_strong"] += TN_values_lasso_strong
                    lasso_output["FP_strong"] += FP_values_lasso_strong
                    lasso_output["FN_strong"] += FN_values_lasso_strong

                TP_values_lasso = TP_values_lasso_2 + TP_values_lasso_3
                TN_values_lasso = TN_values_lasso_2 + TN_values_lasso_3
                FP_values_lasso = FP_values_lasso_2 + FP_values_lasso_3
                FN_values_lasso = FN_values_lasso_2 + FN_values_lasso_3

                two_check = two_real - two_conn_lst_lasso
                three_check = three_real - three_conn_lst_lasso
                right_num_lasso = np.count_nonzero(two_check == 0) + np.count_nonzero(three_check == 0)
                total_num_lasso = len(two_real) + len(three_real)

                lasso_output["A"] += cc
                lasso_output["B"] += dd
                lasso_output["TP"] += TP_values_lasso
                lasso_output["TN"] += TN_values_lasso
                lasso_output["FP"] += FP_values_lasso
                lasso_output["FN"] += FN_values_lasso
                lasso_output["TP_2"] += TP_values_lasso_2
                lasso_output["TN_2"] += TN_values_lasso_2
                lasso_output["FP_2"] += FP_values_lasso_2
                lasso_output["FN_2"] += FN_values_lasso_2
                lasso_output["TP_3"] += TP_values_lasso_3
                lasso_output["TN_3"] += TN_values_lasso_3
                lasso_output["FP_3"] += FP_values_lasso_3
                lasso_output["FN_3"] += FN_values_lasso_3
                lasso_output["right_num"] += right_num_lasso
                lasso_output["total_num"] += total_num_lasso

            if mle_or_ols_results is not None:
                two_conn_lst_mle = mle_conn_lst_total[i]["2"].copy()
                three_conn_lst_mle = mle_conn_lst_total[i]["3"].copy()

                ee = np.count_nonzero(two_conn_lst_mle)
                ff = np.count_nonzero(three_conn_lst_mle)

                TP_values_mle_2 = np.sum(np.logical_and(two_real == 1, two_conn_lst_mle == 1))
                TN_values_mle_2 = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle == 0))
                FP_values_mle_2 = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle == 1))
                FN_values_mle_2 = np.sum(np.logical_and(two_real == 1, two_conn_lst_mle == 0))

                TP_values_mle_3 = np.sum(np.logical_and(three_real == 1, three_conn_lst_mle == 1))
                TN_values_mle_3 = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle == 0))
                FP_values_mle_3 = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle == 1))
                FN_values_mle_3 = np.sum(np.logical_and(three_real == 1, three_conn_lst_mle == 0))

                if weak_strong > 0:
                    TP_values_mle_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                 two_conn_lst_mle == 1))
                    TN_values_mle_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_mle == 0))
                    FP_values_mle_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_mle == 1))
                    FN_values_mle_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                 two_conn_lst_mle == 0))

                    TP_values_mle_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                   two_conn_lst_mle == 1))
                    TN_values_mle_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle == 0))
                    FP_values_mle_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle == 1))
                    FN_values_mle_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                   two_conn_lst_mle == 0))

                    TP_values_mle_3_weak = np.sum(np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                                                 three_conn_lst_mle == 1))
                    TN_values_mle_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_mle == 0))
                    FP_values_mle_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_mle == 1))
                    FN_values_mle_3_weak = np.sum(np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                                                 three_conn_lst_mle == 0))

                    TP_values_mle_3_strong = np.sum(np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                                                   three_conn_lst_mle == 1))
                    TN_values_mle_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle == 0))
                    FP_values_mle_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle == 1))
                    FN_values_mle_3_strong = np.sum(np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                                                   three_conn_lst_mle == 0))

                    TP_values_mle_weak = TP_values_mle_2_weak + TP_values_mle_3_weak
                    TN_values_mle_weak = TN_values_mle_2_weak + TN_values_mle_3_weak
                    FP_values_mle_weak = FP_values_mle_2_weak + FP_values_mle_3_weak
                    FN_values_mle_weak = FN_values_mle_2_weak + FN_values_mle_3_weak
                    TP_values_mle_strong = TP_values_mle_2_strong + TP_values_mle_3_strong
                    TN_values_mle_strong = TN_values_mle_2_strong + TN_values_mle_3_strong
                    FP_values_mle_strong = FP_values_mle_2_strong + FP_values_mle_3_strong
                    FN_values_mle_strong = FN_values_mle_2_strong + FN_values_mle_3_strong

                    mle_output["TP_2_weak"] += TP_values_mle_2_weak
                    mle_output["TN_2_weak"] += TN_values_mle_2_weak
                    mle_output["FP_2_weak"] += FP_values_mle_2_weak
                    mle_output["FN_2_weak"] += FN_values_mle_2_weak
                    mle_output["TP_3_weak"] += TP_values_mle_3_weak
                    mle_output["TN_3_weak"] += TN_values_mle_3_weak
                    mle_output["FP_3_weak"] += FP_values_mle_3_weak
                    mle_output["FN_3_weak"] += FN_values_mle_3_weak
                    mle_output["TP_2_strong"] += TP_values_mle_2_strong
                    mle_output["TN_2_strong"] += TN_values_mle_2_strong
                    mle_output["FP_2_strong"] += FP_values_mle_2_strong
                    mle_output["FN_2_strong"] += FN_values_mle_2_strong
                    mle_output["TP_3_strong"] += TP_values_mle_3_strong
                    mle_output["TN_3_strong"] += TN_values_mle_3_strong
                    mle_output["FP_3_strong"] += FP_values_mle_3_strong
                    mle_output["FN_3_strong"] += FN_values_mle_3_strong
                    mle_output["TP_weak"] += TP_values_mle_weak
                    mle_output["TN_weak"] += TN_values_mle_weak
                    mle_output["FP_weak"] += FP_values_mle_weak
                    mle_output["FN_weak"] += FN_values_mle_weak
                    mle_output["TP_strong"] += TP_values_mle_strong
                    mle_output["TN_strong"] += TN_values_mle_strong
                    mle_output["FP_strong"] += FP_values_mle_strong
                    mle_output["FN_strong"] += FN_values_mle_strong

                TP_values_mle = TP_values_mle_2 + TP_values_mle_3
                TN_values_mle = TN_values_mle_2 + TN_values_mle_3
                FP_values_mle = FP_values_mle_2 + FP_values_mle_3
                FN_values_mle = FN_values_mle_2 + FN_values_mle_3

                two_check_mle = two_real - two_conn_lst_mle
                three_check_mle = three_real - three_conn_lst_mle
                right_num_mle = np.count_nonzero(two_check_mle == 0) + np.count_nonzero(three_check_mle == 0)
                total_num_mle = len(two_real) + len(three_real)

                assert np.sum(two_real) + np.sum(three_real) == TP_values_mle + FN_values_mle, "non0_s boom mle"
                assert np.count_nonzero(two_real == 0) + np.count_nonzero(three_real == 0) == TN_values_mle + \
                       FP_values_mle, "0_s? mle"

                mle_output["A"] += ee
                mle_output["B"] += ff
                mle_output["TP"] += TP_values_mle
                mle_output["TN"] += TN_values_mle
                mle_output["FP"] += FP_values_mle
                mle_output["FN"] += FN_values_mle
                mle_output["TP_2"] += TP_values_mle_2
                mle_output["TN_2"] += TN_values_mle_2
                mle_output["FP_2"] += FP_values_mle_2
                mle_output["FN_2"] += FN_values_mle_2
                mle_output["TP_3"] += TP_values_mle_3
                mle_output["TN_3"] += TN_values_mle_3
                mle_output["FP_3"] += FP_values_mle_3
                mle_output["FN_3"] += FN_values_mle_3
                mle_output["right_num"] += right_num_mle
                mle_output["total_num"] += total_num_mle

                if mle_ori:
                    two_conn_lst_mle_ori = mle_or_ols_results[i]["2"].copy()
                    three_conn_lst_mle_ori = mle_or_ols_results[i]["3"].copy()

                    # # for threshold method
                    # two_conn_lst_mle_ori[abs(two_conn_lst_mle_ori) <= mle_threshold * max_coup] = 0
                    # three_conn_lst_mle_ori[abs(three_conn_lst_mle_ori) <= mle_threshold * max_coup] = 0
                    # two_conn_lst_mle_ori[two_conn_lst_mle_ori != 0] = 1
                    # three_conn_lst_mle_ori[three_conn_lst_mle_ori != 0] = 1

                    # mean&variance method
                    two_conn_lst_mle_ori[np.logical_and(two_conn_lst_mle_ori <= (sp2_mean + 2 * sp2_sd),
                                                        two_conn_lst_mle_ori >= (sp2_mean - 2 * sp2_sd))] = 0
                    three_conn_lst_mle_ori[np.logical_and(three_conn_lst_mle_ori <= (sp3_mean + 2 * sp3_sd),
                                                          three_conn_lst_mle_ori >= (sp3_mean - 2 * sp3_sd))] = 0
                    two_conn_lst_mle_ori[two_conn_lst_mle_ori != 0] = 1
                    three_conn_lst_mle_ori[three_conn_lst_mle_ori != 0] = 1

                    qa = np.count_nonzero(two_conn_lst_mle_ori)
                    aq = np.count_nonzero(three_conn_lst_mle_ori)

                    TP_values_mle_2_ori = np.sum(np.logical_and(two_real == 1, two_conn_lst_mle_ori == 1))
                    TN_values_mle_2_ori = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle_ori == 0))
                    FP_values_mle_2_ori = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle_ori == 1))
                    FN_values_mle_2_ori = np.sum(np.logical_and(two_real == 1, two_conn_lst_mle_ori == 0))

                    TP_values_mle_3_ori = np.sum(np.logical_and(three_real == 1, three_conn_lst_mle_ori == 1))
                    TN_values_mle_3_ori = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle_ori == 0))
                    FP_values_mle_3_ori = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle_ori == 1))
                    FN_values_mle_3_ori = np.sum(np.logical_and(three_real == 1, three_conn_lst_mle_ori == 0))

                    if weak_strong > 0:
                        TP_values_mle_ori_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                         two_conn_lst_mle == 1))
                        TN_values_mle_ori_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_mle == 0))
                        FP_values_mle_ori_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_mle == 1))
                        FN_values_mle_ori_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                         two_conn_lst_mle == 0))

                        TP_values_mle_ori_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                           two_conn_lst_mle == 1))
                        TN_values_mle_ori_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle == 0))
                        FP_values_mle_ori_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_mle == 1))
                        FN_values_mle_ori_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                           two_conn_lst_mle == 0))

                        TP_values_mle_ori_3_weak = np.sum(
                            np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                           three_conn_lst_mle == 1))
                        TN_values_mle_ori_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_mle == 0))
                        FP_values_mle_ori_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_mle == 1))
                        FN_values_mle_ori_3_weak = np.sum(
                            np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                           three_conn_lst_mle == 0))

                        TP_values_mle_ori_3_strong = np.sum(
                            np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                           three_conn_lst_mle == 1))
                        TN_values_mle_ori_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle == 0))
                        FP_values_mle_ori_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_mle == 1))
                        FN_values_mle_ori_3_strong = np.sum(
                            np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                           three_conn_lst_mle == 0))

                        TP_values_mle_ori_weak = TP_values_mle_ori_2_weak + TP_values_mle_ori_3_weak
                        TN_values_mle_ori_weak = TN_values_mle_ori_2_weak + TN_values_mle_ori_3_weak
                        FP_values_mle_ori_weak = FP_values_mle_ori_2_weak + FP_values_mle_ori_3_weak
                        FN_values_mle_ori_weak = FN_values_mle_ori_2_weak + FN_values_mle_ori_3_weak
                        TP_values_mle_ori_strong = TP_values_mle_ori_2_strong + TP_values_mle_ori_3_strong
                        TN_values_mle_ori_strong = TN_values_mle_ori_2_strong + TN_values_mle_ori_3_strong
                        FP_values_mle_ori_strong = FP_values_mle_ori_2_strong + FP_values_mle_ori_3_strong
                        FN_values_mle_ori_strong = FN_values_mle_ori_2_strong + FN_values_mle_ori_3_strong

                        mle_ori_output["TP_2_weak"] += TP_values_mle_ori_2_weak
                        mle_ori_output["TN_2_weak"] += TN_values_mle_ori_2_weak
                        mle_ori_output["FP_2_weak"] += FP_values_mle_ori_2_weak
                        mle_ori_output["FN_2_weak"] += FN_values_mle_ori_2_weak
                        mle_ori_output["TP_3_weak"] += TP_values_mle_ori_3_weak
                        mle_ori_output["TN_3_weak"] += TN_values_mle_ori_3_weak
                        mle_ori_output["FP_3_weak"] += FP_values_mle_ori_3_weak
                        mle_ori_output["FN_3_weak"] += FN_values_mle_ori_3_weak
                        mle_ori_output["TP_2_strong"] += TP_values_mle_ori_2_strong
                        mle_ori_output["TN_2_strong"] += TN_values_mle_ori_2_strong
                        mle_ori_output["FP_2_strong"] += FP_values_mle_ori_2_strong
                        mle_ori_output["FN_2_strong"] += FN_values_mle_ori_2_strong
                        mle_ori_output["TP_3_strong"] += TP_values_mle_ori_3_strong
                        mle_ori_output["TN_3_strong"] += TN_values_mle_ori_3_strong
                        mle_ori_output["FP_3_strong"] += FP_values_mle_ori_3_strong
                        mle_ori_output["FN_3_strong"] += FN_values_mle_ori_3_strong
                        mle_ori_output["TP_weak"] += TP_values_mle_ori_weak
                        mle_ori_output["TN_weak"] += TN_values_mle_ori_weak
                        mle_ori_output["FP_weak"] += FP_values_mle_ori_weak
                        mle_ori_output["FN_weak"] += FN_values_mle_ori_weak
                        mle_ori_output["TP_strong"] += TP_values_mle_ori_strong
                        mle_ori_output["TN_strong"] += TN_values_mle_ori_strong
                        mle_ori_output["FP_strong"] += FP_values_mle_ori_strong
                        mle_ori_output["FN_strong"] += FN_values_mle_ori_strong

                    TP_values_mle_ori = TP_values_mle_2_ori + TP_values_mle_3_ori
                    TN_values_mle_ori = TN_values_mle_2_ori + TN_values_mle_3_ori
                    FP_values_mle_ori = FP_values_mle_2_ori + FP_values_mle_3_ori
                    FN_values_mle_ori = FN_values_mle_2_ori + FN_values_mle_3_ori

                    two_check_mle_ori = two_real - two_conn_lst_mle_ori
                    three_check_mle_ori = three_real - three_conn_lst_mle_ori
                    right_num_mle_ori = np.count_nonzero(two_check_mle_ori == 0) + \
                                        np.count_nonzero(three_check_mle_ori == 0)
                    total_num_mle_ori = len(two_real) + len(three_real)

                    assert np.sum(two_real) + np.sum(three_real) == TP_values_mle_ori + FN_values_mle_ori, \
                        "non0_s boom mle_ori"
                    assert np.count_nonzero(two_real == 0) + np.count_nonzero(three_real == 0) == TN_values_mle_ori + \
                           FP_values_mle_ori, "0_s? mle_ori"

                    mle_ori_output["A"] += qa
                    mle_ori_output["B"] += aq
                    mle_ori_output["TP"] += TP_values_mle_ori
                    mle_ori_output["TN"] += TN_values_mle_ori
                    mle_ori_output["FP"] += FP_values_mle_ori
                    mle_ori_output["FN"] += FN_values_mle_ori
                    mle_ori_output["TP_2"] += TP_values_mle_2_ori
                    mle_ori_output["TN_2"] += TN_values_mle_2_ori
                    mle_ori_output["FP_2"] += FP_values_mle_2_ori
                    mle_ori_output["FN_2"] += FN_values_mle_2_ori
                    mle_ori_output["TP_3"] += TP_values_mle_3_ori
                    mle_ori_output["TN_3"] += TN_values_mle_3_ori
                    mle_ori_output["FP_3"] += FP_values_mle_3_ori
                    mle_ori_output["FN_3"] += FN_values_mle_3_ori
                    mle_ori_output["right_num"] += right_num_mle_ori
                    mle_ori_output["total_num"] += total_num_mle_ori

            if ada_results is not None:
                two_conn_lst_ada = ada_results[i]["2"].copy()
                three_conn_lst_ada = ada_results[i]["3"].copy()

                aa = np.count_nonzero(two_conn_lst_ada)
                bb = np.count_nonzero(three_conn_lst_ada)

                two_conn_lst_ada[two_conn_lst_ada != 0] = 1
                three_conn_lst_ada[three_conn_lst_ada != 0] = 1

                TP_values_ada_2 = np.sum(np.logical_and(two_real == 1, two_conn_lst_ada == 1))
                TN_values_ada_2 = np.sum(np.logical_and(two_real == 0, two_conn_lst_ada == 0))
                FP_values_ada_2 = np.sum(np.logical_and(two_real == 0, two_conn_lst_ada == 1))
                FN_values_ada_2 = np.sum(np.logical_and(two_real == 1, two_conn_lst_ada == 0))

                TP_values_ada_3 = np.sum(np.logical_and(three_real == 1, three_conn_lst_ada == 1))
                TN_values_ada_3 = np.sum(np.logical_and(three_real == 0, three_conn_lst_ada == 0))
                FP_values_ada_3 = np.sum(np.logical_and(three_real == 0, three_conn_lst_ada == 1))
                FN_values_ada_3 = np.sum(np.logical_and(three_real == 1, three_conn_lst_ada == 0))

                if weak_strong > 0:
                    TP_values_ada_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                 two_conn_lst_ada == 1))
                    TN_values_ada_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_ada == 0))
                    FP_values_ada_2_weak = np.sum(np.logical_and(two_real_ws == 0, two_conn_lst_ada == 1))
                    FN_values_ada_2_weak = np.sum(np.logical_and(two_real_ws > 0, two_real_ws <= weak_strong,
                                                                 two_conn_lst_ada == 0))

                    TP_values_ada_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                   two_conn_lst_ada == 1))
                    TN_values_ada_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_ada == 0))
                    FP_values_ada_2_strong = np.sum(np.logical_and(two_real == 0, two_conn_lst_ada == 1))
                    FN_values_ada_2_strong = np.sum(np.logical_and(two_real_ws > 0, two_real_ws > weak_strong,
                                                                   two_conn_lst_ada == 0))

                    TP_values_ada_3_weak = np.sum(np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                                                 three_conn_lst_ada == 1))
                    TN_values_ada_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_ada == 0))
                    FP_values_ada_3_weak = np.sum(np.logical_and(three_real_ws == 0, three_conn_lst_ada == 1))
                    FN_values_ada_3_weak = np.sum(np.logical_and(three_real_ws > 0, three_real_ws <= weak_strong,
                                                                 three_conn_lst_ada == 0))

                    TP_values_ada_3_strong = np.sum(np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                                                   three_conn_lst_ada == 1))
                    TN_values_ada_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_ada == 0))
                    FP_values_ada_3_strong = np.sum(np.logical_and(three_real == 0, three_conn_lst_ada == 1))
                    FN_values_ada_3_strong = np.sum(np.logical_and(three_real_ws > 0, three_real_ws > weak_strong,
                                                                   three_conn_lst_ada == 0))

                    TP_values_ada_weak = TP_values_ada_2_weak + TP_values_ada_3_weak
                    TN_values_ada_weak = TN_values_ada_2_weak + TN_values_ada_3_weak
                    FP_values_ada_weak = FP_values_ada_2_weak + FP_values_ada_3_weak
                    FN_values_ada_weak = FN_values_ada_2_weak + FN_values_ada_3_weak
                    TP_values_ada_strong = TP_values_ada_2_strong + TP_values_ada_3_strong
                    TN_values_ada_strong = TN_values_ada_2_strong + TN_values_ada_3_strong
                    FP_values_ada_strong = FP_values_ada_2_strong + FP_values_ada_3_strong
                    FN_values_ada_strong = FN_values_ada_2_strong + FN_values_ada_3_strong

                    ada_output["TP_2_weak"] += TP_values_ada_2_weak
                    ada_output["TN_2_weak"] += TN_values_ada_2_weak
                    ada_output["FP_2_weak"] += FP_values_ada_2_weak
                    ada_output["FN_2_weak"] += FN_values_ada_2_weak
                    ada_output["TP_3_weak"] += TP_values_ada_3_weak
                    ada_output["TN_3_weak"] += TN_values_ada_3_weak
                    ada_output["FP_3_weak"] += FP_values_ada_3_weak
                    ada_output["FN_3_weak"] += FN_values_ada_3_weak
                    ada_output["TP_2_strong"] += TP_values_ada_2_strong
                    ada_output["TN_2_strong"] += TN_values_ada_2_strong
                    ada_output["FP_2_strong"] += FP_values_ada_2_strong
                    ada_output["FN_2_strong"] += FN_values_ada_2_strong
                    ada_output["TP_3_strong"] += TP_values_ada_3_strong
                    ada_output["TN_3_strong"] += TN_values_ada_3_strong
                    ada_output["FP_3_strong"] += FP_values_ada_3_strong
                    ada_output["FN_3_strong"] += FN_values_ada_3_strong
                    ada_output["TP_weak"] += TP_values_ada_weak
                    ada_output["TN_weak"] += TN_values_ada_weak
                    ada_output["FP_weak"] += FP_values_ada_weak
                    ada_output["FN_weak"] += FN_values_ada_weak
                    ada_output["TP_strong"] += TP_values_ada_strong
                    ada_output["TN_strong"] += TN_values_ada_strong
                    ada_output["FP_strong"] += FP_values_ada_strong
                    ada_output["FN_strong"] += FN_values_ada_strong

                TP_values_ada = TP_values_ada_2 + TP_values_ada_3
                TN_values_ada = TN_values_ada_2 + TN_values_ada_3
                FP_values_ada = FP_values_ada_2 + FP_values_ada_3
                FN_values_ada = FN_values_ada_2 + FN_values_ada_3

                two_check = two_real - two_conn_lst_ada
                three_check = three_real - three_conn_lst_ada
                right_num_ada = np.count_nonzero(two_check == 0) + np.count_nonzero(three_check == 0)
                total_num_ada = len(two_real) + len(three_real)

                # alpha = ada_results[i]["alpha"]
                # print("The alpha is", alpha, "by bic. ")

                ada_output["A"] += aa
                ada_output["B"] += bb
                ada_output["TP"] += TP_values_ada
                ada_output["TN"] += TN_values_ada
                ada_output["FP"] += FP_values_ada
                ada_output["FN"] += FN_values_ada
                ada_output["TP_2"] += TP_values_ada_2
                ada_output["TN_2"] += TN_values_ada_2
                ada_output["FP_2"] += FP_values_ada_2
                ada_output["FN_2"] += FN_values_ada_2
                ada_output["TP_3"] += TP_values_ada_3
                ada_output["TN_3"] += TN_values_ada_3
                ada_output["FP_3"] += FP_values_ada_3
                ada_output["FN_3"] += FN_values_ada_3
                ada_output["right_num"] += right_num_ada
                ada_output["total_num"] += total_num_ada

        all_results = {"mle": mle_output, "lasso": lasso_output, "ada": ada_output, "mle_ori": mle_ori_output}

        if not conn_out:
            return all_results
        else:
            return all_results, mle_conn_lst_total

    def coup_compare_base(self, mle_or_ols_results=None, lasso_results=None, ada_results=None):
        demo_coup2 = self.coupling2
        demo_coup3 = self.coupling3

        demo_coup2_output = np.array([])
        demo_coup3_output = np.array([])

        if mle_or_ols_results is not None:
            assert len(mle_or_ols_results) == self.n_nodes, "mle_or_ols_results BOOM coup_compare"
            mle_coup2_output = np.array([])
            mle_coup3_output = np.array([])
        else:
            mle_coup2_output = None
            mle_coup3_output = None
        if lasso_results is not None:
            assert len(lasso_results) == self.n_nodes, "lasso_results BOOM coup_compare"
            lasso_coup2_output = np.array([])
            lasso_coup3_output = np.array([])
        else:
            lasso_coup2_output = None
            lasso_coup3_output = None
        if ada_results is not None:
            assert len(ada_results) == self.n_nodes, "ada_results BOOM coup_compare"
            ada_coup2_output = np.array([])
            ada_coup3_output = np.array([])
        else:
            ada_coup2_output = None
            ada_coup3_output = None

        for i in range(self.n_nodes):
            others = np.delete(self.all_nodes, i)
            more_others_lst = self.make_more_others_lst(others, i)

            for j in others:
                demo_coup2_output = np.append(demo_coup2_output, demo_coup2[j][i])

                if j == others[-1]:
                    break
                elif j < i:
                    for k in more_others_lst[j]:
                        demo_coup3_output = np.append(demo_coup3_output, demo_coup3[i][j][k])
                else:
                    for k in more_others_lst[j - 1]:
                        demo_coup3_output = np.append(demo_coup3_output, demo_coup3[i][j][k])

            if lasso_results is not None:
                lasso_coup2_output = np.concatenate((lasso_coup2_output, lasso_results[i]["2"]), axis=0)
                lasso_coup3_output = np.concatenate((lasso_coup3_output, lasso_results[i]["3"]), axis=0)

            if mle_or_ols_results is not None:
                mle_coup2_output = np.concatenate((mle_coup2_output, mle_or_ols_results[i]["2"]), axis=0)
                mle_coup3_output = np.concatenate((mle_coup3_output, mle_or_ols_results[i]["3"]), axis=0)

            if ada_results is not None:
                ada_coup2_output = np.concatenate((ada_coup2_output, ada_results[i]["2"]), axis=0)
                ada_coup3_output = np.concatenate((ada_coup3_output, ada_results[i]["3"]), axis=0)

        return demo_coup2_output, demo_coup3_output, mle_coup2_output, mle_coup3_output, lasso_coup2_output, \
            lasso_coup3_output, ada_coup2_output, ada_coup3_output

    def new_rates_easy(self, A: int, B: int):
        if A == 0 and B == 0:
            return np.nan
        return A / (A + 2 * B / (self.n_nodes - 2))

    @staticmethod
    def FPR_easy(FP: int, TN: int):
        return FP / (FP + TN)

    @staticmethod
    def FNR_easy(FN: int, TP: int):
        return FN / (FN + TP)

    @staticmethod
    def MCC_easy(TP: int, TN: int, FP: int, FN: int):
        if TN + FP == 0 or TP + FN == 0:
            return 1
        if TP + FP == 0 or TN + FN == 0:
            return 0
        return (TP * TN - FN * FP) / np.sqrt((TP + FP) * (TN + FP) * (TP + FN) * (TN + FN))

    def conn_type_test(self, mle_or_ols_results=None, lasso_results=None, ada_results=None, sig_level=0.05,
                       mle_ori=False,
                       mle_threshold=0):
        if mle_or_ols_results is not None:
            assert len(mle_or_ols_results) == self.n_nodes, "mle_or_ols_results BOOM conn_criteria"
            mle_conn_lst_total, paramss = self.fdr_control_for_mle(inf_result_real=mle_or_ols_results,
                                                                   sig_level=sig_level)
            mle_2 = 0
            mle_3 = 0
            if mle_ori:
                mle_ori_2 = 0
                mle_ori_3 = 0

                # # get max coup
                # max_coup = 0
                # for i in range(self.n_nodes):
                #     max2_sub = max(mle_or_ols_results[i]["2"])
                #     max3_sub = max(mle_or_ols_results[i]["3"])
                #     max_coup = max(abs(max_coup), max2_sub)
                #     max_coup = max(abs(max_coup), max3_sub)
                # max_coup = abs(max_coup)
                # if mle_threshold == 0:
                #     mle_threshold = 0.1

                # # mean&variance method
                # total_poss = self.n_nodes * (self.n_nodes - 1) + \
                #              self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2
                # sum2 = 0
                # sum3 = 0
                # for i in range(self.n_nodes):
                #     sum2 += mle_or_ols_results[i]["2"].sum()
                #     sum3 += mle_or_ols_results[i]["3"].sum()
                # all_mean = (sum2 + sum3) / total_poss
                # square_sum = 0
                # for i in range(self.n_nodes):
                #     square_sum += ((mle_or_ols_results[i]["2"] - all_mean) ** 2).sum() + \
                #                   ((mle_or_ols_results[i]["3"] - all_mean) ** 2).sum()
                # all_var = np.sqrt(square_sum / total_poss)

                # 1210: update
                sp2_mean = paramss[0]
                sp2_sd = paramss[2]
                sp3_mean = paramss[3]
                sp3_sd = paramss[5]

        if lasso_results is not None:
            assert len(lasso_results) == self.n_nodes, "lasso_results BOOM conn_criteria"
            lasso_2 = 0
            lasso_3 = 0
        if ada_results is not None:
            assert len(ada_results) == self.n_nodes, "ada_results BOOM conn_criteria"
            ada_2 = 0
            ada_3 = 0

        for i in range(self.n_nodes):
            if lasso_results is not None:
                two_conn_lst_lasso = lasso_results[i]["2"]
                three_conn_lst_lasso = lasso_results[i]["3"]

                lasso_2 += np.count_nonzero(two_conn_lst_lasso)
                lasso_3 += np.count_nonzero(three_conn_lst_lasso)

            if mle_or_ols_results is not None:
                two_conn_lst_mle = mle_conn_lst_total[i]["2"]
                three_conn_lst_mle = mle_conn_lst_total[i]["3"]

                mle_2 += np.count_nonzero(two_conn_lst_mle)
                mle_3 += np.count_nonzero(three_conn_lst_mle)
                if mle_ori:
                    two_conn_lst_mle_ori = mle_or_ols_results[i]["2"]
                    three_conn_lst_mle_ori = mle_or_ols_results[i]["3"]

                    # # for threshold method
                    # two_conn_lst_mle_ori[abs(two_conn_lst_mle_ori) <= mle_threshold * max_coup] = 0
                    # three_conn_lst_mle_ori[abs(three_conn_lst_mle_ori) <= mle_threshold * max_coup] = 0

                    # mean&variance method
                    two_conn_lst_mle_ori[np.logical_and(two_conn_lst_mle_ori <= (sp2_mean + 2 * sp2_sd),
                                                        two_conn_lst_mle_ori >= (sp2_mean - 2 * sp2_sd))] = 0
                    three_conn_lst_mle_ori[np.logical_and(three_conn_lst_mle_ori <= (sp3_mean + 2 * sp3_sd),
                                                          three_conn_lst_mle_ori >= (sp3_mean - 2 * sp3_sd))] = 0

                    mle_ori_2 += np.count_nonzero(two_conn_lst_mle_ori)
                    mle_ori_3 += np.count_nonzero(three_conn_lst_mle_ori)

            if ada_results is not None:
                two_conn_lst_ada = ada_results[i]["2"]
                three_conn_lst_ada = ada_results[i]["3"]

                ada_2 += np.count_nonzero(two_conn_lst_ada)
                ada_3 += np.count_nonzero(three_conn_lst_ada)

        n = np.array([((self.n_nodes - 1) * self.n_nodes),
                      (self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2)])
        if mle_or_ols_results is not None:
            x = np.array([mle_2, mle_3])

            _, p_mle = proportions_ztest(x, n)
            # print("p-value for MLE is: ", p_mle)
            if p_mle > sig_level:
                mle_output = "mix"
            elif x[0] / n[0] < x[1] / n[1]:
                mle_output = "3-int"
            elif x[0] / n[0] > x[1] / n[1]:
                mle_output = "pairwise"
            elif x[0] == 0 and x[1] == 0:
                mle_output = "nothing"
            mle_output2 = x[0] / n[0]
            mle_output3 = x[1] / n[1]
            if mle_ori:
                x = np.array([mle_ori_2, mle_ori_3])
                # print(sum(x))
                # print(sum(n))
                # print(x[0])
                # print(x[1])
                # print(n[0])
                # print(n[1])
                # print(x[0]/n[0])
                # print(x[1]/n[1])

                _, p_mle = proportions_ztest(x, n)
                # print("p-value for MLE_ori is: ", p_mle)
                if p_mle > sig_level:
                    mle_ori_output = "mix"
                elif x[0] / n[0] < x[1] / n[1]:
                    mle_ori_output = "3-int"
                elif x[0] / n[0] > x[1] / n[1]:
                    mle_ori_output = "pairwise"
                elif x[0] == 0 and x[1] == 0:
                    mle_ori_output = "nothing"
                mle_ori_output2 = x[0] / n[0]
                mle_ori_output3 = x[1] / n[1]
            else:
                mle_ori_output = None
                mle_ori_output2 = None
                mle_ori_output3 = None
        else:
            mle_output = None
            mle_output2 = None
            mle_output3 = None
        if lasso_results is not None:
            x = np.array([lasso_2, lasso_3])

            _, p_lasso = proportions_ztest(x, n)
            # print("p-value for LASSO is: ", p_lasso)
            if p_lasso > sig_level:
                lasso_output = "mix"
            elif x[0] / n[0] < x[1] / n[1]:
                lasso_output = "3-int"
            elif x[0] / n[0] > x[1] / n[1]:
                lasso_output = "pairwise"
            elif x[0] == 0 and x[1] == 0:
                lasso_output = "nothing"
            lasso_output2 = x[0] / n[0]
            lasso_output3 = x[1] / n[1]
        else:
            lasso_output = None
            lasso_output2 = None
            lasso_output3 = None
        if ada_results is not None:
            x = np.array([ada_2, ada_3])

            _, p_ada = proportions_ztest(x, n)
            # print("p-value for A. LASSO is: ", p_ada)
            if p_ada > sig_level:
                ada_output = "mix"
            elif x[0] / n[0] < x[1] / n[1]:
                ada_output = "3-int"
            elif x[0] / n[0] > x[1] / n[1]:
                ada_output = "pairwise"
            elif x[0] == 0 and x[1] == 0:
                ada_output = "nothing"
            ada_output2 = x[0] / n[0]
            ada_output3 = x[1] / n[1]
        else:
            ada_output = None
            ada_output2 = None
            ada_output3 = None

        all_rsts = [mle_output, mle_ori_output, lasso_output, ada_output, mle_output2, mle_ori_output2, lasso_output2,
                    ada_output2, mle_output3, mle_ori_output3, lasso_output3, ada_output3]
        return all_rsts

    # legacy codes, test for large number node computing
    def old_run(self, infer=False, phases_vec=None, mse=False, starts_from=0.0, inf_last=None, stretch=True):
        """
        phases_vec: 1D ndarray, optional
            States vector of nodes representing the position in radians.
            If not specified, random initialization [0, 2pi].

        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        """
        if phases_vec is None:
            phases_vec = self.init_phase

        demo_coup2 = self.coupling2
        demo_coup3 = self.coupling3
        for i in range(demo_coup2.shape[1]):
            demo_dict2 = {}
            for j in range(demo_coup2.shape[0]):
                if demo_coup2[j][i] != 0:
                    demo_dict2[f"k_{j + 1}{i + 1}"] = demo_coup2[j][i]
            # print(f"The 2-coupling coefficients of node {str(i + 1)} is: ", demo_dict2)

        for i in range(demo_coup3.shape[0]):
            demo_dict3 = {}
            for j in range(demo_coup3.shape[1]):
                for k in range(demo_coup3.shape[2]):
                    if demo_coup3[i][j][k] != 0:
                        demo_dict3[f"k_{j + 1}{k + 1}{i + 1}"] = demo_coup3[i][j][k]
            # print(f"The 3-coupling coefficients of node {str(i + 1)} is: ", demo_dict3)

        # print("================================================================================")

        if self.with_noise is False:
            result = self.integrate(phases_vec)
        else:
            result = self.sde_integrate(phases_vec)

        if result is None:
            raise Exception("Something goes wrong for the integration")

        if not infer and mse:
            raise ValueError("Need to open the 'infer' func in order to get MSE")

        if infer:
            something = self.infer_params_mle_coup(result, starts_from=starts_from, inf_last=inf_last)
            # for i in range(self.n_nodes):
            #     print(f"{str(i + 1)}: ", something[i + 1])
            if mse:
                sum__ = 0
                sum_ = 0
                avg2 = np.average(demo_coup2)
                # print(avg)
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i != j:
                            se_ = (demo_coup2[j][i] - avg2) ** 2
                            sum_ += se_
                            se = (demo_coup2[j][i] - something[i + 1]["k2"][f"k_{j + 1}{i + 1}"]) ** 2
                            sum__ += se
                mse_ = 1 / (self.n_nodes * (self.n_nodes - 1)) * sum__
                var = 1 / (self.n_nodes * (self.n_nodes - 1)) * sum_
                R2 = 1 - mse_ / var
                print("The MSE of connectivities-2 is: ", mse_)
                print("The coefficient of determination-2 is: ", R2)

                sum__ = 0
                sum_ = 0
                avg = np.average(demo_coup3)
                # print(avg)
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        for k in range(j, self.n_nodes):
                            if i != j and i != k and j != k:
                                se_ = (demo_coup3[i][j][k] - avg) ** 2
                                sum_ += se_
                                se = (demo_coup3[i][j][k] - something[i + 1]["k3"][f"k_{j + 1}{k + 1}{i + 1}"]) ** 2
                                sum__ += se
                mse_ = 1 / (self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2) * sum__
                var = 1 / (self.n_nodes * (self.n_nodes - 1) * (self.n_nodes - 2) / 2) * sum_
                R2 = 1 - mse_ / var
                print("The MSE of connectivity-3 is: ", mse_)
                print("The coefficient of determination-3 is: ", R2)
        else:
            something = None

        if stretch:
            index_start = int(starts_from * result.shape[1])
            if inf_last is not None:
                index_end = index_start + int(inf_last * result.shape[1])
            else:
                index_end = int(result.shape[1])
            result = result[:, index_start: index_end]
        return result, something

    def infer_params_mle_coup(self, results: np.array, starts_from=0.0, inf_last=None):
        """
        Testing, only work for coup_dim=2.
        :param inf_last: float64 or None
        :param starts_from: float64
        :param results: numpy.array
        :return: numpy.array of maximum likelihood estimators of nat_freq, coupling strength, and noise deviation
        """
        # assert self.with_noise is True, "Need to infer with noise"

        index_start = int(starts_from * results.shape[1])
        if inf_last is not None:
            index_end = index_start + int(inf_last * results.shape[1])
        else:
            index_end = int(results.shape[1])
        results = results[:, index_start: index_end]

        output = {}
        log_l = 0

        for j in range(results.shape[0]):
            # initial params
            N = results[j].size
            dt = self.dt
            dt_inverse = 1 / dt

            # create lst for every other node, and initialize the phase difference lst
            all_nodes = np.linspace(0, self.n_nodes - 1, self.n_nodes).astype(int)
            others = np.delete(all_nodes, j)
            phi_diff2 = np.zeros(shape=(self.n_nodes - 1, N - 1))

            # create lst for every other node, and initialize the phase difference lst
            # create j lst
            # all_nodes3 = np.linspace(0, results.shape[0] - 1, results.shape[0]).astype(int)
            # others3 = np.delete(all_nodes3, j)
            need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2)
            phi_diff3 = np.zeros(shape=(need_dim, N - 1))
            if self.type3 == 1:
                phi_diff3_sub = np.zeros(shape=(need_dim, N - 1))

            # lst of self phase
            self_phase = results[j]

            # lst of phase differences with respect to other nodes
            for i in others:
                if self.type2 == 1:
                    phi_diff_array = results[i] - self_phase
                elif self.type2 == 2:
                    phi_diff_array = 2 * (results[i] - self_phase)
                elif self.type2 == 0:
                    phi_diff_array = np.zeros(shape=results[i].size)
                else:
                    raise ValueError("'type' can only be 0, 1 or 2. ")
                if j > i:
                    phi_diff2[i] = phi_diff_array[:-1]
                elif j < i:
                    phi_diff2[i - 1] = phi_diff_array[:-1]

            # maybe better to create the dict from here
            # create k lst, and then calculate the phase difference lst
            counter = 0
            more_others_lst = []
            for i in others:
                if j < i:
                    more_others = others[i:]
                elif j > i:
                    more_others = others[i + 1:]
                else:
                    raise Exception("j shouldn't be i, something strange happened. ")
                if more_others.size == 0:
                    break
                more_others_lst.append(more_others)
                for kkk_ in more_others:
                    if self.type3 == 2:
                        phi_diff_array = results[i] + results[kkk_] - 2 * self_phase
                    elif self.type3 == 1:
                        phi_diff_array = 2 * results[i] - results[kkk_] - self_phase
                        phi_diff_array_sub = 2 * results[kkk_] - results[i] - self_phase
                    elif self.type3 == 0:
                        phi_diff_array = np.zeros(shape=results[i].size)
                    else:
                        raise ValueError("'type' can only be 0, 1 or 2. ")
                    phi_diff3[counter] = phi_diff_array[:-1]
                    phi_diff3_sub[counter] = phi_diff_array_sub[:-1]
                    counter += 1

            # lst of self phase differences
            self_diff = np.array([])
            for i_ in range(N - 1):
                self_diff = np.append(self_diff, self_phase[i_ + 1] - self_phase[i_])

            # sin the phase differences - 2
            two_lib = np.zeros(shape=(self.n_nodes - 1, N - 1))
            for i__ in range(phi_diff2.shape[0]):
                sin_diff2 = np.sin(phi_diff2[i__])
                two_lib[i__] = sin_diff2

            # sin the phase differences - 3
            three_lib = np.zeros(shape=(need_dim, N - 1))
            for i__ in range(need_dim):
                sin_diff3 = np.sin(phi_diff3[i__]) + np.sin(phi_diff3_sub[i__])
                three_lib[i__] = sin_diff3

            # solve the log-likelihood equations
            A = np.zeros(shape=(self.n_nodes + need_dim, self.n_nodes + need_dim))
            B = np.zeros(shape=self.n_nodes + need_dim)
            A[0, 0] = N
            B[0] = (self_phase[N - 1] - self_phase[0]) * dt_inverse
            for iii_ in range(1, self.n_nodes + need_dim):
                if iii_ < self.n_nodes:
                    A[iii_, 0] = np.sum(two_lib[iii_ - 1])
                    A[0, iii_] = np.sum(two_lib[iii_ - 1])
                    B[iii_] = np.sum(self_diff * two_lib[iii_ - 1]) * dt_inverse
                    for jjj_ in range(1, self.n_nodes):
                        A[iii_, jjj_] = np.sum(two_lib[iii_ - 1] * two_lib[jjj_ - 1])
                    for jjj_ in range(self.n_nodes, self.n_nodes + need_dim):
                        A[iii_, jjj_] = np.sum(two_lib[iii_ - 1] * three_lib[jjj_ - self.n_nodes])
                else:
                    A[iii_, 0] = np.sum(three_lib[iii_ - self.n_nodes])
                    A[0, iii_] = np.sum(three_lib[iii_ - self.n_nodes])
                    B[iii_] = np.sum(self_diff * three_lib[iii_ - self.n_nodes]) * dt_inverse
                    for jjj_ in range(1, self.n_nodes):
                        A[iii_, jjj_] = np.sum(three_lib[iii_ - self.n_nodes] * two_lib[jjj_ - 1])
                    for jjj_ in range(self.n_nodes, self.n_nodes + need_dim):
                        A[iii_, jjj_] = np.sum(three_lib[iii_ - self.n_nodes] * three_lib[jjj_ - self.n_nodes])
            X = np.linalg.solve(A, B)
            omega = X[0] % (2 * np.pi)
            k2 = X[1:self.n_nodes]
            k3 = X[self.n_nodes:]

            # sort terms needed for the standard deviation calc, and calc
            total_sm = np.zeros(N - 1)
            for kk_ in range(k2.size):
                something = k2[kk_] * two_lib[kk_]
                total_sm += something
            for kk_ in range(k3.size):
                something = k3[kk_] * three_lib[kk_]
                total_sm += something
            se = (self_diff - (omega + total_sm) * dt) ** 2
            variance = (1 / N * dt_inverse) * np.sum(se)
            sd = np.sqrt(variance)

            log_l += - np.sum(se) / (2 * variance * dt) - N / 2 * np.log(2 * np.pi * variance * dt)

            # make dict to sort the coupling terms
            k2_dict = {}
            for i in range(others.size):
                key = f"k_{str(others[i] + 1)}{str(j + 1)}"
                k2_dict[key] = k2[i]

            # make dict to sort the coupling terms
            k3_dict = {}
            counter = 0
            for i in range(others.size - 1):
                for kk__ in range(len(more_others_lst[i])):
                    key = f"k_{str(others[i] + 1)}{str(more_others_lst[i][kk__] + 1)}{str(j + 1)}"
                    k3_dict[key] = k3[counter]
                    counter += 1

            output[j + 1] = {"omega": omega, "k2": k2_dict, "k3": k3_dict, "standard deviation": sd}

        # # derive information criterions
        # k_ = (self.n_nodes + 1) * self.n_nodes
        # aic = 2 * k_ - 2 * log_l
        # bic = k_ * np.log((N - 1) * self.n_nodes) - 2 * log_l
        # output["aic"] = aic
        # output["bic"] = bic

        return output

    def old_solve_ada_lasso(self, starts_from=0.0, inf_last=None):
        act_mat, mle_or_ols_results = self.old_run(infer=True, mse=False, starts_from=starts_from, inf_last=inf_last)

        demo_coup2 = self.coupling2
        demo_coup3 = self.coupling3
        total_results = []

        for i in range(act_mat.shape[0]):
            test = act_mat[i]
            all_nodes = np.linspace(0, self.n_nodes - 1, self.n_nodes).astype(int)

            N = test.size
            others = np.delete(all_nodes, i)
            phi_diff2 = np.zeros(shape=(self.n_nodes - 1, N - 1))

            need_dim = int(((self.n_nodes - 1) * (self.n_nodes - 2)) / 2 + 1)
            phi_diff3 = np.zeros(shape=(need_dim - 1, N - 1))
            phi_diff3_sub = np.zeros(shape=(need_dim - 1, N - 1))

            # lst of self phase differences -- 2
            self_diff = np.array([])
            for i_ in range(N - 1):
                self_diff = np.append(self_diff, test[i_ + 1] - test[i_])
            self_diff = self_diff / self.dt

            # 2
            for j in others:
                if self.type2 == 1:
                    phi_diff_array2 = act_mat[j] - test
                elif self.type2 == 2:
                    phi_diff_array2 = 2 * (act_mat[j] - test)
                else:
                    raise ValueError("'type' can only be 1 or 2. ")
                if i > j:
                    phi_diff2[j] = phi_diff_array2[:-1]
                elif i < j:
                    phi_diff2[j - 1] = phi_diff_array2[:-1]

            # 3
            counter = 0
            more_others_lst = []
            for j in others:
                if i < j:
                    more_others = others[j:]
                elif i > j:
                    more_others = others[j + 1:]
                else:
                    raise Exception("i shouldn't be j, something strange happened. ")
                if more_others.size == 0:
                    break
                more_others_lst.append(more_others)
                for kkk_ in more_others:
                    if self.type3 == 2:
                        phi_diff_array3 = act_mat[j] + act_mat[kkk_] - 2 * test
                    elif self.type3 == 1:
                        phi_diff_array3 = 2 * act_mat[j] - act_mat[kkk_] - test
                        phi_diff_array3_sub = 2 * act_mat[kkk_] - act_mat[j] - test
                    else:
                        raise ValueError("'type' can only be 1 or 2. ")
                    phi_diff3[counter] = phi_diff_array3[:-1]
                    phi_diff3_sub[counter] = phi_diff_array3_sub[:-1]
                    counter += 1

            sin_diff_lib2 = np.zeros(shape=(self.n_nodes - 1, N - 1))
            for i__ in range(phi_diff2.shape[0]):
                sin_diff2 = np.sin(phi_diff2[i__])
                sin_diff_lib2[i__] = sin_diff2

            # sin the phase differences
            sin_diff_lib3 = np.zeros(shape=(need_dim - 1, N - 1))
            for i__ in range(need_dim - 1):
                sin_diff3 = np.sin(phi_diff3[i__]) + np.sin(phi_diff3_sub[i__])
                sin_diff_lib3[i__] = sin_diff3

            mle_estimates2 = ["Estimates-2 - MLE", mle_or_ols_results[i + 1]["omega"] % (2 * np.pi)]
            for j in range(self.n_nodes):
                if i != j:
                    mle_estimates2.append(mle_or_ols_results[i + 1]["k2"][f"k_{j + 1}{i + 1}"])
            mle_2 = np.array(mle_estimates2[2:])
            # twos = np.concatenate((twos, mle_2))

            mle_estimates3 = ["Estimates-3 - MLE"]
            for j in range(self.n_nodes):
                for k in range(j, self.n_nodes):
                    if i != j and i != k and j != k:
                        mle_estimates3.append(mle_or_ols_results[i + 1]["k3"][f"k_{j + 1}{k + 1}{i + 1}"])
            mle_3 = np.array(mle_estimates3[1:])
            # threes = np.concatenate((threes, mle_3))

            sin_diff_lib = np.concatenate((sin_diff_lib2, sin_diff_lib3), axis=0)

            # demo coupling
            others_lst = others.tolist()
            demo_lst2 = ["Real-2", self.natfreqs[i]]
            for j in others_lst:
                demo_lst2.append(demo_coup2[j][i])
            # two_real = np.array(demo_lst2[2:])
            # two_real[two_real != 0] = 1

            demo_lst3 = ["Real-3"]
            for j in others_lst:
                if j == others_lst[-1]:
                    break
                elif j < i:
                    for k in more_others_lst[j]:
                        demo_lst3.append(demo_coup3[i][j][k])
                else:
                    for k in more_others_lst[j - 1]:
                        demo_lst3.append(demo_coup3[i][j][k])
            # three_real = np.array(demo_lst3[1:])
            # three_real[three_real != 0] = 1

            new_X = np.zeros(shape=sin_diff_lib.shape)
            mle_2_array = abs(mle_2)
            mle_3_array = abs(mle_3)
            mle_array = np.append(mle_2_array, mle_3_array)
            assert new_X.shape[0] == len(mle_array), "something wrong"
            for iiii in range(len(mle_array)):
                new_X[iiii] = sin_diff_lib[iiii] * mle_array[iiii]
            new_X = new_X.T

            ada_reg = linear_model.LassoLarsIC(criterion="bic")
            ada_reg.fit(new_X, self_diff)
            two_conn_lst = ada_reg.coef_[:self.n_nodes - 1] * mle_2_array
            three_conn_lst = ada_reg.coef_[self.n_nodes - 1:] * mle_3_array

            results2 = two_conn_lst.tolist()
            results3 = three_conn_lst.tolist()

            if self.type2 == 3 and self.type3 == 3:
                omega = "No omega"
            else:
                omega = ada_reg.intercept_ % (2 * np.pi)
            alpha = ada_reg.alpha_
            results2.insert(0, omega)
            results2.insert(0, "Estimates-2 - Ada. LASSO")
            results3.insert(0, "Estimates-3 - Ada. LASSO")

            # print("The alpha is", alpha, "by bic. ")
            table2 = [demo_lst2, results2, mle_estimates2]
            table3 = [demo_lst3, results3, mle_estimates3]

            col_names2 = ["", "natural frequencies_" + str(i + 1)]
            counter_2 = 0
            for index in others_lst:
                name = "k_" + str(index + 1) + str(i + 1)
                col_names2.append(name)
                counter_2 += 1
            # print(tabulate(table2, headers=col_names2))
            # print("/////////////////////////////////////")

            col_names3 = [""]
            for index in others_lst:
                if index == others_lst[-1]:
                    break
                elif index < i:
                    for inde2 in more_others_lst[index]:
                        name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                        col_names3.append(name)
                else:
                    for inde2 in more_others_lst[index - 1]:
                        name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                        col_names3.append(name)
            # print(tabulate(table3, headers=col_names3))
            # print("====================================================================")

            results = {"2": two_conn_lst, "3": three_conn_lst, "natfreq": omega, "alpha": alpha}
            total_results.append(results)

        return total_results


def plot_activity(activity):
    """
    Plot sin(phase) vs time for each oscillator time series.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(12, 4))
    # ax = plt.subplots(figsize=(12, 4))
    # ax.plot(np.mod(activity.T, 2*np.pi))
    ax.plot(np.sin(activity.T))
    ax.set_xlabel('Time', fontsize=25)
    ax.set_ylabel(r'$\sin(\phi)$', fontsize=25)
    return ax


def plot_activity_lv(activity):
    """
    Plot sin(phase) vs time for each oscillator time series.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(8, 8))
    ax.plot(activity.T)
    ax.set_xlabel('Time', fontsize=25)
    ax.set_ylabel('$x$', fontsize=25)
    return ax


def plot_phase_coherence(activity, x_axis=None, coup2=None, coup3=None, v=None, starts_from=0.0, inf_last=None,
                         color=None, take_slice=0, outer_fig=None):
    """
    Plot order parameter phase_coherence vs time.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    if outer_fig is not None:
        fig, ax = outer_fig[0], outer_fig[1]
    else:
        fig, ax = plt.subplots(figsize=(16, 7), dpi=50)

    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman']
    # ax = plt.subplots(figsize=(8, 3))
    ops = [GeneralInteraction.phase_coherence(vec) for vec in activity.T]
    index_start = int(starts_from * len(ops))
    if inf_last is not None:
        index_end = index_start + int(inf_last * len(ops))
    else:
        index_end = int(len(ops))
    ops_avg = np.average(ops[index_start: index_end])
    if take_slice > 0:
        ops = ops[0::take_slice]
        x_axis = x_axis[0::take_slice]
    print("The average order parameter from the start of this plot is", ops_avg)
    # NUM_COLORS = 20
    # cm = plt.get_cmap('gist_rainbow')
    # ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    # for i in range(NUM_COLORS):
    #     ax.plot(np.arange(10) * (i + 1))
    # ax.plot(ops, 'o')
    if v is not None:
        i = 1
        for v_s in v:
            label_ = 'Snapshot No.' + str(i)
            plt.vlines(v_s, 0, 1, linestyles='--', color='red', label=label_)
            i += 1
    # plt.vlines(v, 0, 1, linestyles='--')
    # plt.legend()
    if x_axis is not None:
        ax.plot(x_axis, ops, linewidth=7.0, color=color)
    else:
        ax.plot(ops, linewidth=7.0, color=color)
    # params = {'legend.fontsize': 16,
    #           'legend.handlelength': 1}
    ax.set_ylabel('Order parameter', fontsize=20)
    if x_axis is not None:
        ax.set_xlabel('Time - T', fontsize=20)
    else:
        ax.set_xlabel('Timesteps', fontsize=20)
    if x_axis is not None:
        ax.set_xlim((x_axis[index_start], x_axis[-1]))
    else:
        ax.set_xlim((index_start, len(ops)))
    ax.set_ylim((-0.01, 1))
    if coup2 is not None and coup3 is not None:
        ax.set_title(f"$K_2={coup2}$, $K_3={coup3}$, $R_1 = {ops_avg}$")
    elif coup2 is not None:
        ax.set_title(f"$K_2={coup2}$, $R_1 = {ops_avg}$")
    elif coup3 is not None:
        ax.set_title(f"$K_3={coup3}$, $R_1 = {ops_avg}$")
    return ax, ops


def avg_phase_coherence(activity, how_last=0.8):
    """
    Return average R in certain part of time series

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    how_last: float64 or Decimal.decimal
        The portion which part of time-series starts at
    return: float64
        Average R
    """
    ops = [GeneralInteraction.phase_coherence(vec) for vec in activity.T]
    ops_min = int(how_last * len(ops))
    ops_avg = np.average(ops[ops_min:])
    return ops_avg


def plot_phase_coherence_3set_all(activity_2, activity_3, activity_mix, x_axis=None, coup2=None, coup3=None,
                                  coup2_mix=None, coup3_mix=None, v=None, starts_from=0.0, inf_last=None, take_slice=1,
                                  xticks=None):
    """
    Plot order parameter phase_coherence vs time.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    fig, axs = plt.subplots(3, figsize=(10, 12), dpi=100)
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman']
    # ax = plt.subplots(figsize=(8, 3))
    ops1 = [GeneralInteraction.phase_coherence(vec) for vec in activity_2.T]
    ops2 = [GeneralInteraction.phase_coherence(vec) for vec in activity_3.T]
    ops3 = [GeneralInteraction.phase_coherence(vec) for vec in activity_mix.T]
    assert len(ops1) == len(ops2) == len(ops3), "Boomba"
    index_start = int(starts_from * len(ops1))
    if inf_last is not None:
        index_end = index_start + int(inf_last * len(ops1))
    else:
        index_end = int(len(ops1))
    ops1_avg = np.average(ops1[index_start: index_end])
    ops2_avg = np.average(ops2[index_start: index_end])
    ops3_avg = np.average(ops3[index_start: index_end])
    if take_slice > 1:
        if type(x_axis[0]) == np.ndarray:
            x_axis[0] = x_axis[0][0::take_slice]
            x_axis[1] = x_axis[1][0::take_slice]
            x_axis[2] = x_axis[2][0::take_slice]
        elif x_axis is not None:
            x_axis = x_axis[0::take_slice]
        ops1_avg = np.average(ops1[index_start:])[0::take_slice]
        ops2_avg = np.average(ops2[index_start:])[0::take_slice]
        ops3_avg = np.average(ops3[index_start:])[0::take_slice]
    print("The pairwise average order parameter from the start of plot is", ops1_avg)
    print("The 3-interaction average order parameter from the start of plot is", ops2_avg)
    print("The mixed average order parameter from the start of plot is", ops3_avg)

    if type(x_axis[0]) == np.ndarray:
        assert len(x_axis) == 3, "x-axis Boomba"
        axs[0].plot(x_axis[0], ops1, label="Pairwise", color="blue", linewidth=7.0)
        axs[1].plot(x_axis[1], ops2, label="3-interaction", color="orange", linewidth=7.0)
        axs[2].plot(x_axis[2], ops3, label="Mixed", color="green", linewidth=7.0)
    elif x_axis is not None:
        axs[0].plot(x_axis, ops1, label="Pairwise", color="blue", linewidth=7.0)
        axs[1].plot(x_axis, ops2, label="3-interaction", color="orange", linewidth=7.0)
        axs[2].plot(x_axis, ops3, label="Mixed", color="green", linewidth=7.0)
    else:
        axs[0].plot(ops1, label="Pairwise", color="blue", linewidth=7.0)
        axs[1].plot(ops2, label="3-interaction", color="orange", linewidth=7.0)
        axs[2].plot(ops3, label="Mixed", color="green", linewidth=7.0)
    if type(v) == list:
        if type(v[0]) == list:
            axs[0].vlines(v[0], 0, 1, linestyles='--', color="red", linewidth=7.0)
            axs[1].vlines(v[1], 0, 1, linestyles='--', color="red", linewidth=7.0)
            axs[2].vlines(v[2], 0, 1, linestyles='--', color="red", linewidth=7.0)
        else:
            axs[0].vlines(v, 0, 1, linestyles='--', color="red", linewidth=7.0)
            axs[1].vlines(v, 0, 1, linestyles='--', color="red", linewidth=7.0)
            axs[2].vlines(v, 0, 1, linestyles='--', color="red", linewidth=7.0)
    params = {'legend.fontsize': 20,
              'legend.handlelength': 3}
    plt.rcParams.update(params)
    axs[0].set_ylabel('Order parameter', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].set_ylim((-0.01, 1))
    if type(x_axis[0]) == np.ndarray:
        axs[0].set_xlim((x_axis[0][index_start], x_axis[0][-1]))
    elif x_axis is not None:
        axs[0].set_xlim((x_axis[index_start], x_axis[-1]))
    else:
        axs[0].set_xlim((index_start, len(ops1)))
    axs[0].legend()
    if coup2 is not None:
        axs[0].set_title(f"$K_2={coup2}$, $R_1={ops1_avg}$", fontsize=20)
    axs[1].set_ylabel('Order parameter', fontsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].set_xlabel('', fontsize=20)
    axs[1].set_ylim((-0.01, 1))
    if type(x_axis[0]) == np.ndarray:
        axs[1].set_xlim((x_axis[1][index_start], x_axis[1][-1]))
    elif x_axis is not None:
        axs[1].set_xlim((x_axis[index_start], x_axis[-1]))
    else:
        axs[1].set_xlim((index_start, len(ops2)))
    axs[1].legend()
    if coup3 is not None:
        axs[1].set_title(f"$K_3={coup3}$, $R_1={ops2_avg}$", fontsize=20)
    axs[2].set_ylabel('Order parameter', fontsize=20)
    axs[2].tick_params(axis='both', which='major', labelsize=20)
    if x_axis is not None:
        axs[2].set_xlabel('Time - T', fontsize=20)
    else:
        axs[2].set_xlabel('Timesteps', fontsize=20)
    axs[2].set_ylim((-0.01, 1))
    if type(x_axis[0]) == np.ndarray:
        axs[2].set_xlim((x_axis[2][index_start], x_axis[2][-1]))
    elif x_axis is not None:
        axs[2].set_xlim((x_axis[index_start], x_axis[-1]))
    else:
        axs[2].set_xlim((index_start, len(ops3)))
    axs[2].legend()
    if coup2_mix is not None and coup3_mix is not None:
        axs[2].set_title(f"$K_2={coup2_mix}$, $K_3={coup3_mix}$, $R_1={ops3_avg}$", fontsize=20)

    for ax_ in axs:
        x_left, x_right = ax_.get_xlim()
        y_low, y_high = ax_.get_ylim()
        ax_.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.33)

    if xticks is not None:
        axs[0].set_xticks(xticks)
        axs[1].set_xticks(xticks)
        axs[2].set_xticks(xticks)
    return None


def Fig1_demo(model_: GeneralInteraction, kind="", print_result=False, mle_ori=False, mle_threshold=0.1):
    mle_or_ols_results_, lasso_results_, ada_results_ = model_.demo_solve(mle=True, ada=True, lasso=True,
                                                                          print_result=print_result)
    main_results_ = model_.conn_criteria_base(mle_or_ols_results_, lasso_results_, ada_results_, mle_ori=mle_ori,
                                              mle_threshold=mle_threshold)
    print(kind + ": ")
    print("New rate for MLE:", model_.new_rates_easy(main_results_["mle"]["A"], main_results_["mle"]["B"]))
    print("FPR for MLE:", GeneralInteraction.FPR_easy(main_results_["mle"]["FP"], main_results_["mle"]["TN"]))
    print("FNR for MLE:", GeneralInteraction.FNR_easy(main_results_["mle"]["FN"], main_results_["mle"]["TP"]))
    print("MCC for MLE:", GeneralInteraction.MCC_easy(main_results_["mle"]["TP"], main_results_["mle"]["TN"],
                                                      main_results_["mle"]["FP"], main_results_["mle"]["FN"]))
    print("New rate for LASSO:", model_.new_rates_easy(main_results_["lasso"]["A"], main_results_["lasso"]["B"]))
    print("FPR for LASSO:",
          GeneralInteraction.FPR_easy(main_results_["lasso"]["FP"], main_results_["lasso"]["TN"]))
    print("FNR for LASSO:",
          GeneralInteraction.FPR_easy(main_results_["lasso"]["FN"], main_results_["lasso"]["TP"]))
    print("MCC for LASSO:",
          GeneralInteraction.MCC_easy(main_results_["lasso"]["TP"], main_results_["lasso"]["TN"],
                                      main_results_["lasso"]["FP"], main_results_["lasso"]["FN"]))
    print("New rate for Adaptive LASSO:", model_.new_rates_easy(main_results_["ada"]["A"], main_results_["ada"]["B"]))
    print("FPR for Adaptive LASSO:",
          GeneralInteraction.FPR_easy(main_results_["ada"]["FP"], main_results_["ada"]["TN"]))
    print("FNR for Adaptive LASSO:",
          GeneralInteraction.FPR_easy(main_results_["ada"]["FN"], main_results_["ada"]["TP"]))
    print("MCC for Adaptive LASSO:",
          GeneralInteraction.MCC_easy(main_results_["ada"]["TP"], main_results_["ada"]["TN"],
                                      main_results_["ada"]["FP"], main_results_["ada"]["FN"]))


def type_test_demo(model_: GeneralInteraction, print_re=False, mle=True, ada=True, lasso=True, mle_ori=False,
                   mle_threshold=0.1):
    mle_or_ols_results_, lasso_results_, ada_results_ = model_.demo_solve(mle=mle, ada=ada, lasso=lasso,
                                                                          print_result=False)
    all_results = model_.conn_type_test(mle_or_ols_results_, lasso_results_, ada_results_, mle_ori=mle_ori,
                                        mle_threshold=mle_threshold)
    if print_re:
        print("MLE result:", all_results[0])
        print("MLE_ori result:", all_results[1])
        print("LASSO result:", all_results[2])
        print("Adative LASSO result:", all_results[3])

    return [all_results[0], round(all_results[4], 5), round(all_results[8], 5)], \
        [all_results[2], round(all_results[6], 5), round(all_results[10], 5)], \
        [all_results[3], round(all_results[7], 5), round(all_results[11], 5)], \
        [all_results[1], round(all_results[5], 5), round(all_results[9], 5)]
