import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objs as go

pio.renderers.default = "notebook"
# sns.set()
import random
import numpy as np
from torch.nn.functional import conv1d
import plotly.express as px
from tqdm import tqdm
import time


def generate_jump(p, s, Norm):
    ### Generate Delta, the direction of the change point located at tau1 ###

    Delta = 2 * torch.bernoulli(torch.tensor(p * [1 / 2])).reshape(-1, 1) - 1

    ### Sparsify Delta
    Delta[torch.randperm(p)[: p - s]] = 0
    return (Norm / Delta.norm()) * Delta


def generate_wcs(p, n, s, Norm, r=None, taus=None, samples=1):
    """generates worst case shape of time series, that is with two change-points."""
    for sample in range(samples):
        ### Choose the scale r randomly ###
        r = int(torch.randint(1, n // 2, size=(1,))) if not r else r

        ### Choose the location of the first change point ###
        if not taus:
            t1 = torch.randint(1, n - r - 1, size=(1,)).item()
            tau1, tau2 = (t1, t1 + r)
        else:
            tau1, tau2 = taus
        # print(f'generated worst case signal with cp at positions {(tau1, tau2)}\r')

        Delta = generate_jump(p, s, Norm)
        Theta = torch.zeros((samples, p, n))
        Theta[sample, :, tau1:tau2] = Delta
    return Theta


def generate_K(p, n, MinNorm, MaxNorm, K, samples=1):
    Theta = torch.zeros(size=(samples, p, n))  # samples is for faster monte carlo
    change_points = torch.randperm(n)[:K].sort()[0]
    for tau in change_points:
        for sample in range(samples):
            Norm = (MaxNorm - MinNorm) * np.random.random() + MinNorm
            s = np.random.randint(1, p + 1)
            Theta_single_cp = torch.zeros_like(Theta)
            Theta_single_cp[sample, :, tau:] = generate_jump(p, s, Norm)
            Theta.add_(Theta_single_cp)
    return Theta, change_points.tolist()


def ending(r):
    return None if r == 1 else -r + 1


def logs_0(p, n, r, delta):
    return np.maximum(int(np.log2(np.log2(n / (r * delta)))), 1)


def logs_m(p, n, r, delta):
    gamma_r = np.log2(n / (r * delta))
    logsm = np.maximum(
        int(np.log2(np.sqrt(p * gamma_r) / (np.maximum(1, np.log2(p) - gamma_r)))), 1
    )
    return np.minimum(logsm, int(np.log2(p)))


def generate_grid(p, n, delta=0.05):
    # CompleteGrid = list(range(n//2))
    SemiDyadicGrid = [2**i for i in range(int(np.log2(n)))]
    Grid = {}
    for r in SemiDyadicGrid:
        Grid[r] = [2**i - 1 for i in range(logs_m(p, n, r, delta) + 1)]
        if 2 ** logs_m(p, n, r, delta) != p:
            Grid[r].append(p - 1)
    return Grid


def compute_cusums(Ys, r):
    samples, p, n = Ys.shape
    Weights = torch.zeros((1, 1, r)) + 1
    Convolutions = conv1d(Ys.reshape(samples * p, 1, n), Weights).reshape(
        samples, p, -1
    )
    ConvolutionsFilled = torch.zeros_like(Ys)
    end = ending(r)
    ConvolutionsFilled[:, :, :end] = Convolutions
    Convolutions = ConvolutionsFilled
    # Cusums = CusumsFilled
    Cusums = torch.zeros_like(Convolutions)
    Cusums[:, :, r:end] = (
        Convolutions[:, :, r:end] - Convolutions[:, :, : -2 * r + 1]
    ) / np.sqrt(2 * r)
    return Cusums


def compute_statistics_r(Y, r, Grid):
    Cusums = compute_cusums(Y, r)
    CusumsSquared = (Cusums**2).sort(dim=1, descending=True)[
        0
    ]  # sort along dimension R^p
    PartialNorms = CusumsSquared.cumsum(dim=1)
    Stats = PartialNorms[:, Grid[r], :]
    return Stats


def compute_statistics(Y, Grid, showbar=False):
    Stats = {}
    for r in Grid:
        if showbar:
            print(f"computing stats for r = {r}         ", end="\r")
        Stats[r] = compute_statistics_r(Y, r, Grid)
    return Stats


def compute_thresholds_r(
    Grid, p, n, r, delta=0.05, batch=50, samples=100, showbar=True
):
    Thresholds_r_batches = []
    for i in range(samples // batch):
        if showbar:
            print(f"r = {r} ; doing batch {i + 1}/{samples//batch}     ", end="\r")
        Noise = torch.randn((batch, p, n))
        Stats_r = compute_statistics_r(Noise, r, Grid)
        Expects_r = Stats_r[:, :, r : ending(r)].mean(dim=(0, 2))
        delta_rs = delta / (2 * len(Grid) * len(Grid[r]))
        Thresholds_r_batches.append(Stats_r[:, :, r : ending(r)].max(dim=2)[0])
    Thresholds_concat = torch.concat(Thresholds_r_batches)
    Thresholds_r = Thresholds_concat.quantile(1 - delta_rs, dim=0)
    Thresholds_r[-1] = Thresholds_concat[:, -1].quantile(
        1 - delta / (2 * len(Grid)), dim=0
    )
    return Thresholds_r, Expects_r


def compute_thresholds(Grid, p, n, delta=0.05, batch=50, samples=100, showbar=True):
    Thresholds = {}
    Expects = {}
    for r in Grid:
        Thresholds[r], Expects[r] = compute_thresholds_r(
            Grid, p, n, r, delta=delta, batch=batch, samples=samples, showbar=True
        )
    return Thresholds, Expects


class ChangePointDetector:
    def __init__(
        self, p, n, delta=0.05, constants=None, samples=100, batch=20, showbar=False
    ):
        self.Grid = generate_grid(p, n)
        self.showbar = showbar
        if constants is None:  # We compute the grid of constants with montecarlo
            # Noise = torch.randn((samples, p, n))
            self.Thresholds = compute_thresholds(
                self.Grid, p, n, delta=delta, samples=samples, batch=batch, showbar=True
            )[0]
        elif len(constants) == 3:
            self.Thresholds = {}
            for r in self.Grid:
                for logs in range(len(self.Grid)):
                    verysparse = (
                        torch.Tensor(
                            logs_0(p, n, r, delta) * [np.log2(n / (r * delta))]
                        )
                        * constants[0]
                    )
                    s = torch.Tensor(self.Grid[r][logs_0(p, n, r, delta) : -1])
                    sparse = s * np.log2(2 * p / s) * constants[1] + s
                    dense = (
                        torch.Tensor([np.sqrt(p * np.log2(n / (r * delta)))])
                        * constants[2]
                        + p
                    )
                    self.Thresholds[r] = torch.concat((verysparse, sparse, dense))

    def fit(self, Y):
        _, p, n = Y.shape
        Stats = compute_statistics(Y, self.Grid, showbar=self.showbar)
        S_left, S_right = [], []
        S = {}
        for r in self.Grid:
            S[r] = []
            S_left_r, S_right_r = [], []
            right = -1
            for l in range(r, n - r):
                if not torch.any(
                    (
                        (torch.Tensor(S_right) >= l - r + 1).bool()
                        & (torch.Tensor(S_right) <= l + r - 1).bool()
                    )
                    | (
                        (torch.Tensor(S_left) >= l - r + 1).bool()
                        & (torch.Tensor(S_left) <= l + r - 1).bool()
                    )
                ):
                    Decision = Stats[r][0, :, l] / self.Thresholds[r]
                    if torch.max(Decision) > 1:
                        logsparsity = next((i for i, v in enumerate(Decision > 1) if v))
                        if logsparsity == len(self.Grid[r]) - 1:
                            sparsity = p
                        else:
                            sparsity = 2**logsparsity
                        if l - r + 1 > right:
                            if right != -1:
                                S_left_r.append(left)
                                S_right_r.append(right)
                                S[r].append((left, right, sparsity))
                            left = l - r + 1
                        right = l + r - 1
            if right != -1:  # add the last change-point detected
                S_left_r.append(left)
                S_right_r.append(right)
                S[r].append((left, right, sparsity))
            S_left.extend(S_left_r)
            S_right.extend(S_right_r)
        self.left = S_left
        self.right = S_right
        self.S = S
        self.Stats = Stats

        self.tau = []
        self.scales = []
        self.sparsities = []
        for r in S:
            for (left, right, s) in S[r]:
                self.tau.append((left + right) // 2)
                self.scales.append(r)
                self.sparsities.append(s)
        if self.tau:
            self.tau, self.scales, self.sparsities = zip(
                *sorted(zip(self.tau, self.scales, self.sparsities))
            )
        else:
            self.tau, self.scales, self.sparsities = ((), (), ())
