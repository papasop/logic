 # spectral_logic_system_v3.py (结构化谱逻辑反馈验证系统)

import cmath
import math
import random
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Config:
    num_nodes = 30
    max_out_degree = 4
    alpha = 0.5
    noise = 0.3
    freq_count = 100
    cluster_k = 5
    recovery_alpha = 0.2
    weight_scale = 1.0
    recovery_trials = 20
    recovery_subset = 100
    keep_ratio = 0.9
    feedback_iters = 10


class DAGBuilder:
    @staticmethod
    def generate_paths(cfg: Config):
        G = nx.DiGraph()
        G.add_nodes_from(range(cfg.num_nodes))
        for i in range(cfg.num_nodes):
            targets = random.sample(range(i + 1, cfg.num_nodes), min(cfg.max_out_degree, cfg.num_nodes - i - 1))
            for t in targets:
                G.add_edge(i, t)
        paths = list(nx.all_simple_paths(G, source=0, target=cfg.num_nodes - 1))
        return G, [tuple(p) for p in paths]


class SpectralAmplitudeModel:
    @staticmethod
    def construct(paths, cfg: Config):
        psi = {}
        for p in paths:
            base_amp = 1 / (1 + len(p))
            theta = cfg.alpha * len(p) + random.uniform(-cfg.noise, cfg.noise)
            psi[p] = base_amp * cmath.exp(1j * theta)
        return psi


class SpectralClustering:
    @staticmethod
    def cluster(psi_dict, cfg: Config):
        F = np.zeros(cfg.freq_count, dtype=complex)
        for p, ψ in psi_dict.items():
            n = len(p)
            for f in range(cfg.freq_count):
                F[f] += ψ * cmath.exp(-2j * np.pi * f * n / cfg.freq_count)
        spectrum = np.abs(F)
        labels = KMeans(n_clusters=cfg.cluster_k, n_init=10, random_state=42).fit_predict(spectrum.reshape(-1, 1))
        cluster_means = {
            i: np.mean([spectrum[j] for j in range(len(spectrum)) if labels[j] == i])
            for i in range(cfg.cluster_k)
        }
        scores = {p: cluster_means[labels[len(p) % cfg.freq_count]] for p in psi_dict}
        return spectrum, labels, scores


class SpectralRecoveryEngine:
    def __init__(self, psi_crash, psi_ref, scores, cfg: Config):
        self.current = psi_crash.copy()
        self.psi_ref = psi_ref
        self.scores = scores
        self.cfg = cfg
        self.entropy_trace = []

    def compute_entropy(self, psi):
        if not psi:
            return 0.0
        norm_sq = sum(abs(v) ** 2 for v in psi.values())
        if norm_sq == 0:
            return 0.0
        return -sum((abs(v) ** 2 / norm_sq) * math.log(abs(v) ** 2 / norm_sq + 1e-12) for v in psi.values())

    def not_gate(self, psi):
        norm_sq = sum(abs(v)**2 for v in psi.values())
        contrib = {
            p: (abs(v)**2 / norm_sq) * math.log(abs(v)**2 / norm_sq + 1e-12)
            for p, v in psi.items()
        }
        sorted_p = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
        cutoff = int(len(sorted_p) * (1 - self.cfg.keep_ratio))
        keep = set(p for p, _ in sorted_p[:cutoff])
        kept = {p: psi[p] for p in psi if p in keep}
        norm = math.sqrt(sum(abs(v)**2 for v in kept.values()))
        return {p: v / norm for p, v in kept.items()} if norm > 0 else psi

    def resonance_weighted_recovery(self, subset_ref):
        recovered = {}
        for p in self.current:
            if p in subset_ref:
                score = self.scores.get(p, 0.0)
                w = 1 + self.cfg.weight_scale * score
                amp_diff = abs(subset_ref[p]) - abs(self.current[p])
                phase_diff = cmath.phase(subset_ref[p]) - cmath.phase(self.current[p])
                adj_amp = abs(self.current[p]) + self.cfg.recovery_alpha * amp_diff * w
                recovered[p] = adj_amp * cmath.exp(1j * (cmath.phase(self.current[p]) + phase_diff * w))
            else:
                recovered[p] = self.current[p]
        return recovered

    def run(self):
        for i in range(self.cfg.feedback_iters):
            H_before = self.compute_entropy(self.current)
            best_H = float('inf')
            best_state = self.current
            for _ in range(self.cfg.recovery_trials):
                subset = random.sample(list(self.psi_ref), min(self.cfg.recovery_subset, len(self.psi_ref)))
                subset_ref = {p: self.psi_ref[p] for p in subset}
                recovered = self.resonance_weighted_recovery(subset_ref)
                H = self.compute_entropy(recovered)
                if H < best_H:
                    best_H = H
                    best_state = recovered
            ΔH = H_before - best_H
            print(f"[Round {i+1}] H_before = {H_before:.4f}, H_after = {best_H:.4f}, ΔH = {ΔH:.4f}")
            if ΔH > 0.002:
                self.current = best_state
            else:
                self.current = self.not_gate(self.current)
                print("→ Applied NOT gate")
            self.entropy_trace.append(self.compute_entropy(self.current))
        return self.entropy_trace


class Visualizer:
    @staticmethod
    def plot_all(entropy_trace, spectrum, labels):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(entropy_trace, marker='o', color='teal')
        plt.title("Entropy Evolution (Greedy Recovery)")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(spectrum)), spectrum, c=labels, cmap='tab10', s=40)
        plt.title("Fourier Spectrum Clustering")
        plt.xlabel("Frequency Index")
        plt.ylabel("|ψ̂(f)|")

        plt.tight_layout()
        plt.show()


# 主运行入口
if __name__ == "__main__":
    cfg = Config()
    G, paths = DAGBuilder.generate_paths(cfg)
    print(f"构建 DAG: {len(paths)} 条路径")
    psi = SpectralAmplitudeModel.construct(paths, cfg)
    spectrum, labels, scores = SpectralClustering.cluster(psi, cfg)
    engine = SpectralRecoveryEngine(psi, psi, scores, cfg)
    trace = engine.run()
    Visualizer.plot_all(trace, spectrum, labels)
