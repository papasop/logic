# Spectral Logic Feedback System

A structure-aware logic simulation framework that implements entropy-driven feedback, resonance-based clustering, and greedy RECOVERY optimization on DAG-structured path systems.

---

## 🔬 Overview

This system implements and validates the full feedback loop of a spectral logic framework, including:

- ✅ Directed acyclic graph (DAG) path generation
- ✅ Complex-valued amplitude modeling (`ψ(p)`)
- ✅ Fourier projection + KMeans clustering
- ✅ Resonance-based path scoring
- ✅ Entropy computation and feedback control
- ✅ Greedy RECOVERY gate with adaptive subset selection
- ✅ NOT gate for entropy-based path pruning
- ✅ Visualization of entropy evolution and spectral clustering

---

## 📁 Files

| File | Description |
|------|-------------|
| `gate.py` | Main Python module: self-contained simulation and plotting |
| `README.md` | This documentation |
| *(Optional)* `figures/` | Output plots (entropy curves, cluster maps) |
| *(Optional)* `data/` | Stored results for batch experiments |

---

## ⚙️ Installation

```bash
# Recommended Python 3.8+
pip install numpy matplotlib scikit-learn networkx
