# Spatial Distortion of Statistical Homogeneity
## L2 Norm Domination, Metric Illusions, and the Ineffectiveness of Smoothing in Compositional Text Mining

This repository provides the core computational codebase and empirical validation scripts for the theoretical framework introduced in the paper:
**"Spatial Distortion of Statistical Homogeneity: L2 Norm Domination, Metric Illusions, and the Ineffectiveness of Smoothing in Compositional Text Mining"**

The provided scripts execute large-scale simulations ($10^3$ to $10^7$ trials) to mathematically and empirically demonstrate the topological collapse of the cosine metric under heavy-tailed (Zipfian) variance, the occurrence of the "Evaluation Illusion" (hacking of the Silhouette score), the failure of standard preprocessing heuristics (Top-N truncation and logarithmic smoothing), and the implementation of the absolute **Topological Sanity Check** via Persistent Homology and Aitchison geometry.

---

## 🛠 Prerequisites and Installation

The codebase relies on Python 3 and several foundational libraries for topological data analysis and manifold learning.

```bash
pip install numpy pandas scipy scikit-learn matplotlib
pip install umap-learn
pip install ripser persim

📁 Repository Structure and Execution SequenceThe validation framework is divided into modular scripts, each designed to empirically prove a specific geometric fallacy or validate the proposed Topological Sanity Check.1. H1_topological_collapse.pyObjective: Validates the phase transition of topological collapse under the cosine metric.Mechanism: Computes the Wasserstein distances ($W_{H_0}$, $W_{H_1}$) between the 0-th and 1-st Persistence Diagrams of a hub-injected skewed space and a theoretically uniform null space.2. H2_global_rank_decoupling.pyObjective: Demonstrates the degradation of global distance meaningfulness.Mechanism: Tracks the asymptotic collapse of Spearman's rank correlation ($\rho \to 0.056$) between high-dimensional cosine distances and low-dimensional embeddings as dimensionality scales.3. H3_local_trust_destruction.pyObjective: Exposes the emergence of artificial gravitational black holes (Hubs) and the destruction of local neighborhoods.Mechanism: Computes Hubness Skewness and the drop in local Trustworthiness scores, proving that $k$-NN structures are hijacked by extreme variance.4. H4_ineffectiveness_of_truncation.pyObjective: Proves the fractal invariance of Zipfian variance and the failure of Top-N frequency cutoffs (e.g., stopword removal).Mechanism: Simulates the removal of the top 5% of dimensions, demonstrating that residual hubness skewness ($\sim 1.85$) and topological fragmentation persist.5. H5_metric_comparison.pyObjective: Contrasts the catastrophic failure of the cosine metric against the geometric preservation of the proposed Aitchison distance.Mechanism: Employs the Centered Log-Ratio (CLR) transformation to neutralize multiplicative shocks into additive bounds, evaluating the structural integrity via Wasserstein $W_{H_0}$.6. H6_tda_sanity_check.pyObjective: Operationalizes the Topological Sanity Check framework.Mechanism: Demonstrates that while UMAP density-based reduction visually fabricates Chimera clusters, the exact computation of the Chimera Gap ($\Gamma_C$) in high-dimensional Aitchison space deterministically unmasks the algorithmic illusion.7. Validation_evaluation_illusion.pyObjective: Exposes the vulnerability of internal, rank-based clustering evaluation metrics.Mechanism: Computes the Silhouette Score on anomaly-injected data to demonstrate that $L_2$ norm destruction artificially inflates the metric (e.g., $0.831$), generating a dangerous false positive for clustering quality.8. Validation_smoothing_and_minority.pyObjective: Proves the ineffectiveness of univariate logarithmic smoothing (TF-IDF style) and validates the Diversity Protection capability of the TDA framework.Mechanism:Shows that UMAP visual distances still collapse to zero despite log-smoothing.Computes the Topological SNR on highly imbalanced populations (5% vs 95%) with disjoint absolute magnitudes, proving that the TDA audit strictly isolates compositional structure without generating scale-induced false positives.🚀 Quick Start Example: TDA Sanity CheckBelow is a minimal reproducible snippet demonstrating the core diagnostic mechanism: extracting the exact Chimera Gap ($\Gamma_C$) from a mathematically compromised space using Aitchison geometry and ripser.Pythonimport numpy as np
from sklearn.metrics import pairwise_distances
from ripser import ripser

def clr_transform(X, epsilon=1e-10):
    """Centered Log-Ratio transformation for Aitchison geometry."""
    X_eps = X + epsilon
    log_X = np.log(X_eps)
    return log_X - np.mean(log_X, axis=1, keepdims=True)

def compute_chimera_gap(X_compositional):
    """
    Computes the Chimera Gap (\Gamma_C) using 0-th Betti number (H0) 
    persistence in the Aitchison space.
    """
    # 1. Transform to Aitchison Euclidean space
    dist_aitchison = pairwise_distances(clr_transform(X_compositional), metric='euclidean')
    
    # 2. Extract Persistent Homology
    ph_res = ripser(dist_aitchison, distance_matrix=True, maxdim=1)['dgms']
    
    # 3. Isolate H0 deaths (excluding infinity)
    h0_deaths = ph_res[0][:, 1]
    finite_h0_deaths = np.sort(h0_deaths[h0_deaths < np.inf])
    
    # 4. The maximum finite death scale represents the Chimera Gap
    chimera_gap = finite_h0_deaths[-1] if len(finite_h0_deaths) > 0 else 0.0
    return chimera_gap

# --- Example Usage ---
# X_matrix represents a row-normalized compositional frequency matrix
# gap = compute_chimera_gap(X_matrix)
# print(f"Topological Chimera Gap: {gap:.3f}")
📜 CitationIf you utilize this framework or codebase in your research, please cite the associated paper:コード スニペット@article{spatial_distortion_2026,
  title={Spatial Distortion of Statistical Homogeneity: $L_2$ Norm Domination, Metric Illusions, and the Ineffectiveness of Smoothing in Compositional Text Mining},
  author={[Author Name]},
  journal={[Journal Name]},
  year={2026}
}
