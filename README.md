# Doppler Law Simulator in MoCap-to-Radar — reproduction code

## Setup

```bash
pip install torch numpy scipy pandas scikit-learn pyyaml tqdm matplotlib huggingface_hub   # tested: python 3.10, torch 2.9.0
python -c "from huggingface_hub import snapshot_download; snapshot_download('ACICULA/mocap2radar', repo_type='dataset', revision='9ff81180fab91238a0125d6e0befe749b52337a1', local_dir='data')"
python scripts/preprocess.py        # data/raw/v3 -> data/processed
```

The ST(Flat) checkpoint (`BaselineMeanPoolAgg` in `src/model/models.py`) is in `models/`
(`baseline_mean_pool_agg.pt`, `model_params.json`, `SHA256SUMS`). All evaluations use the RandomWalk1
recording (2416 windows).

## Code layout

| Part | Scripts | Library |
|---|---|---|
| Data and model | `scripts/preprocess.py` | `src/data/`, `src/model/` |
| 1. Doppler consistency score (DCS) | `scripts/dcs/` | `src/dcs/` |
| 2. Circuit discovery with DCS | `scripts/circuit/` | `src/patching/`, `src/circuit/` |
| 3. Probing and targeted erasure | `scripts/erasure/` | `src/erasure/` |

`src/dcs/` holds the definitions every part uses: the velocity perturbation (`perturb.py`), the Doppler
centroid and the consistency score $M_{\mathcal P}$ (`metric.py`), and the physics model $g$ (`kinematic.py`).

## 1. Doppler consistency score

| Script | What it does |
|---|---|
| `scripts/dcs/compute_m0.py` | $M_{\mathcal P}(f, g)$ of the unpatched model: Doppler centroids of $f(x_\alpha)$ over an $\alpha$ grid, the physics model's slopes $\beta_g(\alpha)$, and the score over each `--spec` alpha set |

```bash
python scripts/dcs/compute_m0.py --alpha-range -1.16 1.16 0.01 --spec paper=-1.16:1.16:0.01~1 --out results/dcs
```

$\alpha \in [-1.16, 1.16]$, step 0.01, $\alpha = 1$ excluded. Output: `results/dcs/paper.json` (`consistency_score` 0.986168).

## 2. Circuit discovery with DCS

The 18 patchable components $U$ are the 8 heads and the MLP of the last spatial and the last temporal layer
(`src/patching/components.py`; code names `h_s_0..7`, `mlp_s`, `h_t_0..7`, `mlp_t`). Patching a set $C$
(`src/patching/hooks.py`) replaces its activations in the forward pass on $x$ with those on $x_\alpha$, and
$M(C)$ is the consistency score of the patched model. A circuit is $\tau$-consistent when patching $C$ recovers
the score and patching $U \setminus C$ does not: $M_{\mathcal P} - M(C) \le (1-\tau) M_{\mathcal P}$ and
$M_{\mathcal P} - M(U \setminus C) \ge \tau M_{\mathcal P}$.

**Search** ($\alpha \in \{0, 0.1, \dots, 0.9, 1.1\}$)

| Script | What it does |
|---|---|
| `scripts/dcs/compute_m0.py` | $M_{\mathcal P}(f, g)$ and $\beta_g$ over the search's alpha grid |
| `scripts/circuit/build_cache.py` | inputs, component activations and clean centroids for every window and alpha (about 580 GB) |
| `scripts/circuit/run_search.py` | greedy minimal $\tau$-consistent circuit for $\tau = 0.3, \dots, 0.9$, then source-randomized controls (10 seeds) |
| `scripts/circuit/make_table.py` | circuit table: $\rho_C$, $\rho_C^{\mathrm{rand}}$, $\rho_{U \setminus C}$, $\rho_{U \setminus C}^{\mathrm{rand}}$ with $\rho = M / M_{\mathcal P}$ |

```bash
python scripts/dcs/compute_m0.py --alpha-range 0 1.1 0.1 --spec search_grid=0:1.1:0.1~1 --out results/dcs
python scripts/circuit/build_cache.py --cache-dir <cache>
python scripts/circuit/run_search.py --cache-dir <cache> --m0 results/dcs/search_grid.json
python scripts/circuit/make_table.py
```

Output: `results/circuit/search.json`, `results/circuit/table_circuits.csv`.

**The $\tau = 0.9$ circuit $C^\dagger$** = T:{MLP, h1, h2, h6, h7}, S:{MLP} (`mlp_t h_t_1 h_t_2 h_t_6 h_t_7 mlp_s`).
These scripts patch on the fly and need no cache.

| Script | What it does |
|---|---|
| `scripts/circuit/compute_phys.py` | physics model $g$ under radial and tangential-pure scaling, $\alpha \in [-2, 2]$ |
| `scripts/circuit/run_sweep.py` | slopes of $f$, $C^\dagger$ and $U \setminus C^\dagger$ under the same scalings |
| `scripts/circuit/circuit_score.py` | $\rho_{C^\dagger}$ and $\rho_{U \setminus C^\dagger}$ from the sweep, search alpha grid |
| `scripts/circuit/plot_main.py` | domain-of-validity figures (radial, tangential-pure) |
| `scripts/circuit/dump_spectrograms.py`, `plot_spectrograms.py` | spectrogram figure at $\alpha = 0.3$: radar, $f(x)$, $f(x_\alpha)$, $C^\dagger$, $U \setminus C^\dagger$ |

```bash
C="--components mlp_t h_t_1 h_t_2 h_t_6 h_t_7 mlp_s"
python scripts/circuit/compute_phys.py --out results/dov
python scripts/circuit/run_sweep.py $C --out results/dov
python scripts/circuit/circuit_score.py --sweep results/dov/sweep.json --phys results/dov/phys.json
python scripts/circuit/plot_main.py --mode radial --sweep results/dov/sweep.json --phys results/dov/phys.json --resolvable 1.16 --out results/dov/radial_main
python scripts/circuit/plot_main.py --mode tangential_pure --sweep results/dov/sweep.json --phys results/dov/phys.json --resolvable 0 --out results/dov/tangential_pure_main
python scripts/circuit/dump_spectrograms.py --name cdagger $C --alphas 0.3
python scripts/circuit/plot_spectrograms.py --name cdagger --alpha 0.3
```

Output: `results/dcs/circuit_score.json` ($\rho_{C^\dagger}$ 0.9009, $\rho_{U \setminus C^\dagger}$ $-1.2321$),
`results/dov/*.png`, `results/spectrograms/cdagger/*.png`.

## 3. Probing and targeted erasure

| Script | What it does |
|---|---|
| `scripts/erasure/run_erase.py` | at one location of $C^\dagger$: held-out ridge-probe $R^2$ of range $r$ and radial velocity $v^{\mathrm{rad}}$, then $M_{\mathcal P}$ after LEACE erasure of one variable there, against three rank-matched random erasers |

Locations: `--site smlp` (S:{MLP}, spatial FFN output) and `--site heads` (T:{h1, h2, h6, h7}, the heads' outputs
before the output projection). Erasers are fitted on RandomWalk2; $M_{\mathcal P}$ uses $\alpha \in \{0.1, \dots, 0.9\}$.

```bash
for site in smlp heads; do for target in range v_rad; do
  python scripts/erasure/run_erase.py --site $site --target $target --out results/erasure
done; done
```

Output: `results/erasure/<site>_<target>.csv`, one row per condition (`native`, `erase_<target>`, `erase_random0..2`)
with `M`, `R2_range` and `R2_v_rad`.
