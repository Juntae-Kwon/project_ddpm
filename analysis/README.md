# Stage-2 diagnostics notebook

`stage2_posterior_diagnostics.ipynb` is a CPU-only analysis of the completed
MALA-within-Gibbs pilot. It does not import JAX, train the DDPM, or run MCMC.

The ignored `stage2_analysis_data/` directory is a compact transfer bundle made
from completed outputs. Rebuild it on HiPerGator with:

```bash
cd /blue/ark007/juntaekwon/projects/DDPM
.venv/bin/python analysis/prepare_stage2_analysis_data.py
```

On a local Mac, put the notebook, `requirements.txt`, and
`stage2_analysis_data/` in the same directory. Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab stage2_posterior_diagnostics.ipynb
```

The bundle contains only posterior summaries, diagnostics, one final terminal
latent snapshot, and existing Stage-1 sanity samples. It excludes model and
MCMC checkpoints.
