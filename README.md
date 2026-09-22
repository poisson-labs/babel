# Babel: Adversarial MARL Communication Robustness (`babel`)

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Poisson Labs Research](https://img.shields.io/badge/Poisson%20Labs-Research-teal.svg)](https://poissonlabs.ai/research/babel-decorative-channel/)

Code, environment dynamics, and experimental configurations backing the Poisson Labs research post:  
**[A Decorative Channel and a Reward Exploit in One of Our Benchmarks](https://poissonlabs.ai/research/babel-decorative-channel/)**

Babel is a benchmark designed to evaluate communication robustness in cooperative multi-agent reinforcement learning (MARL). In cooperative systems, agents often coordinate via discrete, continuous, or natural language communication channels. Babel demonstrates how standard training metrics can easily mask non-functional communication channels and reward exploits.

---

## What the Study Measured

The study evaluates cooperative multi-agent coordination under asymmetric information and adversarial semantic perturbation:

### 1. Environment Substrate (`babel/env/`)
- **Procedural Resource Logistics Graph World:** Procedurally generated connected graphs (12 or 24 nodes) partitioned into resource depots ($D$), demand nodes ($M$), and transit hubs ($T$).
- **Swarm Composition:** A 4-agent team operating under asymmetric role constraints:
  - **Scouts:** Full global visibility into active demand locations, requirements, and expiration timers, but zero actuation ability (cannot move, collect inventory, or service demands).
  - **Responders (Deliverers):** Actuation capabilities to traverse graph edges, pick up resource types ($\alpha, \beta$) from depots, and fulfill demands, but restricted to a 2-hop local observation radius.
- **Episodes & Budgets:** 50-step episodes with strict fuel budgets, node delivery deadlines, and inventory capacities.

### 2. Communication Channels (`babel/channels/`)
- **Symbolic Channel:** Discrete multi-slot communication schema (8-bit / 10-bit) with Gumbel-Softmax discretization.
- **Latent VQ Channel:** Vector-quantized continuous embeddings with codebook commitments.
- **Natural Language Channel:** Continuous prompt-tuning through a frozen causal language model (GPT-2) with discrete autoregressive decoding at evaluation.

### 3. Adversarial Perturbation (`babel/attacks/`)
- **Semantic Man-in-the-Middle Adversary (`EmergentSpoofAttack`):** Rather than injecting random bit noise, the adversary intercepts the Scout's observation, injects hallucinated demand states, and routes them through the Scout's own frozen encoder policy. This produces capability-matched, valid messages that describe non-existent world states.

---

## What It Found

All figures and findings below correspond to the empirical results reported in the published post:

| Configuration | Task Satisfaction | Observed Effect |
|---|---:|---|
| **IPPO (No Communication)** | 21.1% | Independent agents operating with local visibility only |
| **MAPPO (Clean 8-Bit Channel)** | 51.2% | +30.1pp lift over IPPO (apparent communication success) |
| **MAPPO (100% Spoofed Channel)** | 48.4% | −2.8pp drop under full semantic interception |

### Key Findings

1. **The Communication Channel Was Decorative**
   - The +30.1 percentage point gap between IPPO and MAPPO initially suggested that the swarm had developed an effective emergent protocol.
   - However, when the semantic adversary spoofed 100% of the Scout's messages with false intelligence, satisfaction fell by only 2.8 percentage points (from 51.2% to 48.4%).
   - Over 90% of the performance premium survived complete semantic corruption, proving that the Deliverers had learned to ignore the communication channel almost entirely.

2. **The Reward Function Was Exploited via Spatial Partitioning**
   - With parameter sharing and a centralized training critic (CTDE), the Deliverers discovered a cognitive shortcut: learning a near-Hamiltonian spatial sweep across the graph nodes.
   - Given a 50-step episode budget on a 12-node graph, sweeping nodes systematically encounters and fulfills approximately 48% of randomly spawned demands by chance.
   - Because learning a coordinated spatial partition presents a smoother optimization gradient than establishing a co-adapted emergent vocabulary, the swarm bypassed the communication channel entirely.

---

## Methodological Boundaries & Limitations

- **Reward Curves Do Not Verify Communication:** High task satisfaction or a wide performance gap over independent baselines does not demonstrate that information transmitted across a channel is load-bearing. Causal intervention or adversarial perturbation is required to verify protocol reliance.
- **Parameter Sharing Encourages Heuristic Coordination:** Parameter sharing across swarm agents facilitates implicit coordination heuristics (such as spatial sector sweeping) that can bypass explicit communication bottlenecks.

---

## Reproduction Steps

### Environment Setup

Install dependencies using `uv`:

```bash
git clone https://github.com/poisson-labs/babel.git
cd babel
uv sync
```

### Verification Sanity Checks

Run the automated test suite:

```bash
uv run pytest
```

Run code formatting and lint verification:

```bash
uv run ruff check
uv run ruff format --check
```

### Training

Train the IPPO no-communication baseline:

```bash
uv run python scripts/train.py --config-name algo/ippo
```

Train the MAPPO multi-agent team with an 8-bit symbolic communication channel:

```bash
uv run python scripts/train.py --config-name algo/mappo
```

### Adversarial Evaluation Sweep

Evaluate a trained checkpoint under the semantic man-in-the-middle adversary across perturbation intensities ($p_{\text{adv}} \in [0.0, 1.0]$):

```bash
uv run python scripts/monte_sweep.py
```

*Note on external logging:* To log training and evaluation metrics to Weights & Biases, set the `WANDB_API_KEY` environment variable. When unset, runs operate in offline mode.

---

## License & Notice

- Code is licensed under the [MIT License](LICENSE).
- Upstream licenses, citations, and library grants are documented in [NOTICE](NOTICE).

---

## Citation

```bibtex
@misc{kolasinski2026babel,
  author = {Kolasinski, Taylor},
  title = {A Decorative Channel and a Reward Exploit in One of Our Benchmarks},
  year = {2026},
  publisher = {Poisson Labs},
  url = {https://poissonlabs.ai/research/babel-decorative-channel/}
}
```
