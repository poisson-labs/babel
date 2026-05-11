# Babel

Babel is a Poisson Labs research project studying communication-channel robustness in cooperative multi-agent reinforcement learning. Four agents coordinate in a partially observable resource logistics graph world using matched-capacity symbolic, latent vector, or natural-language communication channels, then face stochastic noise, dropout, and semantic adversaries to measure how each modality degrades under attack.

## Build Status

| Gate | Section | Success criterion | Status |
| --- | --- | --- | --- |
| Step 1 | Environment + IPPO baseline | IPPO reaches 25-45% demand satisfaction over 500 eval episodes | passed: 34.3% demand satisfaction, 95% CI 32.9%-35.8% over 1000 eval episodes |
| Step 2 | Symbolic channel + MAPPO | MAPPO+symbolic reaches 70-85% demand satisfaction and 30+ point gap over IPPO | not started |
| Step 3 | Latent VQ channel | Latent VQ MAPPO is within 5 points of symbolic | not started |
| Step 4 | NL channel | NL MAPPO is within 8 points of symbolic and empirical bits/message is about 8 | not started |
| Step 5 | B=4 symbolic sanity check | B=4 symbolic produces the matched-capacity supporting point | not started |
| Step 6 | Attack interface + stochastic attacks | Noise and dropout cause measurable high-intensity degradation | not started |
| Step 7 | Semantic adversary 4a | NL drops by at least 20 percentage points at p_adv=0.25 | not started |
| Step 8 | Semantic adversary 4c | Capability-matched adversary causes measurable degradation | not started |
| Step 9 | Full attack evaluation | Headline Pareto figure is produced and inspected | not started |
| Step 10 | Generalization test | 16-node clean and 4a evaluations complete | not started |
| Step 11 | Forensic analysis + replay clips | Blog-ready exemplar failure episodes and clips are selected | not started |
| Step 12 | Write the post | 5000-6000 word blog post is drafted with figures and clips | not started |

Current gate: §9 Step 2, Symbolic channel + MAPPO. Do not start Step 2 until explicitly prompted.
