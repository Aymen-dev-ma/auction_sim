# ROI-Constrained Bidding via Curriculum-Guided Bayesian Reinforcement Learning
## Brief 
The official PyTorch implementation of the SIGKDD '22 paper: ROI-Constrained Bidding via Curriculum-Guided Bayesian Reinforcement Learning (CBRL). 

PDF available at [ResearchGate](https://www.researchgate.net/publication/361222484_ROI_Constrained_Bidding_via_Curriculum-Guided_Bayesian_Reinforcement_Learning), [Arxiv]().

## Installation

Special dependencies on:
- [dopamine-rl](https://pypi.org/project/dopamine-rl/): for replay buffer
- [pulp](https://pypi.org/project/PuLP/): for solving linear programming
- pytorch
- gym
- seaborn
- ...

## Usage
Due to privacy regulations, the commercial data is currently not available. 

If time permits, I may update the synthetic dataset built on the commercial data. 

The repo now includes:
- CBRL-crude: as proposed in the paper (with its variants).
- CBRL-automated: CBRL with automated curriculum learning, as been briefly discussed in the appendix. 
- [RCPO](https://arxiv.org/abs/1805.11074): adapted version of the CMDP policy optimization alg. to the Constrained Bidding problem. 
- Hard: a hard barrier counterpart to RCPO, also a baseline of CBRL. 
- Ind-RS: a reward shaping baseline of CBRL.
- Ind-ICM: an [ICM](https://arxiv.org/pdf/1705.05363.pdf) baseline of CBRL. 

Refer to the shell for detailed usage doc:
```bash
bash scripts/CBRL.sh 12 [BRL:0/1] 5 1 0.2 [dataset] # crude version, and BRL&CRL variants
bash scripts/CBRL_auto.sh 12 [BRL:0/1] 5 [dataset] # automated CBRL, no hand-tuning curriculum
bash scripts/CBRL_eval.sh ... # to evaluate models on the test set
bash scripts/RCPO.sh ... 
bash scripts/Hard.sh ... 
bash scripts/Ind_RS.sh ...
bash scripts/Ind_ICM.sh ...
```

For prior arts please refer to their official repo. 

# Cite
Haozhe Wang, Chao Du, Panyan Fang, Shuo Yuan, Xuming He, Liang
Wang, Bo Zheng. 2022. ROI Constrained Bidding via Curriculum-Guided
Bayesian Reinforcement Learning. In Proceedings of the 28th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (KDD ’22), August
14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA, 13 pages.
https://doi.org/10.1145/3534678.3539211

# Misc
Contact: jasper.whz@outlook.com, [Linkedin](https://www.linkedin.com/in/haozhe-wang-10877586/).