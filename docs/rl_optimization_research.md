# Reinforcement Learning for DFS Optimization — Literature Review

**Compiled**: 2026-02-17
**Purpose**: Foundation for potential RL agent integration into UrSim backtesting/optimization pipeline

---

## Category 1 — Foundational RL Theory & Deep RL Methods

### 1. Reinforcement Learning: An Introduction (2nd Edition)
- **Authors:** Richard S. Sutton, Andrew G. Barto
- **Year:** 2018 (MIT Press)
- **Key Contribution:** The definitive textbook covering all core RL algorithms — temporal difference learning, Q-learning, policy gradient, Monte Carlo methods, and function approximation. The multi-armed bandit framework (Chapter 2) maps to strategy selection across GPP/Cash modes.
- **DFS Relevance:** Establishes the mathematical foundations for all RL-based DFS work. Bandit framework directly applicable to selecting among UrSim's 5 optimization strategies (combined, kelly, mean-variance, risk-parity, equal-weight).

### 2. Human-Level Control Through Deep Reinforcement Learning (DQN)
- **Authors:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. (DeepMind)
- **Year:** 2015 (*Nature* 518, 529-533)
- **Key Contribution:** Introduced Deep Q-Network (DQN), combining Q-learning with deep CNNs and experience replay to achieve superhuman performance on 49 Atari games.
- **DFS Relevance:** DQN's action-value framework applies to sequential player selection: state = current roster, action = next player to add, reward = expected lineup score. Experience replay enables replaying past DFS slates.

### 3. Proximal Policy Optimization Algorithms (PPO)
- **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
- **Year:** 2017 (arXiv:1707.06347)
- **Key Contribution:** PPO uses clipped probability ratios as a surrogate objective, enabling stable training with multiple gradient updates per batch. Became the default RL algorithm for discrete and continuous control.
- **DFS Relevance:** PPO is the algorithm used in the 2024 "Optimizing Fantasy Sports Team Selection with Deep RL" paper (#22). Clipped objective prevents catastrophic lineup selections during training.

### 4. Simple Statistical Gradient-Following Algorithms for Connectionist RL (REINFORCE)
- **Authors:** Ronald J. Williams
- **Year:** 1992 (*Machine Learning* 8, 229-256)
- **Key Contribution:** Introduced REINFORCE policy gradient algorithm. Established the policy gradient theorem underlying all modern actor-critic and deep policy gradient methods.
- **DFS Relevance:** Foundation for any DFS RL system that parameterizes a "player selection policy" and optimizes via gradient ascent on expected contest payout.

### 5. Continuous Control with Deep Reinforcement Learning (DDPG)
- **Authors:** Timothy P. Lillicrap, Jonathan J. Hunt, et al. (DeepMind)
- **Year:** 2016 (arXiv:1509.02971)
- **Key Contribution:** Actor-critic algorithm for continuous action spaces. Adapted DQN's experience replay and target networks to policy gradient methods.
- **DFS Relevance:** Actor selects player exposure percentages (continuous), critic evaluates expected contest ROI. Directly applicable to Kelly-optimal exposure sizing as a continuous control problem.

### 6. Mastering the Game of Go Without Human Knowledge (AlphaGo Zero)
- **Authors:** David Silver, Julian Schrittwieser, Karen Simonyan, et al. (DeepMind)
- **Year:** 2017 (*Nature* 550, 354-359)
- **Key Contribution:** Self-play + Monte Carlo Tree Search (MCTS) discovered optimal strategies without human training data. Beat all prior versions 100-0.
- **DFS Relevance:** MCTS maps to lineup building: tree nodes = partial rosters, rollouts = MC simulation of lineup scores. Self-play paradigm suggests generating "opponent lineups" for game-theoretic contest simulation.

---

## Category 2 — RL for Combinatorial Optimization (Knapsack, Routing, Scheduling)

### 7. Neural Combinatorial Optimization with Reinforcement Learning
- **Authors:** Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, Samy Bengio (Google Brain)
- **Year:** 2017 (arXiv:1611.09940, ICLR 2017 Workshop)
- **Key Contribution:** Trained pointer network with REINFORCE to solve TSP, achieving near-optimal results up to 100 nodes without hand-crafted heuristics. First demonstration that RL + sequence models rival exact solvers on NP-hard problems.
- **DFS Relevance:** DFS lineup construction is structurally identical to TSP/knapsack: select a sequence of players subject to salary and position constraints. Pointer network directly extends to player selection given partial roster state.

### 8. Pointer Networks
- **Authors:** Oriol Vinyals, Meire Fortunato, Navdeep Jaitly (Google Brain)
- **Year:** 2015 (*NeurIPS 2015*, pp. 2692-2700)
- **Key Contribution:** Attention-as-pointer architecture for variable-size combinatorial problems. Applied to convex hull, triangulation, and TSP.
- **DFS Relevance:** Variable-length input (player pool changes daily) and fixed-size output (8-player roster) is exactly the Pointer Network formulation. Replace supervised TSP with RL + DFS reward signals.

### 9. Attention, Learn to Solve Routing Problems!
- **Authors:** Wouter Kool, Herke van Hoof, Max Welling (University of Amsterdam)
- **Year:** 2019 (*ICLR 2019*, arXiv:1803.08475)
- **Key Contribution:** Transformer-style multi-head attention encoders with REINFORCE + greedy rollout baseline. State-of-the-art on TSP-100 and VRP variants. Learned heuristics generalize across problem instances without retraining.
- **DFS Relevance:** Generalization property is critical: model trained on historical slates should generalize to new slates. Greedy rollout baseline maps to comparing RL policy against ILP-optimal lineup.

### 10. Reinforcement Learning for Solving the Vehicle Routing Problem
- **Authors:** Mohammadreza Nazari, Afshin Oroojlooy, Lawrence V. Snyder, Martin Takac
- **Year:** 2018 (*NeurIPS 2018*)
- **Key Contribution:** Extended pointer network to VRP with capacity constraints. Handles dynamic/stochastic instances. Outperformed Google OR-Tools on medium instances.
- **DFS Relevance:** VRP capacity = salary cap. Dynamic demands (injury scratches) = VRP stochasticity. Architecture directly adaptable to DFS.

### 11. Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon
- **Authors:** Yoshua Bengio, Andrea Lodi, Antoine Prouvost
- **Year:** 2021 (*European Journal of Operational Research*, 290(2), 405-421; arXiv:1811.06128)
- **Key Contribution:** Comprehensive survey organizing ML approaches along two axes: end-to-end vs. learn-to-configure; replace vs. augment vs. initialize the optimizer. Introduced "distribution over problems" framework.
- **DFS Relevance:** Architecture decision map: replace ILP with learned model entirely, or use ML to configure ILP solver. Distribution-over-problems view essential for training on historical slates.

### 12. Reinforcement Learning for Combinatorial Optimization: A Survey
- **Authors:** Nina Mazyavkina, Sergey Sviridov, Sergei Ivanov, Evgeny Burnaev
- **Year:** 2021 (*Computers & Operations Research*, 134, 105400; arXiv:2003.03600)
- **Key Contribution:** Systematic survey covering TSP, VRP, bin packing, job scheduling, graph coloring, and knapsack. Identified graph neural networks as dominant representation.
- **DFS Relevance:** Knapsack results map exactly to DFS. Multi-constraint knapsack (salary + positions + stacking) matches DFS formulation. GNN-based approaches promising for player-team graph structure.

### 13. OR-Gym: A Reinforcement Learning Library for Operations Research Problems
- **Authors:** Christian D. Hubbs, Hector D. Perez, Owais Sarwar, et al.
- **Year:** 2020 (arXiv:2008.06319)
- **Key Contribution:** OpenAI Gym-compatible RL environments for OR problems including knapsack variants, bin packing, supply chains. Benchmarked PPO, A2C, DQN against MILP.
- **DFS Relevance:** Stochastic online knapsack (Knapsack-v3) is the most direct analog to DFS: items arrive sequentially with uncertain value, irrevocable decisions under capacity constraint. Template for DFS RL training environment.

### 14. Discovering Faster Matrix Multiplication Algorithms with RL (AlphaTensor)
- **Authors:** Alhussein Fawzi, Matej Balog, et al. (DeepMind)
- **Year:** 2022 (*Nature* 610, 47-53)
- **Key Contribution:** AlphaTensor discovered provably correct matrix multiplication algorithms faster than any known human method. RL can discover novel algorithms in discrete combinatorial search spaces.
- **DFS Relevance:** Motivates applying RL to discover novel DFS heuristics — stacking patterns, correlation exploitation — that humans haven't formalized.

---

## Category 3 — RL for ILP/MILP Augmentation

### 15. Learning to Branch in Mixed Integer Programming
- **Authors:** Elias B. Khalil, Pierre Le Bodic, Le Song, George Nemhauser, Bistra Dilkina
- **Year:** 2016 (*AAAI 2016*)
- **Key Contribution:** First ML approach to variable branching in branch-and-bound MIP. Trained SVM via imitation learning to replicate strong branching decisions. Significant speedups.
- **DFS Relevance:** DFS ILP solvers spend most time in branch-and-bound. Learning which player binary indicators to branch on first could reduce solve time for 150+ lineup builds from minutes to seconds.

### 16. Exact Combinatorial Optimization with Graph Convolutional Neural Networks
- **Authors:** Maxime Gasse, Didier Chetelat, Nicola Ferroni, Laurent Charlin, Andrea Lodi
- **Year:** 2019 (*NeurIPS 2019*, arXiv:1906.01629)
- **Key Contribution:** Bipartite graph representation (variables + constraints as nodes) processed by GCN. Learned branching policies generalize to larger instances than training.
- **DFS Relevance:** Bipartite graph (players as variable nodes, salary/position/stacking as constraint nodes) is natural for DFS ILP. GCN policy would generalize across different player pools and slates.

### 17. Reinforcement Learning for Integer Programming: Learning to Cut
- **Authors:** Yunhao Tang, Shipra Agrawal, Yuri Faenza
- **Year:** 2020 (*ICML 2020*, pp. 9367-9376)
- **Key Contribution:** RL agent selects cutting planes in the cutting plane method for IP. Adaptive cut selection outperforms fixed heuristics, reducing LP iterations significantly.
- **DFS Relevance:** DFS ILP generates diversity via constraint injection (overlap cuts between lineups). Learning which diversity constraints to inject via RL would accelerate multi-lineup generation.

---

## Category 4 — RL for Portfolio Optimization and Finance

### 18. A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
- **Authors:** Zhengyao Jiang, Dixing Xu, Jinjun Liang
- **Year:** 2017 (arXiv:1706.10059)
- **Key Contribution:** EIIE (Ensemble of Identical Independent Evaluators) combining CNN/RNN/LSTM per asset with portfolio-vector memory. Optimized log-return via policy gradient.
- **DFS Relevance:** Portfolio-vector memory directly applicable to multi-lineup DFS: current lineup portfolio state informs which players are over/under-exposed. Next lineup selection conditioned on portfolio state — the cross-lineup diversification problem.

### 19. Deep RL for Stock Portfolio Optimization by Connecting with Modern Portfolio Theory
- **Authors:** Multiple
- **Year:** 2023 (*Expert Systems with Applications*)
- **Key Contribution:** Connected DRL with Markowitz by incorporating covariance matrix and technical indicators as 3D tensors. Explicit covariance modeling improved Sharpe ratio ~30% over DRL-only baselines.
- **DFS Relevance:** Player covariance problem: same-team players positively correlated (stacking), opposing pitcher/batter negatively correlated. Incorporating correlation matrix into RL state enables discovering stacking patterns without explicit rules.

### 20. Risk-Adjusted Deep RL for Portfolio Optimization: A Multi-Reward Approach
- **Authors:** Multiple
- **Year:** 2025 (*International Journal of Computational Intelligence Systems*)
- **Key Contribution:** Multiple PPO agents with different reward functions (log returns, differential Sharpe, max drawdown) combined via ensemble weighting. Multi-reward agents maintain higher Sharpe + lower drawdowns.
- **DFS Relevance:** GPP requires maximizing ceiling (volatile); cash requires maximizing floor (consistent). Training separate agents per contest type and combining mirrors UrSim's combined/kelly/mean-variance strategy framework.

---

## Category 5 — RL for Daily Fantasy Sports and Sports Analytics

### 21. Competing with Humans at Fantasy Football: Team Formation in Large Partially-Observable Domains
- **Authors:** Tim Matthews, Sarvapali Ramchurn, Georgios Chalkiadakis
- **Year:** 2012 (*AAAI 2012*)
- **Key Contribution:** Formulated season-long fantasy football as Bayesian RL with Dirichlet priors updated from match results. Automated agent finished top 1% of 2.5M participants in English Premier League.
- **DFS Relevance:** Earliest RL treatment of fantasy sports, proving viability. Bayesian belief model over player abilities directly applicable to DFS projection uncertainty modeling.

### 22. Optimizing Fantasy Sports Team Selection with Deep Reinforcement Learning
- **Authors:** Multiple (ACM CODS-COMAD 2025, arXiv:2412.19215)
- **Year:** 2024
- **Key Contribution:** Fantasy cricket DFS as sequential MDP with player-by-player selection. PPO significantly outperformed DQN and classical ILP on test slates, capturing substitution dynamics ILP cannot model.
- **DFS Relevance:** Most recent direct application of deep RL to DFS. Sequential MDP formulation (state = partial roster, action = add player, reward = realized fantasy score) immediately implementable for NBA. **Establishes PPO as preferred algorithm.**

### 23. Optimising Daily Fantasy Sports Teams with Artificial Intelligence
- **Authors:** Multiple (University of Southampton)
- **Year:** 2021 (*International Journal of Computer Science in Sport*)
- **Key Contribution:** Compared ILP, genetic algorithms, and simulated annealing for NFL DFS. AI consistently outperformed average humans. ILP best for single lineups, GAs better for multi-lineup diversity.
- **DFS Relevance:** Benchmark establishing AI superiority over human DFS players. Validates UrSim's hybrid ILP + diversity heuristic approach.

### 24. Optimizing Daily Fantasy Sports Contests Through Stochastic Integer Programming
- **Authors:** Sarah Newell, Todd Easton (Kansas State University)
- **Year:** 2017
- **Key Contribution:** First stochastic IP for DFS, treating player scores as distributions. Proved DFS requires portfolio-level stochastic optimization rather than deterministic lineup maximization.
- **DFS Relevance:** Validates UrSim's MC + ILP architecture. Stochastic IP is the rigorous justification for VaR/CVaR ranking over expected value alone.

### 25. Competing in Daily Fantasy Sports Using Generative Models
- **Authors:** Jiri Mlcoch, Ondrej Hubacek
- **Year:** 2024 (*International Transactions in Operational Research*, 31(3))
- **Key Contribution:** Generative models for player score distributions + mixed-integer quadratic program (MIQP) optimizing expected value AND variance. **34% ROI** in backtesting.
- **DFS Relevance:** Closest prior work to UrSim's quant philosophy. MIQP is the rigorous baseline UrSim's quant engine approximates via MC. 34% ROI is a performance target.

### 26. Optimizing Daily Fantasy Baseball Lineups: A Linear Programming Approach
- **Authors:** Multiple
- **Year:** 2024 (arXiv:2411.11012)
- **Key Contribution:** LP-based MLB DFS optimizer incorporating park factors, handedness splits as constraint-layer features (not projection modifiers). Constraint-layer feature incorporation outperforms direct projection adjustment.
- **DFS Relevance:** MLB-specific validation of projection immutability principle. Park factors encoded as constraints, not projection multipliers — exactly how makrov_cli_adapter.py works.

---

## Category 6 — Multi-Armed Bandits for Parameter Tuning

### 27. Finite-Time Analysis of the Multiarmed Bandit Problem (UCB1)
- **Authors:** Peter Auer, Nicolo Cesa-Bianchi, Paul Fischer
- **Year:** 2002 (*Machine Learning* 47, 235-256)
- **Key Contribution:** UCB1 achieves optimal O(log T) regret with exploration bonus `sqrt(2 ln t / n_i)`. Rigorous exploration-exploitation tradeoff for bounded rewards.
- **DFS Relevance:** Each "arm" = a quant strategy, "reward" = contest ROI. UCB1 adaptively allocates builds to strategies with uncertainty bonuses for under-tested ones. Automated A/B testing of UrSim's 5 optimization strategies.

---

## Bonus — Already Referenced in CLAUDE.md

### 28. Picking Winners in Daily Fantasy Sports Using Integer Programming
- **Authors:** David Scott Hunter, Juan Pablo Vielma, Tauhid Zaman (MIT)
- **Year:** 2016 (arXiv:1604.01455)
- **Key Contribution:** DFS multi-entry portfolio as submodular optimization with jointly Gaussian projections. Won real DraftKings contests.
- **DFS Relevance:** Mathematical foundation of UrSim's multi-lineup generation. Submodularity justifies greedy diversity selection in `select_diverse_lineups()`.

### 29. How to Play Fantasy Sports Strategically (and Win)
- **Authors:** Martin B. Haugh, Raghav Singal (Columbia University)
- **Year:** 2021 (*Management Science* 67(1), 72-92)
- **Key Contribution:** Opponent modeling via Dirichlet-multinomial. **350% ROI** over 17-week NFL season on top-heavy GPP contests.
- **DFS Relevance:** Definitive validation of portfolio-over-single-lineup optimization. Dirichlet regression for opponent modeling is the rigorous implementation of ownership leverage scoring.

### 30. In-Game Soccer Outcome Prediction with Offline Reinforcement Learning
- **Authors:** Multiple
- **Year:** 2024 (*Machine Learning*, Springer)
- **Key Contribution:** Conservative Q-Learning (CQL) for temporal-graph networks. 87% accuracy using offline RL (no live environment interaction required).
- **DFS Relevance:** **Offline RL is the key breakthrough for DFS**: cannot run live contests to collect training data, but have vast historical slate data. CQL allows training a DFS lineup selection policy on historical results without real money.

---

## Synthesis: RL Integration Roadmap for UrSim

### Phase A — Bandit-Based Strategy Selection (Lowest Effort, Highest Impact)
Use UCB1 (#27) to automatically select among UrSim's 5 optimization strategies per slate. Each backtest date provides a reward signal. ~50 lines of code on top of existing backtester.

### Phase B — Sequential MDP Lineup Builder (Medium Effort)
Following #22 (PPO for fantasy cricket), formulate NBA lineup construction as:
- **State**: partial roster (positions filled, salary remaining, players selected)
- **Action**: select next player from available pool
- **Reward**: actual DK points of completed lineup (from backtest data)
- Train PPO agent on 15+ historical slates using offline RL (#30).

### Phase C — Portfolio-Level RL Agent (High Effort, Maximum Differentiation)
Following #18 (EIIE portfolio management), train an agent that:
- Receives current lineup portfolio exposure vector as state
- Selects next lineup to add (maximizing portfolio Sharpe, not individual lineup score)
- Uses covariance-aware state representation (#19) to discover stacking patterns

### Phase D — ILP Augmentation via Learned Heuristics (Research-Grade)
Following #15-17 (learning to branch/cut), train GCN to:
- Select branching variables in PuLP's branch-and-bound
- Choose which diversity constraints to inject
- Reduce multi-lineup solve time from minutes to seconds
