# AI Usage Explanation

TR-Core contains no pretrained models, no LLM weights, no neural network backbones, and no external API calls. The entire architecture -- from the Apriori induction engine and binomial statistical filter to the seven development operations and Gymnasium environments -- was designed, implemented, and validated from scratch in Python.

The induction engine implements level-wise Apriori candidate generation over literal conjunctions (max arity 4). Anti-monotonicity of coverage enables pruning; a minimum coverage floor (c_min = 4) induces a phase transition to subcritical branching, making search tractable. The binomial filter computes exact p-values against the base success rate (alpha = 0.001, minimum hits = 5, precision threshold = 0.95), providing per-candidate Type I error control without asymptotic approximations.

The seven development operations (CREATE, GENERALIZE, SPECIALIZE, SPLIT, COMPOSE, MODIFY, DELETE) modify a rule base with explicit status tracking (LATENT, ACTIVE, RETIRED, DELETED) and evidence counters. COMPOSE -- the conjoining of rules into subprograms -- is the only mechanism enabling structural-OOD generalization.

Two Gymnasium environments were implemented: TRCore-AttributeTaskV3-v0 (8 attributes, 4,608 contexts, rare compositional rules) and TRCore-AttributeTaskV4-v0 (6 attributes, 864 contexts, Structural-OOD protocol with held-out triple). Both expose standard Dict(Discrete) observation and Discrete action spaces.

Experimental infrastructure includes Hydra configuration management, DVC pipeline (14 stages: data to experiments to analysis to figures), permutation testing (20k iterations), Cohen's dz effect sizes, bootstrap confidence intervals, and publication-ready figure generation. Formal complexity analysis (LaTeX) establishes worst-case O(L^K) time, expected O((1+beta*L)^K) under pruning, and exact binomial Type I error bounds.

All code, algorithms, environments, and analyses were authored independently without AI-assisted code generation.