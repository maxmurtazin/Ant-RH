# Symbolic Dynamics, Zeta Zeros, and Learnable Hilbert–Pólya Operators: A Deep Technical Literature Review
### Bridging Aperiodic Substitution Systems, Spectral Theory, and Computational Operator Learning via DTES + RL + ACO

***

## Executive Summary

This review synthesizes research at the intersection of symbolic dynamics, spectral theory of aperiodic Schrödinger operators, the Riemann zeta function, and modern machine learning for operators. The central hypothesis — that a low-complexity symbolic backbone combined with a learnable potential deformation and a reinforcement-learning/ant-colony-optimization (RL/ACO) search can approximate the Hilbert–Pólya operator whose spectrum matches the nontrivial zeta zeros — is framed as a precise inverse spectral learning problem. The review identifies the fundamental obstacle (singular continuous vs. GUE spectral mismatch) and proposes concrete experimental architectures to address it. Key findings are labeled throughout as **[Proven]**, **[Empirical]**, **[Speculative]**, or **[Open Problem]**.

***

## 1. Symbolic Dynamical Systems as Generative Mechanisms

### 1.1 Formal Setup

A *symbolic dynamical system* is a pair \((X, \sigma)\) where \(X \subseteq \mathcal{A}^{\mathbb{Z}}\) is a shift-invariant closed subset of the set of bi-infinite sequences over a finite alphabet \(\mathcal{A}\), and \(\sigma\) is the left shift map \((\sigma x)_n = x_{n+1}\). The dynamical invariants of \((X, \sigma)\) — topological entropy, ergodic measures, spectral type of the Koopman operator on \(L^2\) — provide a compressed description of the complexity of the sequence.[^1][^2]

### 1.2 Key Substitution Systems

Three families dominate the aperiodic operator literature:

| Sequence | Substitution Rule | Diffraction / Spectral Type | Entropy |
|---|---|---|---|
| **Thue–Morse** | \(0 \mapsto 01, 1 \mapsto 10\) | Purely singular continuous[^3] | 0 |
| **Fibonacci** | \(a \mapsto ab, b \mapsto a\) | Purely singular continuous, multifractal[^4] | 0 |
| **Rudin–Shapiro** | \(a \mapsto ab, b \mapsto ac, c \mapsto db, d \mapsto dc\) | Purely absolutely continuous (Lebesgue spectrum)[^5][^6] | 0 |

**[Proven]** All three are strictly ergodic (uniquely ergodic) with zero topological entropy, hence each carries a unique shift-invariant probability measure.[^7][^8]

**[Proven]** The Thue–Morse diffraction measure is purely singular continuous with respect to Lebesgue measure, and the Rudin–Shapiro sequence has purely absolutely continuous diffraction spectrum.[^8][^5]

### 1.3 Autocorrelation and Spectral Compactness

The autocorrelation coefficient of a symbolic sequence \((s_n)\) is \(\eta(m) = \lim_{N\to\infty}\frac{1}{N}\sum_{i=0}^{N-1}s_i s_{i+m}\). The diffraction measure \(\hat{\gamma}\) is the Fourier transform of the autocorrelation measure. For Thue–Morse:[^5]

\[
\hat{\gamma} = \hat{\gamma}_{sc}
\]

(purely singular continuous), while for Rudin–Shapiro, \(\hat{\gamma} = \hat{\gamma}_{ac}\) (Lebesgue).[^3][^5]

**Focus question answered (partially):** Symbolic systems *can* generate complex spectra compactly — but the spectral type is constrained by the substitution rule. Neither Thue–Morse nor Fibonacci produce the continuous/Poisson-to-GUE crossover that characterizes zeta zeros; Rudin–Shapiro's absolutely continuous spectrum is also structurally different. The gap between "complex aperiodic spectrum" and "GUE-distributed spectrum" is the central obstacle of this program.[^8]

***

## 2. Aperiodic Potentials and Spectral Geometry of Schrödinger Operators

### 2.1 Discrete Schrödinger Operator

The central object is the tight-binding Hamiltonian:

\[
(H\psi)_n = \psi_{n+1} + \psi_{n-1} + V_n \psi_n
\]

where \(V_n = f(s_n)\) is determined by the symbolic sequence \((s_n)\). Depending on the coupling constant \(\lambda\) and the substitution generating \((s_n)\), the spectrum \(\sigma(H)\) of this operator can be:[^9][^7]
- **Pure point** (localized eigenstates, Poisson spectral statistics)
- **Absolutely continuous** (extended states, GOE/GUE statistics possible)
- **Singular continuous** (critical states, Cantor-like support, anomalous diffusion)

### 2.2 Thue–Morse Hamiltonian

**[Proven]** For the Thue–Morse Hamiltonian with \(V_n = \lambda(-1)^{s_n}\), the spectrum is purely singular continuous for generic coupling \(\lambda\) (Hof, Knill, Simon).[^10][^7]

**[Proven]** The Hausdorff dimension \(\dim_H \sigma(H_{\text{TM}})\) satisfies a strictly positive lower bound for all coupling constants, and converges to 1 as the order \(m\) tends to infinity.[^11][^12]

The spectral measure is a multifractal Riesz product structure, whose \(L^q\)-spectrum and Fourier dimension can be computed via the eigenvalue of a \(q \times q\) transfer matrix. Recent results (2025) yield explicit formulas for the Birkhoff spectrum and quantization dimension of the Kreĭn–Feller operator associated to the Thue–Morse Riesz product.[^13][^14]

### 2.3 Fibonacci Hamiltonian

**[Proven]** The spectrum of the Fibonacci Hamiltonian is a zero-measure Cantor set for all \(\lambda > 0\) (Sütő, 1989). The states are multifractal, with anomalous transport exponents that interpolate between ballistic and localized.[^4][^15]

**[Proven]** Higher-dimensional Fibonacci quasicrystals (square and cubic) exhibit a transition from singular continuous to absolutely continuous spectrum as dimension increases; the transition boundary is controlled by the Cantor set arithmetic.[^16][^17]

### 2.4 The Spectral Type Gap

**[Critical Observation — Proven]:** None of the standard aperiodic substitution Hamiltonians produce GUE level statistics. Level spacing statistics for quasiperiodic operators interpolate between Poisson (localized) and Wigner-Dyson (delocalized), but remain qualitatively distinct from GUE unless disorder or chaotic dynamics is introduced. This is the fundamental *spectral mismatch* between canonical symbolic systems and the zeta zero distribution.[^18]

***

## 3. The Riemann Zeta Zeros as a Quantum Spectrum

### 3.1 Nontrivial Zeros and the Hilbert–Pólya Vision

The nontrivial zeros of \(\zeta(s)\) lie (conditionally on RH) at \(\rho_n = \tfrac{1}{2} + i\gamma_n\), with \(\gamma_n \in \mathbb{R}\). The Hilbert–Pólya conjecture posits the existence of a self-adjoint operator \(\hat{H}\) on a Hilbert space such that:[^19]

\[
\text{Spec}(\hat{H}) = \{\gamma_n : \zeta(\tfrac{1}{2} + i\gamma_n) = 0\}
\]

If true, RH follows immediately from self-adjointness (which forces real spectrum).[^20][^19]

### 3.2 GUE Universality

**[Empirical — strongly supported]** Montgomery (1973) conjectured and Odlyzko (1987) confirmed numerically that the pair correlation of consecutive \(\gamma_n\) follows the GUE sine-kernel law:

\[
1 - \left(\frac{\sin(\pi u)}{\pi u}\right)^2
\]

as recognized by Freeman Dyson as the pair correlation function of GUE random matrices.[^21][^22][^23]

Odlyzko computed \(10^5\) zeros near height \(10^{12}\) with precision \(10^{-8}\) to confirm this agreement. Extensions to higher zeros at height \(10^{21}\) and beyond strengthen the empirical case considerably.[^23][^24]

**[Recent Proven Result — 2025]** Jerby proved, assuming RH, Montgomery's pair correlation conjecture for the zeros of Hardy's \(Z\)-function by constructing a stochastic process equivalent in law to Dyson Brownian motion with \(\beta = 2\) from a finite-dimensional variational space of approximants. This provides the first rigorous mechanism connecting a deterministic analytic function to GUE ensemble statistics.[^25][^26]

### 3.3 Critical Constraint on Any Candidate Operator

Any computationally constructed or learned Hilbert–Pólya candidate \(H_\theta\) must simultaneously satisfy:
1. **Self-adjointness** (to guarantee \(\gamma_n \in \mathbb{R}\))
2. **Chaotic spectral statistics** (GUE pair correlation, Wigner surmise level spacing)
3. **Weyl law** \(N(\gamma) \sim \frac{\gamma}{2\pi}\log\frac{\gamma}{2\pi e}\) for the zero counting function

None of these three conditions is satisfied by known aperiodic substitution Hamiltonians without modification.[^27]

***

## 4. The Hilbert–Pólya Problem as an Inverse Spectral Problem

### 4.1 Survey of Candidate Operators

The inverse spectral reframing is: *given \(\{\gamma_n\}\), find \(H\) such that \(\text{Spec}(H) = \{\gamma_n\}\)*.

**Berry–Keating \(xp\) operator:** Berry (1985) proposed quantizing the classical Hamiltonian \(H_{cl} = xp\)[^28][^27]. The normal-ordered quantum operator \(\hat{H} = \tfrac{1}{2}(\hat{x}\hat{p} + \hat{p}\hat{x}) = -i\hbar\left(x\partial_x + \tfrac{1}{2}\right)\) has purely continuous spectrum, not a discrete spectrum[^27]. Berry and Keating introduced cutoffs \(|x| \geq \ell_x, |p| \geq \ell_p\) to force discreteness, obtaining an average zero count matching the Riemann formula[^27]. **[Proven]** The modified \(xp\) model does not reproduce the *exact* Riemann zeros[^27].

**Berry–Keating on quantum graphs:** Placing the \(xp\) operator on compact quantum graphs with appropriate boundary conditions yields a discrete spectrum that approximates zeta zero statistics semiclassically. This realizes the quantum chaos conjecture in a controllable setting.[^28]

**Gutzwiller–Selberg trace formula approach:** The Gutzwiller trace formula expresses the quantum density of states as a sum over classical periodic orbits. The Selberg trace formula for hyperbolic surfaces is an *exact* analog: eigenvalues of the Laplace–Beltrami operator correspond to lengths of closed geodesics, with the zeta function analogously encoding the spectrum. **[Proven]** The structural parallel between the Weil explicit formula and the Selberg trace formula motivates the Hilbert–Pólya idea but does not construct the operator.[^29][^30][^31][^32][^19]

**Yakaboylu Hamiltonian (2024–2025):** A recently proposed self-adjoint Hamiltonian is constructed such that eigenfunctions vanish at the origin via the nontrivial Riemann zeros under Dirichlet boundary conditions on \(\mathbb{R}_+\). A similarity transformation is shown to relate it to a self-adjoint operator satisfying the required boundary conditions. A 2025 update claims the existence of a self-adjoint Hamiltonian with eigenvalues \(i(\tfrac{1}{2} - \rho_s)\) for simple zeros, implying that all simple nontrivial zeros lie on the critical line. **[Status: Submitted, not fully peer-reviewed; results suggest all nontrivial zeros are simple]**[^33][^20]

**Connes noncommutative geometry approach:** Connes interprets the zeros as an absorption spectrum on the noncommutative space of adele classes, reformulating RH as the validity of a trace formula. The spectral triple \((A, H, D)\) framework yields a Dirac operator \(D\) with \(\text{Spec}(D) = \{\tfrac{1}{2} + i\gamma_n\}\) contingent on the truth of RH. **[Speculative: mathematically rigorous but conditional; not a constructive Hilbert space operator]**[^34][^35][^36][^37]

***

## 5. Learnable Operators: Modern Perspectives

### 5.1 Neural Operators

**DeepONet:** The Deep Operator Network learns a mapping between function spaces \(\mathcal{U} \to \mathcal{V}\) via a branch network (processing input function values) and a trunk network (processing output domain coordinates). Extensions include Fourier-embedded trunk networks (FEDONet) that improve spectral accuracy via random Fourier features, and physics-informed variants.[^38][^39]

**Fourier Neural Operator (FNO):** FNO parametrizes the integral kernel in Fourier space, making it resolution-invariant for PDE solution operators. It is a subcase of the DeepONet framework. **[Empirical]** Spectral Neural Operators (SNO), using Chebyshev/Fourier series for both domain and codomain, outperform FNO and DeepONet on benchmark PDE operators while eliminating aliasing.[^39][^40][^41]

### 5.2 Operator Eigenvalue Learning

**[Recent — 2023]** Ben-Shaul et al. introduce an unsupervised neural network solver for eigenproblems of self-adjoint differential operators, learning eigenpairs end-to-end without domain discretization. The method uses a Rayleigh quotient formulation and analytically exact partial derivatives via automatic differentiation.[^42][^43]

**[Recent — 2025]** STNet (Spectral Transformation Network) solves operator eigenvalue problems by iteratively applying deflation projections and filter transforms to concentrate learning on desired spectral regions. The approach is particularly well-suited to operators with clustered eigenvalues — relevant to the GUE-like clumping structure of zeta zeros.[^44][^45]

**Operator learning without the adjoint:** Approaches that leverage a prior self-adjoint operator as preconditioner can provably converge even for non-self-adjoint targets, with error decaying at a rate determined by the eigenvalues of the prior. This framework is directly applicable to the Hilbert–Pólya inverse problem where the target spectrum is known but the operator is not.[^46]

### 5.3 Koopman Operator Regression

The Koopman operator \(\mathcal{K}\) acts on scalar observables \(g\) of a dynamical system as \(\mathcal{K}g = g \circ F\), linearizing nonlinear dynamics in infinite-dimensional space. Learning \(\mathcal{K}\) from trajectory data in a reproducing kernel Hilbert space (RKHS) via reduced-rank operator regression yields spectral decompositions linking eigenvalues of \(\mathcal{K}\) to dynamical modes of the system.[^47][^48][^49]

**Relevance to Hilbert–Pólya:** The symbolic shift \(\sigma\) on \((X, \sigma)\) defines a natural Koopman operator whose spectrum encodes the dynamical invariants of the substitution system. Learning the Koopman operator of a deformed substitution dynamical system constitutes an indirect approach to spectral inverse problems.[^48][^50][^51]

***

## 6. Symbolic Backbone + Learnable Deformation: The Core Hypothesis

### 6.1 Construction

The proposed hybrid operator is:

\[
(H_\theta \psi)_n = \psi_{n+1} + \psi_{n-1} + f_\theta(s_n)\psi_n
\]

where \((s_n)\) is a substitution sequence (Thue–Morse, Fibonacci, or composite) and \(f_\theta: \mathcal{A} \to \mathbb{R}\) is a learnable real-valued function parametrized by \(\theta\)[^7]. For \(|\mathcal{A}| = 2\), \(f_\theta\) is simply a learnable real number \(\lambda_\theta\) for each symbol. For \(|\mathcal{A}|\) larger (hierarchical alphabets), \(f_\theta\) becomes a low-dimensional neural network.

**[Observation — Speculative]:** The symbolic backbone provides deterministic, low-entropy structure (measurably different from i.i.d. noise), while the learnable deformation \(f_\theta\) introduces the degrees of freedom needed to tune spectral statistics from singular continuous toward GUE.

### 6.2 The GUE Emergence Problem

The core obstacle is that fixed substitution sequences produce purely singular continuous spectra, whereas zeta zeros follow GUE. Several mechanisms that could bridge this gap are:[^7][^18][^8]

1. **Hierarchical alphabets with disorder:** Using a larger alphabet \(\mathcal{A}\) effectively introduces controlled disorder while preserving long-range aperiodic structure. Transition from singular continuous to absolutely continuous (and hence potentially GUE statistics) has been observed numerically in generalized quasiperiodic models.[^52][^18]

2. **Thue–Morse as a chaos boundary:** The Thue–Morse sequence occupies a borderline between quasiperiodicity and chaos — eigenstates of the quantum baker's map exhibit Thue–Morse structure, and these states are multifractal. **[Empirical]** The quantum baker's map (a fully chaotic system) produces GUE statistics, and its eigenstates contain Thue–Morse structure, suggesting a deeper connection. This constitutes the strongest qualitative evidence for the core hypothesis.[^53][^54]

3. **Rudin–Shapiro as a GUE substrate:** Rudin–Shapiro has absolute continuous spectrum. Starting from Rudin–Shapiro and applying learnable deformations may offer a more direct route to GUE statistics than starting from Thue–Morse.[^6][^5]

### 6.3 Multifractality and Spectral Crossover

**[Proven]** Spectral continuity results (Beckus–Bellissard–De Nittis) show that the spectrum of the Schrödinger operator over an aperiodic system varies continuously as the underlying substitution dynamical system deforms, in the Hausdorff metric on closed sets. This continuity is expressed in the language of groupoid \(C^*\)-algebras. This means the spectral inverse problem is *well-conditioned* in a topological sense — small deformations of the symbolic structure produce small spectral changes — which is a prerequisite for gradient-based learning.[^55][^56]

***

## 7. DTES: Discrete Tropical Energy Surface as Spectral Manifold

### 7.1 Geometric Interpretation

The Discrete Tropical Energy Surface (DTES) is proposed as a latent geometric space encoding the relationship between:
- Symbolic configuration \(s_n\) (discrete geodesic skeleton)
- Spectral eigenvalues \(\gamma_n\) (energy levels)
- Operator structure (geometry generator)

Formally, define:

\[
E_{\text{DTES}}(n) = \Phi(\gamma_n, s_n, \text{local structure})
\]

where \(\Phi\) is a learned or analytically derived embedding. The DTES is hypothesized to be a piecewise-linear (tropical) approximation to the eigenvalue manifold.

### 7.2 Tropical Spectral Theory

Tropical (max-plus) algebra provides a degenerate limit of ordinary algebra in which \(a \oplus b = \max(a,b)\) and \(a \otimes b = a + b\). The tropical eigenvalue problem \(A \otimes x = \lambda \otimes x\) has a unique eigenvalue given by the maximum mean weight of cycles in the weighted graph defined by \(A\).[^57][^58]

**[Proven]** Tropical spectral theory for matrices has been developed (Cuninghame-Green, 1961). Tropical spectral theory for tensors is also established. The connection between tropical eigenvalues and Newton polygon asymptotics of classical characteristic polynomials is rigorous.[^59][^60][^58]

**Relevance:** The piecewise-linear structure of tropical geometry provides a natural framework for approximating spectral manifolds arising from discrete symbolic systems. The DTES as a tropical approximation to the Schrödinger eigenvalue manifold is geometrically natural but **[Speculative]** as a foundation for approaching GUE statistics.

### 7.3 Bellman Fixed Points

The value function of a deterministic infinite-horizon optimization problem satisfies a Bellman equation:

\[
J(x) = \max_{u \in U}\left[f(x,u) + \beta J(g(x,u))\right]
\]

**[Speculative]** The eigenfunctions and eigenvalues of the Bellman operator determine long-horizon behavior of optimization problems. In the DTES framework, identifying the Bellman fixed point with the spectral manifold of \(H_\theta\) would allow RL reward shaping to directly encode spectral matching constraints. The Hamilton–Jacobi–Bellman equation on the DTES surface could then be interpreted as the semiclassical equation for eigenvalue quantization.[^61]

***

## 8. Reinforcement Learning Formulation

### 8.1 MDP Setup

Reformulate the Hilbert–Pólya inverse spectral problem as a Markov Decision Process:

- **State space:** \(s = (s_{n-k}, \ldots, s_{n+k}, \theta_{\text{current}})\) — local symbolic context plus current operator parameters
- **Action space:** \(\mathcal{A} = \Delta\theta \in \mathbb{R}^{d_\theta}\) — parameter updates for \(f_\theta\)
- **Reward:** \(R = -d(\lambda(H_\theta), \{\gamma_n\}_{n=1}^{N})\) where \(d\) is a spectral distance (e.g., Wasserstein-2, sorted \(\ell^2\), or GUE statistics divergence)
- **Transition:** Deterministic update of operator parameters; stochastic perturbation of symbolic sequence sub-blocks

### 8.2 Algorithm Choices

**PPO (Proximal Policy Optimization):** An on-policy algorithm that clips the policy update ratio to prevent destabilizing large gradient steps. PPO with spectral normalization of value function weights provides a principled Lipschitz regularizer for the value function, beneficial when the spectral landscape has sharp features.[^62][^63]

**SAC (Soft Actor-Critic):** An off-policy, maximum-entropy RL algorithm that encourages exploration through entropy regularization. SAC is better suited than PPO when the spectral landscape is multimodal — which is expected given the GUE level repulsion structure of zeta zeros.[^63][^64]

**Deep RL for Hamiltonian Engineering:** Peng et al. (2022) demonstrate that DRL agents can discover Hamiltonian engineering sequences for solid-state NMR quantum simulators that outperform perturbation-theory designs. The discovered sequences exhibit emergent periodic patterns not anticipated analytically — providing experimental precedent for RL discovering structured physical operators.[^65][^66]

**[Speculative]** A hierarchical RL architecture is most appropriate for the multi-scale structure of the zeta zeros: low-level policies optimize local potential values \(f_\theta(s_n)\), while high-level policies adjust the global substitution rule or alphabet, encoding multi-scale spectral constraints simultaneously.[^67][^68]

### 8.3 Spectral Reward Engineering

A key challenge is designing a differentiable reward that penalizes not only eigenvalue mismatch but also incorrect spectral statistics (pair correlation, nearest-neighbor spacing distribution). Suggested reward components:

1. \(R_1 = -\sum_{n=1}^N (\lambda_n(H_\theta) - \gamma_n)^2\) — direct eigenvalue matching
2. \(R_2 = -D_{\text{KL}}(p_{\text{spacing}}(H_\theta) \| p_{\text{GUE}})\) — KL divergence from Wigner surmise
3. \(R_3 = -|r_2(H_\theta) - r_2^{\text{GUE}}|_1\) — pair correlation function mismatch

The composite reward \(R = \alpha R_1 + \beta R_2 + \gamma R_3\) allows tuning the balance between exact eigenvalue matching and statistical structure matching.

***

## 9. ACO / Multi-Agent Search for Spectral Landscapes

### 9.1 ACO Mechanism

In Ant Colony Optimization, artificial agents traverse a graph whose edge weights encode pheromone trails — a positive feedback mechanism that reinforces short/good paths. For the spectral inverse problem:[^69]

- **Graph nodes:** Symbolic sub-blocks or operator parameter values
- **Edges:** Admissible transitions between configurations
- **Pheromone:** \(\tau \propto \text{spectral match quality} = \exp(-R^{-1})\)
- **Update rule:** \(\tau_{ij}^{(t+1)} = (1-\rho)\tau_{ij}^{(t)} + \sum_k \Delta\tau_{ij}^k\) where \(\Delta\tau_{ij}^k\) is deposited by ant \(k\) if it traversed edge \((i,j)\)

**[Proven]** ACO has been applied to spectral variable selection problems (selecting spectroscopic components) with better performance than simulated annealing in avoiding local optima. **[Speculative]** The multi-modal nature of the spectral landscape near zeta zero configurations is particularly well-suited to ACO's global-then-local search pattern.[^70]

### 9.2 ACO for Multi-Modal Spectral Landscapes

GUE eigenvalue repulsion means the spectral landscape has many saddle points corresponding to near-degenerate spectra. ACO's global exploration phase, combined with local intensification, can navigate this landscape more robustly than gradient descent alone. A continuous variant (COAC — Continuous Orthogonal ACO) using an orthogonal design method enables efficient search in continuous parameter spaces.[^71][^69]

### 9.3 Hybridization with RL

A productive hybrid architecture alternates between:
1. **ACO phase:** Global exploration of symbolic sequence space and operator parameter space — agents deposit pheromone proportional to spectral match
2. **RL phase:** Local refinement of operator parameters \(\theta\) given the best symbolic sequence found by ACO
3. **Spectral evaluation:** Diagonalization of \(H_\theta\) (Lanczos algorithm for large systems, or neural network eigenvalue solver)[^42][^44]

***

## 10. Proposed Unified Architecture

### 10.1 Pipeline

```
[Symbolic Sequence Generator]
        ↓ (s_n from substitution rule / evolved alphabet)
[Operator Constructor]
        ↓ (H_θ = tridiagonal matrix with V_n = f_θ(s_n))
[Eigenvalue Solver]
        ↓ (Neural eigenvalue solver (STNet) or Lanczos)
[Spectral Comparator]
        ↓ (Wasserstein-2 to γ_n database, GUE statistics KL divergence)
[RL Policy (PPO/SAC)]   ←→   [ACO Colony]
        ↓                           ↓
[Update f_θ]              [Update substitution rule]
        ↓
[Iterate until R ≥ threshold]
```

### 10.2 Component Specifications

| Component | Method | Key Reference |
|---|---|---|
| Symbolic sequence | Substitution grammar (learnable alphabet) | [^1][^8] |
| Operator construction | Tridiagonal \(H_\theta\) matrix | [^7][^11] |
| Eigenvalue solver | STNet / Neural eigensolver | [^44][^42] |
| Spectral distance | Wasserstein-2 + GUE KL divergence | [^25][^21] |
| RL algorithm | SAC with entropy bonus | [^65][^66] |
| Global search | ACO on symbolic + parameter graph | [^69][^70] |
| Operator learning | FEDONet trunk for \(f_\theta\) | [^38][^40] |

### 10.3 Self-Adjointness Constraint

Maintaining self-adjointness during learning requires \(f_\theta: \mathcal{A} \to \mathbb{R}\) (real-valued potential), which is automatically satisfied for real-valued neural network outputs. The tridiagonal structure with real symmetric off-diagonals (all equal to 1) guarantees self-adjointness for any real potential sequence. **This is a key structural advantage of the symbolic backbone over general operator learning approaches**.[^7]

***

## 11. Critical Obstacles and Research Gaps

### 11.1 Singular Continuous vs. GUE Mismatch [Critical]

**[Proven gap]** Standard substitution Hamiltonians have singular continuous spectra with multifractal level statistics — not GUE. The spectral mismatch is quantitative: nearest-neighbor spacings for singular continuous spectra show sub-Poissonian correlations, while GUE shows Wigner surmise \(p(s) \approx \frac{32}{\pi^2}s^2 e^{-4s^2/\pi}\). No known deformation of a fixed substitution system has been rigorously proved to produce GUE statistics.[^18][^7]

### 11.2 Stability of Learned Spectra

Inverse spectral learning is generically ill-posed: many operators share the same eigenvalue sequence up to unitary equivalence, and small eigenvalue errors amplify into large potential reconstruction errors for high-frequency modes. Regularization constraints (e.g., requiring \(V_n\) to be generated by a substitution grammar of bounded complexity) can reduce ill-posedness at the cost of restricting the hypothesis class.[^72]

### 11.3 Scaling to High-Frequency Zeros

The GUE statistics of \(\gamma_n\) are most pronounced at high \(n\). Computing spectra of large tridiagonal matrices (\(N \sim 10^4\) or greater) with neural eigensolvers while maintaining gradient flow through the diagonalization step is computationally expensive. STNet's spectral transformation approach and Lanczos algorithms with implicit restarts reduce this cost but do not eliminate it.[^45]

### 11.4 Self-Adjointness During Training

Parameterizing \(f_\theta\) as a neural network with real output automatically preserves self-adjointness, but regularization pressure may push \(V_n\) to complex values if the network architecture allows it. Hard constraints via output activation (e.g., tanh scaling to a real interval) or Lagrange multiplier enforcement are necessary.

### 11.5 Operator Learning Without the Adjoint

Recent theory shows that learning non-self-adjoint operators from data is fundamentally harder: recovery requires probing both the operator and its adjoint, with sample complexity scaling with the non-normality. For the Hilbert–Pólya problem, where the target *is* self-adjoint, this additional complexity is avoided — but only if the parameterization enforces self-adjointness throughout training.[^46]

***

## 12. Open Problems

1. **Can deterministic symbolic systems approximate random matrix statistics?**
   **[Open]** The multifractal eigenstates of the quantum baker's map contain Thue–Morse structure, and that map produces GUE statistics. Is there a symbolic Hamiltonian \(H\) with Thue–Morse (or composite aperiodic) potential whose level statistics converge to GUE as the system size \(N \to \infty\)? This would require a proof of delocalization for the Thue–Morse Hamiltonian in the weak-coupling regime — currently open.[^53]

2. **Is a fractal symbolic encoding of zeta zeros possible?**
   **[Speculative]** The Thue–Morse sequence is generated by the rule \(t_n = s_2(n) \bmod 2\) (digit sum modulo 2) and thus implicitly encodes 2-adic structure. The zeta zeros themselves carry arithmetic information via the explicit formula. Is there a symbolic coding of primes — or a substitution rule over a prime-indexed alphabet — that captures the essential spectral statistics of \(\{\gamma_n\}\)?

3. **Can RL discover a Hilbert–Pólya operator?**
   **[Open experimental question]** The RL formulation outlined in Section 8 is computationally feasible for small systems (\(N \leq 200\) eigenvalues). Whether the policy can generalize — i.e., whether an operator learned to match the first 100 zeta zeros also approximately matches zeros 101–500 — is a non-trivial question about the universality of the discovered operator structure.

4. **What minimal structure is required for GUE emergence?**
   **[Partially answered]** Quantum chaos requires classically chaotic dynamics (positive Lyapunov exponents, mixing). The Baker's map is fully hyperbolic. Substitution systems are zero-entropy and have zero Lyapunov exponents. The minimal deformation of a substitution Hamiltonian that produces GUE statistics presumably requires the introduction of at least a positive entropy component. This suggests a hybrid system: a substitution grammar for long-range structure plus a chaotic component for local statistics.[^73]

5. **Does the DTES admit a tropical approximation consistent with GUE asymptotics?**
   **[Open]** Max-plus eigenvalues of matrices approximate classical eigenvalues in the tropical limit. Whether a sequence of tropical approximations to the Hilbert–Pólya operator converges in spectral statistics to GUE is an open problem connecting tropical geometry, random matrix theory, and number theory.[^60][^58]

***

## 13. Suggested Computational Experiments

### Experiment 1: Spectral Statistics Sweep
Compute nearest-neighbor spacing distributions and pair correlation functions for \(H_\theta\) with:
- \(V_n = \lambda(-1)^{s_n}\) (Thue–Morse, varying \(\lambda\))
- \(V_n\) drawn i.i.d. uniform (Anderson model baseline)
- \(V_n = f_\theta(s_n)\) with trained \(f_\theta\)

Compare against GUE Wigner surmise and Poisson. Identify the coupling/deformation range where spectral statistics transition.

### Experiment 2: RL Spectral Matching (Small Scale)
Train a SAC agent on \(N = 50\) eigenvalues. State: local symbolic window of size 7. Action: update \(f_\theta \in \mathbb{R}^{|\mathcal{A}|}\). Reward: \(-W_2(\text{sorted eigenvalues}, \text{sorted }\gamma_{1:50})\). Evaluate whether the learned potential reproduces the pair correlation structure, not just sorted eigenvalue positions.

### Experiment 3: ACO Alphabet Exploration
Run ACO over a space of substitution rules on alphabet size 4. Pheromone proportional to GUE KL divergence of the resulting Hamiltonian's level statistics. Identify which substitution rules consistently produce near-GUE spectra.

### Experiment 4: Koopman Spectral Comparison
Compute the Koopman operator of the shift \(\sigma\) restricted to Thue–Morse vs. Fibonacci vs. Rudin–Shapiro orbit closures. Compare eigenvalues of the Koopman operator against eigenvalues of GUE matrices via Wasserstein distance. This tests whether the dynamical spectrum of the substitution system encodes any GUE structure.

### Experiment 5: Tropical Approximation Sequences
For the Berry–Keating \(xp\) discretization, compute tropical eigenvalues of the associated matrix sequence and test whether they approximate the average zero density \(N(\gamma) \sim \frac{\gamma}{2\pi}\log\frac{\gamma}{2\pi e}\).

***

## Summary of Status Classifications

| Claim | Status |
|---|---|
| Thue–Morse Hamiltonian has purely singular continuous spectrum | **Proven** [^7] |
| GUE statistics of zeta zeros (pair correlation) | **Empirical** [^21][^23] |
| Montgomery pair correlation for \(Z(t)\) assuming RH | **Proven (2025)** [^25] |
| Berry–Keating \(xp\) does not reproduce exact zeta zeros | **Proven** [^27] |
| Yakaboylu self-adjoint Hilbert–Pólya Hamiltonian | **Submitted/Not fully verified** [^20] |
| RL can learn quantum Hamiltonians | **Empirical** [^65][^66] |
| Symbolic systems can approximate GUE statistics | **Open / Speculative** |
| DTES as tropical spectral manifold | **Speculative** |
| ACO effective for multi-modal spectral landscapes | **Partially empirical** [^70] |
| Neural operators can solve operator eigenvalue problems | **Proven** [^42][^43][^44] |

---

## References

1. [[PDF] Examples of Substitution Systems and Their Factors](https://cs.uwaterloo.ca/journals/JIS/VOL16/Baake/baake3.pdf) - Abstract. The theory of substitution sequences and their higher-dimensional analogues is intimately ...

2. [Symbolic dynamics - Scholarpedia](http://www.scholarpedia.org/article/Symbolic_dynamics) - Symbolic dynamics is the study of shift spaces, which consist of infinite or bi-infinite sequences d...

3. [Spectral Theory of Regular Sequences - EMS Press](https://ems.press/content/serial-article-files/26660) - The diffraction measure of the. Thue–Morse sequence with weight function w(a)=1,w(b) = −1 is purely ...

4. [[2012.14744] The Fibonacci quasicrystal: case study of hidden ...](https://arxiv.org/abs/2012.14744) - The Fibonacci noninteracting tight-binding Hamiltonians are characterized by multifractality of spec...

5. [Diffraction spectrum of a Rudin–Shapiro-like sequence](https://oleron.sciencesconf.org/conference/oleron/pages/chan.pdf)

6. [Une nouvelle propriété des suites de Rudin-Shapiro](https://aif.centre-mersenne.org/articles/10.5802/aif.1089/)

7. [[PDF] Singular Continuous Spectrum for Palindromic Schrödinger Operators](http://math.caltech.edu/papers/bsimon/p245.pdf) - In order to get singular continuous spectrum one has also to exclude eigenvalues. ... Spectral prope...

8. [[PDF] The Thue-Morse sequence - IRIF](https://www.irif.fr/~berthe/Articles/ThueMorse.pdf) - The Thue-Morse sequence is a typical example of a k-automatic sequence. Actually, like every fixed p...

9. [arXiv:1201.1423v2  [math-ph]  13 Mar 2012](https://arxiv.org/pdf/1201.1423.pdf)

10. [One-Dimensional Ergodic Schrödinger Operators](https://www.ams.org/books/gsm/249/gsm249-endmatter.pdf) - Grimm, The singular continuous diffraction measure of the Thue-Morse chain ... Kiselev, Imbedded sin...

11. [[PDF] arXiv:2202.08592v1 [math.SP] 17 Feb 2022](https://arxiv.org/pdf/2202.08592.pdf) - This implies that dimH σ(Hm,λ) tends to 1 as m tends to infinity. Key words: one-dimensional Schrödi...

12. [arXiv:1410.1856v1 [math.DS] 17 Sep 2014](https://ia803109.us.archive.org/6/items/arxiv-1410.1856/1410.1856.pdf)

13. [On the generalized dimensions for the Fourier spectrum ...](http://www.stat.physik.uni-potsdam.de/~pikovsky/pdffiles/1999/jpa_32_1532.pdf)

14. [Generalized Thue-Morse measures: spectral and fractal analysis](https://arxiv.org/abs/2509.22109) - We investigate a family of Riesz products and show that they can be regarded as diffraction measures...

15. [[PDF] Zero measure Cantor spectrum for Schrödinger operators with quasi ...](https://www.irif.fr/~numeration/uploads/Main/gohlke_20220111.pdf) - The Fibonacci substitution sequence is a special case. Gohlke. Zero measure Cantor spectrum. Page 10...

16. [arXiv:0712.2840v1  [cond-mat.other]  17 Dec 2007](https://www.arxiv.org/pdf/0712.2840.pdf)

17. [[PDF] Electronic energy spectra of square and cubic Fibonacci quasicrystals](https://www.tau.ac.il/~ronlif/pubs/PhilMag88-2261-2008.pdf) - We use the well-studied Cantor-like energy spectrum of the one-dimensional. Fibonacci quasicrystal t...

18. [[PDF] Probing symmetries of quantum many-body systems through ... - HAL](https://hal.science/hal-02992614/document) - The statistics of gap ratios between consecutive energy levels is a widely used tool, in particular ...

19. [Hilbert–Pólya conjecture - Wikipedia](https://en.wikipedia.org/wiki/Hilbert%E2%80%93P%C3%B3lya_conjecture) - Berry and Keating have conjectured that since this operator is invariant under dilations perhaps the...

20. [On the Existence of the Hilbert-Pólya Hamiltonian](https://arxiv.org/html/2408.15135v10)

21. [Montgomery's pair correlation conjecture - Wikipedia](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture) - In mathematics, Montgomery's pair correlation conjecture is a conjecture made by Hugh Montgomery (19...

22. [[PDF] Quantum chaos, random matrix theory, and the Riemann ζ-function](https://seminaire-poincare.pages.math.cnrs.fr/keating.pdf) - This section describes the fundamental mathematical concepts (i.e. the Riemann zeta function and ran...

23. [[PDF] The 1013 first zeros of the Riemann Zeta function, and zeros ... - Free](http://numbers.computation.free.fr/Constants/Miscellaneous/zetazeros1e13-1e24.pdf) - In 1987, Odlyzko computed numerically 105 zeros of the Riemann Zeta ... In our statistical study to ...

24. [[PDF] On the Distribution of Spacings Between Zeros of the Zeta Function](https://web.williams.edu/Mathematics/sjmiller/public_html/ntprob19/handouts/computational/Odlyzko_DistrSpacingsZerosZeta.pdf) - This makes it possible to compare data for the zeros of the zeta function against the theoretical pr...

25. [Variations of the Hardy Z-Function and the Montgomery ...](https://arxiv.org/abs/2511.18275) - In 1973 Montgomery formulated the pair correlation conjecture, predicting that the local spacing sta...

26. [[PDF] Variations of the Hardy Z-Function and the Montgomery Pair ... - arXiv](https://arxiv.org/pdf/2511.18275.pdf)

27. [[PDF] The Riemann Zeros as Spectrum and the Riemann Hypothesis](https://s3.cern.ch/inspire-prod-files-1/1e65b86fec7566dba4d2d2384183f67b) - The Quantum XP Model. To quantize the xp Hamiltonian, Berry and Keating used the normal ordered oper...

28. [[PDF] the berry-keating operator on compact quantum graphs ... - Uni Ulm](https://www.uni-ulm.de/fileadmin/website_uni_ulm/nawi.inst.260/paper/08/tp08-11.pdf) - In 1985, Berry [10] emphasized that the search for the hypothetical Hilbert-Polya operator in terms ...

29. [Selberg's Trace Formula: An Introduction (II)](https://www.cambridge.org/core/books/hyperbolic-geometry-and-applications-in-quantum-chaos-and-cosmology/selbergs-trace-formula-an-introduction/4035BDCDB226C34FB53F0DBD92EBF40C) - The aim of this short lecture course is to develop Selberg's trace formula for a compact hyperbolic ...

30. [[PDF] arXiv:chao-dyn/9509015v3 17 Nov 1995](https://arxiv.org/pdf/chao-dyn/9509015.pdf)

31. [[PDF] Semiclassical quantization - ChaosBook.org](https://chaosbook.org/version13/chapters/traceSemicl-2p.pdf) - We derive here the Gutzwiller trace formula and the semiclassical zeta func- tion, the central resul...

32. [[PDF] Quantum chaos, random matrix theory, and the Riemann ζ-function](https://cims.nyu.edu/~bourgade/papers/PoincareSeminar.pdf) - To state Selberg's trace formula, we need, as previously, a function ... semiclassical asymptotic ge...

33. [Hamiltonian for the Hilbert-Pólya Conjecture - arXiv](https://arxiv.org/html/2309.00405v5) - Essentially, the Hilbert-Pólya conjecture involves two stages: (I) finding an operator whose eigenva...

34. [Trace formula in noncommutative geometry and the zeros of ... - arXiv](https://arxiv.org/abs/math/9811068) - We give a spectral interpretation of the critical zeros of the Riemann zeta function as an absorptio...

35. [[PDF] Trace Formula in Noncommutative Geometry and - Alain Connes](https://alainconnes.org/wp-content/uploads/selecta.ps-2.pdf) - We shall give in this paper a spectral interpretation of the zeros of the Rie- mann zeta function an...

36. [Alain Connes: Spectral Triples and Zeta Cycles - YouTube](https://www.youtube.com/watch?v=kNXPe1u81pA) - Abstract: This is joint work with C. Consani. When contemplating the low lying zeros of the Riemann ...

37. [[1902.05306] Spectral Action in Noncommutative Geometry - arXiv](https://arxiv.org/abs/1902.05306) - This book offers a guided tour through the mathematical habitat of noncommutative geometry à la Conn...

38. [FEDONet: Fourier-Embedded DeepONet for Spectrally ... - arXiv](https://arxiv.org/html/2509.12344v4) - This work demonstrates the effectiveness of Fourier embeddings in enhancing neural operator learning...

39. [[PDF] Basis-specific Neural Operators - Schools CEA – EDF – INRIA](https://ecoles-cea-edf-inria.fr/files/2025/06/Neural_Operators_new_Part2.pdf) - ... DeepOnet can enhance accuracy and generalization but hybrid training is more robust. ❑ Fourier n...

40. [Spectral Neural Operators - NASA ADS](https://ui.adsabs.harvard.edu/abs/2022arXiv220510573F) - Recently introduced Fourier Neural Operator (FNO) and Deep Operator Network (DeepONet) can provide t...

41. [[PDF] Spectral Neural Operators - arXiv](https://arxiv.org/pdf/2205.10573.pdf) - Recently introduced Fourier Neural Operator. (FNO) and Deep Operator Network (DeepONet) can provide ...

42. [Deep Learning Solution of the Eigenvalue Problem for Differential...](https://openreview.net/forum?id=m4baHw5LZ7M) - We introduce a novel Neural Network (NN)-based solver for the eigenvalue problem of differential sel...

43. [Deep Learning Solution of the Eigenvalue Problem for Differential ...](https://direct.mit.edu/neco/article/35/6/1100/115600/Deep-Learning-Solution-of-the-Eigenvalue-Problem) - We introduce a novel neural network–based solver for the eigenvalue problem of differential self-adj...

44. [[PDF] STNet: Spectral Transformation Network for Solving Operator ...](https://openreview.net/pdf?id=2UpXNbZDyt) - Operator eigenvalue problems play a critical role in various scientific fields and engineering appli...

45. [Spectral Transformation Network for Solving Operator Eigenvalue ...](https://neurips.cc/virtual/2025/poster/116110) - By leveraging approximate eigenvalues and eigenvectors from iterative processes, STNet uses spectral...

46. [[PDF] Operator learning without the adjoint](http://www.jmlr.org/papers/volume25/24-0162/24-0162.pdf) - Abstract. There is a mystery at the heart of operator learning: how can one recover a non-self- adjo...

47. [Learning Dynamical Systems via Koopman Operator Regression in ...](https://arxiv.org/abs/2205.14027) - We study a class of dynamical systems modelled as Markov chains that admit an invariant distribution...

48. [[2102.12086] Modern Koopman Theory for Dynamical Systems - arXiv](https://arxiv.org/abs/2102.12086) - In this review, we provide an overview of modern Koopman operator theory, describing recent theoreti...

49. [[PDF] Learning Dynamical Systems via Koopman Operator Regression in ...](https://proceedings.neurips.cc/paper_files/paper/2022/file/196c4e02b7464c554f0f5646af5d502e-Paper-Conference.pdf) - More precisely, the Koopman operator describes how functions (observables) of the state of the syste...

50. [Operator Learning in Dynamical Systems - Emergent Mind](https://www.emergentmind.com/topics/operator-learning-in-dynamical-systems) - The spectrum of learned operators—e.g., Koopman or transfer operators—organizes and reveals key dyna...

51. [[PDF] The Adaptive Spectral Koopman Method for Dynamical Systems](https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/_DEPARTMENTS/ise/pdf/tech-papers/23/23T_005.pdf) - Abstract. Dynamical systems have a wide range of applications in mechanics, electrical engineering, ...

52. [Crossover in nonstandard random-matrix spectral fluctuations ...](https://link.aps.org/doi/10.1103/PhysRevE.98.022110) - By using the trend modes, we perform data-adaptive unfolding, and we calculate traditional spectral ...

53. [Multifractal eigenstates of quantum chaos and the Thue-Morse sequence](https://www.academia.edu/24758838/Multifractal_eigenstates_of_quantum_chaos_and_the_Thue_Morse_sequence) - We analyze certain eigenstates of the quantum baker's map and demonstrate, using the Walsh-Hadamard ...

54. [Studies in Quantum Chaos: From an almost](https://citeseerx.ist.psu.edu/document?doi=42e1595dfe5da58f45c2547a10f2e1e31831bdce&repid=rep1&type=pdf)

55. [Spectral Continuity for Aperiodic Quantum Systems I. General Theory](https://arxiv.org/abs/1709.00975) - A characterization of the convergence of the spectra by the convergence of the underlying structures...

56. [[1803.03099] Spectral Continuity for Aperiodic Quantum Systems II ...](https://arxiv.org/abs/1803.03099) - The connection of branching vertices in the GAP-graphs and defects is discussed. Comments: 30 pages,...

57. [[PDF] The Tropical Eigenvalue-Vector Problem from Algebraic, Graphical ...](https://scarab.bates.edu/cgi/viewcontent.cgi?article=1119&context=honorstheses) - Incorporating linear algebra, graph theory, and recurrence relations, discrete event systems can be ...

58. [[PDF] Tropical Geometry: - UT Math](https://web.ma.utexas.edu/users/gdavtor/notes/tropical_notes.pdf) - In max-plus, the eigenvalue λ is the maximum mean weighted cycle in the weighted graph determined by...

59. [[1410.5361] Tropical Spectral Theory of Tensors - arXiv.org](https://arxiv.org/abs/1410.5361) - Abstract:We introduce and study tropical eigenpairs of tensors, a generalization of the tropical spe...

60. [[PDF] Max-plus algebra - CMAP](http://www.cmap.polytechnique.fr/~gaubert/siamct09/slidesgaubertsiamct09.pdf) - Let A := a + b, a,b ∈ Cn×n, let λ1,...,λn denote the eigenvalues of A ordered by increasing valuatio...

61. [[PDF] Generalized Solutions of Bellman's Differential Equa- tion](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/kolokoltsov/ch3.pdf) - The theory of new distributions introduced in Chapter 1 can be used to define generalized solutions ...

62. [Improve Your On-Policy Actor-Critic with Positive Advantages - arXiv](https://arxiv.org/html/2306.01460v4) - VSOP consists of three simple modifications to the A3C algorithm: (1) applying a ReLU function to ad...

63. [Actor-Critic Methods: SAC and PPO | Joel's PhD Blog](https://joel-baptista.github.io/phd-weekly-report/posts/ac/) - Both PPO and SAC are designed to optimize stochastic policies for tasks that involve continuous acti...

64. [Effective Reinforcement Learning Control using Conservative Soft ...](https://arxiv.org/html/2505.03356v1) - Traditional methods such as Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) address t...

65. [Deep reinforcement learning for quantum Hamiltonian engineering](https://arxiv.org/abs/2102.13161) - Here we numerically search for Hamiltonian engineering sequences using deep reinforcement learning (...

66. [Deep Reinforcement Learning for Quantum Hamiltonian Engineering](https://link.aps.org/doi/10.1103/PhysRevApplied.18.024033) - Here we numerically search for Hamiltonian engineering sequences using deep reinforcement learning (...

67. [Scalable Hierarchical Reinforcement Learning for Hyper Scale Multi ...](https://arxiv.org/abs/2412.19538) - We construct an efficient multi-stage HRL-based multi-robot task planner for hyper scale MRTP in RMF...

68. [[PDF] Hierarchical Multi-Agent Reinforcement Learning](https://mohammadghavamzadeh.github.io/PUBLICATIONS/jaamas06.pdf) - Abstract. In this paper, we investigate the use of hierarchical reinforcement learning (HRL) to spee...

69. [Ant colony optimization algorithms - Wikipedia](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)

70. [[PDF] Development of an Ant Colony Optimization (ACO) Algorithm Based ...](https://skoge.folk.ntnu.no/prost/proceedings/adchem2015/media/papers/0147.pdf)

71. [Coupling urban cellular automata with ant colony optimization for zoning protected natural areas under a changing landscape](https://www.tandfonline.com/doi/full/10.1080/13658816.2010.481262) - Optimal zoning of protected natural areas is important for conserving ecosystems. It is an NP-hard p...

72. [Inverse Spectral Problems for Differential Operators Research Guide](https://papersflow.ai/research/topics/spectral-theory-in-mathematical-physics/inverse-spectral-problems-for-differential-operators) - Inverse spectral problems for differential operators reconstruct potentials, coefficients, or geomet...

73. [Quantum chaos - Scholarpedia](http://www.scholarpedia.org/article/Quantum_chaos) - Quantum Chaos describes and tries to understand the nature of the wave-like motions for the electron...

