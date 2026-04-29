# Artin Billiard, Quantum Chaos, and the Hilbert–Pólya Program
## A Research-Level Literature Review with DTES, RL, and ACO Perspectives

***

## Executive Overview

This review synthesizes the classical and modern literature on the Artin billiard — geodesic flow on the modular surface — its role as a paradigmatic quantum chaotic system, and the deep but incomplete connections between its spectrum and the nontrivial zeros of the Riemann zeta function. Sections 1–5 cover the mathematically established theory: classical dynamics, symbolic coding, spectral decomposition, the Selberg trace formula, and the Bohigas–Giannoni–Schmit (BGS) conjecture. Sections 6–7 review the Hilbert–Pólya conjecture and principal proposals (Berry–Keating, Connes) that attempt to realize zeta zeros as a spectrum. Sections 8–13 develop a speculative but structured modern framework combining learnable Laplacian deformations, DTES-type latent geometry, reinforcement learning (RL), and ant colony optimization (ACO) for spectral matching. Throughout, proven theorems are distinguished from numerical evidence and open conjectures, and concrete computational experiments are proposed.

***

## 1. Classical Artin Billiard

### 1.1 The System

The Artin billiard was introduced by Emil Artin in 1924 as the geodesic motion of a free particle on the non-compact Riemann surface \(\mathcal{H}/\Gamma\), where \(\mathcal{H} = \{x + iy \mid y > 0\}\) is the Poincaré upper half-plane and \(\Gamma = \mathrm{PSL}(2,\mathbb{Z})\) is the modular group. The Riemannian metric is the canonical Poincaré metric[^1]

\[
ds^2 = \frac{dx^2 + dy^2}{y^2},
\]

which endows \(\mathcal{H}\) with constant sectional curvature \(-1\). The fundamental domain of \(\mathrm{PSL}(2,\mathbb{Z})\) is[^2]

\[
\mathcal{F} = \left\{ z \in \mathcal{H} : |z| > 1,\ |\mathrm{Re}(z)| < \tfrac{1}{2} \right\},
\]

an infinite hyperbolic triangle with a cusp at \(i\infty\) and two elliptic fixed points at \(i\) (order 2) and \(e^{2\pi i/6}\) (order 3). The billiard flow identifies the sides of \(\mathcal{F}\) under the modular transformations \(z \mapsto z+1\) and \(z \mapsto -1/z\), making trajectories that exit one side re-enter from its image.[^3][^1]

The Hamiltonian is the free-particle kinetic energy in the hyperbolic metric,

\[
H(p,q) = \frac{1}{2m} p_i p_j g^{ij}(q),
\]

so the equations of motion reduce to the geodesic equations, whose solutions are semicircles orthogonal to the real axis or vertical lines.[^4][^1]

### 1.2 Ergodicity, Mixing, and Entropy

**Theorem (Artin 1924; Hopf 1939; Anosov 1967).** The geodesic flow on any hyperbolic surface of finite area is ergodic with respect to the Liouville measure. For the Artin billiard specifically, the flow is not merely ergodic but possesses:[^5][^6]

- **Strong mixing of all orders** — time correlations decay;
- **Lebesgue spectrum** — the spectral measure of the flow is absolutely continuous;
- **Positive Kolmogorov–Sinai entropy** \(h_{KS} = 1\) (in units where the curvature radius is 1);
- **Exponential instability** — Lyapunov exponents equal \(\pm 1\), so nearby geodesics diverge as \(e^{|t|}\)[^7][^8].

The Artin billiard thus realizes an **Anosov flow** on its phase space \(T^1\mathcal{F}\): the unit tangent bundle splits into stable, unstable, and flow directions, and the resulting hyperbolic structure is responsible for all ergodic properties. Hadamard in 1898 first observed that negative curvature produces erratic geodesic behaviour; Hopf's 1939 proof of ergodicity used what is now called the Hopf argument; Anosov's 1967 paper provided the general framework for C\(^2\) uniformly hyperbolic flows.[^6][^9][^10][^5]

***

## 2. Symbolic Dynamics of Geodesic Flow

### 2.1 Continued Fraction Coding

A key tool introduced in Artin's original paper is **symbolic dynamics**: the assignment of an infinite symbolic sequence to each geodesic trajectory. Geodesics on \(\mathcal{H}/\Gamma\) correspond bijectively (up to a measure-zero exceptional set) to bi-infinite sequences of continued fraction partial quotients. Specifically, Series (1985) showed that the geodesic flow on the modular surface \(\mathrm{SL}(2,\mathbb{Z})\backslash\mathcal{H}\) is a quotient of a special flow over the **Gauss shift**[^1]

\[
T: (0,1) \to [0,1), \quad T(x) = \frac{1}{x} - \left\lfloor \frac{1}{x} \right\rfloor,
\]

with roof function \(r(x) = \log(1/x)\). Two geodesics are conjugate under \(\mathrm{GL}(2,\mathbb{Z})\) if and only if their symbolic sequences are shift-equivalent. Adler–Flatto and later Katok–Ugarcovici provided complementary codings; Boca–Merriman (2018) extended these to modular surfaces associated with odd and even continued fractions.[^11][^12][^13][^14]

### 2.2 Markov Partitions and Shift Spaces

Sinai (1968) first introduced Markov partitions for Anosov diffeomorphisms; Ratner (1969, 1973) extended the construction to Anosov flows; Smale's horseshoe provided the canonical example of a symbolic representation for a uniformly hyperbolic invariant set. For the Artin billiard flow, the partition into the boundary segments of \(\mathcal{F}\) defines a shift of finite type, and the symbolic dynamics is the **full shift** on a countably infinite alphabet — in contrast to compact hyperbolic surfaces where the alphabet is finite.[^15][^16][^4]

**Comparison with substitution systems.** In deterministic substitution sequences such as Thue–Morse or Fibonacci, the symbolic dynamics is **not chaotic** in the Kolmogorov sense: these sequences have **zero entropy** and a **pure point or mixed spectral measure** (with possible singular continuous component). Chaotic hyperbolic systems generate shift spaces whose Lyapunov exponents are uniformly positive, entropy is maximal on the symbolic level, and correlation functions decay exponentially. Substitution sequences arise in quasi-crystalline settings and are better modeled by symbolic substitution monoids rather than Markov chains. The qualitative gap between these two classes — positive vs. zero topological entropy — is a fundamental divide in the theory of symbolic dynamical systems.[^17]

### 2.3 Dynamical Zeta Functions

The symbolic structure allows explicit computation of the **dynamical zeta function**

\[
\zeta_{\mathrm{dyn}}(s) = \exp\!\sum_{\gamma} \frac{e^{-s\ell(\gamma)}}{|\det(\mathrm{Id} - P_\gamma)|^{1/2}},
\]

where the sum runs over primitive periodic orbits \(\gamma\) with length \(\ell(\gamma)\). Mayer (1991) expressed this as the Fredholm determinant of a transfer operator for the Gauss map, establishing the meromorphic continuation of \(\zeta_{\mathrm{dyn}}\) to the entire complex plane. This is the prototypical example of a thermodynamic formalism connection between hyperbolic dynamics and analytic number theory.[^4]

***

## 3. Quantum Artin Billiard

### 3.1 The Laplace–Beltrami Operator

Quantization of the Artin billiard replaces the classical geodesic flow with the **Laplace–Beltrami operator** on \(\mathcal{F}\):

\[
\Delta = -y^2 \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right).
\]

The eigenvalue equation

\[
\Delta \psi + \lambda \psi = 0, \quad \lambda = \tfrac{1}{4} + r^2, \quad r \in \mathbb{R} \cup i\left[0,\tfrac{1}{2}\right),
\]

is the quantum mechanical version of the Artin billiard. The system is **exactly solvable** in the sense that the eigenfunctions can be expressed in terms of Bessel and Whittaker functions.[^18][^1]

### 3.2 Spectrum and Automorphic Forms

The spectrum of \(\Delta\) on \(L^2(\mathcal{F})\) decomposes into two parts:[^18]

**Discrete spectrum.** The discrete eigenvalues \(\lambda_n = 1/4 + r_n^2\) correspond to **Maass cusp forms** — automorphic functions vanishing at the cusp which satisfy \(\Delta\psi_n = \lambda_n\psi_n\) and lie in \(L^2(\mathcal{F})\). These are among the deepest objects in number theory, connecting spectral geometry to L-functions and the Ramanujan conjecture. Hecke operators \(T_p\) (for primes \(p\)) commute with \(\Delta\), so eigenfunctions can be simultaneously taken as Hecke–Maass forms. The first nontrivial eigenvalue is approximately \(\lambda_1 \approx 91.14\), corresponding to \(r_1 \approx 9.53\).[^19][^20]

**Continuous spectrum.** The continuous spectrum occupies \(

\[
E(z,s) = \sum_{\gamma \in \Gamma_\infty \backslash \Gamma} (\mathrm{Im}\,\gamma z)^s,
\]

and then continued meromorphically to all of \(\mathbb{C}\). On the line \(\mathrm{Re}(s) = 1/2\), the Eisenstein series provide the spectral resolution of the continuous part; the scattering matrix is \(\varphi(s) = \xi(2s-1)/\xi(2s)\) where \(\xi\) is the completed Riemann zeta function[^21][^18]. The **presence of a continuous spectrum** originating from the non-compact cusp is the most important structural obstacle for the Hilbert–Pólya program.

**Quantum unique ergodicity.** Lindenstrauss (2006) proved that Hecke–Maass forms equidistribute: for any compact subset \(A \subset \mathcal{F}\),

\[
\int_A |\psi_n(z)|^2\,d\mu \to \frac{\mu(A)}{\mu(\mathcal{F})}
\]

as \(\lambda_n \to \infty\), where \(\mu\) is the hyperbolic area measure. Soundararajan (2010) complemented this by eliminating the possibility of mass escaping into the cusp, establishing the Quantum Unique Ergodicity (QUE) conjecture of Rudnick–Sarnak for the full modular surface[^22][^23][^24].

***

## 4. Selberg Trace Formula

### 4.1 Statement

The **Selberg trace formula** is the central identity in spectral geometry on hyperbolic surfaces, discovered by Selberg in the early 1950s and proved rigorously shortly thereafter[^25]. For the modular surface (or more generally, a hyperbolic surface of finite area), it takes the schematic form

\[
\sum_n h(r_n) = \frac{\mathrm{Area}(\mathcal{F})}{4\pi} \int_{-\infty}^{\infty} r\tanh(\pi r)\,\hat{h}(r)\,dr + \sum_{\{\gamma\}_{\mathrm{prim}}} \sum_{k=1}^{\infty} \frac{\ell(\gamma)}{2\sinh(k\ell(\gamma)/2)} \hat{h}(k\ell(\gamma)),
\]

where \(h\) is a suitable test function, \(\hat{h}\) its Fourier transform, and the right-hand side sums over primitive conjugacy classes \(\{\gamma\}\) (hyperbolic elements of \(\Gamma\)) with lengths \(\ell(\gamma)\)[^25][^26]. The formula exactly parallels the **explicit formulas** in analytic number theory, which express \(\sum_\rho f(\rho)\) (sum over Riemann zeros) as a sum over prime powers[^27].

### 4.2 Analogy with Number Theory

| Selberg trace formula | Explicit formula for \(\zeta\) |
|---|---|
| Eigenvalues \(\lambda_n = 1/4 + r_n^2\) | Zeros \(\rho_n = 1/2 + i\gamma_n\) |
| Primitive closed geodesics of length \(\ell(\gamma)\) | Primes \(p\) (and prime powers) |
| Length spectrum \(\{e^{\ell(\gamma)}\}\) | Primes \(p^k\) |
| Spectral determinant / Selberg zeta | Riemann zeta \(\zeta(s)\) |

This analogy, noted by Selberg himself and developed extensively by Hejhal (1976, 1983), Gutzwiller (1971), and many others[^28][^29], represents the **geometric realization of a prime-like structure**. The Selberg zeta function

\[
Z(s) = \prod_{\{\gamma\}_{\mathrm{prim}}} \prod_{k=0}^{\infty} \left(1 - e^{-(s+k)\ell(\gamma)}\right)
\]

admits a functional equation and its zeros correspond to eigenvalues of \(\Delta\). For the modular surface specifically, there is an explicit relation between the Selberg zeta function \(Z(s)\) and the Riemann zeta function \(\zeta(s)\)[^30]:

\[
Z(s) = \xi(2s-1)^{-1} \cdot (\text{factors from discrete spectrum and cusp contributions}),
\]

making the analogy between geodesic lengths and primes not merely formal but arithmetically precise.

### 4.3 Gutzwiller Trace Formula (Semiclassical)

For general quantum chaotic systems lacking the exact Selberg structure, Gutzwiller (1971) derived a semiclassical approximation to the density of states:

\[
\rho(E) \approx \bar{\rho}(E) + \frac{1}{\pi\hbar} \mathrm{Re}\sum_{\gamma\,\mathrm{prim}} \sum_{k=1}^{\infty} A_\gamma^k\, e^{i k S_\gamma/\hbar - i\mu_\gamma\pi/2},
\]

where \(S_\gamma\) is the classical action, \(\mu_\gamma\) the Maslov index, and \(A_\gamma\) an amplitude depending on the monodromy[^29][^31]. For geodesic flow on hyperbolic surfaces, the Gutzwiller formula coincides exactly with the Selberg trace formula, making the latter both an exact and semiclassical identity[^32].

***

## 5. Quantum Chaos and GUE Statistics

### 5.1 BGS Conjecture

**Conjecture (Bohigas–Giannoni–Schmit, 1984).** The unfolded eigenvalue statistics of a classically chaotic, time-reversal invariant quantum system coincide with those of the **Gaussian Orthogonal Ensemble (GOE)** of random matrices. For time-reversal breaking systems, the statistics coincide with the **Gaussian Unitary Ensemble (GUE)**[^33][^34].

This conjecture, supported by overwhelming numerical evidence and heuristic arguments based on the Gutzwiller trace formula and diagonal approximations, remains unproven in full generality, though Andreev–Simons (1996) gave a formal derivation via a nonlinear sigma model[^35][^36]. The mechanism relies on the existence of a gap in the Perron–Frobenius spectrum of the underlying classical dynamics[^35].

### 5.2 Montgomery–Dyson Observation

In 1972, Montgomery computed that the pair correlation function of Riemann zeta zeros on the critical line has the form

\[
1 - \left(\frac{\sin(\pi u)}{\pi u}\right)^2,
\]

which Dyson immediately recognized as the GUE two-point correlation function[^37][^38]. This **Montgomery–Dyson connection**, later confirmed numerically for \(10^{20}\)-scale zeros by Odlyzko[^39], is the strongest empirical evidence for the Hilbert–Pólya conjecture and links the Riemann zeros to quantum chaos.

### 5.3 Arithmetic Deviation: The Role of Hecke Operators

The spectral statistics of the **modular surface** deviate from pure GUE in a subtle way. The presence of Hecke operators introduces arithmetic correlations among the eigenvalues \(\lambda_n\), causing deviations from pure RMT at short range — a phenomenon called **arithmetic quantum chaos** or "arithmetic degeneracy" (Bogomolny–Georgeot–Giannoni–Schmit, 1992; Bolte–Steil 1992)[^40][^21]. Numerical computations show that the modular surface eigenvalues agree with GUE only **after** removing these arithmetic degeneracies; the generic non-arithmetic hyperbolic surfaces (Hecke triangle groups with non-arithmetic parameters) appear to show purer GUE statistics[^40][^21]. This is a fundamental structural distinction: the arithmetic nature of \(\mathrm{PSL}(2,\mathbb{Z})\) is both the source of the connection to number theory and an obstacle to clean BGS universality.

***

## 6. Hilbert–Pólya Conjecture: Geometric View

### 6.1 Formulation

**Conjecture (Hilbert–Pólya, ca. 1910–1920; published: Montgomery 1973).** There exists a self-adjoint operator \(H\) on a Hilbert space such that the nontrivial zeros of the Riemann zeta function are \(\rho_n = 1/2 + i\gamma_n\), where \(\{\gamma_n\}\) are the eigenvalues of \(H\)[^27][^41].

A proof would establish the Riemann hypothesis immediately, since eigenvalues of a self-adjoint operator are real, implying \(\gamma_n \in \mathbb{R}\), hence all zeros on the critical line. The conjecture is supported by:

- The Montgomery–Dyson GUE connection;
- Selberg's trace formula analogy;
- The distribution of zeros satisfying Weyl's law: \(N(T) \sim \frac{T}{2\pi}\log\frac{T}{2\pi} - \frac{T}{2\pi}\), consistent with the semiclassical density of states formula[^27].

### 6.2 Can the Modular Laplacian Serve as the Hilbert–Pólya Operator?

The Laplacian \(\Delta\) on \(\mathcal{F}\) has several properties one would want from a Hilbert–Pólya operator: it is self-adjoint, its eigenvalue statistics show GUE-like behavior (after arithmetic corrections), and the Selberg trace formula mirrors the explicit formula. However, it **fails** to be the Hilbert–Pólya operator for the following concrete reasons[^18][^30]:

1. **Mismatch of discrete spectra.** The discrete eigenvalues \(\lambda_n = 1/4 + r_n^2\) are numerically distinct from \(1/4 + \gamma_n^2\). The first eigenvalue \(r_1 \approx 9.53\) does not match the first Riemann zero \(\gamma_1 \approx 14.13\).

2. **Presence of continuous spectrum.** The continuous spectrum \(

3. **Arithmetic structure.** The eigenvalues carry Hecke symmetry; the Riemann zeros do not appear to have an analogous structure.

4. **The scattering matrix.** The determinant of the scattering matrix equals \(\zeta(2s-1)/\zeta(2s)\), and the **resonances** (poles of the scattering matrix, i.e., zeros of \(\zeta\)) contribute to the continuous spectrum rather than appearing as discrete eigenvalues[^42].

**Speculative relation.** Bogomolny and others have proposed that the Riemann zeros appear as **resonances** (poles of the resolvent in the lower half-plane) of the quantum Artin billiard rather than as bound-state eigenvalues. The scattering matrix connection supports this: the Riemann zeros do determine the widths and positions of resonances[^42]. This is a weaker connection than the Hilbert–Pólya program requires.

***

## 7. Bridging Hypothesis: Deformed Hyperbolic Laplacians

### 7.1 The Berry–Keating Proposal

Berry and Keating (1999) conjectured that the classical Hamiltonian \(H = xp\) — a particle on the half-line — is related to the Riemann zeros when appropriately quantized[^43][^44][^45]. The normal-ordered quantum operator is

\[
\hat{H}_{\mathrm{BK}} = \frac{1}{2}(\hat{x}\hat{p} + \hat{p}\hat{x}) = -i\hbar\left(x\frac{d}{dx} + \frac{1}{2}\right),
\]

with classical trajectory \(x(t) = x_0 e^t\), \(p(t) = p_0 e^{-t}\), which is **not periodic** — a fundamental problem since the semiclassical spectrum of a system with no periodic orbits is continuous, not discrete[^43]. Berry and Keating introduced cutoffs \(|x| \geq \ell_x\), \(|p| \geq \ell_p\) to generate a discrete approximation; the resulting semiclassical energies

\[
E_n \sim 2\pi\hbar \cdot n \bigg/ \log\!\left(\frac{E_n}{2\pi\hbar}\right)
\]

reproduce the **average density** of Riemann zeros but not the exact sequence[^43][^46]. Sierra–Rodriguez-Laguna (2011) modified the Hamiltonian to \(H = x(p + \ell_p^2/p)\), which has closed periodic orbits and reproduces the average distribution more accurately[^44]. The exact zeros remain unreproduced by any xp-type model to date.

### 7.2 Connes' Noncommutative Approach

Connes (1998, 1999) gave a spectral realization of the **critical zeros of all Riemann zeta functions and L-functions** as an absorption spectrum on the Hilbert space of the noncommutative space of Adèle classes \(\mathbb{A}_\mathbb{Q}/\mathbb{Q}^*\)[^47][^48]. The key results are:

- The explicit formulas of number theory are realized as a trace formula on the adèle class space;
- The Riemann zeros appear as **missing spectral lines** (absorption spectrum), not eigenvalues;
- The trace formula on the noncommutative space mirrors the Selberg trace formula but with the primes playing the role of closed geodesics[^47][^48].

Connes' approach reduces the Riemann hypothesis to the **positivity of a distributional pairing** (validity of the trace formula for all test functions). While conceptually powerful, it has not yet produced a proof and faces the challenge that the positivity condition is as hard to verify directly as the Riemann hypothesis itself[^49].

### 7.3 Key Obstacles

The major structural obstacles to any hyperbolic-geometric Hilbert–Pólya operator are:

| Obstacle | Description |
|---|---|
| **Continuous spectrum** | The non-compact cusp creates a continuous spectrum \(
***

## 8. Learnable Deformations of the Laplacian

### 8.1 Spectral Geometry with Potentials

Consider the family of perturbed operators

\[
H_\theta = \Delta_{\mathrm{hyp}} + V_\theta(z),
\]

where \(V_\theta : \mathcal{F} \to \mathbb{R}\) is a parameterized potential and \(\Delta_{\mathrm{hyp}}\) is the hyperbolic Laplacian[^50]. The **inverse spectral problem** — recovering \(V_\theta\) from the spectrum — is the central challenge.

**Theorem (Gel'fand–Levitan; Uhlmann etc.)** For compact Riemannian manifolds, the spectrum of \(\Delta + V\) determines \(V\) up to isometry in certain cases (Kac's drum problem analogs). On non-compact hyperbolic surfaces, the problem is significantly harder due to the presence of continuous spectrum and resonances[^51]. Colin de Verdière (1982) proved that perturbations of the Laplacian on hyperbolic surfaces can shift embedded eigenvalues into the continuous spectrum, creating resonances — a generic instability[^50].

A trace formula for a delta potential on a hyperbolic surface of finite volume has been established, providing an analogue of the Selberg trace formula for singular perturbations of the Laplacian[^52][^53]:

\[
\sum_n h\!\left(\sqrt{\lambda_n^{(\theta)} - \tfrac{1}{4}}\right) = \text{(geometric terms)} + \text{(residue from shifted resonances)}.
\]

### 8.2 Operator Learning Framework

Recent advances in **neural operator learning** are directly applicable to the inverse spectral problem. The Geometric Laplace Neural Operator (GLNO) embeds Laplace spectral representations into the eigen-basis of the Laplace–Beltrami operator on Riemannian manifolds, enabling operator learning on non-Euclidean geometries[^54]. The Spectral Correction Network (SC-Net) operates in the spectral domain of a forward operator, learning a filter function \(\Psi_\theta(\sigma_n, y_n)\) that reweights spectral coefficients — a natural architecture for the problem of spectral alignment[^55].

**Key question:** Can one parameterize \(V_\theta\) via a neural network and optimize

\[
\mathcal{L}(\theta) = d\!\left(\mathrm{Spec}(H_\theta),\, \{\gamma_n\}\right)
\]

for some spectral distance \(d\)? This is an operator-valued inverse problem with the following well-posedness issues:

- The discrete spectrum of \(H_\theta\) changes discontinuously as embedded eigenvalues are pulled into the continuous spectrum[^50];
- Self-adjointness requires \(V_\theta\) to be real-valued, restricting the parameter space;
- On the modular surface, the scattering matrix (and hence the resonances) depend on \(V_\theta\) in a nonlinear way through Eisenstein series perturbations[^53].

Physics-informed neural networks (PINNs) and physics-informed Kolmogorov–Arnold networks (PIKANs) have demonstrated effectiveness for solving inverse problems in unbounded domains[^56]; their spectral-domain variants provide a natural toolbox. Holliday–Lindner–Ditto (2023) showed that physics-informed ML can directly compute eigenvalues of the Helmholtz equation on curved domains[^57], providing proof of concept for hyperbolic settings.

***

## 9. DTES Interpretation (Speculative but Structured)

### 9.1 Definition

The **Dynamic Topological Energy Space (DTES)** framework interprets the hyperbolic geometry of the Artin billiard as a **latent spectral manifold** — a structured energy landscape that encodes spectral information geometrically. While DTES as a term does not correspond to a single established publication, the concept is grounded in several well-developed areas:

- **Spectral geometry** — the correspondence between geometric data (curvature, topology) and spectral data (\(\mathrm{Spec}(\Delta)\));
- **Tropical geometry** — piecewise-linear (PL) limits of algebraic geometry, where the tropicalization of a curve encodes combinatorial skeleta[^58][^59];
- **Multi-scale analysis** — hierarchical decomposition of spectral manifolds into regimes corresponding to different energy scales.

### 9.2 Geometric Realization

In the DTES interpretation:

\[
E_{\mathrm{DTES}} \sim \text{hyperbolic metric deformation} = g_{ij}^{(\theta)},
\]

where the metric is continuously deformed from the constant-curvature Poincaré metric. The **cusp** of the modular surface corresponds to a **singular energy region** where the spectral density diverges (associated with the continuous spectrum); **geodesic reflections** at the boundary of \(\mathcal{F}\) represent **energy regime transitions**. The multi-scale structure arises from the hierarchical nature of continued fraction encodings: long geodesics correspond to large partial quotients, encoding coarse-scale behavior, while short segments correspond to fine-scale spectral features.

**Tropical limit.** As a deformation parameter \(\hbar \to 0\), the spectral curves of integrable systems tropicalize to piecewise-linear graphs[^60]. Applied speculatively to the spectral problem, a tropical limit of the operator \(H_\theta\) would replace the elliptic PDE with a max-plus algebraic system — potentially amenable to combinatorial analysis. Maragos–Theodosis (2019) showed that tropical geometry and max-plus algebra share the same structure as mathematical morphology, with applications to PL regression and optimization[^58]. This provides a concrete bridge between the continuous spectral problem and discrete optimization.

***

## 10. RL Formulation for Spectral Optimization

### 10.1 MDP Formulation

The spectral matching problem \(\mathrm{Spec}(H_\theta) \approx \{\gamma_n\}\) can be cast as a **Markov Decision Process**:

| Component | Definition |
|---|---|
| **State** \(s_t\) | Current potential \(V_\theta\) (encoded as function values on a grid), current spectral slice \(\{\lambda_n^{(\theta)}\}_{n \leq N}\) |
| **Action** \(a_t\) | Local perturbation of \(V_\theta\): add/subtract a bump function at location \((x,y) \in \mathcal{F}\), or adjust a Fourier/Laplace-Beltrami coefficient |
| **Reward** | \(R_t = -d\!\left(\{\lambda_n^{(\theta)}\}, \{\gamma_n\}\right)\), e.g., \(-\sum_{n=1}^N |\lambda_n^{(\theta)} - (1/4+\gamma_n^2)|^2\) |
| **Terminal condition** | Agreement within tolerance \(\epsilon\) for the first \(N\) eigenvalues |

### 10.2 Relevant RL Literature

Bukov–Marquardt (2026) provide a comprehensive review of RL in quantum technology, including state preparation, gate optimization, and variational eigensolvers[^61]. The key insight from their survey is that RL agents can navigate the non-convex landscape of quantum control problems where gradient-based methods fail due to local optima and barren plateaus. Physics-informed RL, reviewed by the AEx survey (2025), demonstrates the value of encoding physical constraints (self-adjointness, spectral gap requirements) into the reward and transition functions[^62].

For the spectral optimization problem, the **challenges** are:

- The spectral map \(\theta \mapsto \{\lambda_n^{(\theta)}\}\) is generically non-smooth due to level crossings;
- The continuous spectrum of the unperturbed operator creates an unstable component in the reward landscape;
- High-dimensional parameter spaces (functionals \(V_\theta: \mathcal{F} \to \mathbb{R}\)) require efficient exploration strategies.

**Rigorous data-driven computation of spectral properties of Koopman operators** (Colbrook–Townsend, 2021) provides an important methodological precedent: residual Dynamic Mode Decomposition (ResDMD) computes spectra and pseudospectra of general operators from trajectory data with convergence guarantees[^63], applicable to iterating the RL environment.

### 10.3 RL for Infinite-Dimensional Systems

Recent work on RL for infinite-dimensional systems (JMLR 2024) establishes theoretical foundations for RL policies over function spaces with spectral convergence guarantees[^64], directly relevant to the problem of learning functional potentials \(V_\theta \in L^2(\mathcal{F})\).

***

## 11. ACO / Multi-Agent Spectral Search

### 11.1 Ant Colony Optimization Framework

**Ant Colony Optimization (ACO)**, introduced by Dorigo (1992) and formalized as a metaheuristic by Dorigo–Stützle[^65][^66], is a population-based algorithm in which artificial agents (ants) construct solutions by traversing a graph, depositing pheromone proportional to solution quality. For the spectral matching problem, the natural formulation is:

- **Graph nodes:** discretized parameter values of the potential \(V_\theta\) on a grid over \(\mathcal{F}\), or coefficients in a spectral expansion;
- **Pheromone:** accumulated spectral alignment quality \(-d(\mathrm{Spec}(H_\theta), \{\gamma_n\})\) along parameter paths;
- **Heuristic function:** gradient of \(\mathcal{L}(\theta)\) or sensitivity \(\partial\lambda_n/\partial\theta_k\).

### 11.2 Suitability for Rugged Spectral Landscapes

The spectral landscape \(\theta \mapsto \mathcal{L}(\theta)\) is expected to be highly non-convex due to:

1. **Level crossings** — eigenvalues \(\lambda_n(\theta)\) can switch order or merge with the continuous spectrum as \(\theta\) varies, creating non-smooth transitions;
2. **Arithmetic structure** — the discrete spectrum of arithmetic surfaces is highly constrained, creating narrow "channels" in parameter space where eigenvalues align with Riemann zeros;
3. **Multiple scales** — large eigenvalues require fine-scale potential modifications while small eigenvalues are sensitive to global metric deformations.

ACO's distributed, parallel search with positive feedback is well-suited to such landscapes: the pheromone mechanism allows accumulation of information about globally promising regions without requiring differentiability[^67][^66]. Blum (2005) provides the key benchmarks for ACO on continuous and combinatorial problems[^68]; its relative advantage over gradient descent arises precisely in multi-modal, discontinuous, or high-dimensional settings.

**Comparison with gradient-based methods.** Gradient methods such as Adam or L-BFGS applied to \(\mathcal{L}(\theta)\) face the spectral non-smoothness problem directly: the eigenvalue map is not differentiable at level crossings. Techniques from **non-smooth optimization** (Clarke subdifferential, bundle methods) can handle this formally, but ACO's derivative-free nature provides a practical alternative. A hybrid strategy — ACO for global exploration, gradient descent for local refinement — is the standard best practice in metaheuristic optimization[^67].

***

## 12. Key Obstacles (Synthesis)

| Obstacle | Mathematical Status | Computational Impact |
|---|---|---|
| **Continuous vs. discrete spectrum** | Proven: modular surface has continuous spectrum \(
***

## 13. Open Problems

### 13.1 Fundamental Questions

1. **Compactification.** Can the modular surface be deformed or compactified — removing the cusp — in a way that preserves enough arithmetic structure to retain the Riemann zero statistics? The Teichmüller theory of hyperbolic surfaces provides a parameter space for such deformations[^69], but no arithmetic compactification is known.

2. **Minimal Structure for Zeta Statistics.** What is the minimal operator structure required to reproduce GUE statistics with the correct zero-spacing distribution? Numerical experiments on non-arithmetic hyperbolic surfaces (Hecke triangle groups) suggest that the absence of Hecke operators may be a prerequisite for clean GUE behavior[^21], raising the question of whether a clean Hilbert–Pólya operator must necessarily be **non-arithmetic**.

3. **Geometric Origin of Primes via Geodesics.** The prime geodesic theorem — \(N_{\mathrm{prim}}(\ell) \sim e^\ell / \ell\) — mirrors the prime number theorem \(\pi(x) \sim x/\log x\) via the identification \(p \leftrightarrow e^{\ell(\gamma_p)}\). Is there a canonical bijection between primitive geodesics of the modular surface and primes that is spectral (not merely asymptotic)?

4. **Symbolic Dynamics and Hyperbolic Geometry Unification.** Can the continued fraction symbolic dynamics of geodesic flow be unified with the symbolic dynamics of substitution systems in a framework that interpolates between integrable and chaotic behavior? Tropical geometry's max-plus arithmetic offers a candidate formalism[^60][^70].

5. **Learning the Hilbert–Pólya Operator.** Can a hypernetwork or equivariant neural operator learn a self-adjoint deformation \(H_\theta\) such that \(\mathrm{Spec}(H_\theta)\) approximates the first thousand Riemann zeros to high precision? The computation of Maass forms on the modular surface is exact enough numerically to serve as a baseline[^71][^72].

### 13.2 Proposed Computational Experiments

**Experiment 1: Spectral landscape mapping.** Parameterize \(V_\theta\) as a truncated hyperbolic Fourier series on \(\mathcal{F}\) with \(\theta \in \mathbb{R}^d\) (\(d \sim 100\)). Compute the spectral landscape \(\theta \mapsto \mathcal{L}(\theta)\) using finite element methods on the hyperbolic domain and visualize local minima via principal component analysis of gradient flows. This will quantify the non-convexity and level-crossing structure.

**Experiment 2: RL agent for spectral optimization.** Implement a policy-gradient RL agent (e.g., PPO) with state = first \(N=50\) computed eigenvalues of \(H_\theta\) and action = stochastic perturbation of \(\theta\). Compare convergence speed and solution quality against random search and gradient descent. Test whether physics-informed reward shaping (penalizing loss of self-adjointness or continuous spectrum invasion) improves stability.

**Experiment 3: ACO on Maass form computation.** Apply ACO to the problem of finding parameters \(r_n\) such that the Maass wave equation \(\Delta\psi + (1/4+r^2)\psi = 0\) has a solution on \(\mathcal{F}\) — equivalent to computing eigenvalues. Use the Hejhal algorithm's stepping structure (iterative refinement of the eigenvalue) as the graph structure, with pheromone deposited by successful convergence. Compare efficiency to standard numerical methods[^71][^72].

**Experiment 4: Tropical approximation.** Construct the tropical analogue of the operator \(H_\theta\) by replacing the PDE with a max-plus recurrence on the symbolic sequence space (continued fraction shifts). Compute the "tropical spectrum" and compare with the actual spectrum and with the Riemann zeros. This tests whether the piecewise-linear approximation retains essential spectral information[^58][^60].

**Experiment 5: DTES manifold learning.** Use kernel PCA or diffusion maps on the dataset of perturbed spectra \(\{\mathrm{Spec}(H_\theta)\}_\theta\) to learn the structure of the spectral manifold. Identify whether the Riemann zero sequence lies on (or near) the manifold, and whether geodesics on the DTES manifold correspond to natural deformation paths of the Laplacian.

***

## Summary of Status by Claim Type

| Claim | Status |
|---|---|
| Artin billiard is Anosov, ergodic, strongly mixing | **Proved** (Artin 1924, Hopf 1939, Anosov 1967) |
| Geodesic flow coded by continued fractions / shift on countably infinite alphabet | **Proved** (Series 1985, Adler–Flatto) |
| Selberg trace formula | **Proved** (Selberg ca. 1953; many extensions) |
| BGS conjecture (chaotic spectrum → RMT statistics) | **Unproved**; supported by numerical evidence and heuristic derivation |
| Montgomery–Dyson GUE connection for Riemann zeros | **Unproved** (Montgomery's conjecture); strong numerical evidence |
| QUE for Hecke–Maass forms on modular surface | **Proved** (Lindenstrauss 2006 + Soundararajan 2010) |
| Berry–Keating xp model reproduces exact Riemann zeros | **False**; reproduces only average density |
| Connes' spectral realization of zeta zeros | **Partial**: absorption spectrum realization proved; connection to proof of RH open |
| Learnable \(H_\theta\) with \(\mathrm{Spec}(H_\theta) = \{\gamma_n\}\) | **Speculative**; formulated here as computational program |
| DTES / tropical approximation of spectral landscape | **Speculative**; grounded in established tools (tropical geometry, GLNO) |
| RL/ACO for spectral optimization | **Novel proposal**; supported by analogous RL applications in quantum systems |

---

## References

1. [Artin billiard - Wikipedia](https://en.wikipedia.org/wiki/Artin_billiard)

2. [[PDF] 1. The hyperbolic plane](https://personal.math.ubc.ca/~lior/teaching/0809/620D_F08/HypPlane.pdf) - It is a fundamental domain for the action of Γ. Its boundary is a countable union of geodesic segmen...

3. [[PDF] arXiv:2307.01826v1 [math.NT] 4 Jul 2023](https://arxiv.org/pdf/2307.01826.pdf)

4. [arXiv:chao-dyn/9307001v1  2 Jul 1993](https://arxiv.org/pdf/chao-dyn/9307001.pdf)

5. [[PDF] arXiv:2304.10606v1 [math.DS] 20 Apr 2023](https://arxiv.org/pdf/2304.10606.pdf)

6. [arXiv:1409.8002v4  [math.DS]  9 Mar 2021](https://arxiv.org/pdf/1409.8002v4.pdf)

7. [Correlation Functions of Classical and Quantum Artin System defined on Lobachevsky plane and Scrambling Time](http://theor.jinr.ru/sqs19/talks/Babujian.pdf)

8. [[PDF] universe - Inspire HEP](https://inspirehep.net/files/f2e0a128f0bf51119af24d960a023990)

9. [LONDON MATHEMATICAL SOCIETY LECTURE NOTE SERIES](https://api.pageplace.de/preview/DT0400.9781139382632_A24438319/preview-9781139382632_A24438319.pdf)

10. [[PDF] Partially Hyperbolic Dynamics Federico Rodriguez Hertz Jana ...](https://www.fing.edu.uy/~ures/publicaciones/preprints/phdynamics.pdf)

11. [Coding of geodesics on some modular surfaces and applications to ...](https://www.sciencedirect.com/science/article/pii/S0019357718301988) - A large class of continued fractions has been studied in the context of the geodesic flow and symbol...

12. [arXiv:1711.06965v5  [math.DS]  1 Jul 2019](http://arxiv.org/pdf/1711.06965.pdf)

13. [[PDF] THE MODULAR SURFACE AND CONTINUED FRACTIONS](http://reu.dimacs.rutgers.edu/~ks1613/documents/series.pdf) - We shall describe symbolic dynamics for the first return map P on our special cross-section. Xof^M. ...

14. [[PDF] symbolic dynamics for geodesic flows](https://archive.ymsc.tsinghua.edu.cn/pacm_download/117/6289-11511_2006_Article_BF02392459.pdf) - In this paper we make an explicit geometrical construction of a symbolic dynamics for the geodesic f...

15. [Markov partition - Wikipedia](https://en.wikipedia.org/wiki/Markov_partition) - A Markov partition in mathematics is a tool used in dynamical systems theory, allowing the methods o...

16. [[PDF] symbolic dynamics for nonuniformly hyperbolic systems](https://ymsc.tsinghua.edu.cn/__local/C/BD/6F/657FB269FDDB2028B5B5E6D50C8_E783A8E3_D3065.pdf?e=.pdf)

17. [Symbolic dynamics for a kinds of piecewise smooth maps](https://www.aimsciences.org/article/doi/10.3934/dcdss.2024042) - Symbolic dynamics is effective for the classification of orbital types and their complexity in one-d...

18. [SPECTRAL THEORY OF AUTOMORPHIC FORMS AND ...](https://dchatzakos.math.upatras.gr/wp-content/uploads/2020/02/Notes-for-the-course.pdf)

19. [[PDF] arXiv:1212.3149v1 [math.NT] 13 Dec 2012](https://arxiv.org/pdf/1212.3149.pdf)

20. [[PDF] Arithmetic Quantum Chaos](http://www.its.caltech.edu/~matilde/ArithmeticQuantumChaos.pdf) - Introduction. The central objective in the study of quantum chaos is to characterize universal prope...

21. [[PDF] Computations of automorphic functions on Fuchsian groups](https://www.diva-portal.org/smash/get/diva2:170802/FULLTEXT01.pdf) - This thesis consists of four papers which all deal with computations of automorphic functions on cof...

22. [[0901.4060] Quantum unique ergodicity for SL_2(Z)\H - arXiv](https://arxiv.org/abs/0901.4060) - Abstract: We eliminate the possibility of "escape of mass" for Hecke-Maass forms of large eigenvalue...

23. [[PDF] Quantum unique ergodicity for SL2(Z)"026E30F H](https://annals.math.princeton.edu/wp-content/uploads/annals-v172-n2-p19-p.pdf) - Combined with the work of Lindenstrauss, this establishes the Quantum Unique Ergodicity conjecture o...

24. [Background on the Quantum Unique Ergodicity conjecture](https://aimath.org/news/que/rudnick_summary.html) - We made the conjecture that in certain sufficiently chaotic systems, billiard particles on a negativ...

25. [[PDF] Selberg's Trace Formula: An Introduction - University of Bristol](https://people.maths.bris.ac.uk/~majm/bib/selberg.pdf) - The aim of this short lecture course is to develop Selberg's trace formula for a compact hyperbolic ...

26. [[PDF] The Selberg Trace Formula & Prime Orbit Theorem](https://openresearch-repository.anu.edu.au/bitstreams/f1ba01c2-878e-4b36-b716-2614f8af099a/download) - The purpose of this thesis is to study the asymptotic property of the primitive length spectrum on c...

27. [Hilbert–Pólya conjecture - Wikipedia](https://en.wikipedia.org/wiki/Hilbert%E2%80%93P%C3%B3lya_conjecture)

28. [Selberg Trace Formula for Bordered Riemann Surfaces](https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-163/issue-2/Selberg-trace-formula-for-bordered-Riemann-surfaces--hyperbolic-elliptic/cmp/1104270463.pdf) - The trace formula is formulated for arbitrary Fuchsian groups of the first kind with reflection symm...

29. [Gutzwiller's Semiclassical Trace Formula and Maslov-Type ...](https://arxiv.org/pdf/1608.08294.pdf)

30. [[PDF] A relation between the Riemann zeta-function and the hyperbolic ...](https://www.numdam.org/article/ASNSP_1995_4_22_2_299_0.pdf)

31. [[PDF] The Gutzwiller Trace Formula and the Quantum-Classical ...](http://large.stanford.edu/courses/2020/ph470/rahman/docs/rahman-ph470-stanford-spring-2020.pdf)

32. [Exact Quantum Trace Formula from Complex Periodic Orbits](https://arxiv.org/html/2411.10691v1)

33. [BGS conjecture - Wikipedia](https://en.wikipedia.org/wiki/BGS_conjecture)

34. [Bohigas-Giannoni-Schmit conjecture - Scholarpedia](http://www.scholarpedia.org/article/Bohigas-Giannoni-Schmit_conjecture) - The BGS-conjecture aims to describe are simple quantum mechanical systems for which one can define a...

35. [Quantum Chaos, Irreversible Classical Dynamics, and Random ...](https://ui.adsabs.harvard.edu/abs/1996PhRvL..76.3947A/abstract) - The Bohigas-Giannoni-Schmit conjecture stating that the statistical spectral properties of systems w...

36. [Quantum Chaos, Irreversible Classical Dynamics, and Random ...](https://link.aps.org/doi/10.1103/PhysRevLett.76.3947) - The Bohigas-Giannoni-Schmit conjecture stating that the statistical spectral properties of systems w...

37. [Montgomery's pair correlation conjecture - Wikipedia](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture) - In mathematics, Montgomery's pair correlation conjecture is a conjecture made by Hugh Montgomery (19...

38. [Montgomery's Pair Correlation Conjecture -- from Wolfram MathWorld](https://mathworld.wolfram.com/MontgomerysPairCorrelationConjecture.html) - Montgomery's pair correlation conjecture, published in 1973, asserts that the two-point correlation ...

39. [[PDF] An Approach to the Riemann Hypothesis through Random Matrix ...](https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/c/4287/files/2020/10/232C-Riemann-Hypothesis-and-RMT.pdf) - The empirical evidence, again in Figure 1, shows the pair correlations between the heights of the no...

40. [[PDF] Finite Models for Arithmetical Quantum Chaos - UCSD Math](https://mathweb.ucsd.edu/~aterras/newchaos.pdf)

41. [[PDF] HILBERT–P´OLYA CONJECTURE, ZETA–FUNCTIONS AND ...](https://julioandrade.weebly.com/uploads/4/0/3/2/40324815/4ijmpa.pdf) - The original Hilbert and Pólya conjecture is the assertion that the non–trivial zeros of the Riemann...

42. [Quantum-Mechanical interpretation of Riemann zeta function zeros](https://www.academia.edu/69323341/Quantum_Mechanical_interpretation_of_Riemann_zeta_function_zeros) - We demonstrate that the Riemann zeta function zeros define the position and the widths of the resona...

43. [[PDF] The Riemann Zeros as Spectrum and the Riemann Hypothesis](https://s3.cern.ch/inspire-prod-files-1/1e65b86fec7566dba4d2d2384183f67b) - The Berry–Keating xp model can be implemented quantum mechanically. X The classical xp Hamiltonian m...

44. [The H=xp model revisited and the Riemann zeros](https://arxiv.org/abs/1102.5356) - Berry and Keating conjectured that the classical Hamiltonian H = xp is related to the Riemann zeros....

45. [[PDF] General covariant xp models and the Riemann zeros - arXiv](https://arxiv.org/pdf/1110.3203.pdf) - In 1999 Berry and Keating conjectured that an appropriate quantization of the classical. Hamiltonian...

46. [[PDF] A compact hamiltonian with the same asymptotic mean spectral ...](https://michaelberryphysics.wordpress.com/wp-content/uploads/2013/06/berry4401.pdf) - The Riemann hypothesis [1, 2] states that all complex zeros of the Riemann zeta functions have real ...

47. [Trace formula in noncommutative geometry and the zeros ...](https://arxiv.org/abs/math/9811068) - We give a spectral interpretation of the critical zeros of the Riemann zeta function as an absorptio...

48. [[PDF] Trace Formula in Noncommutative Geometry and - Alain Connes](https://alainconnes.org/wp-content/uploads/selecta.ps-2.pdf) - Abstract. We give a spectral interpretation of the critical zeros of the Rie- mann zeta function as ...

49. [[PDF] Noncommutative Geometry, the spectral standpoint - arXiv](https://arxiv.org/pdf/1910.10407.pdf) - algebraic geometry underlying the spectral realization of the zeros of the Riemann zeta function. .....

50. [[PDF] Spectral deformation of Laplacians on hyperbolic manifolds](http://old.intlpress.com/site/pub/files/_fulltext/journals/cag/1997/0005/0002/CAG-1997-0005-0002-a001.pdf) - We denote by A the unique self-adjoint Laplacian in L2(M) with domain. H2(M), the Sobolev space of o...

51. [[PDF] The spectrum of continuously perturbed operators and the Laplacian ...](https://lu.math.uci.edu/pdfs/publications/2016-2020/59.pdf) - In this article we study the variation in the spectrum of a self-adjoint nonnegative operator on a H...

52. [The trace formula for singular perturbations of the Laplacian on hyperbolic surfaces](https://ar5iv.labs.arxiv.org/html/1002.2930)

53. [arXiv:1002.2930v1  [math-ph]  15 Feb 2010](https://www.arxiv.org/pdf/1002.2930.pdf)

54. [Geometric Laplace Neural Operator - OpenReview](https://openreview.net/forum?id=2Aeje1T4se) - The paper proposes Geometric Laplace Neural Operator (GLNO) as an alternative to Fourier-style opera...

55. [Interpretable Operator Learning for Inverse Problems via Adaptive ...](https://arxiv.org/html/2603.20602v1) - It leverages a lightweight neural network to learn a pointwise adaptive filter function, which rewei...

56. [Physics-informed neural networks to solve inverse problems in ...](https://arxiv.org/html/2512.12074v1) - Our results demonstrate that, in this setting, PINNs provide a more accurate and computationally eff...

57. [[2302.01413] Solving two-dimensional quantum eigenvalue ... - arXiv](https://arxiv.org/abs/2302.01413) - We generalize an unsupervised learning algorithm to find the particles' eigenvalues and eigenfunctio...

58. [Tropical Geometry and Piecewise-Linear Approximation of Curves ...](https://arxiv.org/abs/1912.03891) - Abstract:Tropical Geometry and Mathematical Morphology share the same max-plus and min-plus semiring...

59. [Tropical geometry - Wikipedia](https://en.wikipedia.org/wiki/Tropical_geometry) - Tropical geometry is a variant of algebraic geometry in which polynomial graphs resemble piecewise l...

60. [Tropical curves and integrable piecewise linear maps](https://www.arxiv.org/abs/1111.5771) - We present applications of tropical geometry to some integrable piecewise-linear maps, based on the ...

61. [[2601.18953] Reinforcement Learning for Quantum Technology - arXiv](https://arxiv.org/abs/2601.18953) - We discuss state preparation in few- and many-body quantum systems, the design and optimization of h...

62. [A survey on physics informed reinforcement learning: Review and ...](https://www.sciencedirect.com/science/article/pii/S0957417425017865) - A thorough review of the literature on the fusion of physics information or physics priors in reinfo...

63. [[PDF] Rigorous data‐driven computation of spectral properties of ...](http://www.damtp.cam.ac.uk/user/mjc249/pdfs/RigorousKoopman.pdf) - This allows us to compute the spectral measure associated with the dynam- ics of a protein molecule ...

64. [[PDF] Reinforcement Learning for Infinite-Dimensional Systems](https://www.jmlr.org/papers/volume26/24-1575/24-1575.pdf) - Design of a filtrated RL algorithm for learning optimal policies of parameterized systems with conve...

65. [The Ant Colony Optimization Metaheuristic:](https://citeseerx.ist.psu.edu/document?doi=8578b194c5a6909371c8163337fd50ecfd24e635&repid=rep1&type=pdf)

66. [Ant Colony Optimization - Scholarpedia - IRIDIA](https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/Dor2007sch-aco.pdf)

67. [Ant Colony Optimization: Principles, Variants, and Application ...](https://www.publications.scrs.in/chapter/978-81-975670-0-1/5)

68. [[PDF] Ant colony optimization: Introduction and recent trends - IIIA-CSIC](https://iiia.csic.es/~christian.blum/downloads/blum_aco_PLR2005.pdf)

69. [Degenerating hyperbolic surfaces and spectral gapsfor large genus](https://msp.org/apde/2024/17-4/apde-v17-n4-p06-p.pdf)

70. [[PDF] TROPICAL - Inria](https://radar.inria.fr/rapportsactivite/RA2020/tropical/tropical.pdf) - We represent the behavior of these systems by piecewise affine dynamical systems. ... of the piecewi...

71. [Computation of Maass waveforms with nontrivial multiplier systems](https://www.ams.org/journals/mcom/2008-77-264/S0025-5718-08-02129-7/)

72. [Computing and Verifying Maass Forms](https://davidlowryduda.com/wp-content/uploads/2022/03/BYUMaass-compressed.pdf)

