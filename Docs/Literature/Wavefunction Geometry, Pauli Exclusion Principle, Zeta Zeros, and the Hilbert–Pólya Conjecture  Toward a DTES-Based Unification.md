# Wavefunction Geometry, Pauli Exclusion Principle, Zeta Zeros, and the Hilbert–Pólya Conjecture: Toward a DTES-Based Unification

**A Scientific Literature Review**

***

## Abstract

This review surveys the mathematical and physical structures connecting wavefunction geometry, fermionic antisymmetry, the spectral theory of the Riemann zeta function, and the Hilbert–Pólya conjecture. Across four ostensibly distinct domains — quantum hydrodynamics, many-body physics, analytic number theory, and quantum chaos — a coherent set of structural analogies emerges: nodal sets as geometric constraints, spectral data as arithmetic encodings, and level repulsion as a universal signature of underlying chaotic dynamics. The final sections introduce a speculative but principled synthesis in which a deformation-weighted operator, denoted \(\Delta_S\) or DTES (deformed-translation-symmetry-breaking operator), serves as a candidate framework for a Hilbert–Pólya-type construction. All speculative proposals are explicitly marked and carefully separated from established results.

***

## 1. Wavefunction Geometry

### 1.1 Polar Decomposition and U(1) Bundle Structure

A scalar complex wavefunction \(\psi \in L^2(\mathbb{R}^d, \mathbb{C})\) admits the polar decomposition[^1][^2]

\[
\psi(x) = A(x)\,e^{iS(x)/\hbar}, \qquad A(x) \geq 0,\; S(x) \in \mathbb{R}, \tag{1}
\]

where \(A(x) = |\psi(x)|\) is the amplitude and \(S(x)/\hbar\) is the phase. This decomposition is smooth wherever \(A(x) \neq 0\); at zeros of \(\psi\), the phase \(S\) is undefined and the structure degenerates. Geometrically, (1) identifies \(\psi\) as a section of a complex line bundle over \(\mathbb{R}^d\) with structure group \(\mathrm{U}(1)\)[^3]. The global phase factor \(e^{i\alpha}\) constitutes a \(\mathrm{U}(1)\) gauge freedom; the physically measurable quantities \(\rho = A^2\) and \(\mathbf{v} = \nabla S / m\) are gauge-invariant. The connection one-form on this bundle is \(A_\mu = \partial_\mu S/\hbar\), encoding local phase gradients[^4].

The space of all such wavefunctions (modulo global phase and normalization) is an infinite-dimensional complex projective space \(\mathbb{CP}(\mathcal{H})\), equipped with the Fubini–Study metric. The Madelung transform (discussed below) provides a Kähler morphism from this space into the cotangent bundle of smooth probability densities, equipped with the Fisher–Rao information metric.[^5][^6][^3]

### 1.2 Madelung Transform and Quantum Hydrodynamics

Substituting decomposition (1) into the Schrödinger equation \(i\hbar\partial_t\psi = -(\hbar^2/2m)\Delta\psi + V\psi\) and separating real and imaginary parts yields the **Madelung equations**:[^7][^4][^8]

\[
\partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0, \qquad \rho = A^2, \quad \mathbf{v} = \frac{\nabla S}{m}, \tag{2}
\]

\[
m\partial_t \mathbf{v} + m(\mathbf{v}\cdot\nabla)\mathbf{v} = -\nabla(V + Q), \qquad Q = -\frac{\hbar^2}{2m}\frac{\Delta A}{A}. \tag{3}
\]

Equation (2) is a continuity equation for probability density \(\rho\). Equation (3) is Euler's equation for a compressible irrotational fluid, driven by the classical potential \(V\) and the **quantum potential** \(Q\), which encodes quantum effects through the curvature of the amplitude \(A\). The quantity \(\mathbf{v} = \nabla S/m\) serves as a velocity field on configuration space; since \(\mathbf{v} = \nabla(\cdot)\), the flow is irrotational wherever \(\psi \neq 0\). Vorticity arises only when the phase \(S\) is multivalued — i.e., at quantized vortex lines threading through the nodal set \(\psi^{-1}(0)\).[^9][^10][^8]

The geometric significance of the Madelung transform was clarified by Khesin, Misiolek, and Modin (2018): the transform is a **Kähler map** between the space of wave functions and the cotangent bundle of smooth probability densities, intertwining the symplectic structure of quantum mechanics with the Lie–Poisson structure of compressible fluid dynamics. This result places quantum hydrodynamics on a rigorous infinite-dimensional geometric foundation.[^3][^5]

### 1.3 Phase Gradient as a Classical Flow Field

In the semiclassical (WKB) approximation, the phase \(S(x)\) satisfies the Hamilton–Jacobi equation:[^11][^2][^1]

\[
\frac{|\nabla S|^2}{2m} + V(x) = E, \tag{4}
\]

and \(\nabla S = p(x)\) is the local classical momentum field. The WKB wavefunction takes the form \(\psi \sim A_0(x)\,e^{iS(x)/\hbar}\) with \(A_0\) determined by the transport equation \(\nabla\cdot(A_0^2 \nabla S) = 0\). Classical trajectories are the integral curves of \(\nabla S\); the nodes of \(\psi\) correspond to points where this classical picture breaks down — caustics, turning points, and topological obstructions. In higher dimensions, the phase \(S\) is a section of a Lagrangian submanifold in phase space, and the Maslov index encodes the additional phase acquired at caustics.[^12][^2][^13][^11]

### 1.4 Nodal Sets as Geometric and Topological Objects

The **nodal set** of a wavefunction \(\psi\) is the zero locus \(\mathcal{N}_\psi = \{x \in \mathbb{R}^d : \psi(x) = 0\}\). For eigenfunctions of elliptic differential operators on smooth Riemannian manifolds, nodal sets are smooth hypersurfaces of dimension \(d-1\), except at degenerate points (where the gradient also vanishes). Courant's theorem establishes that the \(n\)-th eigenfunction of a Schrödinger operator on a bounded domain has at most \(n\) nodal domains (connected components of the complement \(\mathbb{R}^d \setminus \mathcal{N}_\psi\)). The topological complexity of nodal sets — their Betti numbers, the number of connected components of nodal domains — encodes spectral information about the operator. In the context of magnetic Schrödinger operators on multiply connected domains, half-integer circulation of the vector potential around holes can be characterised precisely through the structure of the nodal set.[^14][^15][^16][^17]

***

## 2. Pauli Exclusion Principle and Fermionic Geometry

### 2.1 Antisymmetry and the Exterior Algebra

For a system of \(N\) identical fermions (spin-1/2 particles), the physical Hilbert space is the antisymmetric subspace \(\mathcal{H}_\mathrm{F}^N = \bigwedge^N \mathcal{H}_1\), where \(\mathcal{H}_1 = L^2(\mathbb{R}^3 \times \{{\uparrow, \downarrow}\})\) is the single-particle Hilbert space and \(\bigwedge^N\) denotes the \(N\)-th exterior power. The antisymmetry requirement takes the form:[^18][^19]

\[
\Psi(\ldots, x_i, \ldots, x_j, \ldots) = -\Psi(\ldots, x_j, \ldots, x_i, \ldots), \qquad \forall\, i \neq j. \tag{5}
\]

This is equivalent to the statement that \(\Psi \in \bigwedge^N \mathcal{H}_1\): the wavefunction is an \(N\)-form on single-particle space. The Pauli exclusion principle — no two fermions can occupy the same state — follows immediately: if \(x_i = x_j\), then (5) implies \(\Psi = -\Psi = 0\). Condition (5) is formulated independently of any Hamiltonian and constitutes a pure symmetry constraint on state space.[^20][^21]

### 2.2 Nodal Structures from Antisymmetry

The antisymmetry condition (5) forces \(\Psi\) to vanish on all coincidence hyperplanes \(\{x_i = x_j\}\) in the \(3N\)-dimensional configuration space \(\mathbb{R}^{3N}\). More precisely, the **nodal hypersurface** of a fermionic ground state is the \((3N-1)\)-dimensional manifold:[^22][^23][^24]

\[
\mathcal{N}_\Psi = \{(x_1,\ldots,x_N) \in \mathbb{R}^{3N} : \Psi(x_1,\ldots,x_N) = 0\}, \tag{6}
\]

which includes (but is not limited to) the coincidence surfaces. The topology of \(\mathcal{N}_\Psi\) is constrained by symmetry: for the many-body ground state, the **two-nodal-domain conjecture** (proven in several cases) asserts that \(\mathbb{R}^{3N} \setminus \mathcal{N}_\Psi\) consists of exactly two connected domains. This remarkable topological rigidity has far-reaching consequences for quantum Monte Carlo methods: the fixed-node approximation enforces \(\mathcal{N}_\Psi^\mathrm{trial}\) as a Dirichlet boundary condition, converting the antisymmetric sign problem into a local boundary constraint.[^25][^23][^24][^26][^27]

### 2.3 Antisymmetry as a Geometric Constraint in State Space

The fermionic constraint — that \(\Psi\) must change sign under coordinate transpositions — can be reformulated as a **topological boundary condition** on configuration space: physical fermionic states must vanish on the coincidence locus \(\bigcup_{i<j}\{x_i = x_j\}\), which is a union of codimension-3 submanifolds in \(\mathbb{R}^{3N}\). This is structurally analogous to a Dirichlet boundary condition imposed on a domain with a singular boundary. The antisymmetry of the many-body wavefunction thus induces, via (6), a network of hypersurfaces that partitions configuration space and constrains all physical amplitudes. The recent minimal representation theorem of Ryczko and collaborators (2025) establishes that fermionic wavefunctions have an exact and minimal representation in terms of parity-graded symmetric/antisymmetric feature variables, with complexity scaling as \(D \sim N^d\).[^21][^28][^18][^22]

***

## 3. Properties of \(\zeta(1/2 + it)\) and the Spectral Structure of Zeros

### 3.1 Zeros as Nodal Points of a Complex Signal

The Riemann zeta function \(\zeta(s)\) is a meromorphic function of \(s \in \mathbb{C}\), analytic except for a simple pole at \(s = 1\). The non-trivial zeros are complex numbers \(\rho = \sigma + it\) in the critical strip \(0 < \sigma < 1\); by the functional equation \(\zeta(s) = 2^s\pi^{s-1}\sin(\pi s/2)\,\Gamma(1-s)\,\zeta(1-s)\), zeros are symmetric about the critical line \(\mathrm{Re}(s) = 1/2\). The Riemann Hypothesis asserts all non-trivial zeros satisfy \(\sigma = 1/2\).[^29]

On the critical line \(s = 1/2 + it\), the function \(\zeta(1/2+it)\) is a real-valued oscillating function after multiplication by a known phase factor (the Riemann–Siegel \(\vartheta\) function): \(Z(t) = e^{i\vartheta(t)}\zeta(1/2+it) \in \mathbb{R}\). The zeros of \(Z(t)\) — where \(Z(t) = 0\) — correspond precisely to the imaginary parts of the non-trivial zeros on the critical line. In this sense, the Riemann zeros are the **nodal points of the real function \(Z(t)\)** along the critical line, exactly analogous to the zeros of a real-valued wavefunction on a one-dimensional domain. The mean density of zeros near height \(t\) is \(\bar{d}(t) = (1/2\pi)\log(t/2\pi)\), the smooth Riemann–Mangoldt formula, analogous to Weyl's law for eigenvalue density.[^30][^31]

### 3.2 Statistical Structure of Zeros

The statistics of the non-trivial zeros are studied via their normalized spacings. Denoting the imaginary parts in ascending order as \(0 < \gamma_1 \leq \gamma_2 \leq \cdots\), define the normalized gaps \(\tilde{\gamma}_n = \gamma_n \cdot \frac{\log\gamma_n}{2\pi}\). The key result is that the pair correlation function of the normalized zeros, conjectured by Montgomery (1973) and extensively confirmed numerically by Odlyzko (1987), is:[^32][^33]

\[
R_2(u) = 1 - \left(\frac{\sin\pi u}{\pi u}\right)^2. \tag{7}
\]

This is precisely the two-point correlation function of eigenvalues of the **Gaussian Unitary Ensemble (GUE)** of random Hermitian matrices. Higher-order \(n\)-point correlations of the zeros, computed by Rudnick and Sarnak (1996) for test functions with restricted Fourier support, also agree with GUE universally. Katz and Sarnak (1999) extended this analysis to families of \(L\)-functions and showed that the statistical symmetry type — GUE, GOE, or GSE — depends on the arithmetic symmetry of the family. These statistical universality results constitute the primary empirical evidence for the Hilbert–Pólya conjecture.[^34][^35][^32]

***

## 4. The Hilbert–Pólya Conjecture

### 4.1 Statement and Implication for the Riemann Hypothesis

The **Hilbert–Pólya conjecture** asserts the existence of a self-adjoint operator \(\hat{H}\) on a Hilbert space \(\mathcal{H}\) such that its spectrum coincides with the set of imaginary parts of non-trivial zeros of \(\zeta(s)\):[^36][^37]

\[
\sigma(\hat{H}) = \left\{\gamma \in \mathbb{R} : \zeta\!\left(\tfrac{1}{2}+i\gamma\right) = 0\right\}. \tag{8}
\]

The implication for the Riemann Hypothesis is immediate: the spectral theorem for self-adjoint operators guarantees that all eigenvalues of \(\hat{H}\) are real. If \(\gamma_n \in \mathbb{R}\) for all \(n\), then the corresponding zero \(1/2 + i\gamma_n\) has real part exactly \(1/2\) — that is, it lies on the critical line. The logical chain is: *\(\hat{H} = \hat{H}^\dagger\) ⟹ \(\sigma(\hat{H}) \subset \mathbb{R}\) ⟹ all zeros on \(\mathrm{Re}(s)=1/2\) ⟹ RH*.[^38][^39][^36]

The proof programme thus splits into two stages:[^39][^38]
- **Stage I** (*Spectral match*, assuming RH): construct an operator formally satisfying (8).
- **Stage II** (*Self-adjointness*): prove the operator is genuinely self-adjoint in the strict functional-analytic sense, without assuming RH.

Only Stage II would constitute a proof of RH; all known constructions achieve Stage I at best.

### 4.2 Known Partial Constructions

**Berry–Keating (1999):** Building on the structural analogy between the Weil explicit formula and the Gutzwiller trace formula, Berry and Keating proposed that the classical Hamiltonian underlying the Hilbert–Pólya operator is \(H_\mathrm{cl} = xp\). The associated quantization is the symmetric operator:[^31][^40][^30]

\[
\hat{H}_\mathrm{BK} = \tfrac{1}{2}(\hat{x}\hat{p} + \hat{p}\hat{x}) = -i\hbar\!\left(x\frac{d}{dx} + \tfrac{1}{2}\right). \tag{9}
\]

A phase-space regularization with constraints \(|x| \geq \ell_x\), \(|p| \geq \ell_p\) yields a semiclassical spectrum matching the smooth (average) density of Riemann zeros via Weyl's law, but *not* the exact zeros[^41][^31]. The Berry–Keating programme identifies three structural requirements on \(H_\mathrm{cl}\): it should be (a) chaotic, (b) have periodic orbits with periods equal to multiples of \(\log p\) for primes \(p\), and (c) break time-reversal invariance to produce GUE statistics rather than GOE[^31][^30].

**Bender–Brody–Müller (2017):** Bender, Brody, and Müller constructed an explicit Hamiltonian:[^42][^43]

\[
\hat{H}_\mathrm{BBM} = \hat{x}\hat{p} + \hat{p}\hat{x} + \hat{x}^{-1}\!\left(1 - e^{-i\hat{p}}\right)\hat{p}^{-1}, \tag{10}
\]

with classical limit \(H \to 2xp\). While \(\hat{H}_\mathrm{BBM}\) is not Hermitian in the standard sense, \(i\hat{H}_\mathrm{BBM}\) is \(\mathcal{PT}\)-symmetric with broken \(\mathcal{PT}\) symmetry, permitting the possibility of real eigenvalues. The key result: if eigenfunctions satisfy the boundary condition \(\psi_n(0) = 0\), then the eigenvalues are exactly the imaginary parts of the non-trivial zeros of \(\zeta(s)\). However, the self-adjointness of \(\hat{H}_\mathrm{BBM}\) on an appropriate dense domain remains unestablished.[^44][^45][^43][^42]

**Connes (1999):** Connes gave a spectral realization of the critical zeros as an **absorption spectrum** on the noncommutative space of Adèle classes \(X_\mathbb{Q} = \mathbb{A}_\mathbb{Q}/\mathbb{Q}^\times\). The Weil explicit formula is reinterpreted as a trace formula on this space, with critical zeros appearing as missing spectral lines. Connes reduces the RH to a positivity condition on a Weil distribution; verifying this positivity has not been achieved. The adèle class space approach has deep connections to the Weil proof of RH for function fields and motivates the search for a geometric framework unifying arithmetic and spectral geometry.[^46][^47][^48]

***

## 5. Connections to Physics

### 5.1 Quantum Chaos and Spectral Statistics

The central paradigm connecting quantum mechanics to the Hilbert–Pólya conjecture is the **Bohigas–Giannoni–Schmit (BGS) conjecture** (1984): quantum systems with classically chaotic dynamics exhibit level spacing statistics described by random matrix theory. For systems with broken time-reversal invariance, the relevant ensemble is the GUE, producing the Wigner distribution \(P(s) \propto s^2 e^{-\pi s^2/4}\) for nearest-neighbor spacings and the pair correlation (7) for two-point statistics.[^49][^50]

The spectral statistics of Riemann zeros match GUE rather than GOE, which constrains the unknown Hilbert–Pólya operator to govern a quantum chaotic system with broken time-reversal invariance. Berry and Keating identify a magnetic field (or a spin-orbit interaction) as the natural mechanism for time-reversal breaking in \(H_\mathrm{cl} = xp\). Numerical studies of quantized chaotic billiards — the Sinai billiard, the cardioid billiard — confirm GUE statistics when time-reversal symmetry is explicitly broken by a magnetic field.[^51][^40][^50][^33][^30][^31]

### 5.2 The Selberg–Gutzwiller–Weil Analogy

The central structural analogy in the field is between three trace formulas of identical form:[^52][^53][^54]

| Formula | Spectrum side | Orbit/prime side |
|---|---|---|
| Weil explicit formula | Riemann zeros | Prime numbers |
| Selberg trace formula | Laplacian eigenvalues on hyperbolic surface | Closed geodesics |
| Gutzwiller trace formula | Quantum energy levels (chaotic system) | Classical periodic orbits |

In all three cases, the general structure is \(\sum_n f(E_n) = \text{smooth term} + \sum_\gamma A_\gamma \hat{f}(\ell_\gamma)\), where the sum over orbits/primes is a Fourier-type dual of the spectral sum. The Selberg trace formula is rigorously established for compact hyperbolic surfaces; Selberg's zeta function is its generating function. The Gutzwiller formula is semiclassical. The Weil formula is exact but number-theoretic. This structural isomorphism motivates the interpretation of primes as the "periodic orbits" of the unknown Riemann dynamics.[^54][^30][^52]

### 5.3 GUE Statistics and Random Matrix Universality

Montgomery's pair correlation theorem (1973) established (conditionally on RH) that the normalized two-point density of Riemann zeros satisfies (7) for test functions with Fourier support in \((-1,1)\). Freeman Dyson immediately identified the right-hand side as the GUE pair correlation function. Odlyzko's large-scale computations (1987–present) — studying tens of millions of zeros near \(10^{20}\) and \(10^{22}\) — confirmed this agreement to extraordinary precision, with the full nearest-neighbor distribution, \(n\)-point correlations, and spectral rigidity all consistent with GUE predictions.[^55][^56][^33][^57][^32]

The GUE universality is not unique to \(\zeta(s)\): Rudnick and Sarnak (1996) proved GUE \(n\)-point correlations for automorphic \(L\)-functions. Katz and Sarnak (1999) further showed that the low-lying zeros of families of \(L\)-functions are distributed according to the eigenvalue statistics of compact classical groups — \(\mathrm{U}(N)\), \(\mathrm{O}(N)\), \(\mathrm{USp}(2N)\) — depending on the symmetry type of the family. These results suggest that the GUE behavior of the Riemann zeros is a special case of a vastly more general law connecting L-functions and random matrices.[^58][^59][^60][^61][^35][^34]

***

## 6. DTES-Based Synthesis *(Speculative)*

> **Status: The framework in this section is a novel speculative synthesis. It is not established in the literature and should be interpreted as a research hypothesis, not an established result.**

### 6.1 The DTES Operator

Recent work (Khachatryan, 2025) proposes a structural reformulation of quantum mechanics based on the breakdown of translation symmetry of the vacuum wave operator. The central object is the **deformed translation-symmetry-breaking (DTES) weighted Laplacian**:[^62]

\[
\Delta_S = \nabla \cdot \left(e^{-2S}\nabla\right), \tag{11}
\]

where \(S(x)\) is a scalar "tempo field" encoding the geometric deformation. This operator is a standard weighted Laplacian (also known as the Witten Laplacian or drift Laplacian in other contexts), with \(e^{-2S}\) playing the role of a Riemannian density weight. The eigenvalue problem for \(\Delta_S\) is:[^63]

\[
-\Delta_S \phi = \lambda \phi \quad \Leftrightarrow \quad -\nabla\cdot(e^{-2S}\nabla\phi) = \lambda\phi. \tag{12}
\]

Inertial mass, localization, and the quantum potential all emerge from the spectral properties of \(\Delta_S\) through the symmetry-breaking structure, and the Schrödinger equation is derived as the unique linear closure of wave dynamics under this weighted Laplacian.[^62]

### 6.2 Amplitude as an Exponential Energy Function

Within the DTES framework, the amplitude field in the Madelung decomposition (1) is interpreted as:[^62]

\[
A(x) \approx e^{-E_S(x)}, \qquad E_S(x) = S(x), \tag{13}
\]

where \(E_S\) is the DTES energy function. This identification is natural: in ground states, \(A(x) = |\psi(x)| \propto e^{-V_\mathrm{eff}(x)/\hbar}\) in the semiclassical limit. The quantum potential then becomes \(Q = -(\hbar^2/2m)\Delta A/A = (\hbar^2/2m)[\Delta S - |\nabla S|^2]\), which is precisely the curvature correction generated by the DTES structure. The zeros of \(\psi\) — where \(A(x) = 0\), i.e., \(E_S(x) \to +\infty\) — are **forbidden regions** in the DTES energy landscape, analogous to hard-wall boundaries or infinite potentials.

### 6.3 Phase as a Path Integral over DTES Geometry

*[Speculative]* The phase \(S(x)\) in (1) plays a dual role in the DTES framework: as the classical action (Hamilton–Jacobi function) and as the weight function in \(\Delta_S\). The path integral representation

\[
\psi(x) = \int \mathcal{D}[\mathrm{paths}]\; e^{iS[\mathrm{path}]/\hbar}
\]

suggests that \(S\) encodes the geometry of all paths flowing through the DTES-deformed configuration space. In regions of high \(S\), the metric weight \(e^{-2S}\) suppresses the Laplacian, effectively confining diffusion; in regions of low \(S\), propagation is amplified. This creates a differential geometry of accessibility on \(\mathbb{R}^d\) whose topology is precisely the nodal structure of \(\psi\).[^62]

### 6.4 Pauli Principle as Hard Constraints in DTES

*[Speculative]* The fermionic antisymmetry (5) and its induced nodal hypersurface (6) can be reinterpreted within the DTES framework as **infinite barriers in the DTES energy landscape**. On the coincidence loci \(\{x_i = x_j\}\), the amplitude \(A \to 0\), meaning \(E_S \to \infty\). These divergences in \(E_S\) along the exchange-symmetry hyperplanes act as hard geometric constraints that deform the effective metric on \(\mathbb{R}^{3N}\), creating a configuration manifold with codimension-3 singularities along \(\bigcup_{i < j}\{x_i = x_j\}\). The two-nodal-domain conjecture would, in this language, correspond to the statement that the DTES-deformed configuration space of a fermionic ground state has exactly two connected components of finite accessibility.[^26][^27]

### 6.5 Zeta Zeros as Spectral Nodal Structure of DTES

*[Speculative]* The most ambitious synthesis of this framework is the conjecture that there exists a DTES operator \(\Delta_S\) on an appropriate Hilbert space such that its spectral nodal structure — the set of \(\lambda\) at which \(\ker(-\Delta_S - \lambda)\) is nontrivial — coincides with the imaginary parts of the Riemann zeros. This would proceed in two steps:

1. Identify the weight function \(S\) with the "arithmetic geometry" of the prime number distribution, analogously to how Connes identifies the adèle class space as the geometric realization of the Riemann zeros.[^47][^46]

2. Verify that \(\Delta_S\) is self-adjoint on a dense domain in \(L^2(\mathbb{R}, e^{-2S}dx)\), which would then imply — by the spectral theorem — that all spectral values are real, i.e., all zeros are on the critical line.

The attractiveness of this approach lies in the fact that weighted Laplacians of the form (11) are well-studied in geometric analysis and arise naturally in contexts ranging from Kähler–Einstein geometry to manifold learning. The minimax spectral estimation theory for weighted Laplacians (Chaubet–Divol, 2025) provides tools for relating the spectrum of \(\Delta_f = \Delta + \nabla\log f \cdot \nabla\) to the geometry of the underlying density \(f\), which could in principle be adapted to arithmetic settings.[^64][^63]

***

## 7. Toward a Unifying Hypothesis

### 7.1 Structural Parallels

The following table places the four domains — wavefunction geometry, fermionic antisymmetry, Riemann zero structure, and Hilbert–Pólya theory — in a unified structural framework:

| Domain | Spectral object | Nodal structure | Governing operator | Key constraint |
|---|---|---|---|---|
| Single-particle QM | Energy eigenvalues \(E_n\) | Zeros of eigenfunctions in \(\mathbb{R}^d\) | Schrödinger \(-\Delta + V\) | Self-adjointness |
| Many-body fermions | Many-body energy levels | Nodal hypersurface \(\mathcal{N}_\Psi \subset \mathbb{R}^{3N}\) | \(N\)-body Hamiltonian | Antisymmetry (5) |
| Riemann zeta function | Imaginary parts \(\gamma_n\) of zeros | Zeros of \(Z(t)\) on critical line | Unknown \(\hat{H}\) | RH = all \(\gamma_n \in \mathbb{R}\) |
| Random matrix theory | GUE eigenvalues | Level repulsion, pair correlation (7) | Random Hermitian matrix | No time-reversal symmetry |

The parallel between rows 1 and 3 is the Hilbert–Pólya conjecture in its strongest form. The parallel between rows 2 and 3 is more speculative: both involve a nodal set enforced by a global algebraic constraint (antisymmetry vs. the functional equation of \(\zeta\)), and both nodal structures encode profound arithmetic/geometric information.[^27][^18]

### 7.2 Proposed Unifying Hypothesis *(Speculative)*

*[Speculative]* The proposed synthesis asserts:

> **Hypothesis (DTES Unification):** Wavefunction nodal geometry, fermionic exchange constraints, and the spectral structure of the Riemann zeta function all arise as instances of a single geometric-spectral principle: the deformation of the Laplacian by an arithmetic or physical potential \(S\), producing a self-adjoint operator \(\Delta_S\) whose nodal set and discrete spectrum encode the relevant physical or number-theoretic data.

Specifically:
- In single-particle QM: \(S(x)\) is the WKB action, and the spectrum of \(-\Delta_S\) gives the energy levels.
- In many-body QM: \(S(x_1,\ldots,x_N)\) is the multi-particle action with singularities at coincidence loci, and the nodal hypersurface of the ground state eigenfunction reflects the fermionic antisymmetry.
- In the Hilbert–Pólya scenario: \(S\) would be determined by the prime-counting function or an arithmetic flow, and the spectrum of \(-\Delta_S\) would reproduce the Riemann zeros.

The operationally precise formulation would require: (i) identification of the correct Hilbert space (likely \(L^2(\mathbb{A}_\mathbb{Q}/\mathbb{Q}^\times, \mu)\) for some arithmetic measure \(\mu\), following Connes); (ii) definition of \(S\) in terms of the arithmetic geometry; (iii) proof of essential self-adjointness of \(\Delta_S\) on a natural dense domain.[^46][^47]

***

## 8. Open Problems

### 8.1 Existence of a Concrete Self-Adjoint Operator

The central open problem remains the construction of a self-adjoint operator satisfying (8). Known candidates — Berry–Keating, BBM, Connes — each achieve a partial realization:[^45][^30][^31][^42][^44][^46]

- The Berry–Keating \(xp\) model reproduces only the **average** zero density via Weyl's law, not the exact zeros.[^41][^31]
- The BBM Hamiltonian (10) has eigenvalues matching the zeros if the boundary condition \(\psi_n(0) = 0\) is imposed, but self-adjointness on a rigorously defined Hilbert space domain remains unproven.[^44]
- The Connes absorption spectrum framework reduces RH to a positivity statement on a Weil distribution; this positivity has not been verified.[^47][^46]

Recent claims (arXiv:2408.15135, Khachatryan 2025; Cambridge Open Engage 2026) assert progress toward constructing a self-adjoint Hilbert–Pólya Hamiltonian, but these results await peer review.[^65][^38]

### 8.2 Rigorous Link Between Nodal Geometry and Zeta Zeros

No rigorous connection between the geometric/topological structure of nodal sets (in the sense of Section 1.4 or Section 2.2) and the arithmetic structure of the Riemann zeros has been established. The analogy is compelling — both involve discrete structures (zero counts, nodal domain counts) that obey universal laws — but the precise map between nodal topology and zero statistics remains entirely open. Courant's theorem and the two-nodal-domain conjecture suggest that nodal structure of physical eigenfunctions is highly constrained; whether any such constraint applies in a number-theoretic setting is unknown.[^17][^26]

### 8.3 Bridging Quantum Mechanics, Number Theory, and Geometry

The three relevant trace formulas (Selberg, Gutzwiller, Weil) share a common structure, but the mechanism producing this structure in each case is different. A unified mathematical framework — a single theorem specializing to all three — does not exist. Connes' noncommutative geometry approach comes closest, providing a common language of spectral triples and trace formulas, but does not yet yield a proof. The Weil proof for function fields proceeds via the Frobenius action on cohomology of a curve; the analogue for \(\mathbb{Q}\) — what would play the role of the Frobenius — has not been identified.[^48][^52][^54][^46]

### 8.4 Feasibility of Computational and Spectral Reconstruction

*[Speculative]* A computational programme suggested by the DTES framework would attempt to reconstruct, from the known Riemann zeros \(\gamma_n\), the weight function \(S(x)\) such that \(\Delta_S\) has spectrum \(\{\gamma_n\}\). This is an **inverse spectral problem** for weighted Laplacians. Spectral deformation theory for Schrödinger operators (Simon et al., Caltech preprints) shows that isospectral deformations of Dirichlet data can be characterized via Weyl–Titchmarsh theory; analogous techniques applied to arithmetic settings might constrain what \(S\) can be. The minimax estimation rates for weighted Laplacian spectra suggest that, given sufficiently many zeros, the weight function \(S\) could in principle be reconstructed to increasing precision — though whether the resulting \(S\) has arithmetic meaning remains speculative.[^66][^63]

***

## Summary of Established vs. Speculative Results

| Claim | Status |
|---|---|
| Wavefunction polar decomposition (1) and Madelung hydrodynamics (2)–(3) | Established[^7][^5][^4] |
| Madelung transform is a Kähler map (Khesin–Misiolek–Modin 2018) | Established[^5][^6][^3] |
| Nodal sets of Schrödinger eigenfunctions are smooth hypersurfaces | Established[^14][^16] |
| Antisymmetry forces \(\Psi = 0\) on coincidence loci | Established[^18][^20] |
| Two-nodal-domain conjecture for fermionic ground states | Partially proven; conjectural in general[^26][^27] |
| Fixed-node QMC as antisymmetry-to-boundary-condition reduction | Established methodology[^25][^23][^24] |
| Montgomery pair correlation = GUE (conditional on RH) | Proven for restricted test functions[^32] |
| Odlyzko GUE numerics for Riemann zeros | Established numerically[^32][^33] |
| Berry–Keating \(xp\) conjecture | Speculative; partial support[^31][^30][^40] |
| BBM Hamiltonian eigenvalues match zeta zeros | Established under boundary condition; self-adjointness unproven[^42][^45][^44] |
| Connes absorption spectrum reduces RH to positivity | Established reduction; positivity unproven[^46][^47] |
| DTES operator as candidate Hilbert–Pólya operator | Speculative/research hypothesis[^62] |
| Pauli antisymmetry ↔ DTES forbidden regions | Speculative analogy |
| Zeta zeros ↔ DTES spectral nodal structure | Speculative hypothesis |

***

## Key References

- **Montgomery, H.** (1973). The pair correlation of the zeros of the zeta function. *Analytic Number Theory*, AMS.[^32]
- **Odlyzko, A.M.** (1987). On the distribution of spacings between zeros of the zeta function. *Math. Comp.* 48, 273–308.[^33]
- **Berry, M.V. and Keating, J.P.** (1999). The Riemann zeros and eigenvalue asymptotics. *SIAM Review* 41(2), 236–266.[^40][^30][^31]
- **Connes, A.** (1999). Trace formula in noncommutative geometry and the zeros of the Riemann zeta function. *Selecta Math.* 5, 29–106. arXiv:math/9811068.[^46][^47]
- **Bender, C.M., Brody, D.C., Müller, M.P.** (2017). Hamiltonian for the zeros of the Riemann zeta function. *Phys. Rev. Lett.* 118, 130201. arXiv:1608.03679.[^42][^45]
- **Khesin, B., Misiolek, G., Modin, K.** (2018). Geometric hydrodynamics via Madelung transform. *PNAS* 115(24), 6165–6170.[^4][^5]
- **Katz, N.M. and Sarnak, P.** (1999). Zeros of zeta functions and symmetry. *Bull. AMS* 36, 1–26.[^35][^34]
- **Bressanini, D.** (2012). Implications of the two nodal domains conjecture for ground state fermionic wave functions. *Phys. Rev. B* 86, 115120.[^26][^27]
- **Rudnick, Z. and Sarnak, P.** (1996). Zeros of principal L-functions and random matrix theory. *Duke Math. J.* 81, 269–322.[^35]
- **Connes, A., Consani, C., Marcolli, M.** (2007). The Weil proof and the geometry of the adèles class space. arXiv:math/0703Connes.[^48]
- **Khachatryan, A.** (2025). Quantum mechanics from broken translation symmetry: a spectral–geometric framework. arXiv:2501.[^62]
- **Ryczko, K. et al.** (2025). A minimal and universal representation of fermionic wavefunctions. arXiv:2510.11431.[^21]

---

## References

1. [[PDF] Quantum Physics III Chapter 3: Semiclassical Approximation](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/bf207c35150e1f5d93ef05d4664f406d_MIT8_06S18ch3.pdf) - The real part of S, divided by h, is the phase of the wavefunction. The imaginary part of S(x) deter...

2. [[PDF] Chapter 14 The WKB Method](https://faculty.washington.edu/seattle/physics541/14text.pdf)

3. [[PDF] Geometry of the Madelung transform - arXiv](https://arxiv.org/pdf/1807.07172.pdf) - Thus the Madelung transform relates the standard symplectic structure on the space of wave functions...

4. [Geometric hydrodynamics via Madelung transform - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6004435/) - Geometry has always played a fundamental role in theoretical physics via symmetries and conservation...

5. [Geometric hydrodynamics via Madelung transform | PNAS](https://www.pnas.org/doi/10.1073/pnas.1719346115) - We introduce a geometric framework to study Newton’s equations on infinite-dimensional configuration...

6. [Digital Object Identifier (DOI) https://doi.org/10.1007/s00205-019-01397-2](https://www.math.toronto.edu/khesin/papers/madelungARMA.pdf)

7. [arXiv:2104.13185v1  [math-ph]  27 Apr 2021](https://arxiv.org/pdf/2104.13185.pdf)

8. [[PDF] Analytic Solutions of the Madelung Equation - KFKI](https://www.kfki.hu/~barnai/madelung.pdf) - There is a direct connection between the zeros of the Madelung fluid density and the magnitude of th...

9. [Topology made visible through standing waves in a spinning fluid](https://www.nature.com/articles/s42005-026-02603-w) - Here, we demonstrate that standing surface waves scattered by a single vortex give rise to global no...

10. [A geometric effect of quantum particles originated](https://arxiv.org/pdf/2402.11624.pdf)

11. [[PDF] Semiclassical approximations in wave mechanics - Michael Berry](https://michaelberryphysics.wordpress.com/wp-content/uploads/2013/07/berry023.pdf) - The phase occurring in the WKB solutions (2.18) is the classical action S, measured in units of E,. ...

12. [WKB for semiclassical operators: How to fly over caustics (and more)](https://arxiv.org/html/2603.25601v1) - This framework provides a rigorous proof of the Bohr–Sommerfeld–Einstein–Brillouin–Keller quantizati...

13. [[PDF] Lectures on the Geometry of Quantization](https://math.berkeley.edu/~alanw/GofQ.pdf) - 1 In these notes, we show how symplectic geometry arises from the study of semi-classical solutions ...

14. [[PDF] Nodal sets for the groundstate of the Schrödinger operator with zero ...](https://arxiv.org/pdf/math/9807064.pdf) - Abstract. We investigate nodal sets of magnetic Schrödinger operators with zero magnetic field, acti...

15. [On the relation between nodal structures in quantum wave functions ...](https://pubs.aip.org/aip/adv/article/13/12/125307/2928620/On-the-relation-between-nodal-structures-in) - We study the influence of nodal structures in two-dimensional quantum mechanical densities on wave p...

16. [Topology of the nodal sets of eigenfunctions of Schrödinger operators](https://www.youtube.com/watch?v=pue3EoCXs7s) - ... zeros in the eigenfunctions of the harmonic oscillator. J. Eur. Math. Soc. 20, 301-314. [2] Enci...

17. [[PDF] On Courant's nodal domain property for linear combinations of ... - HAL](https://hal.science/hal-01718768/file/berard-helffer-ecp-II-balslev-190926-ha.pdf) - In this case, the function w has a closed nodal line and two nodal domains. Note: We know that dim (...

18. [Pauli exclusion principle - Wikipedia](https://en.wikipedia.org/wiki/Pauli_exclusion_principle) - Two or more identical particles with half-integer spins (ie fermions) cannot simultaneously occupy t...

19. [8.5: Wavefunctions must be Antisymmetric to Interchange of any Two ...](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/08:_Multielectron_Atoms/8.05:_Wavefunctions_must_be_Antisymmetric_to_Interchange_of_any_Two_Electrons) - The Pauli Exclusion Principle is simply the requirement that the wavefunction be antisymmetric for e...

20. [1.4 The Pauli Exclusion Principle](https://www.schulz.chemie.uni-rostock.de/storages/uni-rostock/Alle_MNF/Chemie_Schulz/Computerchemie_2/pauli.html) - The Pauli exclusion principle dictates that no two Fermions can be in the same quantum state. This f...

21. [A minimal and universal representation of fermionic ...](https://arxiv.org/html/2510.11431v1)

22. [Weighted nodal domain averages of eigenstates for quantum Monte Carlo and beyond](https://www.osti.gov/servlets/purl/1976953)

23. [3. Construction And...](https://www.emergentmind.com/topics/fixed-node-approximation) - The fixed-node approximation stabilizes quantum Monte Carlo calculations by constraining nodal surfa...

24. [Fixed-node methods and geminal nodes (or Topology of ...](http://www.mcc.uiuc.edu/summerschool/2007/qmc/tutorials/FixedNode_Mitas.pdf)

25. [Wave functions for quantum Monte Carlo calculations in ...](https://www.fzu.cz/~kolorenc/papers/PhysRevB_82_115108_2010.pdf)

26. [Implications of the two nodal domains conjecture for ground state ...](https://link.aps.org/doi/10.1103/PhysRevB.86.115120) - The aim of this paper is to stimulate the investigation of the properties of the nodes of many-body ...

27. [Implications of the two nodal domains conjecture for ground state fermionic wave functions](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.86.115120) - The nodes of many-body wave functions are mathematical objects important in many different fields of...

28. [From Wavefunction Sign Structure to Static Correlation - arXiv](https://arxiv.org/html/2511.01569v2) - Bressanini (2012) Implications of the two nodal domains conjecture for ground-state fermionic wave f...

29. [[PDF] positioning the zeta function zeros with an optimum frame - HAL](https://hal.science/hal-04036089/file/Riemann_locus01.pdf) - Abstract. The purpose of this article is to design a frame of reference enabling to locate the Zeta ...

30. [Riemann zeros and eigenvalue asymptotics - Technion](https://cris.technion.ac.il/en/publications/riemann-zeros-and-eigenvalue-asymptotics-2/) - Berry, M. V. ; Keating, J. P. / Riemann zeros and eigenvalue asymptotics. In: SIAM Review. 1999 ; Vo...

31. [The Riemann Zeros and Eigenvalue Asymptotics | SIAM Review](https://epubs.siam.org/doi/10.1137/S0036144598347497) - The Riemann-Siegel formula for the zeta function is described in detail. Its interpretation as a rel...

32. [Montgomery's pair correlation conjecture - Wikipedia](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture) - In mathematics, Montgomery's pair correlation conjecture is a conjecture made by Hugh Montgomery (19...

33. [MATHEMATICS OF COMPUTATION](https://www.physics.rutgers.edu/grad/682/papers/zeta.pdf)

34. [[PDF] Zeros of Zeta Functions and Symmetry](https://web.williams.edu/Mathematics/sjmiller/public_html/ntprob19/handouts/general/KatzSarnak_Zeroes%20of%20zeta%20functions%20and%20symmetry%20BAMS%201999.pdf) - We call this phenomenon - that the high zeroes of any fixed L(s, f), f a cusp form on GLm/Q obey GUE...

35. [Zeros of Zeta Functions and Symmetry](https://www.ams.org/journals/bull/1999-36-01/S0273-0979-99-00766-1/S0273-0979-99-00766-1.pdf)

36. [Hilbert–Pólya conjecture - Wikipedia](https://en.wikipedia.org/wiki/Hilbert%E2%80%93P%C3%B3lya_conjecture) - History. edit. In a letter to Andrew Odlyzko, dated January 3, 1982, George Pólya said that while he...

37. [Hilbert-Pólya Conjecture -- from Wolfram MathWorld](https://mathworld.wolfram.com/Hilbert-PolyaConjecture.html) - Hilbert-Pólya Conjecture. The nontrivial zeros of the Riemann zeta function correspond to the eigenv...

38. [Reality of the Eigenvalues of the Hilbert-Pólya Hamiltonian - arXiv](https://arxiv.org/html/2408.15135v4) - Essentially, the HPC involves two stages: (I) finding an operator whose eigenvalues correspond to th...

39. [Hamiltonian for the Hilbert-Pólya Conjecture - arXiv](https://arxiv.org/html/2309.00405v5) - Essentially, the Hilbert-Pólya conjecture involves two stages: (I) finding an operator whose eigenva...

40. [[PDF] The Riemann Zeros and Eigenvalue Asymptotics - shiftleft.com](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/98/HPL-BRIMS-98-26.pdf) - We speculate that the Riemann dynamics is related to the trajectories generated by the classical ham...

41. [[PDF] The Riemann Zeros as Spectrum and the Riemann Hypothesis](https://s3.cern.ch/inspire-prod-files-1/1e65b86fec7566dba4d2d2384183f67b) - In this paper we shall review the progress made along this direction starting from the famous xp mod...

42. [[1608.03679] Hamiltonian for the zeros of the Riemann zeta function](https://arxiv.org/abs/1608.03679) - Hamiltonian for the zeros of the Riemann zeta function. Authors:Carl M. Bender, Dorje C. Brody, Mark...

43. [[PDF] Hamiltonian for the zeros of the Riemann zeta function](https://bura.brunel.ac.uk/bitstream/2438/14197/1/FullText.pdf) - If the analysis presented here can be made rigorous to show that ˆH is manifestly self-adjoint, then...

44. [Self-adjointness of the Bender–Brody–Müller operator](https://www.emergentmind.com/open-problems/self-adjointness-bender-brody-mueller-operator) - Establish whether the Hamiltonian proposed by Carl M. Bender, Dorje C. Brody, and Markus P. Müller (...

45. [Hamiltonian for the Zeros of the Riemann Zeta Function](https://link.aps.org/doi/10.1103/PhysRevLett.118.130201) - A Hamilonian with PT-symmetry invariant properties is investigated as a candidate for an operator th...

46. [Trace formula in noncommutative geometry and the zeros of ... - arXiv](https://arxiv.org/abs/math/9811068) - We give a geometric interpretation of the explicit formulas of number theory as a trace formula on t...

47. [Trace Formula in Noncommutative Geometry and](https://alainconnes.org/wp-content/uploads/selecta.ps-2.pdf)

48. [[PDF] The Weil Proof and the Geometry of the Adèles Class Space](https://www.its.caltech.edu/~matilde/WeilProofAdelesClassSpace.pdf) - This paper explores analogies between the Weil proof of the Riemann hypothesis for function fields a...

49. [[PDF] Observation of Wigner-Dyson level statistics in a classically ...](https://link.aps.org/accepted/10.1103/PhysRevE.103.062211)

50. [PHYSICAL REVIEW E 94, 062214 (2016)](https://chaos1.la.asu.edu/~yclai/papers/PRE_2016_YLXHDGL.pdf)

51. [Spectral Statistics in the Quantized Cardioid Billiard - PUBDB](https://bib-pubdb1.desy.de/record/593099/files/DESY-94-213.pdf)

52. [Selberg trace formula - Wikipedia](https://en.wikipedia.org/wiki/Selberg_trace_formula) - Motivated by the analogy, Selberg introduced the Selberg zeta function of a Riemann surface, whose a...

53. [Selberg trace formula in nLab](https://ncatlab.org/nlab/show/Selberg+trace+formula) - The Selberg trace formula (Selberg 1956) is an expression for certain sums of eigenvalues of the Lap...

54. [[PDF] Selberg's Trace Formula: An Introduction - University of Bristol](https://people.maths.bris.ac.uk/~majm/bib/selberg.pdf) - The aim of this short lecture course is to develop Selberg's trace formula for a compact hyperbolic ...

55. [From Prime Numbers to Nuclear Physics and Beyond | Ideas](https://www.ias.edu/ideas/2013/primes-random-matrices) - In early April 1972, Hugh Montgomery, who had been a Member in the School of Mathematics the previou...

56. [[PDF] Notes on L-functions and Random Matrix Theory](https://aimath.org/~kaur/publications/58.pdf) - Sarnak and Katz developed their theory of symmetry types of families of. L-functions by studying zet...

57. [[PDF] arXiv:math/0601653v2 [math.NT] 30 Jan 2006](https://arxiv.org/pdf/math/0601653.pdf)

58. [[PDF] Arithmetic consequences of the GUE conjecture for zeta zeros](http://user.math.uzh.ch/rodgers/ArithmeticGUE.pdf) - Conditioned on the Riemann hypothesis, we show that the conjecture that the zeros of the Riemann zet...

59. [[PDF] L-functions and Random Matrix Theory](https://www.aimath.org/WWN/lrmt/lrmt.pdf) - Goldston,. Gonek, and Montgomery have shown that the pair correlation conjecture is equivalent to a ...

60. [c:\yandy\texte\cmp\22021401\cmp737.dvi](https://people.maths.bris.ac.uk/~mancs/papers/RMTLfunction.pdf)

61. [[PDF] Random Matrix Theory and L-functions at s =1/2 - shiftleft.com](http://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2000/HPL-BRIMS-2000-05.pdf) - Recent results of Katz and Sarnak [9,10] suggest that the low- lying zeros of families of L-function...

62. [[PDF] Quantum Mechanics from Broken Translation Symmetry: A Spectral ...](https://philarchive.org/archive/KHAQMF) - The quantum potential is shown to emerge directly from the curvature penalty encoded in ∆S. A univer...

63. [Minimax spectral estimation of weighted Laplace operators - arXiv](https://arxiv.org/abs/2511.22694) - These operators arise as continuum limits of graph Laplacian matrices and provide valuable geometric...

64. [[1311.0038] On the Spectrum of weighted Laplacian operator and its ...](https://arxiv.org/abs/1311.0038) - The purpose of this paper is to provide a new proof of Bando-Mabuchi's uniqueness theorem of Kähler ...

65. [The existence of Hilbert-Pólya operator | Mathematics | Cambridge Open Engage](https://client.prod.orp.cambridge.org/engage/coe/article-details/69d390ed810b9dcc82e69f7e) - Hilbert-Pólya conjecture is proved by constructing Hilbert-Pólya operator, the self-adjoint operator...

66. [[PDF] spectral deformations of one-dimensional schrodinger operators](http://math.caltech.edu/SimonPapers/258.pdf)

