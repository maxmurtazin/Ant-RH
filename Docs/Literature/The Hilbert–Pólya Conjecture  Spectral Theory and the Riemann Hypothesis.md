# The Hilbert–Pólya Conjecture: Spectral Theory and the Riemann Hypothesis

## Overview

The **Hilbert–Pólya conjecture** proposes that the non-trivial zeros of the Riemann zeta function \(\zeta(s)\) are the eigenvalues of a self-adjoint operator on some Hilbert space. If true, the Riemann Hypothesis (RH) would follow automatically from the spectral theorem, since self-adjoint operators can only have real eigenvalues. The conjecture forges a bridge between analytic number theory, quantum mechanics, and random matrix theory — and despite decades of effort and compelling numerical evidence, the operator has never been rigorously constructed.[^1][^2][^3][^4]

***

## 1. Basic Idea

### What the Conjecture States

The Riemann zeta function \(\zeta(s)\) has two kinds of zeros:
- **Trivial zeros** at \(s = -2, -4, -6, \ldots\)
- **Non-trivial zeros** at complex numbers \(\rho = \sigma + it\) with \(0 < \sigma < 1\)

The **Riemann Hypothesis** asserts that all non-trivial zeros satisfy \(\sigma = 1/2\), i.e., they lie on the *critical line* \(\mathrm{Re}(s) = 1/2\).[^5][^1]

The Hilbert–Pólya conjecture states: there exists a self-adjoint (Hermitian) operator \(\hat{H}\) on a Hilbert space \(\mathcal{H}\), whose eigenvalues \(\{\gamma_n\}\) are exactly the imaginary parts of the non-trivial zeros. Writing \(\rho_n = 1/2 + i\gamma_n\), the conjecture means:[^2][^1]

\[
\hat{H}\,\psi_n = \gamma_n\,\psi_n, \qquad \text{where } \zeta\!\left(\tfrac{1}{2}+i\gamma_n\right)=0. \tag{1}
\]

### Why It Would Imply the Riemann Hypothesis

The **spectral theorem** guarantees that every self-adjoint operator on a Hilbert space has only **real** eigenvalues. If the \(\gamma_n\) in (1) are the imaginary parts of the non-trivial zeros, and they are eigenvalues of a self-adjoint operator, then they must be real — meaning every zero has the form \(1/2 + i\gamma_n\) with \(\gamma_n \in \mathbb{R}\). That is precisely the Riemann Hypothesis.[^3][^4][^6]

The implication runs: *self-adjointness of \(\hat{H}\) ⟹ all eigenvalues \(\gamma_n \in \mathbb{R}\) ⟹ all zeros on the critical line ⟹ RH*.[^4]

**Historical origin.** Around 1912–1914, George Pólya (recounted in a 1982 letter to Andrew Odlyzko) suggested this spectral idea when asked by Edmund Landau for a physical reason the RH should be true. David Hilbert had independently speculated, according to a story relayed by Ernst Hellinger, that the RH might follow from Fredholm's theory of integral operators with symmetric kernels. The conjecture was first published by Hugh Montgomery in 1973.[^1]

***

## 2. Mathematical Formulation

### Nontrivial Zeros and Eigenvalues of a Self-Adjoint Operator

Formally, the conjecture asserts the existence of a densely defined, self-adjoint operator \(\hat{H}: \mathcal{D}(\hat{H}) \subset \mathcal{H} \to \mathcal{H}\) satisfying \(\hat{H} = \hat{H}^\dagger\) on its domain, with spectrum

\[
\sigma(\hat{H}) = \left\{\gamma \in \mathbb{R} : \zeta\!\left(\tfrac{1}{2}+i\gamma\right)=0\right\}. \tag{2}
\]

Equivalently, one may phrase this as: the operator \(Q = \frac{1}{2} + i\hat{H}\) has spectrum equal to the set of non-trivial zeros of \(\zeta(s)\).[^7][^6]

The conjecture is structured around two distinct stages:[^6][^4]

- **Stage I** — *Spectral match*: find a (possibly only formally symmetric) operator whose eigenvalues match the \(\gamma_n\), *assuming* RH is true.
- **Stage II** — *Self-adjointness*: prove that this operator is genuinely self-adjoint (not merely symmetric) on a well-defined dense domain — this is the step that would actually imply RH.

Proving self-adjointness in the strict functional-analytic sense requires verifying that the operator equals its own adjoint on a specific dense domain, that its deficiency indices \((n_+, n_-)\) are both zero, and that the resolvent exists for non-real \(\lambda\). This is far harder than constructing a formally symmetric operator.[^3][^6]

### Why Self-Adjointness Forces Zeros onto the Critical Line

For a self-adjoint operator \(\hat{H}\), the spectral theorem asserts that:

\[
\hat{H} = \int_{\mathbb{R}} \lambda \, dE(\lambda),
\]

where \(E(\lambda)\) is the spectral resolution of the identity (a projection-valued measure on \(\mathbb{R}\)). All spectral values are **real**. Therefore, any eigenvalue \(\gamma_n\) is real, and the associated zero \(\rho_n = 1/2 + i\gamma_n\) has real part exactly \(1/2\).[^8][^6][^1]

If, on the other hand, the RH were false and some zero had \(\mathrm{Re}(\rho) \neq 1/2\), then the imaginary part of that zero would not be a real eigenvalue — it could not appear in the spectrum of any self-adjoint operator. So the existence of a self-adjoint \(\hat{H}\) with spectrum (2) is precisely equivalent to the RH.[^4]

***

## 3. Intuition: Zeros as a Quantum Spectrum

### Quantum Energy Levels

An isolated quantum system — say, a particle confined in a potential well — has discrete, real energy levels \(\{E_n\}\), solutions to the Schrödinger equation \(\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle\) with \(\hat{H}\) Hermitian. The Hilbert–Pólya conjecture literally suggests: *the Riemann zeros \(\gamma_n\) are the energy levels of some quantum system we have not yet identified*[^9][^10].

This analogy is strengthened by the **Weil explicit formula**, which expresses the prime-counting function in terms of the non-trivial zeros:[^11]

\[
\psi_0(x) = x - \sum_{\rho}\frac{x^\rho}{\rho} - \log(2\pi) - \tfrac{1}{2}\log\!\left(1 - x^{-2}\right). \tag{3}
\]

Here, the sum over non-trivial zeros \(\rho\) plays the role that periodic orbits play in quantum mechanics via the **Gutzwiller trace formula**: the zeros are the "resonant frequencies" encoding the arithmetic structure of the primes. Just as you can recover the geometry of a billiard from its vibration spectrum, you can recover the distribution of primes from the zeta zeros.[^10][^9]

### Three Structurally Identical Trace Formulas

The analogy between zeros and energy levels is made precise by observing that three important formulas in mathematics and physics share the same structure — *spectrum ↔ periodic orbits*:[^9][^10]

| Formula | Spectrum | "Periodic Orbits" |
|---|---|---|
| Weil explicit formula | Riemann zeros | Prime numbers |
| Selberg trace formula | Laplacian eigenvalues on a hyperbolic surface | Closed geodesics |
| Gutzwiller trace formula | Quantum energy levels of a chaotic system | Classical periodic orbits |

This triple coincidence suggests that the hypothetical Hilbert–Pólya operator should be the quantization of a **classically chaotic Hamiltonian** whose periodic orbits are tied to the primes.[^10][^9]

***

## 4. Connection to Physics and Quantum Chaos

### Quantum Chaos and Level Statistics

In quantum mechanics, the statistical distribution of energy level spacings depends critically on whether the underlying classical dynamics is **integrable** or **chaotic**:[^12][^13]

- **Integrable classical dynamics** → quantum levels show **Poisson statistics**: spacings are independent, levels can accidentally coincide (no repulsion).
- **Chaotic classical dynamics** → quantum levels show **Wigner–Dyson statistics**: strong level repulsion, eigenvalues push each other apart.

The Bohigas–Giannoni–Schmit conjecture (1984) formalizes this: quantum systems with classically chaotic counterparts obey random matrix theory (RMT) level statistics. The specific ensemble depends on symmetry:[^12]

| Ensemble | Physical Symmetry | Level Repulsion |
|---|---|---|
| GOE (Gaussian Orthogonal) | Time-reversal invariant, integer spin | \(P(s) \sim s\) |
| GUE (Gaussian Unitary) | No time-reversal symmetry | \(P(s) \sim s^2\) |
| GSE (Gaussian Symplectic) | Time-reversal invariant, half-integer spin | \(P(s) \sim s^4\) |

### Gaussian Unitary Ensemble (GUE) and Zeta Zeros

The key empirical fact is that the spacing statistics of the Riemann zeros match the **GUE** ensemble — not the GOE or GSE. The pair correlation function of large GUE random Hermitian matrices is:[^14][^15][^11]

\[
R_2(u) = 1 - \left(\frac{\sin \pi u}{\pi u}\right)^2. \tag{4}
\]

Montgomery proved (conditionally on RH) that the Riemann zeros have this exact pair correlation. The GUE match implies the unknown Hilbert–Pólya Hamiltonian should govern a quantum chaotic system with **broken time-reversal invariance**.[^15][^11][^10]

***

## 5. Known Results

### Montgomery Pair Correlation (1973)

Hugh Montgomery proved the following theorem, assuming the Riemann Hypothesis:[^16][^15]

\[
\lim_{T\to\infty} \frac{\#\left\{(\gamma, \gamma') : 0 < \gamma,\gamma' \leq T,\; \frac{2\pi\alpha}{\log T} \leq \gamma-\gamma' \leq \frac{2\pi\beta}{\log T}\right\}}{\frac{T}{2\pi}\log T} = \int_\alpha^\beta\left[1-\left(\frac{\sin\pi u}{\pi u}\right)^2\right]du. \tag{5}
\]

This says the normalized two-point spacing distribution of Riemann zeros is the GUE pair correlation function (4). The result was initially proved only for test functions whose Fourier transforms are supported in \((-1,1)\); the full conjecture for all test functions remains open.[^16]

The discovery of this connection is legendary: in April 1972 at the Institute for Advanced Study, Montgomery showed his result to Freeman Dyson over tea. Dyson immediately recognized the right-hand side of (5) as the GUE formula he had computed years earlier for nuclear physics — a connection Montgomery had not anticipated.[^17][^1]

### Odlyzko's Numerical Experiments (1987–2000s)

Andrew Odlyzko computed tens of millions of Riemann zeros at extraordinary heights — including zeros near the \(10^{20}\)-th and \(10^{22}\)-nd zero — and compared their spacing statistics to GUE predictions. The agreement is visually and statistically overwhelming:[^18][^19][^20][^15]

- Nearest-neighbor spacing distributions match the GUE prediction precisely.
- The pair correlation function matches (5) to very high precision.
- Higher-order \(n\)-point correlations also agree with GUE.
- Large-scale verifications consistently yield \(\langle r \rangle \approx 0.601\), consistent with GUE.[^21]

These experiments remain the strongest empirical evidence for the Hilbert–Pólya conjecture, though they do not constitute a proof.[^18]

### Berry–Keating Proposal (1999)

Building on the quantum chaos analogy, Michael Berry and Jon Keating proposed that the classical Hamiltonian underlying the Hilbert–Pólya operator is:[^22][^10]

\[
H_\mathrm{cl} = xp, \tag{6}
\]

where \(x\) is position and \(p\) is momentum. This generates **classical dilation** (scaling) dynamics: \(\dot{x} = x\), \(\dot{p} = -p\), with trajectories \(x(t) = x_0 e^t\), \(p(t) = p_0 e^{-t}\). The trajectories are hyperbolic (unstable) — the hallmark of classical chaos.[^23]

The quantization of \(xp\) gives the symmetric operator:[^24][^10]

\[
\hat{H}_\mathrm{BK} = \tfrac{1}{2}(\hat{x}\hat{p} + \hat{p}\hat{x}) = -i\hbar\!\left(x\frac{d}{dx} + \tfrac{1}{2}\right). \tag{7}
\]

Berry and Keating showed that a regularization of (6) — constraining phase space to \(|x| \geq \ell_x\) and \(|p| \geq \ell_p\) — gives a semiclassical spectrum that reproduces the **average density** of Riemann zeros, matching the smooth Riemann–Mangoldt formula[^22][^25]. However, the exact zeros do not appear in the spectrum of any known modification of the \(xp\) Hamiltonians[^23].

### Bender–Brody–Müller Operator (2017)

Bender, Brody, and Müller constructed an explicitly defined Hamiltonian with classical limit \(H = 2xp\):[^26][^24]

\[
\hat{H}_\mathrm{BBM} = \hat{x}\hat{p} + \hat{p}\hat{x} + \hat{x}^{-1}(1 - e^{-i\hat{p}})\hat{p}^{-1}. \tag{8}
\]

This operator is not Hermitian in the conventional sense, but \(i\hat{H}_\mathrm{BBM}\) is \(\mathcal{PT}\)-symmetric (invariant under parity–time reflection) with broken \(\mathcal{PT}\) symmetry. They showed: if the eigenfunctions satisfy the boundary condition \(\psi_n(0) = 0\), then the eigenvalues are precisely the imaginary parts of the non-trivial zeros. If the operator can be shown to be self-adjoint on an appropriate inner product, the RH follows — but this self-adjointness remains unproven.[^27][^28][^24][^26]

### Connes' Noncommutative Geometry Approach (1999)

Alain Connes gave a spectral realization of the zeros via a trace formula on the **noncommutative space of Adèle classes** \(X_\mathbb{Q} = \mathbb{A}_\mathbb{Q}/\mathbb{Q}^\times\). In this framework, the critical zeros appear as an **absorption spectrum** — missing lines in a trace formula — rather than as an emission spectrum of eigenvalues. The Weil explicit formula becomes a trace formula on this noncommutative space. Connes reduced the RH to a positivity condition on a certain distribution; verifying this positivity remains open.[^29][^30][^31][^32]

***

## 6. The Open Problem: What Is Still Missing

### Why No Operator Has Been Rigorously Constructed

Decades of effort have identified the following key obstacles:[^23][^6][^4]

1. **The xp spectrum is continuous, not discrete.** The naive quantization of \(H_\mathrm{cl} = xp\) on \(L^2(\mathbb{R}^+)\) has a purely continuous spectrum (the entire real line). Discretizing it requires boundary conditions or modifications that are necessarily *ad hoc* and break the natural symmetry.[^25][^23]

2. **Modified xp models match only the average zero density.** All known modifications of the Berry–Keating \(xp\) Hamiltonian reproduce the smooth (Weyl-law) counting function for zeros, but *not the exact zeros themselves*. As Sierra's review concludes: "There is no trace of the exact Riemann zeros in the spectrum of the modified \(xp\) models".[^23]

3. **Self-adjointness is a hard analytic problem.** Constructing a formally symmetric operator whose eigenvalues look like the \(\gamma_n\) under heuristic argument is relatively tractable. The genuinely difficult challenge is verifying **self-adjointness** in the strict sense: checking that the domain is dense, that the deficiency indices vanish, and that the adjoint coincides with the operator. For the BBM operator, this step remains unresolved.[^27][^6]

4. **No canonical physical interpretation.** One would need to identify a natural quantum mechanical system — a specific potential, geometry, or field theory — whose energy levels are *exactly* the Riemann zeros. No geometrically or physically natural choice has succeeded. The Connes approach works in noncommutative geometry but requires a positivity verification that is likewise unproven.[^29][^9][^10]

5. **Off-critical zeros would require complex eigenvalues.** The framework is inherently circular: to *construct* the operator, one typically assumes RH (Stage I), then hopes to prove self-adjointness (Stage II) without circular reasoning. It is not yet clear how to break this circularity rigorously.[^6][^4]

### Why This Remains Unsolved

The fundamental difficulty is that the Riemann zeros are a number-theoretic object of extreme arithmetic richness, defined by the distribution of primes. A self-adjoint operator, by contrast, is an analytic object defined by geometry or physics. Bridging these two worlds — showing that one specific operator in one specific Hilbert space has *exactly* the same spectrum as the zeros — requires an unexpected confluence of analytic number theory, functional analysis, and mathematical physics that has not yet materialized. As Odlyzko's experiments and Montgomery's theorem show, the statistical *fingerprint* of the zeros matches GUE universality with stunning precision — but universality applies to broad classes of chaotic systems, and does not by itself identify the unique physical system behind the zeros.[^20][^9][^3][^18]

***

## Key References

- **Montgomery (1973)**: "The pair correlation of the zeros of the zeta function," *Analytic Number Theory*, AMS — introduces the GUE connection[^15]
- **Odlyzko (1987)**: "On the distribution of spacings between zeros of the zeta function," *Math. Comp.* 48 — numerical GUE verification[^11]
- **Berry & Keating (1999)**: "The Riemann zeros and eigenvalue asymptotics," *SIAM Review* 41 — the \(xp\) Hamiltonian proposal[^10]
- **Connes (1999)**: "Trace formula in noncommutative geometry and the zeros of the Riemann zeta function," *Selecta Math.* — absorption spectrum approach[^29]
- **Bender, Brody & Müller (2017)**: "Hamiltonian for the zeros of the Riemann zeta function," *Phys. Rev. Lett.* 118, 130201 — explicit \(\mathcal{PT}\)-symmetric Hamiltonian[^26]
- **Sierra (2019)**: "The Riemann zeros as spectrum and the Riemann hypothesis," *Symmetry* 11 — comprehensive review of \(xp\) models[^23]
- **Katz & Sarnak (1999)**: "Zeros of zeta functions and symmetry," *Bull. AMS* 36 — symmetry classification of zeta zeros[^19]

---

## References

1. [Hilbert–Pólya conjecture - Wikipedia](https://en.wikipedia.org/wiki/Hilbert%E2%80%93P%C3%B3lya_conjecture)

2. [Hilbert-Pólya Conjecture -- from Wolfram MathWorld](https://mathworld.wolfram.com/Hilbert-PolyaConjecture.html) - Hilbert-Pólya Conjecture. The nontrivial zeros of the Riemann zeta function correspond to the eigenv...

3. [[TeX] A Self-Adjoint Operator with Spectrum Given by the Riemann Zeta ...](https://zenodo.org/records/19203882/files/A_Self_Adjoint_Operator.tex?download=1) - ... Self-Adjoint Operator with Spectrum Given by the Riemann Zeta Zeros. A Self ... operator; spectr...

4. [Hamiltonian for the Hilbert-Pólya Conjecture - arXiv](https://arxiv.org/html/2309.00405v5) - Essentially, the Hilbert-Pólya conjecture involves two stages: (I) finding an operator whose eigenva...

5. [Physicists make major breakthrough towards Hilbert-Pólya conjecture](https://www.brunel.ac.uk/news-and-events/news/articles/Physicists-make-major-breakthrough-towards-proof-of-Riemann-hypothesis) - Scientists in the UK, US and Canada have made a significant breakthrough in attempting to establish ...

6. [Reality of the Eigenvalues of the Hilbert-Pólya Hamiltonian](https://arxiv.org/html/2408.15135v4)

7. [[PDF] Nontrivial Riemann Zeros as Spectrum - arXiv](https://arxiv.org/pdf/2408.15135.pdf) - Under the same positivity condition, the intertwining relation yields a self- adjoint operator whose...

8. [[PDF] arXiv:1211.5198v2 [math-ph] 11 Jul 2013](https://arxiv.org/pdf/1211.5198.pdf)

9. [Riemann zeros and quantum chaos - Scholarpedia](http://www.scholarpedia.org/article/Riemann_zeros_and_quantum_chaos) - This article will describe properties of Riemann zeros and their links with field of quantum chaos. ...

10. [The Riemann Zeros and Eigenvalue Asymptotics | SIAM Review](https://epubs.siam.org/doi/10.1137/S0036144598347497) - The Riemann-Siegel formula for the zeta function is described in detail. Its interpretation as a rel...

11. [[PDF] On the Distribution of Spacings Between Zeros of the Zeta Function](https://www.physics.rutgers.edu/grad/682/papers/zeta.pdf) - GUE properties of eigenvalues might apply at least approximately to the zeros of the zeta function. ...

12. [Quantum Chaos & Level Statistics - Iris](https://iris.joshua-becker.com/lab/quantum-chaos-level-statistics/) - Wigner-Dyson vs Poisson — chaos leaves a fingerprint in quantum spectra. Billiard Geometry. Stadium ...

13. [Quantum Chaos - Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/archives/win2024/entries/chaos/quantum-chaos.html) - The reverse transition also is possible, where a chaotic billiards ... level statistics for the nonc...

14. [Montgomery's Pair Correlation Conjecture -- from Wolfram MathWorld](https://mathworld.wolfram.com/MontgomerysPairCorrelationConjecture.html) - Montgomery's pair correlation conjecture, published in 1973, asserts that the two-point correlation ...

15. [Montgomery's pair correlation conjecture - Wikipedia](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture) - In mathematics, Montgomery's pair correlation conjecture is a conjecture made by Hugh Montgomery (19...

16. [[2501.14545] Pair Correlation of Zeros of the Riemann Zeta Function I](https://arxiv.org/abs/2501.14545) - Assuming the Riemann Hypothesis (RH), Montgomery proved a theorem in 1973 concerning the pair correl...

17. [From Prime Numbers to Nuclear Physics and Beyond | Ideas](https://www.ias.edu/ideas/2013/primes-random-matrices) - In early April 1972, Hugh Montgomery, who had been a Member in the School of Mathematics the previou...

18. [[PDF] arXiv:math/0601653v2 [math.NT] 30 Jan 2006](https://arxiv.org/pdf/math/0601653.pdf)

19. [Zeros of Zeta Functions and Symmetry](https://www.ams.org/journals/bull/1999-36-01/S0273-0979-99-00766-1/S0273-0979-99-00766-1.pdf)

20. [BULLETIN (New Series) OF THE](https://web.williams.edu/Mathematics/sjmiller/public_html/ntprob17/handouts/general/KatzSarnak_Zeroes%20of%20zeta%20functions%20and%20symmetry%20BAMS%201999.pdf)

21. [Large-Scale Verification of GUE Statistics in Riemann Zero Spacings](https://www.academia.edu/144852412/Large_Scale_Verification_of_GUE_Statistics_in_Riemann_Zero_Spacings) - We present a comprehensive statistical analysis of spacing between the first 1000 nontrivial zeros o...

22. [[0712.0705] A quantum mechanical model of the Riemann zeros](https://arxiv.org/abs/0712.0705) - In 1999 Berry and Keating showed that a regularization of the 1D classical Hamiltonian H = xp gives ...

23. [[PDF] The Riemann Zeros as Spectrum and the Riemann Hypothesis](https://s3.cern.ch/inspire-prod-files-1/1e65b86fec7566dba4d2d2384183f67b) - The Berry–Keating xp model can be implemented quantum mechanically. X The classical xp Hamiltonian m...

24. [Hamiltonian for the Zeros of the Riemann Zeta Function](https://link.aps.org/doi/10.1103/PhysRevLett.118.130201) - A Hamilonian with PT-symmetry invariant properties is investigated as a candidate for an operator th...

25. [[PDF] arXiv:1102.5356v1 [math-ph] 25 Feb 2011](https://arxiv.org/pdf/1102.5356.pdf)

26. [[1608.03679] Hamiltonian for the zeros of the Riemann zeta function](https://arxiv.org/abs/1608.03679) - Hamiltonian for the zeros of the Riemann zeta function. Authors:Carl M. Bender, Dorje C. Brody, Mark...

27. [Self-adjointness of the Bender–Brody–Müller operator](https://www.emergentmind.com/open-problems/self-adjointness-bender-brody-mueller-operator) - Establish whether the Hamiltonian proposed by Carl M. Bender, Dorje C. Brody, and Markus P. Müller (...

28. [[PDF] Hamiltonian for the zeros of the Riemann zeta function - ICMAT](https://www.icmat.es/congresos/2017/darryl70/Slides/Brody.pdf) - Reference: Bender, C.M., Brody, D.C. & Müller, M.P. “Hamiltonian for the zeros of the Riemann zeta f...

29. [Trace formula in noncommutative geometry and the zeros of ... - arXiv](https://arxiv.org/abs/math/9811068) - We give a spectral interpretation of the critical zeros of the Riemann zeta function as an absorptio...

30. [[PDF] Trace Formula in Noncommutative Geometry and - Alain Connes](https://alainconnes.org/wp-content/uploads/selecta.ps-2.pdf) - In other words, the spectral interpretation of the zeros of the Riemann zeta function should be as a...

31. [[PDF] The Riemann-Roch strategy A. Connes (Collaboration with C ...](https://aimath.org/wp-content/uploads/bristol-2018-slides/Connes-talk.pdf) - The adele class space and the explicit formulas. Let K be a global field, the adele class space of K...

32. [[PDF] The Weil Proof and the Geometry of the Adèles Class Space](https://www.its.caltech.edu/~matilde/WeilProofAdelesClassSpace.pdf) - This paper explores analogies between the Weil proof of the Riemann hypothesis for function fields a...

