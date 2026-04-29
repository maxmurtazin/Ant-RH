# Topological Quantum Computing: Braid Groups, Non-Abelian Anyons, and Operator Representations

## A Rigorous Literature Review

***

## 1. Foundations of Topological Quantum Computing

### 1.1 Topological Phases of Matter

A topological phase of matter is a gapped quantum phase whose ground-state properties are invariant under any local, adiabatic perturbation that does not close the gap. Such systems are characterized not by a local order parameter in the Landau–Ginzburg sense, but by global topological invariants — quantities that can only change through a phase transition. Formally, two Hamiltonians \(H_0\) and \(H_1\) belong to the same topological phase if there exists a smooth, gapped interpolation \(H(s)\), \(s \in [0,1]\), connecting them.[^1][^2]

The ground-state degeneracy of a topological system depends on the topology of the underlying manifold. For Kitaev's toric code on a torus \(\mathbb{T}^2\), the ground space is exactly four-fold degenerate, corresponding to two logical qubits, and this degeneracy is stable against arbitrary local perturbations exponentially small in the system size \(L\):[^3]

\[
\Delta E \sim e^{-\alpha L}, \quad \alpha > 0 \quad (1)
\]

This exponential suppression is the core promise of topological quantum memory.[^4][^5]

### 1.2 Role of Non-Abelian Anyons

In two spatial dimensions, particle exchange is governed by the fundamental group of the configuration space:[^1]

\[
\pi_1\!\left(\mathrm{Conf}_N(\mathbb{R}^2)\right) \cong B_N \quad (2)
\]

where \(B_N\) is the braid group on \(N\) strands. For Abelian anyons, monodromy acts as a scalar phase \(e^{i\theta}\). For **non-Abelian anyons**, the degenerate ground-state manifold \(\mathcal{V}\) has dimension \(d > 1\), and braiding acts as a non-commutative unitary transformation:[^6]

\[
\rho(\sigma_i) : \mathcal{V} \to \mathcal{V}, \quad \rho(\sigma_i)\rho(\sigma_j) \neq \rho(\sigma_j)\rho(\sigma_i) \quad \text{in general} \quad (3)
\]

The key insight of Kitaev, Freedman, Larsen, and Wang (2003) is that quantum information encoded in the degenerate ground space and manipulated by braiding is immune to local errors, since any local operator cannot distinguish between ground states — a property called **topological protection**.[^7][^8]

### 1.3 Formal Definition of Braiding in Spacetime

Consider \(N\) non-Abelian anyons with fixed positions in \(\mathbb{R}^2\). Their collective worldlines trace paths in \(\mathbb{R}^2 \times [0,T]\) forming a braid in \(\mathbb{R}^3\). Formally, a braid on \(N\) strands is a mapping:

\[
\beta: \{1,\ldots,N\} \times [0,1] \to \mathbb{R}^2 \times [0,1] \quad (4)
\]

such that the \(i\)-th strand \(\beta_i(t)\) satisfies \(\beta_i(0) = (x_i, 0)\), \(\beta_i(1) = (x_{\pi(i)}, 1)\) for some permutation \(\pi\), and strands never intersect. The operation performed on the quantum state depends **only on the topology** of the resulting braid — it is invariant under continuous deformations that keep endpoints fixed. This topological robustness is the fundamental advantage of TQC.[^9][^1]

### 1.4 Mathematical Structure of the Braid Group \(B_n\)

The braid group \(B_n\) is presented by generators \(\sigma_1, \sigma_2, \ldots, \sigma_{n-1}\) satisfying the **Artin relations**:[^2][^1]

\[
\sigma_i \sigma_{i+1} \sigma_i = \sigma_{i+1} \sigma_i \sigma_{i+1}, \quad 1 \leq i \leq n-2 \quad (5)
\]
\[
\sigma_i \sigma_j = \sigma_j \sigma_i, \quad |i-j| \geq 2 \quad (6)
\]

Here \(\sigma_i\) represents the elementary crossing of the \(i\)-th strand over the \((i+1)\)-th strand. Equation (5) is the **braid relation** (Yang–Baxter), while equation (6) is the **commutativity of distant crossings**. There is a surjection \(B_n \twoheadrightarrow S_n\) (onto the symmetric group) obtained by forgetting over/under crossing information; the kernel is the **pure braid group** \(P_n\). Unlike \(S_n\), the braid group is infinite and torsion-free, and does not satisfy the relation \(\sigma_i^2 = 1\).[^1]

***

## 2. Braid Group Representations

### 2.1 Generators and Artin Relations

A **representation** of \(B_n\) on a Hilbert space \(\mathcal{H}\) is a group homomorphism:[^10]

\[
\rho: B_n \to U(\mathcal{H}) \quad (7)
\]

where \(U(\mathcal{H})\) denotes the unitary group. For TQC, each generator \(\sigma_i\) maps to a physically realizable unitary gate. The matrices \(R_i = \rho(\sigma_i)\) must satisfy the matrix form of the Artin relations:

\[
R_i R_{i+1} R_i = R_{i+1} R_i R_{i+1}, \quad R_i R_j = R_j R_i \; (|i-j|\geq 2) \quad (8)
\]

A representation is **irreducible** if the only invariant subspaces are \(\{0\}\) and \(\mathcal{H}\), and is **unitary** if each \(R_i\) is unitary. For quantum computation, unitarity is essential to preserve probabilities.[^11][^10]

### 2.2 The Burau Representation

The **Burau representation** \(\psi: B_n \to GL_n(\mathbb{Z}[t, t^{-1}])\) is defined via the action of \(B_n\) on the homology of a certain covering space of the punctured disk. The unreduced Burau matrices for \(\sigma_i\) are \(n \times n\) with entries:[^12]

\[
\psi(\sigma_i)_{jk} = \begin{cases}
1-t & j=k=i \\
t & j=i, k=i+1 \\
1 & j=k\neq i \\
0 & \text{otherwise}
\end{cases}
\quad (9)
\]

The reduced Burau representation is an \((n-1)\)-dimensional quotient. Its connection to knot theory is fundamental: the Alexander polynomial \(\Delta_K(t)\) of a knot arising as the closure of braid \(\beta \in B_n\) can be expressed in terms of \(\det(I - \psi(\beta))\).[^13][^12]

A central question is **faithfulness**: does \(\psi\) injectively represent \(B_n\)? It is faithful for \(n \leq 3\). Bigelow (1999) proved it is **unfaithful for \(n \geq 5\)**. The case \(n=4\) remains open as of 2024, despite computational attacks. Recent work by Bapat and Queffélec (2024) extended the faithfulness question to Artin–Tits groups, proving unfaithfulness in affine type \(\widetilde{A}_3\) using categorical methods generalizing Bigelow's curve strategy.[^14][^13]

### 2.3 The Jones Representation

Jones (1983) discovered a sequence of representations in the study of von Neumann algebras, now central to TQC. The **Jones representation** \(\rho_r: B_n \to U(\mathcal{H})\) is defined via the **Temperley–Lieb algebra** \(TL_n(A)\) at \(A = e^{\pm 2\pi i/r}\). The key theorem is:[^10][^11]

> **Theorem (Jones–Freedman–Larsen–Wang):** For \(q = e^{\pm 2\pi i/r}\) with \(r \neq 1,2,3,4,6\), the Jones representation of \(B_n\) has a **dense image** in the appropriate unitary group.

Density of the image means that any target unitary can be approximated arbitrarily well by a braid word — a necessary condition for universality. The Jones polynomial \(V_K(t)\) of a knot \(K\) at a root of unity can be computed as a specific matrix trace of \(\rho(\beta)\), where \(\beta\) is any braid whose closure gives \(K\). Evaluating the Jones polynomial is \(\#P\)-hard classically, motivating quantum algorithms using TQC.[^15][^16]

### 2.4 Chern–Simons / TQFT Constructions

Witten (1988–89) showed that \((2+1)\)-dimensional Chern–Simons gauge theory with compact gauge group \(G\) at level \(k\) provides a physical realization of TQFT. The Chern–Simons action is:[^17]

\[
S_{CS}[A] = \frac{k}{4\pi} \int_M \mathrm{tr}\!\left(A \wedge dA + \frac{2}{3} A \wedge A \wedge A\right) \quad (10)
\]

The partition function of this theory on a 3-manifold \(M\) is a topological invariant. For \(G = SU(2)\) at level \(k\), the Hilbert space of states on a Riemann surface \(\Sigma\) is identified with the space of **conformal blocks** of the \(SU(2)_k\) WZW model on \(\Sigma\). The braiding matrices acting on conformal blocks realize unitary representations of \(B_n\), and for \(k=2\) these correspond to Ising anyons, while for \(k=3\) they give Fibonacci anyons. As a symmetric monoidal functor:[^18][^19][^17][^1]

\[
Z: n\text{Cob} \to \text{Hilb} \quad (11)
\]

assigns Hilbert spaces to manifolds and unitary operators to cobordisms, formalizing the abstract structure of TQC.

### 2.5 Braids as Quantum Gates

The computational model assigns qubits to fusion spaces of anyon pairs and unitary gates to braid operations. For \(2N\) Fibonacci anyons, the Hilbert space \(\mathcal{F}_{2N}\) has dimension growing as the Fibonacci sequence \(F_{2N-1}\), with golden-ratio scaling \(\sim \phi^{2N}\) where \(\phi = (1+\sqrt{5})/2\). The elementary braiding matrices for adjacent Fibonacci anyons are:[^19][^20][^21][^7][^1]

\[
B_1 = \begin{pmatrix} e^{-4\pi i/5} & 0 \\ 0 & e^{3\pi i/5} \end{pmatrix}, \quad B_2 = \frac{1}{\phi}\begin{pmatrix} e^{3\pi i/5}/\phi & e^{-4\pi i/5} \\ e^{-4\pi i/5} & -1/\phi \end{pmatrix} \quad (12)
\]

The **Solovay–Kitaev theorem** guarantees that any target \(U \in SU(2)\) can be approximated to precision \(\varepsilon\) by a braid word of length \(O(\log^{3.97}(1/\varepsilon))\):[^22][^23]

\[
\|U - \rho(\sigma_{i_1}^{\pm 1}\cdots \sigma_{i_L}^{\pm 1})\| < \varepsilon, \quad L = O\!\left(\log^{3.97}\!\tfrac{1}{\varepsilon}\right) \quad (13)
\]

***

## 3. Anyons and Physical Realizations

### 3.1 Fractional Quantum Hall Effect

The fractional quantum Hall (FQH) effect, discovered at filling fractions \(\nu = p/q\), supports quasiparticles with fractional charge \(e^* = e/q\) and anyonic statistics. The most important candidate for non-Abelian anyons is the **Moore–Read Pfaffian state** at \(\nu = 5/2\):[^24][^25][^26][^1]

\[
\Psi_{5/2}(z_1,\ldots,z_N) = \mathrm{Pf}\!\left(\frac{1}{z_i - z_j}\right) \prod_{i<j}(z_i - z_j)^2 e^{-\sum|z_i|^2/4} \quad (14)
\]

The quasihole excitations of this state have charge \(e/4\) and obey **Ising non-Abelian statistics**: \(2N\) such quasiparticles support a Hilbert space of dimension \(2^{N-1}\). The fusion rules are:[^25][^26]

\[
\sigma \times \sigma = \mathbf{1} + \psi, \quad \sigma \times \psi = \sigma, \quad \psi \times \psi = \mathbf{1} \quad (15)
\]

where \(\sigma\) denotes the Ising anyon and \(\psi\) is a Majorana fermion. Recent work on parton states (2026) numerically computed braiding matrices for a broad family of non-Abelian FQH states and provided a general framework for diagnosing non-Abelian characteristics in candidate states.[^27][^24][^1]

### 3.2 Majorana Zero Modes

Kitaev (2001) introduced a 1D \(p\)-wave superconductor (the **Kitaev chain**) as a model supporting Majorana zero modes (MZMs) at its ends. The Hamiltonian is:

\[
H = -\mu\sum_j \hat{n}_j - \sum_j\!\left(t c_j^\dagger c_{j+1} + \Delta c_j c_{j+1} + \mathrm{h.c.}\right) \quad (16)
\]

In the topological phase \((|\mu| < 2t)\), the two end modes \(\gamma_1, \gamma_2\) satisfy the Majorana condition \(\gamma_j = \gamma_j^\dagger\) and commutation relation \(\{\gamma_i, \gamma_j\} = 2\delta_{ij}\)[^28]. A topological qubit is encoded in the fermion parity \(i\gamma_1\gamma_2\). Braiding of MZMs in a T-junction or 2D network implements unitary transformations in the degenerate ground space[^29][^1].

Practical realizations use semiconductor nanowires (e.g., InAs) with strong spin-orbit coupling, proximity-coupled to an \(s\)-wave superconductor in a magnetic field. The effective Hamiltonian near the nanowire end acquires a \(p\)-wave pairing term, realizing the Kitaev model.[^30][^31]

### 3.3 Experimental Status and Limitations

Microsoft unveiled the **Majorana 1** chip in February 2025: the world's first quantum processing unit powered by a Topological Core architecture, featuring an indium arsenide–aluminum heterostructure designed to host and measure MZMs. The Nature paper accompanying the announcement describes single-shot parity readout with ~1% error probability and quasiparticle poisoning rates of approximately once per millisecond. However, independent analysis by Legg (2025) challenged the **topological gap protocol (TGP)** underpinning Microsoft's verification methodology, arguing it cannot definitively distinguish true topological phases from trivial Andreev bound states.[^32][^33][^34][^35]

In parallel, Google AI demonstrated **non-Abelian braiding of graph vertices** on a superconducting processor (Nature, 2023), and a subsequent experiment reported realization of Fibonacci anyon topological order with a superconducting processor, demonstrating universal braiding statistics and measuring topological entanglement entropy. Multiple Majorana zero modes in a single vortex of superconducting SnTe were identified by a HKUST-led collaboration (2024), using crystal symmetry to control MZM coupling.[^36][^37][^6]

### 3.4 Scalability Challenges

Scalability of TQC faces several formidable obstacles:
- **Ising/Majorana anyons are not computationally universal**: braiding alone generates only the Clifford group, which is efficiently simulable classically (Gottesman–Knill theorem). Supplementary operations — magic state injection or measurement-only protocols — are required.[^38][^19]
- **Finite temperature**: topological protection at temperature \(T > 0\) is lost; thermal fluctuations can nucleate anyon pairs, and passive protection requires \(T \ll \Delta\) where \(\Delta\) is the excitation gap.[^39][^40]
- **Non-abelian braiding in 1D**: true 2D platforms are needed for full non-Abelian exchange; nanowire networks only simulate braiding via T-junctions, introducing overhead.[^41]
- **Fidelity of anyon identification**: as noted above, zero-bias conductance peaks — the primary experimental signature — can arise from trivial Andreev states.[^34][^30]

***

## 4. Topological Protection and Fault Tolerance

### 4.1 Why Topology Protects Quantum Information

The ground-state degeneracy in a topologically ordered system is **locally indistinguishable**: no local operator \(\mathcal{O}\) supported on a region of diameter \(< L\) can distinguish between ground states[^4][^7]. Formally, for ground states \(|\psi_\alpha\rangle\) and \(|\psi_\beta\rangle\):

\[
\langle \psi_\alpha | \mathcal{O} | \psi_\beta \rangle = c \cdot \delta_{\alpha\beta} + O(e^{-L/\xi}) \quad (17)
\]

where \(\xi\) is a correlation length. Any error operator acting on fewer than \(\mathcal{O}(L)\) sites is effectively invisible to the encoded information. This is stronger than conventional quantum error-correcting codes, where protection is active.[^4][^3]

### 4.2 Error Models vs. Topological Invariance

A recent 2026 paper (Lyons and Brown, arXiv:2602.11258) rigorously proved that fault-tolerant universal quantum computation is achievable by braiding anyons with circuit-level noise below a threshold. The error model includes both measurement errors and gate errors. The error-correction scheme operates by braiding anyons in spacetime, treating error chains as a matching problem on a syndrome graph. The fault-tolerance threshold is finite and comparable to surface codes.[^42]

For Ising anyons specifically, Dauphinais and Poulin (2016) proved existence of a fault-tolerant threshold using a local cellular automaton correction scheme valid for both Abelian and non-Abelian anyons, including measurement errors.[^40][^39]

### 4.3 Limitations

**Non-universality** remains the central limitation. For Ising anyons, the braid group image lies in the **Clifford group** — a set of measure zero in \(U(2^n)\). The state of the art as of 2025 is the construction by Iulianelli, Kim, Sussan, and Lauda: extending the Ising framework to a **non-semisimple TQFT** by introducing a new anyon type ("neglecton") achieves universality through braiding alone. Universality via Fibonacci anyons is theoretically established but Fibonacci anyons have not been unambiguously realized in experiment.[^20][^43][^21][^44][^45][^19]

**Measurement issues** also constrain TQC: fusion-based measurement of anyon charge requires bringing anyons together, which risks accidental braiding and introduces correlated errors.[^1]

***

## 5. Connection to Spectral Theory and the Hilbert–Pólya Program

### 5.1 The Hilbert–Pólya Conjecture

The Hilbert–Pólya conjecture posits that the non-trivial zeros \(\frac{1}{2} + i\gamma_n\) of the Riemann zeta function are eigenvalues of a self-adjoint operator \(H\) on some Hilbert space:[^46]

\[
H |\psi_n\rangle = \gamma_n |\psi_n\rangle, \quad \gamma_n \in \mathbb{R} \quad (18)
\]

Self-adjointness forces real eigenvalues, which would place all zeros on the critical line — proving the Riemann hypothesis as a corollary. Montgomery (1973) showed that the pair correlation of zeta zeros matches the **GUE (Gaussian Unitary Ensemble)** statistics of random matrix eigenvalues. Since GUE arises naturally from quantum systems with time-reversal symmetry broken — characteristic of quantum chaotic systems — this provided strong circumstantial evidence for a quantum mechanical origin of the zeros.[^47][^46]

### 5.2 Braid Group Operators and Self-Adjoint Constructions

A natural proposal in the TQC–spectral theory interface is to construct operators of the form:

\[
H = \sum_{i=1}^{n-1} w_i\, \rho(\sigma_i) \quad (19)
\]

where \(\rho: B_n \to U(\mathcal{H})\) is a unitary representation and \(w_i \in \mathbb{C}\) are weights. For \(H\) to be **self-adjoint**, we need \(H = H^\dagger\), which requires:

\[
\sum_i w_i \rho(\sigma_i) = \sum_i \bar{w}_i \rho(\sigma_i)^\dagger = \sum_i \bar{w}_i \rho(\sigma_i^{-1}) \quad (20)
\]

This is satisfiable if, for example, \(w_i = \bar{w}_i\) (real weights) and the representation satisfies \(\rho(\sigma_i)^\dagger = \rho(\sigma_i^{-1})\) — which holds for unitary representations where \(\rho(\sigma_i)\) is itself a unitary operator. However, such an operator has spectrum lying on the **unit circle** in \(\mathbb{C}\) (since unitary matrices have unimodular eigenvalues), not on the real line, unless one instead considers the Hermitian part:[^48]

\[
H_{\text{sym}} = \frac{1}{2i}\sum_i w_i\!\left(\rho(\sigma_i) - \rho(\sigma_i)^\dagger\right) \quad (21)
\]

This produces a self-adjoint operator by construction, with real spectrum. The spectral content of \(H_{\text{sym}}\) under Jones representations has not been systematically studied in relation to zeta zeros; this constitutes an open direction.

An alternative approach, inspired by the Berry–Keating model, uses \(H = xp + px\) reinterpreted in the braid context, assigning a differential operator structure to braid generators via the Magnus expansion of \(B_n\) into formal power series in the free group algebra.[^47]

### 5.3 Quantum Chaos and GUE Statistics

The statistical universality connecting zeta zeros to GUE is the same universality governing eigenvalues of generic quantum chaotic Hamiltonians. In TQC language, the relevant question is: do the eigenvalues of the operator \(\sum_i w_i \rho(\sigma_i)\) — for a random (Haar-distributed) unitary representation — exhibit GUE level statistics? This is a well-posed spectral problem. Numerical experiments on random braid-derived operators could test this, and would constitute a new bridge between TQC and the Hilbert–Pólya program.[^46][^47]

A chaotic operator approach (arXiv:2404.00583, 2025) attempts to construct a Hermitian operator whose eigenvalues exhibit GUE statistics and whose functional form parallels the Riemann zeta function's explicit formula. The construction yields an unbounded self-adjoint operator whose spectral resolution aligns with expected zeta zeros numerically in finite truncations.[^49]

***

## 6. Connection to Dynamical Systems and Geometry

### 6.1 Artin Billiard and Geodesic Flow

The **Artin billiard** describes geodesic motion of a free particle on the non-compact Riemann surface \(\mathcal{H}/\mathrm{PSL}(2,\mathbb{Z})\), where \(\mathcal{H}\) is the upper half-plane with Poincaré metric:[^50]

\[
ds^2 = \frac{dx^2 + dy^2}{y^2}, \quad H = \frac{p_x^2 + p_y^2}{2m}\cdot y^2 \quad (22)
\]

This is a paradigmatic example of quantum chaos: the classical flow is strongly mixing (Anosov), and the quantum eigenvalue spectrum exhibits GUE statistics. The symbolic dynamics of the geodesic flow on the modular surface are encoded by continued fraction expansions, providing a natural connection to braid words via the Artin representation of the mapping class group.[^51][^50]

Periodic geodesic orbits on the modular surface correspond to **hyperbolic conjugacy classes** of \(\mathrm{PSL}(2,\mathbb{Z})\), and their lengths are \(\log \varepsilon_\gamma\) where \(\varepsilon_\gamma\) is the associated unit in a real quadratic field. These orbits can be encoded as braid words in the pure braid group \(P_3\) via the correspondence between the mapping class group of the once-punctured torus and \(\mathrm{PSL}(2,\mathbb{Z})\).[^50][^51]

### 6.2 Periodic Orbits and Braid Words

The orbit–braid correspondence is made precise as follows. A closed geodesic \(\gamma\) on a hyperbolic surface \(S\) with punctures defines a free homotopy class in \(\pi_1(S)\), which maps under the Dehn–Nielsen–Baer theorem to a mapping class in \(\mathrm{MCG}(S)\). For the modular surface, \(\mathrm{MCG}(\mathbb{T}^2) \cong \mathrm{SL}(2,\mathbb{Z})\), and elements of \(\mathrm{SL}(2,\mathbb{Z})\) are expressible in terms of the generators \(T = \bigl(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\bigr)\) and \(S = \bigl(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\bigr)\), which correspond to Dehn twists — the surface analogue of braid generators.[^52]

### 6.3 Selberg Trace Formula as Bridge

The **Selberg trace formula** for a compact hyperbolic surface of genus \(g \geq 2\):[^53][^54]

\[
\sum_n h(r_n) = \frac{\mathrm{Area}(S)}{4\pi}\int_{-\infty}^\infty r\, h(r)\tanh(\pi r)\, dr + \sum_{\{\gamma\}_{\text{prim}}} \sum_{m=1}^\infty \frac{\ell(\gamma)}{2\sinh(m\ell(\gamma)/2)} \hat{h}(m\ell(\gamma)) \quad (23)
\]

relates the **spectral side** (eigenvalues \(\lambda_n = \frac{1}{4} + r_n^2\) of the Laplacian) to the **geometric side** (lengths \(\ell(\gamma)\) of primitive closed geodesics). This is structurally analogous to the explicit formula in number theory, which relates zeros of the Riemann zeta function to prime numbers. The Selberg zeta function:[^54][^55][^53][^46]

\[
Z(s) = \prod_{\{\gamma\}_{\text{prim}}} \prod_{k=0}^\infty \left(1 - e^{-(s+k)\ell(\gamma)}\right) \quad (24)
\]

has zeros at \(s = \frac{1}{2} \pm ir_n\), precisely encoding the Laplacian spectrum. This structure suggests a program: map braid words (via the orbit–braid correspondence) to operators whose spectral zeta function mirrors \(Z(s)\), potentially connecting to the Riemann zeta function's zero distribution.

***

## 7. Algorithmic and Computational Perspectives

### 7.1 Braid-Based Quantum Computation Models

Two primary models exist for braid-based TQC. In the **anyon braiding model**, a universal gate set is realized by composing elementary braid operations; universality is achieved for Fibonacci anyons (whose braid group image is dense in \(SU(2)\)) but not for Ising anyons (Clifford group only). The **measurement-only model** of Bonderson, Freedman, and Nayak (2008) achieves topological computation via sequences of topological charge measurements without requiring any adiabatic anyon transport, thereby removing a major engineering burden.[^56][^19][^7][^1]

Quantum compiling — finding a braid word \(\beta \in B_n\) such that \(\|\rho(\beta) - U_{\text{target}}\| < \varepsilon\) — is the core algorithmic challenge. The standard approach uses the Solovay–Kitaev algorithm[^22], which runs in time \(O(\log^{2.71}(1/\varepsilon))\) and produces words of length \(O(\log^{3.97}(1/\varepsilon))\). For Fibonacci anyons, recent work (2025) using a **genetic algorithm (GA)-enhanced Solovay–Kitaev** achieved gate distances of \(5.9 \times 10^{-7}\) with initial braid lengths \(l_0 = 50\), surpassing Monte Carlo approaches and matching deep RL methods for braid lengths exceeding 25[^57].

### 7.2 Classical Simulation of TQC

Classical simulation of TQC is feasible for small systems but generically hard. For Fibonacci anyons and Clifford-group Ising anyons:
- Clifford circuits (Ising braids) are efficiently simulable in \(O(n^2)\) time via the stabilizer formalism.[^58]
- Fibonacci anyon braiding is **BQP-complete** to simulate, since the braid group image is dense in \(SU(2)\).[^19][^56]
- State vector simulation scales as \(O(2^n)\) in memory; tensor network methods (MPS, PEPS) are viable for systems with limited entanglement.[^59][^58]

For systems of 50+ qubits, classical simulation becomes intractable, providing the quantum advantage frontier.[^59]

### 7.3 Variational and Learning-Based Approaches

Variational Quantum Eigensolver (VQE) methods have been adapted to identify and simulate topological phases. Okada et al. (2022/2023) introduced **classically-optimized VQE (CO-VQE)**, where optimization is conducted classically on circuits of constant or logarithmic depth, and quantum computers are used only for nonlocal measurements. This approach successfully identified topological phases in spin models using nonlocal order parameters and unsupervised machine learning on quantum state inner products.[^60][^61][^62]

Reinforcement learning (RL) has been applied to quantum circuit optimization. Fösel et al. (2021) demonstrated RL-based circuit optimization achieving 27% depth reduction and 15% gate count reduction on 12-qubit random circuits. More recently, Riu et al. (2023/2025) proposed RL with ZX-calculus graph simplification, training on 5-qubit Clifford+T circuits and generalizing to 80-qubit, 2100-gate circuits. These methods can in principle be applied to braid-word optimization in TQC.[^63][^64]

***

## 8. Emerging Directions

### 8.1 Reinforcement Learning over Braid Words

The problem of braid-word quantum compiling is naturally framed as a **Markov decision process (MDP)**:[^64][^63]
- **State**: current braid word \(\beta = \sigma_{i_1}^{\pm 1}\cdots\sigma_{i_L}^{\pm 1}\)
- **Action**: append or replace a generator \(\sigma_i^{\pm 1}\)
- **Reward**: \(r = -\|\rho(\beta) - U_{\text{target}}\|\) (negative gate distance)
- **Goal**: find \(\beta\) maximizing total reward

Deep RL agents (e.g., using PPO or actor-critic with graph neural network policy) explore the exponentially large braid word space. Long et al. (2025) demonstrated GA-enhanced SKA achieving precision \(5.9 \times 10^{-7}\) for Fibonacci anyons, and noted that deep RL approaches produce comparable or superior results for long braid words. An open research direction is to use RL agents that natively exploit the algebraic structure of \(B_n\) — incorporating Markov moves, braid isotopies (equations (5)–(6)), and Markov stabilization — as environment symmetries for data-efficient learning.[^57]

### 8.2 Ant Colony Optimization for Braid Discovery

Ant colony optimization (ACO) is a swarm-intelligence heuristic where artificial ants deposit pheromone on high-quality paths. In the braid-word context, each "path" corresponds to a sequence of generators; pheromone reinforces generator subsequences that decrease gate distance. A quantum ACO variant using qubit-encoded ant states could, in principle, exploit superposition to explore exponentially many braid words in parallel. The key formulation is:[^65][^66][^67][^68]

- **Pheromone matrix**: \(\tau_{ij}^{(t)}\) = pheromone level for appending generator \(\sigma_j^{\pm 1}\) after position \(i\)
- **Transition probability**: \(p_{ij} \propto \tau_{ij}^\alpha \cdot \eta_{ij}^\beta\) where \(\eta_{ij}\) is a heuristic (e.g., inverse gate distance gradient)
- **Update rule**: \(\tau_{ij}^{(t+1)} = (1-\rho)\tau_{ij}^{(t)} + \Delta\tau_{ij}\) where \(\Delta\tau_{ij} \propto r_\beta\) for braid \(\beta\) using edge \((i,j)\)

This approach remains largely unexplored for TQC specifically; existing quantum ACO work focuses on combinatorial optimization problems such as TSP.[^66][^65]

### 8.3 Trajectories in State Space → Braid Words → Operators

A novel structural idea is to **map continuous dynamical trajectories to braid words**. Given a time-parameterized path \(\gamma: [0,T] \to \mathcal{M}\) in a state space manifold \(\mathcal{M}\), one can:

1. **Discretize** the path into segments \(\gamma_1, \ldots, \gamma_k\) corresponding to topological sectors.
2. **Encode** the homotopy class of adjacent trajectories as a generator \(\sigma_i^{\pm 1}\) when trajectory \(\gamma_i\) winds around configuration \(i\).
3. **Assemble** the braid word \(\beta = \sigma_{i_1}^{\pm 1}\cdots\sigma_{i_k}^{\pm 1} \in B_n\).
4. **Compute** the operator \(\rho(\beta)\) using the desired representation.

This pipeline connects geometric flows — including geodesic flows on modular surfaces, Hamiltonian flows in phase space, or neural network loss-landscape trajectories — to braid representations and thence to unitary quantum gates. The spectral properties of the resulting operators then encode information about the underlying dynamical system's topology.[^51][^50]

### 8.4 Software Topological Quantum Computing

An emerging paradigm, which may be called **Software TQC (S-TQC)**, proposes to implement the logical structure of topological protection without physical anyons. The idea is to:

1. Classically **simulate** the fusion and braiding rules of an anyon model (e.g., Fibonacci anyons) on a conventional computer.
2. Encode logical qubits in the fusion space of simulated anyons, representing quantum states as vectors in \(\mathcal{F}_{2N}\).
3. Apply quantum gates by multiplying by braid matrices \(B_i\) (equation (12)).
4. Perform "error correction" by projecting back onto the code space using fusion rules.

This approach is computationally limited (scaling exponentially in qubit number for non-Clifford braids), but serves as a testbed for algorithms and error models. It is related to, but distinct from, classical simulation of TQC circuits: S-TQC uses the topological language as a native programming model rather than merely simulating a quantum circuit. This connects to the quantum double models of Kitaev and to tensor-network descriptions of topological order.[^58][^59]

### 8.5 Connections to Piecewise-Linear and Tropical Geometry

Tropical geometry replaces the standard \((+, \times)\) arithmetic with the tropical semiring \((\min, +)\), yielding piecewise-linear (PL) counterparts of algebraic curves and varieties:[^69][^70]

\[
f \oplus g = \min(f, g), \quad f \otimes g = f + g \quad (25)
\]

Tropical polynomials are piecewise-linear functions with integer slopes. The connection to TQC arises via the **tropicalization** of braid group algebras: replacing the usual polynomial ring in braid group generators with the tropical semiring produces PL analogues of knot invariants.[^70]

Concretely, the Alexander polynomial — which arises from the Burau representation — can be tropicalized to a piecewise-linear function encoding the lengths of geodesics in the Cayley graph of \(B_n\). Tropical curves arising from amoebas of the Alexander polynomial's Newton polygon can serve as models for phase space boundaries in S-TQC. The max-plus algebra underlying tropical geometry is also structurally isomorphic to the Maslov dequantization limit of quantum mechanics (\(\hbar \to 0\) in the heat equation), suggesting a semiclassical TQC framework.[^71]

Furthermore, Maragos and Theodosis (2019) connected tropical geometry and max-plus algebra to piecewise-linear regression and machine learning optimization, opening an avenue for **tropical learning-based braid compilers**: braid words are encoded as tropical polynomials, and optimization over braid space becomes a max-plus convex regression problem.[^72][^71]

***

## 9. Synthesis: Open Problems and Opportunities

### 9.1 Key Open Problems

| Domain | Open Problem | Status |
|---|---|---|
| Mathematics | Faithfulness of Burau representation for \(n=4\) | Open since 1930s[^14][^13] |
| Physics | Unambiguous detection of non-Abelian anyons in solid-state systems | Contentious (2025)[^34][^35] |
| TQC | Universal fault-tolerant TQC with realizable systems | Theoretical roadmap; \(\geq 10^6\) qubits needed[^32] |
| Number theory | Whether braid-derived operators can realize Hilbert–Pólya | Completely open[^46][^73] |
| Algorithms | Optimal classical algorithm for braid-word compiling | Best known: \(O(\log^{3.97}(1/\varepsilon))\)[^22] |
| Simulation | Tensor-network simulation of Fibonacci TQC at \(n>20\) | Computationally intractable[^59] |
| Geometry | Explicit isomorphism between braid words and geodesic words on modular surface | Partially known via Garside structure[^52] |

### 9.2 Where Theory Leads Experiment

Theory is substantially ahead of experiment in several directions:
- **Fibonacci anyons**: full universality proven, but no physical realization has demonstrated fusion rules to sufficient fidelity.[^20][^19]
- **Non-semisimple TQFTs**: Iulianelli et al. (2025) showed universality of Ising anyons extended by one new type, but the corresponding physical system is unknown.[^43][^44][^45]
- **Fault-tolerant threshold proofs**: rigorous proofs of finite threshold for non-Abelian anyon error correction precede any experimental implementation by at least a decade.[^39][^42]
- **Topological quantum memories at finite \(T\)**: theory predicts thermal instability, but active error correction protocols are well-developed theoretically.[^42][^39]

### 9.3 Opportunities for Computational Approaches

The intersection of TQC with machine learning, combinatorial optimization, and dynamical systems offers several concrete research opportunities:

- **RL-braid compilers** exploiting the Artin relations as environment symmetries could dramatically reduce sample complexity.[^63][^64][^57]
- **Tropical geometry models** of braid spaces could provide new PL invariants of quantum circuits, faster to compute than Jones polynomial evaluations.[^72][^69]
- **Spectral operator construction** \(H = \sum_i w_i \rho(\sigma_i)\) may yield new integrable Hamiltonians or random matrix models testable against GUE statistics.[^49][^46][^47]
- **Classical S-TQC** on GPU clusters can serve as a validation platform for error-correction protocols and anyon braiding algorithms, operating in topological language without physical hardware.[^58][^59]
- **ACO-based braid search** with pheromone evaporation tuned to the algebraic distance in \(B_n\) (Garside normal form length) is an unexplored combination of swarm intelligence and algebraic combinatorics.[^68][^65][^66]

The trajectory \(\mathcal{M} \to B_n \to U(\mathcal{H})\) proposed in Section 8.3 provides a general-purpose pipeline applicable to: quantum algorithm discovery, machine learning of quantum phases, and potentially to spectral models for number-theoretic zeta functions — making TQC an unexpectedly fertile meeting ground for topology, spectral theory, tropical geometry, and modern optimization.

---

## References

1. [Non-Abelian anyons and topological quantum computation](https://link.aps.org/doi/10.1103/RevModPhys.80.1083) - In this review article, current research in this field is described, focusing on the general theoret...

2. [Non-Abelian Anyons and Topological Quantum Computation - arXiv](https://arxiv.org/abs/0707.1889) - In this review article, we describe current research in this field, focusing on the general theoreti...

3. [Fault-tolerant quantum computation by anyons](https://scienceplusplus.org/visions/assets/Kitaev2003.pdf)

4. [Fault-tolerant quantum computation by anyons - ScienceDirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0003491602000180) - An arbitrary quantum circuit can be simulated using imperfect gates, provided these gates are close ...

5. [ToricCode](https://mptoolkit.qusim.net/HOWTO/ToricCode&rut=1d5135c84037785a80abb9e1cf599bdd426c07bfb5dfac69636e035483965641)

6. [Non-Abelian braiding of graph vertices in a ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10247379/) - Indistinguishability of particles is a fundamental principle of quantum mechanics1. For all elementa...

7. [[quant-ph/0101025] Topological Quantum Computation - arXiv](https://arxiv.org/abs/quant-ph/0101025) - Title:Topological Quantum Computation. Authors:Michael H. Freedman, Alexei Kitaev, Michael J. Larsen...

8. [[PDF] topological quantum computation - Mathematics Department](https://web.math.ucsb.edu/~zhenghwa/data/research/pub/TQC-03.pdf) - Abstract. The theory of quantum computation can be constructed from the abstract study of anyonic sy...

9. [Introduction to topological quantum computation with non-Abelian anyons](http://arxiv.org/pdf/1802.06176.pdf)

10. [arXiv:1604.06429v1  [math.QA]  21 Apr 2016](http://arxiv.org/pdf/1604.06429.pdf)

11. [[PDF] arXiv:1604.06429v1 [math.QA] 21 Apr 2016](https://arxiv.org/pdf/1604.06429.pdf)

12. [Burau representation - Wikipedia](https://en.wikipedia.org/wiki/Burau_representation)

13. [[PDF] faithful specializations of the burau representation - MIT](https://www.mit.edu/~anser/files/burau.pdf)

14. [Some remarks about the faithfulness of the Burau representation of Artin–Tits groups](https://arxiv.org/html/2409.00144v1)

15. [arXiv:quant-ph/0210095v1  12 Oct 2002](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=db0afbcf170e3a0d92413b2c04e6fa186221d907)

16. [The Jones polynomial: quantum algorithms and ...](https://kam.mff.cuni.cz/~loebl/clanky/caltech.pdf)

17. [Chern-Simons theory in nLab](https://ncatlab.org/nlab/show/Chern-Simons+theory) - The beautiful thing about Chern–Simons theory is that Witten was able use the locality property of t...

18. [[PDF] Topological Quantum Field via Chern-Simons Theory, part1 - arXiv](https://arxiv.org/pdf/1112.0373.pdf) - To understand what does Chern-Simons with compact Lie group(does not like. Dijkgraaf-Witten model wi...

19. [Quantum Computing Modalities: Fibonacci Anyons](https://postquantum.com/quantum-modalities/fibonacci-anyons/) - Universal Quantum Gates via Braiding: Any quantum gate can be implemented by braiding Fibonacci anyo...

20. [[PDF] The construction of a universal quantum gate set for the SU(2)k (k=5 ...](https://arxiv.org/pdf/2505.01774.pdf) - The k=3 case represents the Fibonacci anyon model – the simplest known non-Abelian system enabling u...

21. [[PDF] universality of fibonacci anyons in topological quantum computing](https://ncatlab.org/nlab/files/Simeon-UniversalityOfFibonacciAnyons.pdf) - More specif- ically, we will talk about approximating quantum circuits with braids, showing that if ...

22. [[quant-ph/0505030] The Solovay-Kitaev algorithm - arXiv](https://arxiv.org/abs/quant-ph/0505030) - This pedagogical review presents the proof of the Solovay-Kitaev theorem in the form of an efficient...

23. [SolovayKitaev (latest version) | IBM Quantum Documentation](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.SolovayKitaev) - The Solovay-Kitaev theorem [1] states that any single qubit gate can be approximated to arbitrary pr...

24. [Non-Abelian fusion and braiding in many-body parton states - arXiv](https://arxiv.org/html/2601.16819v1) - Fractional quantum Hall (FQH) states host fractionally charged anyons with exotic exchange statistic...

25. [[PDF] The Moore-Read Quantum Hall State: An Overview](https://www.lancaster.ac.uk/users/esqn/windsor10/lectures/cooper.pdf) - In a non-abelian quantum Hall state, quasi-particles obey non-abelian ... The prediction for the ν=5...

26. [[PDF] Fractional quantum Hall effect at the filling factor ν=5/2 - Virginia Tech](https://scarola.phys.vt.edu/content/dam/scarola_phys_vt_edu/papers/3-s2.0-B9780323908009001359-main.pdf) - If observed, non-Abelian anyons could offer fundamental building blocks of a topological quantum com...

27. [[PDF] Non-Abelian Anyons in the Quantum Hall Effect](https://www.pg.infn.it/QID2011/Talks/Cappelli.pdf) - ... Non-Abelian fractional statistics. ○ described by Moore-Read “Pfaffian state” ~ Ising CFT x boso...

28. [Majorana zero modes in multiplicative topological phases](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.014516?ft=1) - Topological qubits composed of unpaired Majorana zero modes are under intense experimental and theor...

29. [Microsoft's Majorana Topological Chip -- An Advance 17 Years in ...](https://thequantuminsider.com/2025/02/19/microsofts-majorana-topological-chip-an-advance-17-years-in-the-making/) - Microsoft has made progress toward building a more stable quantum computer by successfully measuring...

30. [Current Development Status](https://postquantum.com/quantum-modalities/majorana-qubits/) - Majorana qubits are quantum bits encoded using Majorana zero modes, exotic quasiparticles that are t...

31. [Topological quantum processor uses Majorana zero modes for fault-tolerant computing](https://phys.org/news/2025-02-topological-quantum-processor-majorana-modes.html) - In a leap forward for quantum computing, a Microsoft team led by UC Santa Barbara physicists on Wedn...

32. [Microsoft's Majorana 1 chip carves new path for quantum computing](https://news.microsoft.com/source/features/innovation/microsofts-majorana-1-chip-carves-new-path-for-quantum-computing/) - Microsoft's topological qubit architecture has aluminum nanowires joined together to form an H. Each...

33. [Microsoft unveils Majorana 1, the world's first quantum processor ...](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/) - Built with a breakthrough class of materials called a topoconductor, Majorana 1 marks a transformati...

34. [Microsoft quantum computing 'breakthrough' faces fresh challenge](https://www.nature.com/articles/d41586-025-00683-2) - Analysis pokes holes in protocol that underpins Microsoft's claim to have created the first topologi...

35. [Microsoft claims quantum-computing breakthrough - Nature](https://www.nature.com/articles/d41586-025-00527-z) - The tech giant aims to make 'topological' quantum computers that will reach useful scales faster tha...

36. [Non-Abelian braiding of Fibonacci anyons with a superconducting processor](https://www.emergentmind.com/papers/2404.00091) - Non-Abelian topological orders offer an intriguing path towards fault-tolerant quantum computation, ...

37. [Physics researchers identify new multiple Majorana zero modes in ...](https://www.sciencedaily.com/releases/2024/08/240829132424.htm) - A collaborative research team has identified the world's first multiple Majorana zero modes (MZMs) i...

38. [Quantum Gates from Non-Semisimple Ising Anyons | PDF - Scribd](https://www.scribd.com/document/950862856/Topological-Quantum-Compilation-for-Non-semisimple-Ising-Anyons) - This document presents a numerical construction of a universal quantum gate set for topological quan...

39. [Fault-Tolerant Quantum Error Correction for non-Abelian Anyons](https://arxiv.org/abs/1607.02159) - We present a scheme to protect the information stored in a system supporting non-cyclic anyons again...

40. [Fault-Tolerant Quantum Error Correction for non-Abelian ...](https://arxiv.org/pdf/1607.02159.pdf)

41. [[PDF] Topological Quantum Computation—From Basic Concepts to First ...](https://allen.physics.ucsd.edu/research/Stern_topological_QC_review.pdf) - Kitaev, Ann. Phys. 303, 2 (2003). 4. M. H. Freedman, M. Larsen, Z. Wang, Comm. Math. Phys.

42. [Quantum computing with anyons is fault tolerant - arXiv](https://arxiv.org/html/2602.11258v1) - Here, we show how to carry out error correction throughout a quantum computation by a universal set ...

43. [Universal quantum computation using Ising anyons from a non ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325951/) - The standard Fibonacci anyon theory is a chiral topological phase that cannot be realized by a commu...

44. [Ising Anyons & Neglectons Enable Universal Quantum](https://quantumzeitgeist.com/quantum-universal-computation-ising-anyons-neglectons/) - This document details a comprehensive investigation into topological quantum computation, specifical...

45. [Universal quantum computation using Ising anyons from a ... - Nature](https://www.nature.com/articles/s41467-025-61342-8) - We propose a framework for topological quantum computation using newly discovered non-semisimple ana...

46. [Hilbert–Pólya conjecture - Wikipedia](https://en.wikipedia.org/wiki/Hilbert%E2%80%93P%C3%B3lya_conjecture) - In mathematics, the Hilbert–Pólya conjecture states that the non-trivial zeros of the Riemann zeta f...

47. [[PDF] Quantum chaos, random matrix theory, and the Riemann ζ-function](https://seminaire-poincare.pages.math.cnrs.fr/keating.pdf) - This unexpected result gave new insight into the Hilbert-Pólya suggestion that the zeta zeros might ...

48. [[PDF] 1. Spectral theory of bounded self-adjoint operators](https://loss.math.gatech.edu/14SPRINGTEA/spectraltheory.pdf) - Spectral theory of bounded self-adjoint operators. In the essential ideas I follow the book of Rober...

49. [If our chaotic operator is derived correctly, then the Riemann ... - arXiv](https://arxiv.org/html/2404.00583v2) - This conjecture lies at the heart of analytic number theory and has deep consequences for the distri...

50. [Artin billiard - Wikipedia](https://en.wikipedia.org/wiki/Artin_billiard)

51. [arXiv:chao-dyn/9307001v1  2 Jul 1993](https://arxiv.org/pdf/chao-dyn/9307001.pdf)

52. [[PDF] the language of geodesics for garside groups - Brandeis](https://people.brandeis.edu/~charney/papers/geodesicfinalrev.pdf)

53. [Selberg trace formula - Wikipedia](https://en.wikipedia.org/wiki/Selberg_trace_formula) - In mathematics, the Selberg trace formula, introduced by Selberg (1956), is an expression for the ch...

54. [[PDF] Selberg's Trace Formula: An Introduction - University of Bristol](https://people.maths.bris.ac.uk/~majm/bib/selberg.pdf) - The aim of this short lecture course is to develop Selberg's trace formula for a compact hyperbolic ...

55. [[PDF] The Selberg Trace Formula & Prime Orbit Theorem](https://openresearch-repository.anu.edu.au/bitstreams/f1ba01c2-878e-4b36-b716-2614f8af099a/download) - The major tool we will use is the Selberg trace formula, which states the trace of a certain compact...

56. [[PDF] Universal topological quantum computing](https://www.cs.tufts.edu/comp/150QC/Report3MichaelJ.pdf) - [Freedman et al., 2003] Freedman, M., Kitaev, A., Larsen, M., and Wang, Z. (2003). Topological quant...

57. [[2501.01746] Genetic algorithm enhanced Solovay-Kitaev ...](https://arxiv.org/abs/2501.01746) - Quantum compiling, which aims to approximate target qubit gates by finding optimal sequences (braidw...

58. [Classical Simulation of Quantum Circuits](https://mqt.readthedocs.io/en/stable/handbook/02_simulation.html) - The MQT offers the classical quantum circuit simulator DDSIM that can be used to perform various qua...

59. [The Value of Classical Quantum Simulators - IonQ](https://www.ionq.com/resources/the-value-of-classical-quantum-simulators) - Quantum simulators—software programs that allow you to use a classical computer to run quantum circu...

60. [[2202.02909] Identification of topological phases using classically ...](https://arxiv.org/abs/2202.02909) - Variational quantum eigensolver (VQE) is regarded as a promising candidate of hybrid quantum-classic...

61. [Classically optimized variational quantum eigensolver with ...](https://link.aps.org/doi/10.1103/PhysRevResearch.5.043217) - As a proof of concept for our method, we present numerical simulations on 1D and 2D quantum spin mod...

62. [Efficient variational quantum circuit structure for correlated ...](https://link.aps.org/doi/10.1103/PhysRevB.108.075127) - We propose an efficient circuit structure of variational quantum circuit Ansätze used for the variat...

63. [Quantum circuit optimization with deep reinforcement learning - arXiv](https://arxiv.org/abs/2103.07585) - We present an approach to quantum circuit optimization based on reinforcement learning. We demonstra...

64. [Reinforcement Learning Based Quantum Circuit Optimization via ZX ...](https://arxiv.org/abs/2312.11597) - We propose a novel Reinforcement Learning (RL) method for optimizing quantum circuits using graph-th...

65. [A Novel Quantum Algorithm for Ant Colony Optimization - arXiv](https://arxiv.org/html/2403.00367v1) - In this paper, we introduce a hybrid quantum-classical algorithm by combining the clustering algorit...

66. [[PDF] Improved ant colony optimization for quantum cost reduction](https://beei.org/index.php/EEI/article/download/1657/2022)

67. [[PDF] quantum ant colony optimization algorithm based on bloch ... - NNW](https://www.nnw.cz/doi/2012/NNW.2012.22.019.pdf) - The simulation results show that the proposed algorithm is superior to other quantum-behaved optimiz...

68. [A novel quantum algorithm for ant colony optimisation - IET Journals](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/qtc2.12023) - This study's main contribution is to propose a fully quantum algorithm to solve ACO, enhancing the q...

69. [Tropical geometry - Wikipedia](https://en.wikipedia.org/wiki/Tropical_geometry) - Tropical geometry is a variant of algebraic geometry in which polynomial graphs resemble piecewise l...

70. [[PDF] An introduction to tropical geometry: theory and applications Lecture 1](https://www.fields.utoronto.ca/talk-media/1/30/92/slides.pdf) - A piecewise linear shadow of algebraic geometry. Fatemeh Mohammadi. Tropical Geometry. January 15, 2...

71. [[PDF] Tropical Geometry and Piecewise-Linear Approximation of Curves ...](https://robotics.ntua.gr/wp-content/uploads/sites/2/MaragosTheodosis_TropicalApproximation_Springer.pdf) - In this chapter we sum- marize some of their main ideas and common (geometric and algebraic) structu...

72. [Tropical Geometry and Piecewise-Linear Approximation of Curves ...](https://arxiv.org/abs/1912.03891) - In this chapter we summarize some of their main ideas and common (geometric and algebraic) structure...

73. [[1305.3342] Hilbert-Pólya Conjecture, Zeta-Functions and Bosonic ...](https://arxiv.org/abs/1305.3342) - In this paper we show that the functional integrals associated with a hypothetical class of physical...

