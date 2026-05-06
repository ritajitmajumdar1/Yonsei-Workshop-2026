import numpy as np
from pyscf import gto
from pyscf.cc import ccsd as ccsd_mod


__all__ = ["run_ccsd_from_hg", "print_largest_amplitudes", "transform_integrals_to_mo"]


def _to_chemist_eri(g, g_format="user_operator_order"):
    """
    Convert the user-provided 4-index tensor g to PySCF chemist notation eri[p,q,r,s] = (pq|rs).

    Parameters
    ----------
    g : ndarray, shape (norb, norb, norb, norb)
        Two-electron tensor.
    g_format : str
        - "chemist":
            g[p,q,r,s] already equals (pq|rs).
        - "user_operator_order":
            g is defined by your Hamiltonian

                H2 = 1/2 * sum_{i,j,k,l} g[i,j,k,l] a_i^\dagger a_j^\dagger a_k a_l

            whereas PySCF chemist notation corresponds to

                H2 = 1/2 * sum_{p,q,r,s} (pq|rs) a_p^\dagger a_r^\dagger a_s a_q

            Matching operator order gives
                (pq|rs) = g[p, r, s, q]

            so we use
                eri = g.transpose(0, 3, 1, 2)

    Returns
    -------
    eri : ndarray, shape (norb, norb, norb, norb)
        Two-electron tensor in chemist notation.
    """
    g = np.asarray(g)

    if g_format == "chemist":
        eri = g
    elif g_format == "user_operator_order":
        eri = np.transpose(g, (0, 3, 1, 2))
    else:
        raise ValueError(
            "Unsupported g_format. Use 'chemist' or 'user_operator_order'."
        )

    return np.asarray(eri)


def _build_custom_rhf(h1, eri_chemist, nelec, ecore=0.0, verbose=4):
    """
    Build a custom RHF object from one- and two-electron integrals
    in an orthonormal spatial-orbital basis.
    """
    h1 = np.asarray(h1, dtype=np.float64)
    eri_chemist = np.asarray(eri_chemist, dtype=np.float64)

    norb = h1.shape[0]
    if h1.shape != (norb, norb):
        raise ValueError("h1 must have shape (norb, norb).")
    if eri_chemist.shape != (norb, norb, norb, norb):
        raise ValueError("eri_chemist must have shape (norb, norb, norb, norb).")

    # RHF closed-shell assumption
    if nelec % 2 != 0:
        raise ValueError(
            "This driver assumes a closed-shell even-electron system. "
            "For open-shell or true spin-orbital Hamiltonians, use UCCSD/GCCSD instead."
        )

    # Basic Hermiticity check for h1
    if not np.allclose(h1, h1.T, atol=1e-10):
        raise ValueError("h1 must be symmetric / Hermitian in this RHF driver.")

    # Build a dummy Mole and override the integral providers
    mol = gto.M()
    mol.nelectron = int(nelec)
    mol.spin = 0
    mol.nao = norb
    mol.incore_anyway = True
    mol.energy_nuc = lambda *args: float(ecore)

    mf = mol.RHF()
    mf.verbose = verbose
    mf.init_guess = "1e"
    mf.max_cycle = 200

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)  # Basis is orthonormal.
    mf._eri = eri_chemist

    # Make overlap integrals consistent with the orthonormal-basis assumption
    original_intor_symmetric = mf.mol.intor_symmetric
    mf.mol.intor_symmetric = (
        lambda intor, **kwargs:
        np.eye(norb) if intor == "int1e_ovlp"
        else original_intor_symmetric(intor, **kwargs)
    )

    return mf


def run_ccsd_from_hg(
    h,
    g,
    nelec,
    ecore=0.0,
    g_format="chemist",
    ccsd_conv_tol=1e-9,
    ccsd_conv_tol_normt=1e-7,
    verbose=4,
):
    """
    Solve RHF-CCSD from a given one-body matrix h and two-body tensor g.

    Parameters
    ----------
    h : ndarray, shape (norb, norb)
        One-electron integrals.
    g : ndarray, shape (norb, norb, norb, norb)
        Two-electron tensor.
    nelec : int
        Total number of electrons. Must be even in this RHF driver.
    ecore : float
        Constant energy offset (for example nuclear repulsion / frozen-core constant).
    g_format : str
        'user_operator_order' or 'chemist'
    ccsd_conv_tol : float
        CCSD energy convergence threshold.
    ccsd_conv_tol_normt : float
        CCSD amplitude convergence threshold.
    verbose : int
        PySCF verbosity.

    Returns
    -------
    result : dict
        Dictionary with HF energy, CCSD energies, amplitudes, and objects.
    """
    h = np.asarray(h, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix.")
    norb = h.shape[0]
    if g.shape != (norb, norb, norb, norb):
        raise ValueError("g must have shape (norb, norb, norb, norb).")

    eri_chemist = _to_chemist_eri(g, g_format=g_format)
    mf = _build_custom_rhf(h, eri_chemist, nelec=nelec, ecore=ecore, verbose=verbose)

    # Mean-field reference
    e_hf = mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge.")

    # CCSD
    mycc = mf.CCSD()
    mycc.conv_tol = ccsd_conv_tol
    mycc.conv_tol_normt = ccsd_conv_tol_normt

    e_corr, t1, t2 = mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge.")

    return {
        "e_hf": float(e_hf),
        "e_corr": float(e_corr),
        "e_tot": float(e_hf + e_corr),
        "t1": t1,
        "t2": t2,
        "mf": mf,
        "cc": mycc,
    }


def print_largest_amplitudes(t1, t2, nt1=10, nt2=10):
    """Print the largest-magnitude T1 and T2 amplitudes."""
    print("\nLargest T1 amplitudes:")
    flat_t1 = []
    for i in range(t1.shape[0]):
        for a in range(t1.shape[1]):
            flat_t1.append((abs(t1[i, a]), i, a, t1[i, a]))
    flat_t1.sort(reverse=True, key=lambda x: x[0])

    for _, i, a, val in flat_t1[:nt1]:
        print(f"  t1[{i},{a}] = {val:+.12e}")

    print("\nLargest T2 amplitudes:")
    flat_t2 = []
    for i in range(t2.shape[0]):
        for j in range(t2.shape[1]):
            for a in range(t2.shape[2]):
                for b in range(t2.shape[3]):
                    flat_t2.append((abs(t2[i, j, a, b]), i, j, a, b, t2[i, j, a, b]))
    flat_t2.sort(reverse=True, key=lambda x: x[0])

    for _, i, j, a, b, val in flat_t2[:nt2]:
        print(f"  t2[{i},{j},{a},{b}] = {val:+.12e}")



def transform_integrals_to_mo(hcore, eri, C):
    """
    Rotate one- and two-electron integrals to the MO basis.

    h_MO[i,j] = sum_pq C[p,i] h[p,q] C[q,j]
    eri_MO[i,j,k,l] = sum_pqrs C[p,i] C[q,j] C[r,k] C[s,l] eri[p,q,r,s]
    """
    h_mo = C.T @ hcore @ C
    eri_mo = np.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, eri, optimize=True)
    return h_mo, eri_mo



# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Replace these with your actual integrals
    #
    # h[p,q] : one-electron matrix
    # g[i,j,k,l] : two-electron tensor in one of the two supported conventions
    #
    # This tiny example is just a placeholder closed-shell 2-orbital model.
    h = np.array([
        [-1.0,  0.1],
        [ 0.1,  0.5],
    ], dtype=float)

    g = np.zeros((2, 2, 2, 2), dtype=float)

    # Example values in *your* operator-order convention:
    # H2 = 1/2 sum g[i,j,k,l] a_i^† a_j^† a_k a_l
    g[0, 1, 1, 0] = 0.60
    g[1, 0, 0, 1] = 0.60
    g[0, 1, 0, 1] = 0.20
    g[1, 0, 1, 0] = 0.20

    nelec = 2
    ecore = 0.0

    result = run_ccsd_from_hg(
        h=h,
        g=g,
        nelec=nelec,
        ecore=ecore,
        g_format="user_operator_order",  # or "chemist"
        verbose=4,
    )

    print("\n=== Energies ===")
    print(f"HF energy        : {result['e_hf']:+.12f}")
    print(f"CCSD corr energy : {result['e_corr']:+.12f}")
    print(f"CCSD total energy: {result['e_tot']:+.12f}")

    print("\n=== Amplitude shapes ===")
    print("t1 shape:", result["t1"].shape)
    print("t2 shape:", result["t2"].shape)
    print("l1 shape:", result["l1"].shape)
    print("l2 shape:", result["l2"].shape)
    print("amplitude vector length:", result["amp_vector"].shape[0])

    print("\n=== Norms ===")
    print("||t1|| =", np.linalg.norm(result["t1"]))
    print("||t2|| =", np.linalg.norm(result["t2"]))
    print("||l1|| =", np.linalg.norm(result["l1"]))
    print("||l2|| =", np.linalg.norm(result["l2"]))

    print_largest_amplitudes(result["t1"], result["t2"], nt1=10, nt2=10)