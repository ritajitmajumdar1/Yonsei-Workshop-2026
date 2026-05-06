
from __future__ import annotations

import matplotlib.pyplot as plt

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

__all__ = [
    "build_polyacene_ppp",
    "polyacene_geometry",
    "PPPIntegrals",
    "spinorb_ppp_to_spatial",
    "plot_coordinate",
    "plot_site_population",
    "plot_orbital_spread",
    "orbital_spread_from_coefficients",
]


@dataclass
class PPPIntegrals:
    """
    PPP tensors in the spin-orbital basis:

        H = sum_{p,q} h[p,q] a_p^\\dagger a_q
          + 1/2 * sum_{i,j,k,l} g[i,j,k,l] a_i^\\dagger a_j^\\dagger a_k a_l
          + constant

    Here p,q,... are spin-orbital indices:
        p = 2*site + spin,   spin = 0 (alpha), 1 (beta)

    `g_sparse[(i,j,k,l)]` stores only nonzero entries.
    """
    coords: np.ndarray                # (n_sites, 2), Cartesian positions in Angstrom
    edges: list[tuple[int, int]]      # nearest-neighbor carbon-carbon bonds
    h: np.ndarray                     # (2*n_sites, 2*n_sites)
    g_sparse: Dict[Tuple[int, int, int, int], float]
    constant: float                   # scalar shift from background charges
    V: np.ndarray                     # (n_sites, n_sites) intersite Coulomb matrix (V_ii = 0)


def ohno_potential(distance_angstrom: float, U: float, kappa: float = 1.0) -> float:
    """
    Ohno interpolation formula in eV.

    V(R) = U / [kappa * sqrt(1 + (U*R/14.397)^2)]

    Parameters
    ----------
    distance_angstrom : float
        Distance R_ij in Angstrom.
    U : float
        On-site repulsion in eV.
    kappa : float
        Screening factor (default 1.0).
    """
    return U / (kappa * math.sqrt(1.0 + (U * distance_angstrom / 14.397) ** 2))


def _regular_hexagon_offsets(bond_length: float) -> np.ndarray:
    """Return the six vertices of a regular hexagon with vertical left/right edges."""
    return np.array(
        [
            [ math.sqrt(3) / 2,  0.5],
            [ 0.0,               1.0],
            [-math.sqrt(3) / 2,  0.5],
            [-math.sqrt(3) / 2, -0.5],
            [ 0.0,              -1.0],
            [ math.sqrt(3) / 2, -0.5],
        ],
        dtype=float,
    ) * bond_length


def _build_fused_hexagon_geometry(centers: list[np.ndarray], bond_length: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Merge regular hexagons centered at `centers` into a fused-ring carbon graph."""
    offsets = _regular_hexagon_offsets(bond_length)
    ring_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    coord_to_index: dict[tuple[float, float], int] = {}
    coords: list[np.ndarray] = []
    edges: set[tuple[int, int]] = set()

    for center in centers:
        ring_vertex_ids = []
        for off in offsets:
            xy = tuple(np.round(center + off, 12))
            if xy not in coord_to_index:
                coord_to_index[xy] = len(coords)
                coords.append(np.array(xy, dtype=float))
            ring_vertex_ids.append(coord_to_index[xy])

        for a, b in ring_edges:
            i, j = ring_vertex_ids[a], ring_vertex_ids[b]
            edges.add(tuple(sorted((i, j))))

    # Reorder sites from left to right; for the same x, list the upper site first.
    order = sorted(range(len(coords)), key=lambda i: (coords[i][0], -coords[i][1]))
    old_to_new = {old: new for new, old in enumerate(order)}

    coords_array = np.array([coords[i] for i in order], dtype=float)
    edges_list = sorted((old_to_new[i], old_to_new[j]) for i, j in edges)
    return coords_array, edges_list


def polyacene_geometry(
    n_rings: int,
    bond_length: float = 1.4,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Build idealized 2D coordinates and nearest-neighbor edges for linearly fused acenes.

    Number of carbon sites = ``4*n_rings + 2``.

    Examples
    --------
    ``n_rings = 1`` -> benzene (6 sites)
    ``n_rings = 2`` -> naphthalene (10 sites)
    ``n_rings = 3`` -> anthracene (14 sites)
    """
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1")

    centers = [
        np.array([m * math.sqrt(3) * bond_length, 0.0], dtype=float)
        for m in range(n_rings)
    ]
    return _build_fused_hexagon_geometry(centers=centers, bond_length=bond_length)


def add_density_density_term(
    g_sparse: Dict[Tuple[int, int, int, int], float],
    p: int,
    q: int,
    strength: float,
) -> None:
    """
    Add W * n_p n_q to the tensor representation

        1/2 * sum_{i,j,k,l} g[i,j,k,l] a_i^\\dagger a_j^\\dagger a_k a_l

    for p != q.

    Since n_p n_q = a_p^\\dagger a_q^\\dagger a_q a_p for p != q,
    we store both:
        g[p,q,q,p] += W
        g[q,p,p,q] += W

    so that the prefactor 1/2 reproduces exactly W * n_p n_q.
    """
    if p == q:
        raise ValueError("Density-density term requires p != q")
    g_sparse[(p, q, q, p)] = g_sparse.get((p, q, q, p), 0.0) + strength
    g_sparse[(q, p, p, q)] = g_sparse.get((q, p, p, q), 0.0) + strength


def sparse_to_dense_g(
    g_sparse: Dict[Tuple[int, int, int, int], float],
    n_spin_orbitals: int,
) -> np.ndarray:
    """Convert sparse 4-index tensor dictionary to a dense ndarray."""
    g = np.zeros((n_spin_orbitals,) * 4, dtype=float)
    for (i, j, k, l), value in g_sparse.items():
        g[i, j, k, l] = value
    return g


def build_polyacene_ppp(
    n_rings: int,
    t0: float = -2.4,
    U: float = 11.26,
    bond_length: float = 1.4,
    kappa: float = 1.0,
    site_energies: Iterable[float] | None = None,
    z: Iterable[float] | None = None,
    coulomb_fn: Callable[[float, float, float], float] = ohno_potential,
    dense_g: bool = False,
) -> PPPIntegrals | tuple[PPPIntegrals, np.ndarray]:
    """
    Build PPP h_{pq} and g_{ijkl} for linearly fused polyacene in the spin-orbital basis.

    Hamiltonian convention
    ----------------------
    H = sum_{<i,j>,sigma} t0 (a^dagger_{i,sigma} a_{j,sigma} + h.c.)
      + sum_i U * n_{i,alpha} n_{i,beta}
      + sum_{i<j} V_ij (n_i - z_i) (n_j - z_j)

    converted to

    H = sum_{p,q} h[p,q] a_p^\\dagger a_q
      + 1/2 * sum_{i,j,k,l} g[i,j,k,l] a_i^\\dagger a_j^\\dagger a_k a_l
      + constant

    Notes
    -----
    1. The returned `h` and `g_sparse` are in the spin-orbital basis:
           site i, alpha -> 2*i
           site i, beta  -> 2*i + 1

    2. The intersite coulomb expansion contributes a one-body diagonal shift:
           h_{iσ,iσ} += -sum_{j != i} z_j V_ij + epsilon_i

    3. The returned scalar `constant` is
           sum_{i<j} z_i z_j V_ij

    4. The sign of `t0` is not enforced. For the usual bonding convention,
       choose `t0 < 0`.
    """
    coords, edges = polyacene_geometry(
        n_rings=n_rings,
        bond_length=bond_length,
    )
    n_sites = coords.shape[0]
    n_spin_orb = 2 * n_sites

    if site_energies is None:
        epsilon = np.zeros(n_sites, dtype=float)
    else:
        epsilon = np.asarray(list(site_energies), dtype=float)
        if epsilon.shape != (n_sites,):
            raise ValueError(f"site_energies must have length {n_sites}")

    if z is None:
        z_arr = np.ones(n_sites, dtype=float)
    else:
        z_arr = np.asarray(list(z), dtype=float)
        if z_arr.shape != (n_sites,):
            raise ValueError(f"z must have length {n_sites}")

    # Intersite Coulomb matrix V_ij, with V_ii = 0 by convention.
    V = np.zeros((n_sites, n_sites), dtype=float)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            Rij = float(np.linalg.norm(coords[i] - coords[j]))
            Vij = coulomb_fn(Rij, U, kappa)
            V[i, j] = V[j, i] = Vij

    h = np.zeros((n_spin_orb, n_spin_orb), dtype=float)
    g_sparse: Dict[Tuple[int, int, int, int], float] = {}

    # 1) One-body hopping: same spin only.
    for i, j in edges:
        for spin in (0, 1):
            p = 2 * i + spin
            q = 2 * j + spin
            h[p, q] += t0
            h[q, p] += t0

    # 2) One-body diagonal from site energies and intersite Coulomb interaction.
    for i in range(n_sites):
        diag_shift = epsilon[i] - np.dot(V[i], z_arr)
        for spin in (0, 1):
            p = 2 * i + spin
            h[p, p] += diag_shift

    # 3) On-site U term: U * n_{i,alpha} n_{i,beta}
    for i in range(n_sites):
        p_alpha = 2 * i
        p_beta = 2 * i + 1
        add_density_density_term(g_sparse, p_alpha, p_beta, U)

    # 4) Intersite Coulomb terms: V_ij * n_i n_j
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            Vij = V[i, j]
            for spin_i in (0, 1):
                for spin_j in (0, 1):
                    p = 2 * i + spin_i
                    q = 2 * j + spin_j
                    add_density_density_term(g_sparse, p, q, Vij)

    # 5) Constant from background charges.
    constant = 0.0
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            constant += z_arr[i] * z_arr[j] * V[i, j]

    result = PPPIntegrals(
        coords=coords,
        edges=edges,
        h=h,
        g_sparse=g_sparse,
        constant=constant,
        V=V,
    )

    if dense_g:
        g_dense = sparse_to_dense_g(g_sparse, n_spin_orb)
        return result, g_dense
    return result


def spinorb_ppp_to_spatial(ppp_spin, U):
    """Convert the spin-orbital PPP object to spatial-orbital hcore/eri tensors."""
    n_sites = ppp_spin.coords.shape[0]

    h_alpha = np.asarray(ppp_spin.h[0::2, 0::2], dtype=float)
    h_beta = np.asarray(ppp_spin.h[1::2, 1::2], dtype=float)
    if not np.allclose(h_alpha, h_beta):
        raise ValueError("Alpha and beta one-body blocks are not identical.")

    eri = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=float)
    for p in range(n_sites):
        eri[p, p, p, p] = U
    for p in range(n_sites):
        for q in range(n_sites):
            if p != q:
                eri[p, p, q, q] = ppp_spin.V[p, q]

    return h_alpha, eri, float(ppp_spin.constant)



def _validate_site_array(values: np.ndarray, n_sites: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n_sites:
        raise ValueError(f"{name} must be a 1D array of length {n_sites}")
    return arr


def _validate_coeff_matrix(C: np.ndarray, n_sites: int) -> np.ndarray:
    coeff = np.asarray(C)
    if coeff.ndim != 2 or coeff.shape[0] != n_sites:
        raise ValueError(
            f"C must have shape (n_sites, n_orbitals) with n_sites={n_sites}; got {coeff.shape}"
        )
    return coeff


def _draw_polyacene_skeleton(ax, coords: np.ndarray, edges: Iterable[tuple[int, int]]) -> None:
    for i, j in edges:
        xi, yi = coords[i]
        xj, yj = coords[j]
        ax.plot([xi, xj], [yi, yj], color="0.65", lw=2.0, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def orbital_spread_from_coefficients(C: np.ndarray, orbital_index: int) -> np.ndarray:
    """Return the site weights |C_{p,mu}|^2 of a single spatial orbital."""
    coeff = np.asarray(C)
    if coeff.ndim != 2:
        raise ValueError(f"C must be a rank-2 array; got shape {coeff.shape}")
    n_orbitals = coeff.shape[1]
    if not 0 <= orbital_index < n_orbitals:
        raise IndexError(f"orbital_index must be in [0, {n_orbitals}); got {orbital_index}")
    return np.abs(coeff[:, orbital_index]) ** 2


def plot_site_population(
    ppp_spin: PPPIntegrals,
    site_population: np.ndarray,
    *,
    ax=None,
    title: str = "Site-orbital population",
    cmap: str = "twilight_shifted",
    scale: float = 1000.0,
    annotate: bool = True,
    show_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Visualize site populations on the polyacene geometry.

    Parameters
    ----------
    ppp_spin
        PPP geometry/tensor container returned by :func:`build_polyacene_ppp`.
    site_population
        Length-``n_sites`` array such as a site occupancy or orbital weight.
    ax
        Optional Matplotlib axes. When omitted, a new figure is created.
    scale
        Marker area prefactor. Actual marker sizes are ``scale * population``.
    """
    coords = np.asarray(ppp_spin.coords, dtype=float)
    pop = _validate_site_array(site_population, coords.shape[0], "site_population")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        created_fig = True
    else:
        fig = ax.figure

    ax.margins(x=0.15, y=0.15)

    _draw_polyacene_skeleton(ax, coords, ppp_spin.edges)

    sizes = scale * np.clip(pop, a_min=0.0, a_max=None)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=pop,
        s=sizes,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    if annotate:
        dy = 0.03 * np.ptp(coords[:, 1]) if np.ptp(coords[:, 1]) > 0 else 0.15
        for i, ((x, y), value) in enumerate(zip(coords, pop)):
            ax.text(x, y + dy, f"{i}: {value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, y=1.03)
    if show_colorbar:
        fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02, label="Population")

    if created_fig:
        plt.show()
    return ax


def plot_orbital_spread(
    ppp_spin: PPPIntegrals,
    C: np.ndarray,
    orbital_indices: int | Iterable[int],
    *,
    cols: int = 2,
    figsize: tuple[float, float] | None = None,
    cmap: str = "twilight_shifted",
    scale: float = 1000.0,
    annotate: bool = True,
    show_colorbar: bool = True,
    min_marker_size: float = 0.0,
):
    """
    Visualize orbital spreads |C_{p,mu}|^2 on the polyacene geometry.

    Parameters
    ----------
    C
        Orbital coefficient matrix with shape ``(n_sites, n_orbitals)``.
        Columns correspond to RHF spatial orbitals expanded in the site basis.
    orbital_indices
        One orbital index or an iterable of orbital indices to visualize.
    """
    coeff = _validate_coeff_matrix(C, ppp_spin.coords.shape[0])
    if isinstance(orbital_indices, int):
        indices = [orbital_indices]
    else:
        indices = list(orbital_indices)
    if not indices:
        raise ValueError("orbital_indices must contain at least one orbital index")

    coords = np.asarray(ppp_spin.coords, dtype=float)
    n_panels = len(indices)
    cols = max(1, min(cols, n_panels))
    rows = math.ceil(n_panels / cols)
    if figsize is None:
        figsize = (4.8 * cols, 2.9 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax, mu in zip(axes_flat, indices):
        coeff_mu = np.asarray(coeff[:, mu])
        amplitude = np.abs(coeff_mu) ** 2
        phase = np.angle(coeff_mu)

        _draw_polyacene_skeleton(ax, coords, ppp_spin.edges)
        sizes = min_marker_size + scale * amplitude
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=phase,
            s=sizes,
            cmap=cmap,
            vmin=-np.pi,
            vmax=np.pi,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

        dy = 0.06 * np.ptp(coords[:, 1]) if np.ptp(coords[:, 1]) > 0 else 0.15
        if annotate:
            for i, ((x, y), amp, ph) in enumerate(zip(coords, amplitude, phase)):
                ax.text(
                    x,
                    y + dy,
                    f"{i}: {amp:.3f}\n{ph:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        xs = coords[:, 0]
        ys = coords[:, 1]
        pad = 0.35
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.set_title(rf"Orbital $\mu = {mu}$: size $|C_{{p\mu}}|^2$, color $\arg(C_{{p\mu}})$", pad=16)

        if show_colorbar:
            cbar = fig.colorbar(
                scatter,
                ax=ax,
                shrink=0.85,
                pad=0.02,
                ticks=[-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi],
            )
            cbar.ax.set_yticklabels(
                [r"$-\pi\equiv\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi\equiv-\pi$"]
            )
            cbar.set_label("Phase [rad] (mod $2\pi$)")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return axes

def plot_coordinate(ppp_spin):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    _draw_polyacene_skeleton(ax, np.asarray(ppp_spin.coords, dtype=float), ppp_spin.edges)
    ax.scatter(ppp_spin.coords[:, 0], ppp_spin.coords[:, 1], s=80, color="tab:blue", edgecolors="black", zorder=3)
    ax.margins(x=0.15, y=0.15)
    dy = 0.03 * np.ptp(ppp_spin.coords[:, 1]) if np.ptp(ppp_spin.coords[:, 1]) > 0 else 0.15
    for i, (x, y) in enumerate(ppp_spin.coords):
        ax.text(x, y + dy, str(i), ha="center", va="bottom", fontsize=9)
    ax.set_title("PPP graph (site basis)", y=1.03)
    plt.show()


if __name__ == "__main__":
    # Example: anthracene (3 fused benzene rings, 14 pi orbitals, 28 spin orbitals)
    result, g_dense = build_polyacene_ppp(
        n_rings=3,
        t0=-2.4,
        U=11.26,
        bond_length=1.4,
        kappa=1.0,
        dense_g=True
    )

    print("Number of carbon sites:", result.coords.shape[0])
    print("Number of spin orbitals:", result.h.shape[0])
    print("Nearest-neighbor bonds:", len(result.edges))
    print("Constant energy shift:", result.constant)
    print("h shape:", result.h.shape)
    print("g shape:", g_dense.shape)

    # Show a few nonzero tensor entries
    print("\nA few nonzero g[i,j,k,l] entries:")
    shown = 0
    for key, value in result.g_sparse.items():
        if abs(value) > 1e-12:
            print(f"g{key} = {value:.6f}")
            shown += 1
            if shown >= 12:
                break

    # Example visualizations based on the site-basis one-body block.
    h_site = result.h[0::2, 0::2]
    _, C_demo = np.linalg.eigh(h_site)
    plot_coordinate(result)
    plot_orbital_spread(result, C_demo, orbital_indices=[0, 1])
    plot_site_population(result, np.sum(np.abs(C_demo[:, :2]) ** 2, axis=1), title="Combined population of the two lowest one-body orbitals")
