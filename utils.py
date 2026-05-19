from __future__ import annotations

import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from qiskit.transpiler import CouplingMap

__all__ = [
    "make_heavy_hex_grid",
    "make_square_grid",
    "plot_coupling_map",
    "lucj_ab_pair_paths_dataframe",
    "plot_lucj_mapping_from_result",
]


def _to_directed_edges(
    undirected_edges: set[tuple[int, int]],
    *,
    bidirectional: bool,
) -> list[tuple[int, int]]:
    edges = sorted(undirected_edges)
    if bidirectional:
        return edges + [(v, u) for u, v in edges]
    return edges


def make_heavy_hex_grid(
    rows: int,
    cols: int,
    *,
    bidirectional: bool = True,
    hex_radius: float = 1.0,
) -> tuple[CouplingMap, dict[int, tuple[float, float]]]:
    """Generate a heavy-hex patch with ``rows * cols`` hexagonal cells.

    The generator starts from a pointy-top hexagonal lattice and subdivides
    every hex-lattice edge by inserting one additional qubit at its midpoint:

        corner -- edge_qubit -- corner

    The returned labels are construction-order labels for plotting/debugging;
    they are not intended to match IBM device labels.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols should be positive integers.")
    if hex_radius <= 0:
        raise ValueError("hex_radius should be positive.")

    sqrt3 = math.sqrt(3.0)
    vertex_angles = [math.radians(angle) for angle in (90, 30, -30, -90, -150, 150)]

    corner_id: dict[tuple[int, int], int] = {}
    pos: dict[int, tuple[float, float]] = {}
    hex_edges: set[tuple[int, int]] = set()

    def rounded_key(x: float, y: float) -> tuple[int, int]:
        scale = 10**9
        return (round(x * scale), round(y * scale))

    def get_corner_node(x: float, y: float) -> int:
        key = rounded_key(x, y)
        if key not in corner_id:
            node = len(corner_id)
            corner_id[key] = node
            pos[node] = (x, y)
        return corner_id[key]

    for row in range(rows):
        for col in range(cols):
            center_x = sqrt3 * hex_radius * (col + 0.5 * row)
            center_y = -1.5 * hex_radius * row

            corners = [
                get_corner_node(
                    center_x + hex_radius * math.cos(theta),
                    center_y + hex_radius * math.sin(theta),
                )
                for theta in vertex_angles
            ]

            for u, v in zip(corners, corners[1:] + corners[:1]):
                hex_edges.add(tuple(sorted((u, v))))

    undirected_edges: set[tuple[int, int]] = set()
    next_node = len(pos)

    for u, v in sorted(hex_edges):
        x_mid = 0.5 * (pos[u][0] + pos[v][0])
        y_mid = 0.5 * (pos[u][1] + pos[v][1])

        edge_node = next_node
        next_node += 1
        pos[edge_node] = (x_mid, y_mid)

        undirected_edges.add(tuple(sorted((u, edge_node))))
        undirected_edges.add(tuple(sorted((edge_node, v))))

    directed_edges = _to_directed_edges(
        undirected_edges,
        bidirectional=bidirectional,
    )
    return CouplingMap(directed_edges), pos


def make_square_grid(
    rows: int,
    cols: int,
    *,
    bidirectional: bool = True,
) -> tuple[CouplingMap, dict[int, tuple[float, float]]]:
    """Generate a rows x cols square-lattice coupling map.

    Qubit indexing is row-major. Edges connect nearest neighbors horizontally
    and vertically.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols should be positive integers.")

    def node(row: int, col: int) -> int:
        return row * cols + col

    undirected_edges: set[tuple[int, int]] = set()

    for row in range(rows):
        for col in range(cols - 1):
            undirected_edges.add((node(row, col), node(row, col + 1)))

    for row in range(rows - 1):
        for col in range(cols):
            undirected_edges.add((node(row, col), node(row + 1, col)))

    directed_edges = _to_directed_edges(
        undirected_edges,
        bidirectional=bidirectional,
    )

    coupling_map = CouplingMap(directed_edges)
    pos = {
        node(row, col): (float(col), -float(row))
        for row in range(rows)
        for col in range(cols)
    }

    return coupling_map, pos


def _get_backend_coupling_map(backend):
    if hasattr(backend, "coupling_map") and backend.coupling_map is not None:
        return backend.coupling_map

    if hasattr(backend, "target"):
        return backend.target.build_coupling_map()

    raise ValueError("Could not extract coupling_map from backend.")


def _coupling_map_to_graph(coupling_map: CouplingMap) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(coupling_map.physical_qubits)
    graph.add_edges_from(
        {
            tuple(sorted((u, v)))
            for u, v in coupling_map.get_edges()
            if u != v
        }
    )
    return graph


def _get_lucj_logical_to_physical(result, *, use_final_layout: bool = False):
    circuit = result["circuit"]
    isa_circuit = result["isa_circuit"]
    norb = result.get("norb", circuit.num_qubits // 2)

    layout = isa_circuit.layout
    n_lucj_qubits = 2 * norb

    if layout is None:
        return {i: i for i in range(n_lucj_qubits)}

    if use_final_layout:
        final_layout = layout.final_index_layout(filter_ancillas=True)
        return {i: final_layout[i] for i in range(n_lucj_qubits)}

    initial_layout = layout.initial_virtual_layout(filter_ancillas=True)
    virtual_to_physical = initial_layout.get_virtual_bits()

    return {
        i: virtual_to_physical[circuit.qubits[i]]
        for i in range(n_lucj_qubits)
    }


def lucj_ab_pair_paths_dataframe(
    result,
    backend,
    use_final_layout: bool = False,
) -> pd.DataFrame:
    """Return physical shortest paths for the accepted LUCJ alpha-beta pairs."""

    norb = result.get("norb", result["circuit"].num_qubits // 2)
    coupling_map = _get_backend_coupling_map(backend)
    graph = _coupling_map_to_graph(coupling_map)
    logical_to_physical = _get_lucj_logical_to_physical(
        result,
        use_final_layout=use_final_layout,
    )

    rows = []
    for a, b in result.get("ab_pairs", []):
        alpha_logical = a
        beta_logical = norb + b
        alpha_physical = logical_to_physical[alpha_logical]
        beta_physical = logical_to_physical[beta_logical]

        try:
            path = nx.shortest_path(graph, alpha_physical, beta_physical)
            distance = len(path) - 1
        except nx.NetworkXNoPath:
            path = []
            distance = None

        rows.append(
            {
                "ab_pair": (a, b),
                "alpha": f"alpha_{a}",
                "beta": f"beta_{b}",
                "alpha_physical": alpha_physical,
                "beta_physical": beta_physical,
                "distance": distance,
                "path": path,
            }
        )

    return pd.DataFrame(rows)


def plot_lucj_mapping_from_result(
    result,
    *,
    backend=None,
    pos=None,
    use_final_layout: bool = False,
    show_ab_paths: bool = True,
    figsize=None,
    node_size: int = 400,
    margin: float = 0.5,
):
    """Plot LUCJ spin-orbital qubits on the backend coupling map."""

    if backend is None:
        backend = result.get("backend", None)

    if backend is None:
        raise ValueError(
            "backend is not included in result. "
            "Either add 'backend' to the return dict or pass backend=backend."
        )

    norb = result.get("norb", result["circuit"].num_qubits // 2)
    coupling_map = _get_backend_coupling_map(backend)
    logical_to_physical = _get_lucj_logical_to_physical(
        result,
        use_final_layout=use_final_layout,
    )

    physical_to_lucj_label = {}
    for logical, physical in logical_to_physical.items():
        if logical < norb:
            label = f"α{logical}"
        else:
            label = f"β{logical - norb}"
        physical_to_lucj_label.setdefault(physical, []).append(label)

    graph = _coupling_map_to_graph(coupling_map)

    if pos is None:
        pos = nx.kamada_kawai_layout(graph)

    labels = {}
    node_colors = []

    for node in graph.nodes():
        if node in physical_to_lucj_label:
            lucj_label = ",".join(physical_to_lucj_label[node])
            labels[node] = f"{node}\n{lucj_label}"

            has_alpha = any(x.startswith("α") for x in physical_to_lucj_label[node])
            has_beta = any(x.startswith("β") for x in physical_to_lucj_label[node])

            if has_alpha and has_beta:
                node_colors.append("plum")
            elif has_alpha:
                node_colors.append("lightskyblue")
            else:
                node_colors.append("lightcoral")
        else:
            labels[node] = str(node)
            node_colors.append("lightgray")

    if figsize is None:
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        figsize = (
            max(6, 0.8 * (max(xs) - min(xs) + 1)),
            max(4, 0.8 * (max(ys) - min(ys) + 1)),
        )

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="gray",
        width=1.2,
    )

    if show_ab_paths and result.get("ab_pairs"):
        ab_paths = lucj_ab_pair_paths_dataframe(
            result,
            backend=backend,
            use_final_layout=use_final_layout,
        )
        highlighted_edges: set[tuple[int, int]] = set()
        for path in ab_paths["path"]:
            for u, v in zip(path[:-1], path[1:]):
                highlighted_edges.add(tuple(sorted((u, v))))

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=sorted(highlighted_edges),
            edge_color="purple",
            width=3.0,
        )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.0,
        node_size=node_size,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels=labels,
        font_size=9,
    )

    title = "LUCJ logical-to-physical qubit mapping"
    if use_final_layout:
        title += " after routing"
    else:
        title += " initial layout"

    y_min, y_max = min(ys), max(ys)
    y_span = max(y_max - y_min, 1.0)
    ax.set_ylim(
        y_min - margin * y_span,
        y_max + margin * y_span,
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()

    return logical_to_physical


def plot_coupling_map(
    coupling_map: CouplingMap,
    pos: dict[int, tuple[float, float]],
    *,
    directed: bool = False,
    label_qubits: bool = True,
    node_size: int = 200,
    font_size: int = 9,
    figsize: tuple[float, float] | None = None,
    margin: float = 0.2,
    ax=None,
):
    """Plot a Qiskit CouplingMap using manually supplied node positions."""

    if directed:
        graph = nx.DiGraph()
        edges = list(coupling_map.get_edges())
    else:
        graph = nx.Graph()
        edges = {
            tuple(sorted((u, v)))
            for u, v in coupling_map.get_edges()
            if u != v
        }

    graph.add_nodes_from(coupling_map.physical_qubits)
    graph.add_edges_from(edges)

    missing_nodes = set(graph.nodes()) - set(pos)
    if missing_nodes:
        raise ValueError(
            f"pos is missing coordinates for qubits: {sorted(missing_nodes)}"
        )

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]

    if figsize is None:
        figsize = (
            max(5.0, 0.8 * (max(xs) - min(xs) + 1)),
            max(3.0, 0.8 * (max(ys) - min(ys) + 1)),
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=1.2,
        arrows=directed,
        arrowsize=12,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.0,
    )

    if label_qubits:
        nx.draw_networkx_labels(
            graph,
            pos,
            ax=ax,
            labels={q: str(q) for q in graph.nodes()},
            font_size=font_size,
        )

    y_min, y_max = min(ys), max(ys)
    y_span = max(y_max - y_min, 1.0)
    ax.set_ylim(
        y_min - margin * y_span,
        y_max + margin * y_span,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax
