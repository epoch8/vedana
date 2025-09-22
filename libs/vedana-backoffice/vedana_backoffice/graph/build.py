from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class StepMeta:
    index: int
    name: str
    step_type: str
    inputs: List[str]
    outputs: List[str]
    labels: List[Tuple[str, str]]


NodeDict = Dict[str, Any]
EdgeDict = Dict[str, Any]


@dataclass
class CanonicalGraph:
    steps: List[StepMeta]
    # Indexes for fast joins
    table_to_producers: Dict[str, Set[int]]
    table_to_consumers: Dict[str, Set[int]]
    step_name_by_index: Dict[int, str]
    step_type_by_index: Dict[int, str]


def build_canonical(steps_meta: Iterable[dict]) -> CanonicalGraph:
    """Build a pipeline representation from step metadata.

    - All steps with their metadata (name, type, inputs, outputs, labels)
    - Indexed mappings for fast lookups (table -> producers/consumers)
    - Pre-computed relationships for efficient edge derivation

    Args:
        steps_meta: Iterable of step metadata dictionaries from pipeline introspection

    Returns:
        CanonicalGraph: Normalized pipeline representation with indexed relationships
    """
    steps: List[StepMeta] = []
    table_to_producers: Dict[str, Set[int]] = {}
    table_to_consumers: Dict[str, Set[int]] = {}
    step_name_by_index: Dict[int, str] = {}
    step_type_by_index: Dict[int, str] = {}

    for m in steps_meta:
        try:
            idx = int(m.get("index", -1))
        except Exception:
            idx = -1
        if idx < 0:
            continue
        name = str(m.get("name", f"step_{idx}"))
        step_type = str(m.get("step_type", ""))
        inputs = [str(x) for x in (m.get("inputs") or [])]
        outputs = [str(x) for x in (m.get("outputs") or [])]
        labels_raw = m.get("labels") or []
        labels = [(str(a), str(b)) for a, b in labels_raw]
        sm = StepMeta(index=idx, name=name, step_type=step_type, inputs=inputs, outputs=outputs, labels=labels)
        steps.append(sm)
        step_name_by_index[idx] = name
        step_type_by_index[idx] = step_type

        for t in outputs:
            table_to_producers.setdefault(t, set()).add(idx)
        for t in inputs:
            table_to_consumers.setdefault(t, set()).add(idx)

    return CanonicalGraph(
        steps=steps,
        table_to_producers=table_to_producers,
        table_to_consumers=table_to_consumers,
        step_name_by_index=step_name_by_index,
        step_type_by_index=step_type_by_index,
    )


def derive_step_edges(cg: CanonicalGraph) -> List[Tuple[int, int, List[str]]]:
    """Edges between steps, labeled by shared tables.

    Use table indexes to connect producers -> consumers.
    """
    edges_map: Dict[Tuple[int, int], Set[str]] = {}
    for table, producers in cg.table_to_producers.items():
        consumers = cg.table_to_consumers.get(table, set())
        if not consumers:
            continue
        for s in producers:
            for t in consumers:
                if s == t:
                    continue
                edges_map.setdefault((s, t), set()).add(table)
    edges: List[Tuple[int, int, List[str]]] = []
    for (s, t), tables in edges_map.items():
        edges.append((s, t, sorted(list(tables))))
    return edges


def derive_table_edges(cg: CanonicalGraph) -> List[Tuple[int, int, str]]:
    """Edges between tables; labeled by step name.

    For each step, connect each input table (or -1 for BatchGenerate's) to each output table.
    """
    table_names = sorted(list(set([t for s in cg.steps for t in s.inputs + s.outputs])))
    name_to_id: Dict[str, int] = {name: i for i, name in enumerate(table_names)}

    edges: List[Tuple[int, int, str]] = []
    for sm in cg.steps:
        in_ids = [name_to_id[t] for t in sm.inputs if t in name_to_id]
        out_ids = [name_to_id[t] for t in sm.outputs if t in name_to_id]
        if not in_ids:
            for ot in out_ids:
                edges.append((-1, ot, sm.name))
        else:
            for it in in_ids:
                for ot in out_ids:
                    edges.append((it, ot, sm.name))
    return edges
