"""The 18 patchable components U and their names.

spec name  -> used in code, JSON and CSV            (h_t_6, mlp_t, h_s_4, mlp_s)
hook name  -> what patching_context understands     (head_6, temporal_ffn_delta, ...)
paper name -> format_circuit                        (T:h6, T:MLP, S:h4, S:MLP)
"""
from __future__ import annotations

SPEC_TO_HOOK: dict[str, str] = {
    **{f"h_s_{k}": f"spatial_head_{k}" for k in range(8)},
    "mlp_s": "spatial_ffn_delta",
    **{f"h_t_{k}": f"head_{k}" for k in range(8)},
    "mlp_t": "temporal_ffn_delta",
}
ALL_COMPONENTS: list[str] = list(SPEC_TO_HOOK)
assert len(ALL_COMPONENTS) == 18 and len(set(SPEC_TO_HOOK.values())) == 18


def to_hook_names(components: list[str]) -> list[str]:
    return [SPEC_TO_HOOK[c] for c in components]


def complement(components: list[str]) -> list[str]:
    """U \\ C in canonical order."""
    s = set(components)
    return [c for c in ALL_COMPONENTS if c not in s]


def format_circuit(components: list[str]) -> str:
    """'T:{MLP, h6, h7}, S:{MLP}' — MLP first, heads ascending, temporal branch first."""
    out = []
    for branch, mlp, head in (("T", "mlp_t", "h_t_"), ("S", "mlp_s", "h_s_")):
        items = (["MLP"] if mlp in components else []) + \
                [f"h{c[-1]}" for c in sorted(c for c in components if c.startswith(head))]
        if items:
            out.append(f"{branch}:{{{', '.join(items)}}}")
    return ", ".join(out)
