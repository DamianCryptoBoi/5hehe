# Reactive stat board

Nodes are mutable source stats or derived displays. Displays retain their shown values until a cutoff is crossed. Dependency weights may also be reassigned between generations.

Write one complete Rust program containing `fn main()`. Use only the standard library.

Input:

- `N M Q` — nodes, dependency edges, and batches.
- `N` descriptions for node IDs `1..N`:
  - `S cutoff initial` for a source.
  - `A cutoff base` for a display.
- `M` edges, numbered `1..M` in input order. Each is `u v w`, meaning display `v` depends on node `u` with weight `w`.
- `Q` batches. Each starts with `k l`, followed by `k` pairs `s proposed`, then `l` pairs `e new_weight`. Pairs may cross line boundaries.

Every edge target is a display, every source has indegree zero, and the dependency graph is acyclic. Within a batch, source IDs are distinct and edge IDs are distinct.

Initially, each source shows `initial`. In dependency order, each display initially shows exactly
`base + sum(weight * shown[predecessor])`.
Initial cutoffs are not applied.

Process each batch as one simultaneous generation:

1. Compare every source proposal with that source’s shown value before the batch. Accept it exactly when the absolute difference is strictly greater than its cutoff. Rejected proposals do not alter the source and never accumulate.
2. Assign every listed edge its proposed weight. An assignment equal to its previous weight is not a weight change.
3. All accepted source changes and actual weight changes take effect simultaneously. Thus, if a source and one of its outgoing weights are both changed, subsequent formulas use both new values.
4. A display is recomputed exactly once if at least one direct dependency’s shown value changed or at least one incoming edge’s weight changed. Recomputations respect dependency order.
5. Recompute the full formula using current shown predecessor values and current weights. The display adopts the result exactly when its absolute difference from its old shown value is strictly greater than its cutoff. Otherwise its shown value remains unchanged. Only an adopted shown-value change triggers dependent displays; hidden differences may accumulate in the formula and cross the cutoff during a later recomputation.

After each batch print:
`R r id1 ... idr C c jd1 ... jdc`

`R` contains every recomputed display. `C` contains every node whose shown value changed, including sources. Each list is in increasing node-ID order and is preceded by its count. Weight changes are not included in `C`.

Constraints:

- `1 <= N,Q <= 200000`, `0 <= M <= 200000`
- Total `k` and total `l` over all batches are each at most `200000`
- Cutoffs are in `[0,10^18]`; all other numeric input values have absolute value at most `10^18`
- Every required product, formula value, and shown value fits signed 64-bit range
- Across all batches, at most `1000000` displays are recomputed, and the sum of outdegrees over all occurrences of changed nodes is at most `1000000`

Output is compared by splitting only on ASCII whitespace bytes `0x09` through `0x0D` and `0x20`, then comparing tokens exactly.
