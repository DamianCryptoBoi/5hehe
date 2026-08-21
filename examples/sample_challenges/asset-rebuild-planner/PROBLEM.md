# Asset rebuild planner

Define `plan_rebuild(local_hashes, dependencies, cache, transaction_log)`.

`local_hashes` maps every current asset name to its current local-hash string. `dependencies` has exactly the same keys and maps each asset to a list of distinct current asset names it depends on. Self-dependencies are allowed.

A `cache` value contains:

- `"local"`: its recorded local hash;
- `"dependencies"`: distinct `[dependency_name, recorded_full_hash]` pairs in arbitrary order;
- `"full"`: its recorded full hash.

All strings use UTF-8. Let `frame(s)` be the 8-byte unsigned big-endian byte length followed by the encoded string. Given local hash `L` and dependency/full-hash pairs sorted by dependency name, the full hash is lowercase SHA-256 of:

`b"ASSETv1\0" + frame(L) + count + frame(name1) + frame(hash1) + ...`

where `count` is an 8-byte unsigned big-endian integer.

Each transaction-log record is `[event, operation_id, asset_name]`, all strings, in chronological order. An event is structurally invalid if:

- it is neither `"begin"` nor `"end"`;
- a `"begin"` reuses any previously begun operation ID;
- a `"begin"` names an asset already having an unfinished operation;
- an `"end"` has no matching unfinished operation ID, or names a different asset.

If any event is structurally invalid, recovery is `"invalid"`; discard every distinct name occurring in the cache or as a log asset, and no cache card is reusable. Otherwise, if operations remain unfinished, recovery is `"recoverable"`; discard exactly their asset names, and cards for those names are unusable. Otherwise recovery is `"valid"` and nothing is discarded. Discard lists are sorted by Python string ordering. A discarded dependency may be rebuilt to the same full hash, allowing an undiscarded dependent’s card to remain reusable.

A usable card is reusable exactly when its recorded local hash matches, its dependency names equal the current names, its recorded `"full"` matches the formula applied to its recorded data, and every recorded dependency hash equals that dependency’s newly determined full hash. Otherwise the asset is rebuilt and receives the formula’s current full hash.

Process assets only after all dependencies. Among eligible assets choose the smallest name by Python string ordering; this determines rebuild order.

For an acyclic graph return:
`{"status":"ok", "recovery":kind, "discard":names, "rebuild":names, "full_hashes":pairs}`
where pairs contain every `[name, full_hash]`, sorted by name.

If any cycle exists, return:
`{"status":"cycles", "recovery":kind, "discard":names, "cycles":groups}`.
Each group is a maximal mutually reachable set, including multi-asset groups and self-dependent singletons only. Sort each group and then the groups lexicographically. Cache validity does not affect cycle detection.

There are at most 100,000 current assets, 300,000 current dependency edges, 300,000 cached dependency pairs, and 500,000 log records. Use only the Python standard library and perform no I/O.
