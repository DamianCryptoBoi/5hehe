# Extent journal

Implement `extent_journal(chunk_size, address_limit, large_threshold, events)`.

Memory consists of byte addresses in `[0, address_limit)`. `MAX = 2**63 - 1`.
The first three arguments are integers satisfying `1 <= chunk_size <= address_limit <= MAX` and `1 <= large_threshold <= MAX`. At most 200,000 well-formed event tuples are provided. Allocation IDs and arena IDs are nonnegative integers.

A logical size is converted to a physical size by evaluating `size + chunk_size - 1`, then rounding down to a multiple of `chunk_size`. Conversion is invalid if `size < large_threshold`, `size > MAX`, or the addition would exceed `MAX`. Invalid operations are rejected before changing any state.

Process events in order:

- `("claim", allocation_id, arena_id, start, size)` creates an extent if the ID is inactive, `start` is nonnegative and chunk-aligned, conversion succeeds, the physical end does not exceed `address_limit`, and the half-open physical interval overlaps no active extent. Append `("claimed", start, end)` on success or `("rejected",)` otherwise.
- `("resize", allocation_id, size)` changes an active allocation's logical and physical sizes. Reject it if the ID is inactive or conversion fails. If the new physical span is no larger than the old span, or can expand at the current start without crossing `address_limit` or the next active extent, it remains at its current start. Otherwise, attempt allocate-copy-release fallback: while the old extent remains occupied, choose the lowest chunk-aligned start whose new physical interval is within memory and overlaps no active extent. If no such start exists, reject the event. On success, atomically replace the old extent and append `("resized", old_start, old_end, new_start, new_end)`. A rejected resize changes nothing.
- `("release", allocation_id)` removes an active extent. Append `("released", start, end)` on success or `("rejected",)` if the ID is inactive. A released ID may later be claimed again.
- `("lookup", address)` examines the active physical extent containing `address`, including rounded padding. Append `None` if none exists. Otherwise append `("extent", allocation_id, arena_id, start, logical_size, end, address-start)`.

Intervals are half-open, so adjacent extents do not overlap. The old extent is not available as fallback space during its own resize. Every event is atomic. Return one result per event, in order. Use only the Python standard library and perform no I/O.
