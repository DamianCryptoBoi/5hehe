# Sparse circular array

Write one complete Rust program containing `fn main()` that processes a sparse circular array.

The array has fixed logical length `N` and indices `0..N-1`. Each slot is empty or stores a signed 64-bit value. A cursor always points to one index, initially `0`.

Initially, `M` distinct occupied indices are listed. Their input order is their age order: first is oldest, last is newest.

Process `Q` operations in order:

- `MOVE p`: set the cursor to `p`.
- `SHIFT d`: simultaneously move every occupied value and the cursor forward by `d` positions modulo `N`. Thus an item at index `i` moves to `(i+d) mod N`. Ages are unchanged.
- `INSERT x`: starting at the cursor, inspect indices forward with wraparound. Store `x` in the first empty slot. It becomes newest, and the cursor moves to the next index after that slot. Output the chosen index. If every slot is occupied, output `-1` and do not move the cursor.
- `RING x`: store `x` exactly at the cursor. If occupied, the previous value is removed. The new value becomes newest. Output its index, then advance the cursor by one with wraparound.
- `DELETE a b`: delete every occupied slot on the inclusive forward circular range from `a` through `b`. If `a <= b`, this is `[a,b]`; otherwise it is `[a,N-1]` followed by `[0,b]`. The cursor does not move.

Deleting or overwriting removes the previous item from age order. Moving or shifting never changes ages. All operation indices refer to the current indices after all preceding shifts.

Input consists of:

`N Q M`

then `M` lines:

`index value`

then `Q` operation lines in the formats above.

Constraints:

- `1 <= N <= 10^18`
- `0 <= M <= min(N, 200000)`
- `0 <= Q <= 200000`
- indices and `d` are in `0..N-1`
- initial indices are distinct
- values are signed 64-bit integers

For each `INSERT` or `RING`, output one line with its required result. After all operations, output `K`, the number of occupied slots, followed by `K` lines `index value`, ordered from oldest remaining item to newest. Report final current indices after all shifts. No other operations produce output.

Only the Rust standard library may be used. Output is judged by splitting on ASCII whitespace bytes `0x09` through `0x0D` and `0x20`, then comparing tokens exactly.
