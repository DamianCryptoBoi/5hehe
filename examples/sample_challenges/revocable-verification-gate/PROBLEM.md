# Revocable weighted verification gate

A service processes events in chronological order. Request events name a network address and an email address and have a positive cost. Every request is assigned an ID, starting from `1`, in request-event order.

Two independent quotas apply:

- For each exact network-address string, accepted requests may have total active cost at most `BI`. They remain active for `WI` time units.
- For each normalized email address, accepted requests may have total active cost at most `BE`. They remain active for `WE` time units.

An email is normalized by converting ASCII uppercase letters to lowercase; all other characters are unchanged. A request accepted at time `s` is active for a quota at time `x` exactly when `s <= x < s + W` for that quota.

An accepted request may later be revoked. Revocation immediately removes any still-active contribution from both quotas. It remains revoked permanently, even if its quota contributions had already expired.

Events with equal timestamps are processed in input order.

For a request event:

1. If `proof` is `0`, output `CAPTCHA`; the request is not recorded.
2. Otherwise, if `cost > BI` or `cost > BE`, output `NEVER`; the request is not recorded.
3. Otherwise, if adding its cost respects both quotas, output `ACCEPT` and record it atomically.
4. Otherwise, output `RETRY z`, where `z` is the earliest integer time at or after its timestamp when this request would respect both quotas, assuming the rejected request is not recorded, no later events occur, and all currently recorded requests expire normally.

For a revocation event naming request ID `id`:

- If that request has already been processed, was accepted, and has not previously been revoked, output `REVOKED` and revoke it.
- Otherwise, output `INVALID` and change nothing.

## Input

The first line contains:

`N WI BI WE BE`

Each of the next `N` lines is one event:

- `R t ip email cost proof` — request
- `X t id` — revocation

Constraints:

- `1 <= N <= 300000`
- `1 <= WI, WE, BI, BE, cost <= 10^18`
- `0 <= t <= 4*10^18`; event timestamps are nondecreasing
- `proof` is `0` or `1`; `id` is a positive integer
- Strings are nonempty ASCII without whitespace
- Total string length is at most `5*10^6`
- Every `t + WI` and `t + WE` fits in `u64`

## Output

Print one result per event using exactly the forms above. Output is compared as ASCII-whitespace-separated tokens.

Submit one complete Rust program using only the standard library and containing `fn main()`.
