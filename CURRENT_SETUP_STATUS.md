# Current setup status — Topstep 50K

**As of 2026-09-10: research/paper only.  No live configuration is approved.**

The previous 84% pass-rate recommendation is retired.  It did not survive the
corrected-data and current-rule audit.  Do not deploy the accelerated ML config,
the ORB/MSITE/GIRE sizes, or the weekend-hold book based on older reports.

The only currently defensible candidate is a sparse one-contract paper book:

- Monday MNQ, long 09:31-15:59 ET;
- pre-FOMC MNQ, long 18:01 ET the prior day through 13:55 ET decision day;
- $250 protective stop per contract;
- no session-to-session carry and no position through the FOMC announcement.

It has positive historical evidence, but **not** a high probability of passing
within three months.  The full audit, path estimates, invalidated candidates,
and reproducible commands are in `EDGE_AUDIT_2026-09-10.md`.
