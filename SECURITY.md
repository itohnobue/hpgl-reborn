# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.6.x   | :white_check_mark: |
| 1.5.x   | :white_check_mark: |
| < 1.5.0 | :x:                |

## Reporting a Vulnerability

**Do not open a public issue.** Instead, email the maintainers directly at:

- **hpgl@ufanipi.ru**

Please include:

- A detailed description of the vulnerability
- Steps to reproduce (minimal Python script preferred)
- Affected HPGL version and platform (OS, Python/NumPy versions)
- Any suggested fix (if available)

We will acknowledge your report within 5 business days and aim to publish a fix within 30 days. We will credit you in the release notes unless you request anonymity.

## Security Design

HPGL Reborn incorporates the following security measures:

### Path Traversal Prevention

All file I/O functions (`load_cont_property`, `write_property`, `LoadGslibFile`, etc.) route paths through `PathValidator.validate_filepath` in `src/geo_bsd/validation.py`, which:

- Rejects paths containing `..` before normalization
- Resolves all paths to absolute form
- Verifies resolved paths are within allowed directories (when `basedir` is specified)
- Rejects paths outside the package directory

### Safe Native Library Loading

`hpgl_wrap._safe_load_library` and `cvariogram` load the C++ shared library with directory-containment validation, preventing DLL sideloading and symlink escapes (CVSS 7.8).

### Use-After-Free Prevention

All ctypes structures that the C++ backend holds pointers to have Python-side array references pinned via `_array_refs` to prevent premature garbage collection.

### Stale Error Suppression

The `_snapshot_hpgl_error` / `_check_hpgl_error` mechanism (with thread-safe locking) prevents stale C++ exception messages from propagating across unrelated function calls.

### Memory Safety

- GSLIB file parser caps property count (`num_p`) to prevent memory exhaustion from malicious headers.
- Grid size validation enforces a 1-billion-cell maximum to prevent OOM on malicious input.
- `read_inc_file_float` / `read_inc_file_byte` validate element counts against `c_int` max.

## OWASP Coverage

| Threat | Status |
|--------|--------|
| A01: Broken Access Control | N/A (library, not a service) |
| A02: Cryptographic Failures | N/A |
| A03: Injection | Path traversal guarded; no SQL/command execution |
| A04: Insecure Design | N/A |
| A05: Security Misconfiguration | N/A |
| A06: Vulnerable Components | Dependencies managed via `uv.lock` |
| A07: Auth Failures | N/A |
| A08: Software & Data Integrity | Library loading validated |
| A09: Logging & Monitoring | Python `logging` module; configurable output handlers |
| A10: SSRF | N/A |
