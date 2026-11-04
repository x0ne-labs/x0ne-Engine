---
name: Bug report
about: Something refuses when it should serve, or the reverse
labels: bug
---

**Do not report a vulnerability here** — see `SECURITY.md`.

**What happened**
The command, the exit code, and the last lines of stderr. Exit codes are
distinct on purpose (`x0ne-infer --help` lists them).

**What you expected**

**Environment**
- x0ne-infer version or commit:
- Machine type / cloud (e.g. GCP a3-highgpu-1g, TDX):
- `--device` (`cuda` or `cuda:0`):
- NVIDIA driver version, if a GPU is involved:

**Is the machine a genuine confidential VM?**
A refusal on a machine that is not one is the intended behaviour, not a bug.
