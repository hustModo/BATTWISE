# Installation

## System requirements

- Python 3.10 is the observed application/benchmark interpreter version.
- Node.js 20 is required for the React development server or frontend build.
- Linux is the validated server platform. Other operating systems have not been
  validated for the process-isolation and shell-script integration.
- A CPU is sufficient for the offline CDDI demo and deterministic code tests.
- An LLM service and the user's own credentials are required for interactive
  algorithm discovery. They are not required for the offline demo.
- PyTorch is optional for FC-VAE/DECM; it is not required for the CDDI demo.

## Create a separate environment

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root requirements describe the observed code-validation environment. They
supersede the older backend pins retained in `app/backend/requirements.txt` for
provenance. Do not install a source tree's historical requirements into a running
production environment merely to inspect this release.

## Frontend

```bash
cd app/frontend
npm ci
npm run build
cd ../..
```

The production build can be served by FastAPI without a separate Vite process.
For development, see `APPLICATION.md`.

## Optional deep-representation experiments

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-benchmark.txt
```

The optional file adds PyTorch to the observed core environment. See
`REPRODUCIBILITY.md` for the distinction between currently validated package
metadata and a fully reconstructed historical formal environment.
Do not describe an untested dependency change as an exact reproduction.

## Time and hardware reporting

Installation time depends on package caches, network access, and the selected
PyTorch wheel. A clean installation time has not been measured. Do not substitute
the demo runtime for an installation-time claim. Validation times measured for
this distribution are recorded in `VALIDATION.md` when available.
