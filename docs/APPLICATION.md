# Running the application

From a new environment with dependencies installed:

```bash
export BATRECYCLER_PYTHON="$(python -c 'import sys; print(sys.executable)')"
bash deploy/run_dev.sh backend
```

In another terminal:

```bash
bash deploy/run_dev.sh frontend
```

Open `http://127.0.0.1:5173`. The API listens on `127.0.0.1:8765`.
If the frontend has been built, `http://127.0.0.1:8765` also serves the UI.
Backend auto-reload is disabled because Designer/Compute intentionally write
runtime Python files. Restart manually after changing backend source.

## Credentials

Copy `.env.example` to `.env` and fill it locally, or use the application's
settings panel. Keep keys out of source, command history, commits, issues, and
documentation. The encrypted settings store and its encryption key are runtime
files under `app/backend/workspace/config/`; neither is shipped here.

Provider/model values are runtime configuration, not a claim that a particular
external model alias remains available. Configure a model offered by your provider.

## Data and outputs

The built-in First-200s/All-200 dataset locations are described in
`DATA_FORMAT.md`. They are empty in this distribution. Opening the UI does not
supply measured research data. Full pipeline actions require compatible input
files or a user-uploaded dataset.

New outputs are session-local. Code downloads are named `cddi_function.py`; result
tables use `CDDI`, and merged provenance uses `CDDI_source`. Historical DDIA/MMD
columns and route values remain accepted for compatibility.

The application has no production authentication layer. The launch instructions
bind it to loopback for local or SSH-forwarded research use.
