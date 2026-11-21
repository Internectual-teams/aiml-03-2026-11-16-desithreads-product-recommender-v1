## Running tests

From the repository root, install the locked dependencies:

```powershell
uv sync --locked
```

Run all tests:

```powershell
uv run pytest -v
```

Alternatively, activate the project environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
pytest
```

The tests use a small hand-made similarity matrix and temporary CSV/NumPy
files. They do not require real product data or a running Flask server.

Coverage includes ranking, self-exclusion, unique IDs, result limits,
fewer available products, ties, product details, and HTTP 200/400/404
responses.