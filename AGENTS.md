# AGENTS.md

Repository-specific instructions for automated coding agents.

## Questions vs. Coding
- If the user or developer is asking a question (including clarification), answer the question directly first.
- Do not start editing files or running a coding session unless the user explicitly asks you to implement a change (or clearly requests code).
- If you think code changes might be helpful, explain the idea briefly and ask for confirmation before modifying the repo.

## Development Environment (Python)
- Always run project tooling (scripts, tests, linters, `pyright`, etc.) inside the repo virtual environment at `.venv` (activate it, or call binaries via `.venv/bin/...`).
- Prefix Python tooling with `PYTHONPATH=.` (for `pytest`, `coverage`, `python -m build`, etc.) so editable-install shims resolve correctly (pytest bootstrap plugin, Open WebUI shims, etc.).
- Example:
  - `source .venv/bin/activate && PYTHONPATH=. pytest tests/test_pipe_guards.py`

## Coding Style
- Use the existing Python style: 4-space indents and typed signatures.
- Naming:
  - Helpers: `snake_case`
  - Models/types: `PascalCase`
  - Constants: `UPPER_CASE` near the manifest docstring
- Keep import grouping: stdlib → third-party → Open WebUI.
- Comment only when the logic is non-obvious.
- Prefer linting with `ruff` or `flake8`.
- Avoid formatters that would rewrite the manifest header or valve tables.

## Testing
- The test suite uses `pytest` + `pytest_asyncio`.
- Test file naming: `tests/test_<feature>.py` (extend the subsystem you touched, e.g. `test_streaming_queues.py`, `test_pipe_guards.py`).
- Reuse fixtures from `tests/conftest.py` (fake Redis, ULIDs, FastAPI requests, etc.).
- Workflow:
  - Run the touched test file first.
  - Then run the whole suite: `PYTHONPATH=. .venv/bin/pytest tests -q`
  - Update coverage with `PYTHONPATH=. .venv/bin/coverage run -m pytest` before editing anything under `coverage_annotate/`.

## Error Checks (After Edits)
- After making edits, also run quick error checks:
  - Bytecode compile check: `.venv/bin/python -m compileall .`
  - Type check (Pylance/pyright): `.venv/bin/pyright`

## Commits & PRs
- Use lightweight Conventional Commits: `fix:`, `feat:`, `chore:`.
- Keep subjects imperative, under ~72 characters, and scoped.
- Never run `git commit` until the user approves the exact commit message/details.
  - If the user says “commit” but hasn’t approved a message yet, propose the subject + short body (what/why) and ask for approval before committing.
- Do not include a `Tests:` section in commit messages.
- Do not include verification notes in commit messages (e.g. “checks run”, “tested with”, commands, etc.); keep commits to change descriptions only.
- PRs should:
  - Explain the behavior change.
  - List valves/docs touched.
  - Paste the exact test commands or Open WebUI flows exercised.
  - Update relevant docs in the same PR.

## Backups (Before Editing Files)
- Before editing/overwriting/renaming/deleting any file, create a snapshot under `backups/`.
- Naming format: `backups/<relative-path>-YYYY-MM-DD-HH-MM-SS` (24-hour clock).
- Create a new backup each edit session.

## Local Reference Material
- `.external/` is read-only reference material at the repo root.
- Use it to:
  - Browse Open WebUI sources: `.external/open-webui`
  - Consult OpenRouter docs: `.external/openrouter_docs`
  - Inspect latest `/models` dump: `.external/models-openrouter-*.json`
  - Check the live model catalog JSON: `https://openrouter.ai/api/frontend/models`

## Security & Configuration
- Never commit real secrets, api keys, passwords, usernames, real names, PII data.
- If changing SSRF guards, artifact persistence, breakers, or adding new valves:
  - Add/extend tests (e.g. `test_security_methods.py`, `test_artifact_helpers.py`).
  - Include a short operator note.
  - Capture new valve defaults and migrations in the valve atlas before shipping.
