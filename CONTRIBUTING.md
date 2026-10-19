# Contributing to Sonicproof

Thanks for helping make generated audio easier to trust. Bug reports, fixes, docs and new checks are all welcome.

## Set up

You need [uv](https://docs.astral.sh/uv/), Python 3.11+, and Node.js 20+ if you touch the workbench UI.

```bash
git clone https://github.com/kanaworksai-hue/sonicproof.git && cd sonicproof
uv sync --group dev                  # everything except the model: fast, no PyTorch
uv sync --extra model --group dev    # add PyTorch + sentence-transformers to run the model
uv run sp doctor --load              # downloads EmbeddingGemma 2 (~1.5 GB) once
```

`ffmpeg` on your `PATH` enables the M4A / video code paths and their tests; without it those tests are skipped.

## Tests

There are two tiers:

- **Without the model** — `uv sync --group dev && uv run pytest`. This is what CI runs on Linux, macOS and
  Windows. Tests here must not need the model: use the fake embedders and `monkeypatch` helpers already in
  `tests/`, and keep each test fast.
- **With the model** — after `uv sync --extra model --group dev` and `sp doctor --load`, the same
  `uv run pytest` also runs the model tests (they are skipped when the model or PyTorch is missing).
  Run this tier before changing scoring, calibration, the vocabulary or the embedder.

Tests must never play sound.

## Lint and format

```bash
uv run ruff check src tests examples
uv run ruff format --check src tests examples     # or `uv run ruff format …` to fix
```

## Workbench UI

The UI sources live in `web/` (Vite + TypeScript, no framework). The built bundle in `src/sonicproof/web/` is
shipped inside the Python package, so **if you change anything in `web/`, rebuild and commit the bundle**:

```bash
cd web && npm ci && npm run build      # writes src/sonicproof/web/
npm run dev                            # hot reload; proxies /api to `sp serve` on :7788
```

CI rebuilds the UI and fails if the committed bundle is out of date. User-facing strings go in
`web/src/i18n/en.ts` and `ja.ts`; the type checker makes sure both languages have the same keys.

## What not to commit

- No audio, video or model files. Put test material in `samples/` (ignored by git). Small fixtures that a
  test really needs (< 200 KB, made by you or under a permissive license) can go in `tests/fixtures/`.
- No copyrighted sound effects or music, also not in issues.
- No local paths, hostnames or credentials.

## Pull requests

- One topic per pull request; keep it small enough to review.
- Describe what changed and why, and how you tested it (which tier, which OS).
- Add or update tests for behavior changes, and update `README.md` / the agent skill
  (`src/sonicproof/skill/SKILL.md`) when commands, flags or output change.
- `--json` output and MCP tool results are an interface for scripts and agents: add fields rather than
  renaming or removing them, and call out any breaking change in the pull request.
- Docs, comments and user-facing text are in English (the workbench is also translated to Japanese).
- By contributing you agree that your contribution is licensed under the [MIT License](https://github.com/kanaworksai-hue/sonicproof/blob/main/LICENSE).
