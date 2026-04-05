# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python web app for image color analysis. Core backend and plotting logic live in [`color_analysis.py`](/Users/marat/Documents/6-CODE/COLOR/color_analysis.py). The browser UI is in [`web/index.html`](/Users/marat/Documents/6-CODE/COLOR/web/index.html), [`web/collection.html`](/Users/marat/Documents/6-CODE/COLOR/web/collection.html), and [`web/styles.css`](/Users/marat/Documents/6-CODE/COLOR/web/styles.css). Use [`run_web.sh`](/Users/marat/Documents/6-CODE/COLOR/run_web.sh) as the standard local entry point. Keep new server logic in Python helpers instead of expanding the request handler with large inline blocks.

Single-image analysis and collection analysis share the same backend module. Collection analysis is expected to expose a richer chart set, and frontend chart cards should support click-to-enlarge previews rather than static thumbnails only.

## Build, Test, and Development Commands
- `./run_web.sh` starts the local web app on `127.0.0.1:8000`.
- `HOST=0.0.0.0 PORT=9000 ./run_web.sh` runs the server on a custom host or port.
- `python3 color_analysis.py ./images` runs batch analysis against a folder of JPG files.
- `python3 -m py_compile color_analysis.py` performs a quick syntax check.

The project currently relies on `numpy`, `matplotlib`, and `Pillow`. `scikit-image` is optional and improves Lab conversion accuracy.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and `snake_case` for functions, variables, and file names. Keep plotting helpers focused on one chart each, and keep configuration constants near the top of [`color_analysis.py`](/Users/marat/Documents/6-CODE/COLOR/color_analysis.py). Frontend code should stay dependency-free unless there is a strong reason to add a build step.

When doing NumPy channel math, avoid patterns that compute invalid branches eagerly. Prefer `np.divide(..., where=...)` or equivalent masked operations over `np.where(condition, safe_value, risky_expression)` when the risky branch can emit divide-by-zero or invalid-value warnings.

## Testing Guidelines
There is no formal automated test suite yet. Minimum verification for changes:

- `python3 -m py_compile color_analysis.py`
- Start the app with `./run_web.sh`
- Upload one sample image and confirm both preview and analysis panel render
- Open the single-image chart lightbox and confirm enlarged preview works
- Upload one sample folder in the collection page and confirm the expected chart count renders and enlarged previews work

If automated tests are added later, place them under `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
Use concise Conventional Commit-style messages such as `feat: add upload validation` or `fix: stabilize luma chart rendering`. Pull requests should describe user-visible changes, list verification steps, and include screenshots when the UI or chart output changes.

## Configuration & Generated Files
Do not commit `__pycache__/`, `.mplconfig/`, or generated analysis images. Keep machine-specific paths out of the codebase; prefer relative paths and environment variables like `HOST` and `PORT`.
