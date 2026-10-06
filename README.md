# Sleishm4n.github.io

Personal site for Sam Leishman, served at <https://sleishm4n.github.io>.

## Layout

```text
index.html            # root (redirects to /mentats for now)
mentats/              # showcase page for the mentats Rust NN library
  index.html, style.css, app.js
  img/                # gallery images copied from mentats/images
  data/*.json         # all numbers on the page come from here
tools/mentats/        # (planned) scripts that generate mentats/data/*.json
```

## Data files (`mentats/data/`)

| File | Contents | Produced by |
|------|----------|-------------|
| `models.json` | per-model test accuracy + per-epoch loss/accuracy/time | `tools/mentats/collect_metrics.py` (seeded from README) |
| `kernels.json` | Criterion medians for matmul/conv/layout kernels + hardware | `tools/mentats/criterion_to_json.py` |
| `history.json` | tracked benchmarks re-measured at each milestone commit | `tools/mentats/backfill.py` |
| `reference.json` | mentats vs single-thread PyTorch | `tools/mentats/reference/` |
| `stats.json` | LOC, tests, coverage, deps | `.github/workflows/pages.yml` |

Every file has a `"placeholder"` flag. When it's `true`, the page shows a
"placeholder data" badge on that section.