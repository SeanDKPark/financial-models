---
description: Inspect project codebase and update docs/architecture.md with an up-to-date Mermaid diagram.
---

Inspect the codebase across `src/finmodels/` (including curves, interpolation, models, pricer, and visualization) and test suites:

- Analyze actual class hierarchies, module boundaries, and data flow contracts.
- Check which curve models inherit from `BaseYieldCurve`.
- Verify input pipelines (market data -> bootstrapper/calibrator -> curve interpolator -> valuation/reporting).
- Update `docs/architecture.md` with a clean, fully accurate Mermaid.js diagram and brief layer explanations.
- Ensure the diagram reflects only what actually exists or is explicitly stubbed in code.
- Report a summary of what changed or was verified in the diagram.
