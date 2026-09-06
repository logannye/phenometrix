# Contributing

Thank you for contributing to PhenoMetrix.

## Before opening a change

1. Read the [architecture](docs/architecture.md) and
   [safety requirements](docs/safety.md), then review the
   [development baseline](docs/development-baseline.md). Historical designs in
   `docs/superpowers/` preserve earlier decisions and may describe superseded
   behavior; use the baseline and current architecture for implementation status.
2. State the intended use of the change.
3. Identify which of the three product capabilities it strengthens.
4. Document expected failures, quality requirements, and validation status.
5. Use synthetic data only.

## Pull requests

Every pull request should explain:

- the user or workflow problem;
- the affected system boundary;
- the data consumed and emitted;
- consent and retention implications;
- new clinical or safety claims, if any;
- how the change was checked.

Any agent activity shown in the interface must correspond to a real event,
decision, action, or verified outcome. Do not add simulated chain-of-thought or
decorative activity that cannot be audited.

If a change does not improve Ambient Capture, Personal Trajectory, Clinician
Evidence Card, or a required safety foundation, leave it out of the MVP.

Changes introducing a consequential clinical decision must include a human
review gate and may not execute the decision automatically.

## Checks

```bash
pnpm verify
```

This runs all automated gates, including the isolated Python service tests.
The individual commands are:

```bash
pnpm test
pnpm test:browser
pnpm test:voice
```

`pnpm test` runs the structure and asset validator, unit tests, TypeScript
typechecking, and the production build. Browser tests verify the ambient
lifecycle separately. Run the Python command when the optional research
sidecar is affected.
