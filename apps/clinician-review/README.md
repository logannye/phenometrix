# Clinician evidence panel

This directory implements an embeddable treatment-alignment evidence panel and
a standalone **synthetic** preview. It is a research surface; it does not
interpret measurements as disease severity or treatment efficacy.

`mountTreatmentResponsePanel(element, {scope, load, onUnavailable})` mounts the
panel in a host-provided chart element. The host supplies its existing signed
clinical session and loader; `packages/encounter-client` provides a service-backed
loader. No extra patient login, capture workflow, mandatory provider review, or
patient-facing retry prompt is added.

The panel displays calendar points or days since a qualified treatment anchor,
native-unit values, the frozen pretreatment reference, phase coverage, explicit
exclusions, source/quality details, and supplied review status. Unknown treatment
timing keeps calendar observations and hides treatment alignment. It does not
draw fitted curves or fill gaps. Scope, digest, and source consistency are
validated before rendering. Call `panel.refresh()` on initial chart load and
existing chart refresh events; a response with pending analysis clears the old
panel. Unavailable results fail quietly through host telemetry.

The encounter service exposes authenticated durable review writes bound to an
exact run and content digest. This panel currently reads supplied reviews; the
host wires its review action to that API. The preview does not authenticate a
clinician or persist review decisions.

## Synthetic preview

```bash
pnpm dev:evidence
```

Open `http://127.0.0.1:4175`. The bundled HFS measurements are fabricated fixtures,
not recordings, clinical data, or evidence of model performance. The preview
illustrates two facial rows; the service HFS protocol selects four existing
left/right eye aperture and lid closure completeness measurements. These do not
quantify spasms.

## Verification

```bash
pnpm --filter @phenometrix/clinician-review test:unit
pnpm --filter @phenometrix/clinician-review typecheck
pnpm --filter @phenometrix/clinician-review test:browser
pnpm --filter @phenometrix/clinician-review build
```

See the [integration runbook](../../docs/encounter-integration.md) and
[validation boundaries](../../docs/validation.md) for the host/service setup and
remaining real-hardware, institutional, and clinical requirements.

The original `apps/capture-web` six-row unilateral facial movement demo remains
separate. Its **Accept research card** / **Dismiss research card** controls
change only page memory; they are not authenticated or durable reviews.
