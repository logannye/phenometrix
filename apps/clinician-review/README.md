# Clinician review surface

No separate clinician-review application is implemented.

`apps/capture-web` renders a deterministic 27-outcome report for each of two
live sessions and, after explicit in-memory reference acceptance, a fixed
six-row unilateral facial movement research card. That page includes **Accept
research card** and **Dismiss research card**, but these controls only change an
unauthenticated value in the current page's memory. They are not clinician
approval, signature, governed adjudication, or a durable audit record.

The card contains no generated narrative or clinical interpretation. It shows
raw current-minus-reference differences only for strictly compatible measured
sources; withheld and incompatible rows have no delta. There is no persistence,
export, EHR integration, or separate review workflow. This directory remains a
documentation-only marker for the future boundary of the third named
capability.
