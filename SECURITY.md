# Security policy

## Prototype status

This repository is a research prototype and is not approved for production
health data or protected health information.

## Do not submit

- real patient data;
- camera or microphone recordings;
- medical records or identifiers;
- credentials, tokens, or private endpoints;
- security reports in public issues.

## Reporting a vulnerability

Report suspected vulnerabilities privately through GitHub's security advisory
feature for this repository. Include reproduction steps, affected components,
and the potential impact. Do not include real health data.

## Baseline expectations

- least-privilege access;
- encrypted transport and storage;
- explicit retention and deletion policies;
- immutable audit events;
- scoped service identities;
- dependency and secret scanning before production use;
- an HTTP response CSP (including `frame-ancestors`) from the deployment edge;
- threat modeling for media capture, prompt injection, model supply chain, and
  cross-tenant data access.
