# VAPA public core release `public_core_v1`

Version `1.0.0`

This package contains public interoperability contracts for the VAPA method: prompt
templates and closed JSON schemas for calculator, verifier, and task manifests. Every
asset is authenticated by `manifest.json`.

This is explicitly a **non-paper-exact** release. It does not contain the unreleased
author verifier predicates, reliability weights, calculator catalog, task programs,
model revisions, or protected patient records. A downstream run must not relabel these
assets as paper-exact.

The assets are ordinary package data and are loaded with `importlib.resources` by
`vapa.releases.load_public_core_release`.
