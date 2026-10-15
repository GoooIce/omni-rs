# Working in omni-rs

- Start with [README.md](README.md) for entry points and [CODING_STANDARDS.md](CODING_STANDARDS.md) for design decisions.
- Use [docs/agent-workflow.md](docs/agent-workflow.md) for checks, bounded searches and interruption recovery. The manifests in [scripts/checks.json](scripts/checks.json) are independent Cargo projects; a root Cargo command does not cover all of them.
- Native behavior and existing regression commands live in [prototypes/makepad/README.md](prototypes/makepad/README.md). Numerical and hardware evidence lives in [VALIDATION.md](VALIDATION.md) and the relevant module document under `docs/`.
- Continue within the user's approved scope. Preserve accepted recommendations and completed checks across turns; ask only when an unresolved choice changes that scope or observable behavior.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in this repo, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage vocabulary: `needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
