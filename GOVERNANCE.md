# HKL-1 Project Governance

This document describes the governance model for the HKL-1 project. It is designed to be lightweight, transparent, and welcoming to contributors of all backgrounds.

---

## 🌟 Project Values

- **Excellence**: We maintain zero warnings, zero dependencies, and high test coverage.
- **Transparency**: Decisions are made in public on GitHub issues and discussions.
- **Inclusivity**: We welcome contributors regardless of experience, background, or identity.
- **Pragmatism**: We prioritize working, tested code over architectural purity.

---

## 👥 Roles

### Users
Anyone who uses HKL-1 in their projects. Users are encouraged to:
- Open issues for bugs or feature requests
- Participate in discussions
- Share their use cases and experiences

### Contributors
Anyone who contributes to the project through issues, discussions, documentation, code, or testing. Contributors:
- Follow the [code of conduct](CODE_OF_CONDUCT.md)
- Adhere to [contributing guidelines](CONTRIBUTING.md)
- May be invited to become committers based on sustained contributions

### Committers
Contributors with direct commit access to the repository. Committers:
- Review and merge pull requests
- Triage issues
- Ensure code quality and test coverage
- Are expected to participate in project discussions
- Can be nominated by existing committers

### Maintainers
The core team responsible for the project's direction and health. Maintainers:
- Set the technical roadmap
- Make final decisions on contentious issues
- Manage releases and versioning
- Onboard new committers
- Handle security reports

---

## 🗳️ Decision Making

### Consensus-Based
Decisions are made by consensus among committers and maintainers. We strive for:

1. **Discussion**: Open an issue or discussion
2. **Proposal**: Clear description of the proposed change
3. **Feedback**: Minimum 72 hours for comments
4. **Consensus**: Agreement among active participants
5. **Decision**: If consensus cannot be reached, maintainers decide

### Voting
When consensus cannot be reached:
- Committers and maintainers vote
- Simple majority wins
- Maintainers have veto power for project-critical decisions

### Areas of Autonomy

| Area | Decision Maker |
|---|---|
| Bug fixes | Committer review + merge |
| New features | Issue discussion + committer consensus |
| API changes | Maintainer approval required |
| Breaking changes | RFC-style proposal + maintainer consensus |
| Release schedule | Maintainers |
| Governance changes | Maintainers + committer majority |

---

## 📋 Release Process

HKL-1 follows [Semantic Versioning 2.0.0](https://semver.org/):

- **Patch** (0.1.x): Bug fixes, documentation, test additions
- **Minor** (0.x.0): New features, non-breaking API changes
- **Major** (x.0.0): Breaking changes, significant architecture shifts

### Pre-Release (Current)
While HKL-1 is in pre-1.0 development (0.x.x):
- Breaking changes are allowed in minor versions
- Deprecation notices should be provided when possible
- Major milestones are documented in [CHANGELOG.md](CHANGELOG.md)

### Release Checklist
1. All tests pass: `cargo test --lib`
2. Zero clippy warnings
3. CHANGELOG updated
4. Version bumped in `Cargo.toml`
5. Tagged in git: `git tag v0.x.x`
6. GitHub Release created with release notes

---

## 🤝 Contribution Ladder

```
User → Contributor → Committer → Maintainer
```

### Contributor → Committer
- 5+ merged PRs
- Consistent code quality
- Active in discussions/reviews
- Nominated by a committer or maintainer
- Unanimous committer approval

### Committer → Maintainer
- Sustained contribution over 6+ months
- Deep understanding of the codebase
- Leadership in project discussions
- Nominated by a maintainer
- Unanimous maintainer approval

---

## 🔒 Conflict Resolution

1. **Direct conversation**: Discuss with the involved parties
2. **Mediation**: Involve a third committer
3. **Maintainer decision**: If unresolved, maintainers decide
4. **Escalation**: Governance changes require full maintainer consensus

---

## 📝 Changes to Governance

Proposals to change governance follow the same process as major decisions:
1. Open an issue with the governance change proposal
2. 1-week comment period
3. Committer vote (majority)
4. Maintainer approval (unanimous)

---

## 📬 Contact

- **Technical discussions**: [GitHub Issues](https://github.com/Hakille-ai/HKL-1/issues)
- **Community discussions**: [GitHub Discussions](https://github.com/Hakille-ai/HKL-1/discussions)
- **Security**: [SECURITY.md](SECURITY.md)
- **Maintainers**: maintainers@hkl1.dev

---

<p align="center">
  <sub>This governance model is adapted from the <a href="https://github.com/rust-lang/rfcs">Rust RFCs</a> process and the <a href="https://www.contributor-covenant.org/">Contributor Covenant</a>.</sub>
</p>
