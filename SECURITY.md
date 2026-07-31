# HKL-1 Security Policy

**TL;DR**: We take security seriously. Please report vulnerabilities privately to **security@hkl1.dev**. We aim to respond within 48 hours.

---

## 🔐 Supported Versions

| Version | Supported |
|---|---|
| Latest (main branch) | ✅ Active development — security fixes applied immediately |
| 0.x.x (pre-release) | ⚠️ Best effort — upgrade to latest commit |
| Older versions | ❌ Not supported |

---

## 🛡️ What We Consider a Security Issue

- **Memory safety**: Unsound `unsafe` code, buffer overflows, use-after-free
- **Cryptographic flaws**: Weaknesses in ChaCha20, HMAC-SHA256, PRNG (XorShift64Star)
- **PUF key extraction**: Attacks on device-unique key generation or ephemeral storage
- **Flash persistence**: Unauthorized read/write of dump slots, checksum bypass
- **Boot integrity**: Bypass of emergency checks, unauthorized OTA rollback
- **Secure erase**: Incomplete erasure allowing data recovery
- **Swarm security**: Injection of malicious weight deltas, peer impersonation

### What is NOT a Security Issue

- Failing `cargo test` (open a regular [issue](https://github.com/Hakille-ai/HKL-1/issues))
- Compilation errors on unsupported platforms
- Feature requests for additional cryptographic algorithms

---

## 📧 Reporting a Vulnerability

### Private Reporting

**Please do NOT report security vulnerabilities in public issues.**

Send an email to **security@hkl1.dev** with:

```
Subject: [HKL-1 Security] Brief description

Body:
- Type of vulnerability
- Affected component(s) (file + line numbers)
- Description of the issue
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if known)
```

### PGP Key

```
Key ID:      0x0000000000000000
Fingerprint: 0000 0000 0000 0000 0000  0000 0000 0000 0000 0000
Download:    https://hkl1.dev/security-pgp.asc
```

> *(Note: PGP key will be published when the project is released. For now, email is sufficient.)*

### Response Timeline

| Timeframe | Action |
|---|---|
| 48 hours | Acknowledgment of receipt |
| 7 days | Initial assessment and severity classification |
| 30 days | Fix developed and tested |
| 60 days | Public disclosure (coordinated with reporter) |

---

## ⚙️ Responsible Disclosure

We follow a coordinated disclosure process:

1. **Report** → You report privately
2. **Triage** → We assess severity and impact
3. **Fix** → We develop and test a fix
4. **Release** → We push a fix (patch release or main branch)
5. **Disclosure** → We publish a security advisory describing the issue (after fix is available)

We ask reporters to wait for the fix to be released before public disclosure.

---

## 🏆 Recognition

We maintain a **Security Hall of Fame** of researchers who responsibly disclose vulnerabilities:

| Researcher | Finding | Date |
|---|---|---|
| *(Be the first!)* | | |

If you'd like to remain anonymous, let us know and we will respect your wishes.

---

## 🔒 Security Features

HKL-1 includes these built-in security mechanisms:

| Feature | Description |
|---|---|
| **ChaCha20 Encryption** | Optional flash dump encryption, 256-bit key |
| **PUF** | Device-unique key from SRAM/ring-oscillator, never stored |
| **Ephemeral Key Manager** | Keys erased on power loss |
| **Secure Erase** | XOR overwrite + verification |
| **HMAC-SHA256** | Firmware integrity validation (OTA) |
| **CRC32** | Dump integrity checksums |
| **Watchdog** | Graduated recovery on anomaly detection |
| **Entropy Monitor** | Detects low-entropy (stagnation) states |
| **Emergency Checks** | Boot-time system integrity validation |
| **Memory Protection** | MPU configuration (STM32F7) |

---

## 📚 Related Documentation

- [Architecture: Crypto](../docs/core.md#crypto) — ChaCha20, PUF, HMAC-SHA256
- [System: Persistence](../docs/system.md#persistence) — Secure erase, flash slots
- [System: Boot](../docs/system.md#boot) — Emergency checks
- [System: Watchdog](../docs/system.md#watchdog) — Graduated recovery
- [Safety: Hardware Resilience](../docs/safety.md#hardware-resilience) — ECC, migration

---

## 🗓️ Security Updates

Security advisories are published at:

- **GitHub Security Advisories**: https://github.com/Hakille-ai/HKL-1/security/advisories
- **Mailing list**: security-announce@hkl1.dev (coming soon)

---

<p align="center">
  <sub>Thank you for helping keep HKL-1 and its users safe.</sub>
</p>
