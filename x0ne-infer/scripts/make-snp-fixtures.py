#!/usr/bin/env python3
"""TEST certificate chains for x0ne-infer's SEV-SNP verifier (src/snp.rs).

NOT AMD. Writes, into <out-dir>, DER certificates shaped like AMD's KDS
chain (ARK -> ASK -> VCEK, RSASSA-PSS SHA-384 salt 48, a P-384 VCEK carrying
AMD's extensions under 1.3.6.1.4.1.3704) so the tests can sign reports the
real chain never would: debug, a migration agent, a low TCB, another session,
and CRLs AMD never issued (a revoked ASK or ASVK, an expired window, another
signer).

  test_genoa_ark.der, test_genoa_ask.der     a TEST chain for product Genoa
  test_turin_ark.der, test_turin_ask.der     a TEST chain for product Turin
  test_genoa_vcek_current.der                Genoa, TCB at the default minimum or above
  test_genoa_vcek_old.der                    Genoa, TCB below it (SNP 0x17, microcode 0x54)
  test_turin_vcek.der                        Turin layout (FMC byte, 8-byte hwID)
  test_genoa_vcek_rogue.der                  the current VCEK's key and subject, signed by
                                             an ASK of the same name under another ARK
  test_genoa_asvk.der                        the VLEK signing key (ASVK) of the Genoa chain
  test_genoa_vlek.der                        a VLEK under it, Genoa, current TCB, cspId
                                             "CN=test-csp.x0ne.invalid", no hwID
  test_genoa_crl.der                         the Genoa ARK's CRL, 2026-01-27 to 2026-03-10,
                                             one unrelated serial (0x0F) revoked
  test_genoa_crl_ask_revoked.der             the same window, the ASK (serial 2) revoked
  test_genoa_crl_asvk_revoked.der            the same window, the ASVK (serial 3) revoked
  test_genoa_crl_expired.der                 2025-11-01 to 2026-01-01
  test_genoa_crl_rogue.der                   test_genoa_crl.der's content, signed by the
                                             other ARK of the same name
  test_genoa_crl_critical.der                the same window, an unknown critical extension
  test_turin_crl.der                         the Turin ARK's CRL, nothing revoked

The VCEK private keys are fixed public scalars (SCALARS below), repeated in
src/snp.rs's tests, which sign reports with them: no private key file is
written. The RSA keys of the ARKs and ASKs are random and discarded, so a
re-run writes other certificates; the tests depend only on the files.

usage: make-snp-fixtures.py <out-dir>   (python3 with cryptography >= 41)
"""
import datetime
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import NameOID

AMD = "1.3.6.1.4.1.3704.1."
PSS = padding.PSS(mgf=padding.MGF1(hashes.SHA384()), salt_length=48)
SCALARS = {"current": 0x5A, "old": 0x3C, "turin": 0x6B, "vlek": 0x7C}
GENOA_CHIP_ID = bytes((0xC0 + i) % 256 for i in range(64))
TURIN_HWID = bytes.fromhex("7a1e5c266c0108db")


def when(y, m, d):
    return datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc)


def name(cn):
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "x0ne-Engine tests"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TEST fixture, not Advanced Micro Devices"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def der_int(v):
    body = v.to_bytes(max(1, (v.bit_length() + 8) // 8), "big")
    return bytes([0x02, len(body)]) + body


def ia5(s):
    return bytes([0x16, len(s)]) + s.encode("ascii")


def ext(oid, value):
    return x509.UnrecognizedExtension(x509.ObjectIdentifier(AMD + oid), value)


def ca(subject, issuer, key, issuer_key, serial, pathlen):
    b = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(when(2025, 1, 1))
        .not_valid_after(when(2045, 1, 1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=pathlen), critical=False)
        .add_extension(
            x509.KeyUsage(False, False, False, False, False, True, True, False, False),
            critical=False,
        )
    )
    return b.sign(issuer_key, hashes.SHA384(), rsa_padding=PSS)


def chain(product):
    ark_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    ask_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    ark_name, ask_name = name(f"TEST-ARK-{product}"), name(f"TEST-SEV-{product}")
    ark = ca(ark_name, ark_name, ark_key, ark_key, 1, None)
    ask = ca(ask_name, ark_name, ask_key, ark_key, 2, 0)
    return ark, ask, ask_name, ask_key, ark_name, ark_key


def crl(issuer_name, issuer_key, this_update, next_update, revoked, critical_ext=False):
    b = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer_name)
        .last_update(this_update)
        .next_update(next_update)
        .add_extension(x509.CRLNumber(1), critical=False)
    )
    for serial in revoked:
        b = b.add_revoked_certificate(
            x509.RevokedCertificateBuilder().serial_number(serial).revocation_date(when(2026, 1, 20)).build()
        )
    if critical_ext:
        b = b.add_extension(x509.UnrecognizedExtension(x509.ObjectIdentifier("1.3.6.1.4.1.3704.99.1"), b"\x05\x00"), critical=True)
    return b.sign(issuer_key, hashes.SHA384(), rsa_padding=PSS)


def vcek(scalar, issuer_name, issuer_key, product, spl, hwid, serial, csp_id=None):
    key = ec.derive_private_key(int.from_bytes(bytes([scalar]) * 48, "big"), ec.SECP384R1())
    b = (
        x509.CertificateBuilder()
        .subject_name(name("TEST-SEV-VLEK" if csp_id else "TEST-SEV-VCEK"))
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(when(2025, 1, 1))
        .not_valid_after(when(2032, 1, 1))
        .add_extension(ext("1", der_int(0)), critical=False)
        .add_extension(ext("2", ia5(product)), critical=False)
    )
    for oid, v in spl.items():
        b = b.add_extension(ext(oid, der_int(v)), critical=False)
    if csp_id is None:
        b = b.add_extension(ext("4", hwid), critical=False)
    else:
        b = b.add_extension(ext("5", ia5(csp_id)), critical=False)
    return b.sign(issuer_key, hashes.SHA384(), rsa_padding=PSS)


def genoa_spl(bl, tee, snp, ucode):
    # Genoa and Milan: blSPL .3.1, teeSPL .3.2, spl_4..7 .3.4-.3.7, snpSPL .3.3, ucodeSPL .3.8
    return {"3.1": bl, "3.2": tee, "3.4": 0, "3.5": 0, "3.6": 0, "3.7": 0, "3.3": snp, "3.8": ucode}


def turin_spl(fmc, bl, tee, snp, ucode):
    # Turin: fmcSPL .3.9, then the same names; no .3.4 (byte 3 is SNP)
    return {"3.9": fmc, "3.1": bl, "3.2": tee, "3.3": snp, "3.5": 0, "3.6": 0, "3.7": 0, "3.8": ucode}


def main(out):
    def write(file, cert):
        with open(f"{out}/{file}", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.DER))

    def write_crl(file, c):
        with open(f"{out}/{file}", "wb") as f:
            f.write(c.public_bytes(serialization.Encoding.DER))

    g_ark, g_ask, g_ask_name, g_ask_key, g_ark_name, g_ark_key = chain("Genoa")
    t_ark, t_ask, t_ask_name, t_ask_key, t_ark_name, t_ark_key = chain("Turin")
    write("test_genoa_ark.der", g_ark)
    write("test_genoa_ask.der", g_ask)
    write("test_turin_ark.der", t_ark)
    write("test_turin_ask.der", t_ask)
    write(
        "test_genoa_vcek_current.der",
        vcek(SCALARS["current"], g_ask_name, g_ask_key, "Genoa", genoa_spl(0x0A, 0, 0x1B, 0x58), GENOA_CHIP_ID, 10),
    )
    write(
        "test_genoa_vcek_old.der",
        vcek(SCALARS["old"], g_ask_name, g_ask_key, "Genoa", genoa_spl(0x0A, 0, 0x17, 0x54), GENOA_CHIP_ID, 11),
    )
    write(
        "test_turin_vcek.der",
        vcek(SCALARS["turin"], t_ask_name, t_ask_key, "Turin", turin_spl(1, 1, 1, 0x04, 0x52), TURIN_HWID, 12),
    )
    # Same ASK subject, another ARK: only the signature tells it from the pinned one.
    _, _, rogue_ask_name, rogue_ask_key, rogue_ark_name, rogue_ark_key = chain("Genoa")
    assert rogue_ask_name == g_ask_name and rogue_ark_name == g_ark_name
    # The VLEK chain: ARK -> ASVK -> VLEK, the ASVK beside the ASK.
    asvk_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    asvk_name = name("TEST-SEV-VLEK-Genoa")
    write("test_genoa_asvk.der", ca(asvk_name, g_ark_name, asvk_key, g_ark_key, 3, 0))
    write(
        "test_genoa_vlek.der",
        vcek(SCALARS["vlek"], asvk_name, asvk_key, "Genoa", genoa_spl(0x0A, 0, 0x1B, 0x58), None, 13,
             csp_id="CN=test-csp.x0ne.invalid"),
    )
    window = (when(2026, 1, 27), when(2026, 3, 10))
    write_crl("test_genoa_crl.der", crl(g_ark_name, g_ark_key, *window, [0x0F]))
    write_crl("test_genoa_crl_ask_revoked.der", crl(g_ark_name, g_ark_key, *window, [0x0F, 2]))
    write_crl("test_genoa_crl_asvk_revoked.der", crl(g_ark_name, g_ark_key, *window, [3]))
    write_crl("test_genoa_crl_expired.der", crl(g_ark_name, g_ark_key, when(2025, 11, 1), when(2026, 1, 1), [0x0F]))
    write_crl("test_genoa_crl_rogue.der", crl(rogue_ark_name, rogue_ark_key, *window, [0x0F]))
    write_crl("test_genoa_crl_critical.der", crl(g_ark_name, g_ark_key, *window, [0x0F], critical_ext=True))
    write_crl("test_turin_crl.der", crl(t_ark_name, t_ark_key, *window, []))
    write(
        "test_genoa_vcek_rogue.der",
        vcek(SCALARS["current"], rogue_ask_name, rogue_ask_key, "Genoa", genoa_spl(0x0A, 0, 0x1B, 0x58), GENOA_CHIP_ID, 10),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
