#!/usr/bin/env python3
"""An implementation of the TDVF measurement independent of x0ne-infer's.

  tdvf_reference.py measure <firmware.fd>   sections, SHA-384, MRTD, CFV digest
  tdvf_reference.py make <out.fd>           writes the synthetic image

Written from google/gce-tcb-verifier (ovmf/fw_guid_table_ops.go,
ovmf/tdx_data.go, tdx/measurement.go, read 2026-10-22), standard library
only. On Google's binary 53c558fe... it prints MRTD c1ee9c16... (the quotes
of the GCE boots of 2026-10-17 and -18) and CFV 4e1221cf... (the RTMR0
EV_EFI_PLATFORM_FIRMWARE_BLOB2 event of their logs), measured 2026-10-22.
x0ne-infer's tdvf.rs builds the synthetic image too (tdvf::tests::synthetic)
and pins what this script prints for it.
"""
import hashlib
import struct
import sys
import uuid


def efi_guid(s):
    u = uuid.UUID(s).bytes
    return struct.pack("<IHH", *struct.unpack(">IHH", u[:8])) + u[8:]


FOOTER = efi_guid("96b582de-1fb2-45f7-baea-a366c55a082d")
OFFSET_BLOCK = efi_guid("e47a6535-984a-4798-865e-4685a7bf8ec2")
METADATA = efi_guid("e9eaf9f3-168e-44d5-a8eb-7f4d8738f6ae")
KINDS = {0: "BFV", 1: "CFV", 2: "TDHOB", 3: "TempMem"}


def sections(fw):
    end = len(fw) - 0x20
    size, guid = struct.unpack("<H16s", fw[end - 18:end])
    assert guid == FOOTER, "no GUID table footer"
    table = fw[end - size:end - 18]
    blocks, n = {}, len(table)
    while n > 0:
        s, g = struct.unpack("<H16s", table[n - 18:n])
        blocks[g] = table[n - s:n - 18]
        n -= s
    (offset,) = struct.unpack("<I", blocks[OFFSET_BLOCK][:4])
    desc = len(fw) - offset
    assert fw[desc - 16:desc] == METADATA, "no metadata GUID"
    sig, length, version, count = struct.unpack("<4sIII", fw[desc:desc + 16])
    assert (sig, version, length) == (b"TDVF", 1, 16 + 32 * count)
    return [
        struct.unpack("<IIQQII", fw[desc + 16 + 32 * k:desc + 48 + 32 * k])
        for k in range(count)
    ]


def mrtd(fw, secs):
    h = hashlib.sha384()
    for off, raw, base, msize, kind, attr in secs:
        for i in range(0, msize, 256):
            if i % 4096 == 0:
                b = bytearray(128)
                b[0:12] = b"MEM.PAGE.ADD"
                b[16:24] = struct.pack("<Q", base + i)
                h.update(b)
            if attr & 1:
                b = bytearray(128)
                b[0:9] = b"MR.EXTEND"
                b[16:24] = struct.pack("<Q", base + i)
                h.update(b)
                h.update(fw[off + i:off + i + 256])
    return h.hexdigest()


def measure(path):
    fw = open(path, "rb").read()
    secs = sections(fw)
    for off, raw, base, msize, kind, attr in secs:
        print(f"{KINDS.get(kind, kind):8} data {off:#x}+{raw:#x} memory {base:#x}+{msize:#x} attr {attr:#x}")
    print("sha384", hashlib.sha384(fw).hexdigest())
    print("sha256", hashlib.sha256(fw).hexdigest())
    print("mrtd  ", mrtd(fw, secs))
    for off, raw, base, msize, kind, attr in secs:
        if kind == 1:
            print("cfv   ", hashlib.sha384(fw[off:off + raw]).hexdigest())


def make(path):
    """16 KiB: a CFV page, three BFV pages marked EXTEND_MR, a TD HOB and
    temporary memory; the descriptor at 0x2F10, inside the BFV, as Google's."""
    n = 0x4000
    fw = bytearray((i * 7 + 3) & 0xFF for i in range(n))
    cfv = b"x0ne synthetic CFV for tests; not a firmware. "
    fw[0:len(cfv)] = cfv
    secs = [
        (0x1000, 0x3000, 0xFFFFD000, 0x3000, 0, 1),
        (0x0, 0x1000, 0xFFFFC000, 0x1000, 1, 0),
        (0, 0, 0x809000, 0x2000, 2, 0),
        (0, 0, 0x800000, 0x6000, 3, 0),
    ]
    desc = 0x2F10
    fw[desc - 16:desc] = METADATA
    fw[desc:desc + 16] = struct.pack("<4sIII", b"TDVF", 16 + 32 * len(secs), 1, len(secs))
    for k, s in enumerate(secs):
        fw[desc + 16 + 32 * k:desc + 48 + 32 * k] = struct.pack("<IIQQII", *s)
    block = struct.pack("<I", n - desc) + struct.pack("<H", 22) + OFFSET_BLOCK
    end = n - 0x20
    fw[end - 18:end] = struct.pack("<H", 18 + len(block)) + FOOTER
    fw[end - 18 - len(block):end - 18] = block
    open(path, "wb").write(bytes(fw))


if __name__ == "__main__":
    {"measure": measure, "make": make}[sys.argv[1]](sys.argv[2])
