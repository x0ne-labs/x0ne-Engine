//! The GPU half of the key binding (`e2e::gpu_nonce`, `e2e::key_binding_gpu`):
//! the layout spelled out, and every byte of the GPU evidence in the hash,
//! each field's bounds included, so no two evidence lists bind alike.

use sha2::{Digest, Sha256, Sha512};
use x0ne_sdk::e2e::{
    gpu_nonce, key_binding, key_binding_gpu, Boot, BINDING_LABEL, GPU_BINDING_LABEL,
    GPU_NONCE_LABEL,
};

const CONFIG: &[u8] = b"\x01\x00\x20key-config-bytes";
const NONCE: [u8; 32] = [0x11; 32];
const PK: [u8; 32] = [0x22; 32];
const BOOT: Boot = Boot {
    nonce: [0x33; 32],
    tls_spki_sha256: [0x44; 32],
    config_sha256: [0x55; 32],
};

#[test]
fn the_gpu_nonce_is_sha256_over_its_spelled_layout() {
    let mut h = Sha256::new();
    h.update(b"x0ne-e2e/v1/gpu\0");
    h.update((CONFIG.len() as u16).to_be_bytes());
    h.update(CONFIG);
    h.update(NONCE);
    h.update(PK);
    let want: [u8; 32] = h.finalize().into();
    assert_eq!(GPU_NONCE_LABEL, b"x0ne-e2e/v1/gpu");
    assert_eq!(gpu_nonce(CONFIG, &NONCE, &PK), want);
    assert_ne!(
        gpu_nonce(CONFIG, &[0x12; 32], &PK),
        want,
        "per caller nonce"
    );
    assert_ne!(
        gpu_nonce(CONFIG, &NONCE, &[0x23; 32]),
        want,
        "per receipt key"
    );
}

#[test]
fn the_gpu_binding_is_sha512_over_its_spelled_layout() {
    let (report, chain) = (b"report".as_slice(), b"chain-pem".as_slice());
    let mut h = Sha512::new();
    h.update(b"x0ne-e2e/v2/ohttp-gpu-key\0");
    h.update((CONFIG.len() as u16).to_be_bytes());
    h.update(CONFIG);
    h.update(NONCE);
    h.update(PK);
    h.update(BOOT.nonce);
    h.update(BOOT.tls_spki_sha256);
    h.update(BOOT.config_sha256);
    h.update(1u64.to_be_bytes());
    h.update((report.len() as u64).to_be_bytes());
    h.update(report);
    h.update((chain.len() as u64).to_be_bytes());
    h.update(chain);
    let want: [u8; 64] = h.finalize().into();
    assert_eq!(GPU_BINDING_LABEL, b"x0ne-e2e/v2/ohttp-gpu-key");
    assert_eq!(
        key_binding_gpu(CONFIG, &NONCE, &PK, &BOOT, &[(report, chain)]),
        want
    );
}

/// SECURITY: a byte flipped anywhere, a byte moved from one field to the
/// next, two GPUs reordered or one dropped, no GPU at all: each binds
/// otherwise, and none binds as the CPU key binding does.
#[test]
fn every_byte_and_bound_of_the_gpu_evidence_moves_the_binding() {
    let a = (b"report-a".to_vec(), b"chain-a".to_vec());
    let b = (b"report-b".to_vec(), b"chain-b".to_vec());
    let bind = |gpus: &[(Vec<u8>, Vec<u8>)]| {
        let views: Vec<(&[u8], &[u8])> = gpus
            .iter()
            .map(|(r, c)| (r.as_slice(), c.as_slice()))
            .collect();
        key_binding_gpu(CONFIG, &NONCE, &PK, &BOOT, &views)
    };
    let base = bind(&[a.clone(), b.clone()]);
    let mut seen = vec![base, key_binding(CONFIG, &NONCE, &PK, &BOOT), bind(&[])];
    for gpu in 0..2 {
        for field in 0..2 {
            let len = if field == 0 { 8 } else { 7 };
            for i in 0..len {
                let mut gpus = [a.clone(), b.clone()];
                let f = if field == 0 {
                    &mut gpus[gpu].0
                } else {
                    &mut gpus[gpu].1
                };
                f[i] ^= 1;
                seen.push(bind(&gpus));
            }
        }
    }
    // The last report byte moved to the head of its chain: same concatenation.
    let mut moved = [a.clone(), b.clone()];
    let last = moved[0].0.pop().unwrap();
    moved[0].1.insert(0, last);
    seen.push(bind(&moved));
    seen.push(bind(&[b.clone(), a.clone()]));
    seen.push(bind(std::slice::from_ref(&a)));
    // Each field of the TD's boot binding, the GPUs unchanged.
    let views = [
        (a.0.as_slice(), a.1.as_slice()),
        (b.0.as_slice(), b.1.as_slice()),
    ];
    for boot in [
        Boot {
            nonce: [0x34; 32],
            ..BOOT
        },
        Boot {
            tls_spki_sha256: [0x45; 32],
            ..BOOT
        },
        Boot {
            config_sha256: [0x56; 32],
            ..BOOT
        },
    ] {
        seen.push(key_binding_gpu(CONFIG, &NONCE, &PK, &boot, &views));
    }
    let mut distinct = seen.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), seen.len(), "two evidence lists bind alike");
}

#[test]
fn the_e2e_labels_are_no_prefix_of_one_another() {
    let labels = [BINDING_LABEL, GPU_BINDING_LABEL, GPU_NONCE_LABEL];
    for (i, x) in labels.iter().enumerate() {
        for y in labels.iter().skip(i + 1) {
            assert!(!x.starts_with(y) && !y.starts_with(x), "{x:?} {y:?}");
        }
    }
}
