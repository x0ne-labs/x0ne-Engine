// SPDX-License-Identifier: Apache-2.0
//! Hex decoding of input nobody has vouched for yet. Compiled into x0ne-sdk
//! too (`#[path]`, as `cloud_config`, the path `tcb.rs` calls): no crate
//! paths here.

/// Hex of either case to bytes, for input nobody has vouched for yet (a
/// receipt, Intel's collateral). Reads the bytes of `s`: every byte must be a
/// hex digit and the count even, so a multibyte character is refused, never
/// sliced through (2026-10-23: `&s[0..2]` on "€…" panicked at a char boundary).
pub fn hex_bytes(s: &str) -> Result<Vec<u8>, String> {
    let b = s.as_bytes();
    if !b.len().is_multiple_of(2) {
        return Err(format!("odd length {}", b.len()));
    }
    b.as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            let digit = |c: u8| char::from(c).to_digit(16).map(|d| d as u8);
            match (digit(pair[0]), digit(pair[1])) {
                (Some(hi), Some(lo)) => Ok(hi << 4 | lo),
                _ => Err(format!("byte {} is not a hex digit", 2 * i)),
            }
        })
        .collect()
}

/// `hex_bytes` for a fixed size: exactly `2 * N` hex characters, either case.
pub fn hex_array<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != N * 2 {
        return Err(format!(
            "expected {} hex characters, got {} bytes",
            N * 2,
            s.len()
        ));
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&hex_bytes(s)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder every untrusted hex field goes through: it indexes bytes,
    /// not chars, so "€" at the exact byte length is an error, not a panic.
    #[test]
    fn hex_bytes_refuses_non_ascii_odd_and_non_hex_and_takes_either_case() {
        assert_eq!(hex_bytes("DC9e").unwrap(), vec![0xdc, 0x9e]);
        assert_eq!(hex_bytes("").unwrap(), Vec::<u8>::new());
        assert_eq!(hex_bytes("€0").unwrap_err(), "byte 0 is not a hex digit");
        assert_eq!(hex_bytes("0€").unwrap_err(), "byte 0 is not a hex digit");
        assert_eq!(hex_bytes("€").unwrap_err(), "odd length 3");
        assert_eq!(hex_bytes("abc").unwrap_err(), "odd length 3");
        assert_eq!(hex_bytes("ab0g").unwrap_err(), "byte 2 is not a hex digit");
        // from_str_radix took a sign; the byte decoder does not.
        assert_eq!(hex_bytes("+f").unwrap_err(), "byte 0 is not a hex digit");
        let euro64 = format!("€{}", "0".repeat(61));
        assert_eq!(euro64.len(), 64);
        assert!(hex_array::<32>(&euro64).is_err());
        assert!(hex_array::<32>(&"0".repeat(63)).is_err());
        assert!(hex_array::<32>(&"0".repeat(65)).is_err());
        assert_eq!(hex_array::<2>("ABcd").unwrap(), [0xab, 0xcd]);
    }
}
