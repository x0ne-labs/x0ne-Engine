//! Every field `tdx_quote::parse` exposes, read from the REAL vendored
//! v4 quote and compared with the bytes at the Intel TD Quote v4 offsets,
//! written here independently of the parser's constants.

use x0ne_confidential::tdx_quote;

const QUOTE: &[u8] = include_bytes!("pcs_live/quote.dat");
const HEADER: usize = 48;

/// Intel TD Quote v4, TD report body (offsets relative to the body).
const TD_ATTRIBUTES: (usize, usize) = (120, 8);
const MR_TD: (usize, usize) = (136, 48);
const MR_OWNER_CONFIG: (usize, usize) = (280, 48);
const RTMR: [(usize, usize); 4] = [(328, 48), (376, 48), (424, 48), (472, 48)];
const REPORT_DATA: (usize, usize) = (520, 64);

fn body((off, len): (usize, usize)) -> &'static [u8] {
    &QUOTE[HEADER + off..HEADER + off + len]
}

#[test]
fn every_parsed_field_is_the_bytes_at_the_intel_v4_offset() {
    let c = tdx_quote::parse(QUOTE).expect("the vendored quote parses");
    assert_eq!(
        c.tee_type,
        u32::from_le_bytes(QUOTE[4..8].try_into().unwrap())
    );
    assert_eq!(c.td_attributes.as_slice(), body(TD_ATTRIBUTES));
    assert_eq!(c.mr_td.as_slice(), body(MR_TD));
    for (i, r) in RTMR.iter().enumerate() {
        assert_eq!(
            tdx_quote::hex(&c.rtmr[i]),
            tdx_quote::hex(body(*r)),
            "rtmr{i}"
        );
    }
    assert_eq!(c.report_data.as_slice(), body(REPORT_DATA));
}

/// In this quote MROWNERCONFIG is all zero and RTMR0 is not: a parser that
/// reads rtmr0 at 280 yields zeros here.
#[test]
fn rtmr0_is_not_mrownerconfig() {
    assert!(body(MR_OWNER_CONFIG).iter().all(|&b| b == 0));
    let c = tdx_quote::parse(QUOTE).unwrap();
    assert!(c.rtmr[0].iter().any(|&b| b != 0), "rtmr0 read as zeros");
}
