//! This machine's public IPv4 address, for the session's firewall rule.
//!
//! One TXT query for `o-o.myaddr.l.google.com`, sent over UDP to Google's own
//! name server, which answers with the address the query came from. Google
//! already sees the relay's connections to the VM: no other party is asked.
//! Google does not document this name; on 2026-10-30 its answer equalled the
//! address an HTTPS echo service saw from the same machine. When it fails,
//! `--allow-from` names the rule's sources instead.
//!
//! Nothing authenticates the answer. A socket connected to the server, a random
//! id and an ephemeral source port keep out who cannot see the query; someone
//! on the path answers as the server does (docs/cloud.md, "The firewall rule").

use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

/// The name whose TXT record is the address the query came from.
pub const QNAME: &str = "o-o.myaddr.l.google.com";
/// Google's authoritative server: a recursive resolver would answer with its
/// own address, not this machine's.
pub const SERVER: &str = "ns1.google.com:53";
/// One wait per query; two queries at most.
pub const WAIT: Duration = Duration::from_secs(3);
const TRIES: usize = 2;
const TYPE_TXT: u16 = 16;
const CLASS_IN: u16 = 1;

/// The query with `id`: no recursion asked, no EDNS, so one TXT comes back.
pub fn query(id: u16) -> Vec<u8> {
    let mut q = id.to_be_bytes().to_vec();
    q.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    for label in QNAME.split('.') {
        q.push(label.len() as u8);
        q.extend_from_slice(label.as_bytes());
    }
    q.push(0);
    q.extend_from_slice(&TYPE_TXT.to_be_bytes());
    q.extend_from_slice(&CLASS_IN.to_be_bytes());
    q
}

fn u16_at(msg: &[u8], i: usize) -> Result<u16, String> {
    msg.get(i..i + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .ok_or_else(|| "the answer is cut short".to_string())
}

/// The name at `i` (compression followed, 16 jumps at most) and where the
/// record goes on after it.
fn read_name(msg: &[u8], mut i: usize) -> Result<(String, usize), String> {
    let cut = || "the answer is cut short".to_string();
    let mut name = String::new();
    let mut after = None;
    let mut jumps = 0;
    loop {
        let len = *msg.get(i).ok_or_else(cut)? as usize;
        match len & 0xc0 {
            0xc0 => {
                let low = *msg.get(i + 1).ok_or_else(cut)? as usize;
                after.get_or_insert(i + 2);
                jumps += 1;
                if jumps > 16 {
                    return Err("a name in the answer loops".into());
                }
                i = ((len & 0x3f) << 8) | low;
            }
            0 if len == 0 => return Ok((name, after.unwrap_or(i + 1))),
            0 => {
                let label = msg.get(i + 1..i + 1 + len).ok_or_else(cut)?;
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(&String::from_utf8_lossy(label));
                if name.len() > 255 {
                    return Err("a name in the answer is too long".into());
                }
                i += 1 + len;
            }
            _ => return Err("a name in the answer has an unknown label type".into()),
        }
    }
}

/// The IPv4 address an answer to [`query`]`(id)` carries, or why there is none.
pub fn parse_answer(msg: &[u8], id: u16) -> Result<Ipv4Addr, String> {
    if msg.len() < 12 {
        return Err("the answer is shorter than a DNS header".into());
    }
    if u16_at(msg, 0)? != id {
        return Err("the answer is to another query".into());
    }
    let flags = u16_at(msg, 2)?;
    if flags & 0x8000 == 0 {
        return Err("a query came back, not an answer".into());
    }
    if flags & 0x0200 != 0 {
        return Err("the answer is truncated".into());
    }
    if flags & 0x000f != 0 {
        return Err(format!("the server answered DNS error {}", flags & 0x000f));
    }
    if u16_at(msg, 4)? != 1 {
        return Err("the answer does not repeat the question".into());
    }
    let records = u16_at(msg, 6)?;
    let (name, mut i) = read_name(msg, 12)?;
    if !name.eq_ignore_ascii_case(QNAME)
        || u16_at(msg, i)? != TYPE_TXT
        || u16_at(msg, i + 2)? != CLASS_IN
    {
        return Err("the answer is to another question".into());
    }
    i += 4;
    for _ in 0..records {
        let (_, at) = read_name(msg, i)?;
        let len = u16_at(msg, at + 8)? as usize;
        let data = msg
            .get(at + 10..at + 10 + len)
            .ok_or_else(|| "the answer is cut short".to_string())?;
        let txt = u16_at(msg, at)? == TYPE_TXT && u16_at(msg, at + 2)? == CLASS_IN;
        i = at + 10 + len;
        if !txt {
            continue;
        }
        let mut text = Vec::new();
        let mut j = 0;
        while j < data.len() {
            let n = data[j] as usize;
            let part = data
                .get(j + 1..j + 1 + n)
                .ok_or_else(|| "a TXT record is cut short".to_string())?;
            text.extend_from_slice(part);
            j += 1 + n;
        }
        if let Some(ip) = std::str::from_utf8(&text)
            .ok()
            .and_then(|t| t.parse::<Ipv4Addr>().ok())
        {
            return Ok(ip);
        }
    }
    Err("the answer holds no IPv4 address".into())
}

/// An address a firewall rule can name for this machine: public unicast.
pub fn check_public(ip: Ipv4Addr) -> Result<Ipv4Addr, String> {
    let o = ip.octets();
    let what = if o[0] == 0 {
        "unspecified"
    } else if ip.is_loopback() {
        "loopback"
    } else if ip.is_private() {
        "private"
    } else if o[0] == 100 && o[1] & 0xc0 == 64 {
        "carrier-grade NAT"
    } else if ip.is_link_local() {
        "link-local"
    } else if ip.is_multicast() {
        "multicast"
    } else if o[0] >= 240 {
        "reserved"
    } else {
        return Ok(ip);
    };
    Err(format!("{ip} is {what}, not a public address"))
}

/// Asks `server` (`host:port`, over IPv4) at most twice, `wait` each time.
pub async fn public_ipv4(server: &str, wait: Duration) -> Result<Ipv4Addr, String> {
    let to = tokio::net::lookup_host(server)
        .await
        .map_err(|e| format!("{server}: {e}"))?
        .find(SocketAddr::is_ipv4)
        .ok_or_else(|| format!("{server} has no IPv4 address"))?;
    let sock = tokio::net::UdpSocket::bind(("0.0.0.0", 0))
        .await
        .map_err(|e| format!("UDP socket: {e}"))?;
    sock.connect(to)
        .await
        .map_err(|e| format!("{server}: {e}"))?;
    let mut buf = [0u8; 1232];
    let mut last = format!("no answer from {server}");
    for _ in 0..TRIES {
        let hex = crate::random_hex(2).map_err(|e| format!("/dev/urandom: {e}"))?;
        let id = u16::from_str_radix(&hex, 16).map_err(|e| e.to_string())?;
        sock.send(&query(id))
            .await
            .map_err(|e| format!("{server}: {e}"))?;
        let until = tokio::time::Instant::now() + wait;
        loop {
            match tokio::time::timeout_at(until, sock.recv(&mut buf)).await {
                Err(_) => {
                    last = format!("no answer from {server} within {} s", wait.as_secs_f32());
                    break;
                }
                // A port closed on the way answers ICMP: said, then asked again.
                Ok(Err(e)) => {
                    last = format!("{server}: {e}");
                    break;
                }
                // A late answer to the previous query: not this one's.
                Ok(Ok(n)) if n < 2 || buf[..2] != id.to_be_bytes() => {}
                Ok(Ok(n)) => {
                    let ip = parse_answer(&buf[..n], id).map_err(|e| format!("{server}: {e}"))?;
                    return check_public(ip);
                }
            }
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: u16 = 0x5a3c;

    /// The answer Google's server gives, shaped as `dig` showed it on
    /// 2026-10-30: the question echoed, one TXT under a compressed name.
    fn answer(id: u16, flags: u16, texts: &[&[u8]]) -> Vec<u8> {
        let q = query(id);
        let mut r = q[..2].to_vec();
        r.extend_from_slice(&flags.to_be_bytes());
        r.extend_from_slice(&[0, 1]);
        r.extend_from_slice(&(texts.len() as u16).to_be_bytes());
        r.extend_from_slice(&[0, 0, 0, 0]);
        r.extend_from_slice(&q[12..]);
        for t in texts {
            r.extend_from_slice(&[0xc0, 0x0c, 0, 16, 0, 1, 0, 0, 0, 60]);
            r.extend_from_slice(&(t.len() as u16 + 1).to_be_bytes());
            r.push(t.len() as u8);
            r.extend_from_slice(t);
        }
        r
    }

    /// The exact bytes `dig +noedns +norecurse -t TXT` sends for the name.
    #[test]
    fn the_query_is_one_txt_question_without_recursion() {
        let mut want = vec![0x5a, 0x3c, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
        for l in ["o-o", "myaddr", "l", "google", "com"] {
            want.push(l.len() as u8);
            want.extend_from_slice(l.as_bytes());
        }
        want.extend_from_slice(&[0, 0, 16, 0, 1]);
        assert_eq!(query(ID), want);
    }

    #[test]
    fn the_address_is_read_from_the_txt_answer() {
        let a = answer(ID, 0x8400, &[b"198.51.100.23"]);
        assert_eq!(parse_answer(&a, ID), Ok(Ipv4Addr::new(198, 51, 100, 23)));
        // A client subnet record first (EDNS answers carry one): skipped.
        let a = answer(
            ID,
            0x8400,
            &[b"edns0-client-subnet 198.51.100.0/24", b"198.51.100.23"],
        );
        assert_eq!(parse_answer(&a, ID), Ok(Ipv4Addr::new(198, 51, 100, 23)));
    }

    /// Anything but a whole, matching, successful answer is refused.
    #[test]
    fn a_foreign_cut_or_failed_answer_is_refused() {
        let good = answer(ID, 0x8400, &[b"198.51.100.23"]);
        assert!(parse_answer(&good, ID ^ 1).is_err(), "another id");
        assert!(
            parse_answer(&answer(ID, 0x0400, &[b"1.2.3.4"]), ID).is_err(),
            "a query, not an answer"
        );
        assert!(
            parse_answer(&answer(ID, 0x8600, &[b"1.2.3.4"]), ID).is_err(),
            "truncated"
        );
        assert!(
            parse_answer(&answer(ID, 0x8403, &[]), ID).is_err(),
            "NXDOMAIN"
        );
        assert!(
            parse_answer(&answer(ID, 0x8400, &[]), ID).is_err(),
            "no record"
        );
        assert!(
            parse_answer(&answer(ID, 0x8400, &[b"2001:db8::1"]), ID).is_err(),
            "IPv6"
        );
        assert!(
            parse_answer(&answer(ID, 0x8400, &[b"not an address"]), ID).is_err(),
            "text"
        );
        for cut in [0, 5, 11, 12, 20, good.len() - 1] {
            assert!(parse_answer(&good[..cut], ID).is_err(), "cut at {cut}");
        }
        let mut other = good.clone();
        other[13] = b'x';
        assert!(parse_answer(&other, ID).is_err(), "another name");
        let mut lp = good.clone();
        let at = 12 + QNAME.len() + 2 + 4;
        lp[at] = 0xc0;
        lp[at + 1] = at as u8;
        assert!(parse_answer(&lp, ID).is_err(), "a name pointing at itself");
    }

    #[test]
    fn only_a_public_unicast_address_is_taken() {
        for ip in ["198.51.100.23", "8.8.8.8", "100.63.255.255", "100.128.0.1"] {
            let ip: Ipv4Addr = ip.parse().unwrap();
            assert_eq!(check_public(ip), Ok(ip), "{ip}");
        }
        // Octets, not dotted strings: the publication gate refuses dotted
        // private addresses anywhere in the tree.
        for ([a, b, c, d], says) in [
            ([0, 0, 0, 0], "unspecified"),
            ([0, 1, 2, 3], "unspecified"),
            ([127, 0, 0, 1], "loopback"),
            ([10, 1, 2, 3], "private"),
            ([172, 16, 0, 1], "private"),
            ([192, 168, 1, 1], "private"),
            ([100, 64, 7, 8], "carrier-grade NAT"),
            ([100, 127, 255, 255], "carrier-grade NAT"),
            ([169, 254, 169, 254], "link-local"),
            ([224, 0, 0, 1], "multicast"),
            ([240, 0, 0, 1], "reserved"),
            ([255, 255, 255, 255], "reserved"),
        ] {
            let ip = Ipv4Addr::new(a, b, c, d);
            let e = check_public(ip).unwrap_err();
            assert!(e.contains(says), "{ip}: {e}");
        }
    }

    /// The whole exchange against a local server: a stale answer to another
    /// id is skipped, the matching one is read.
    #[tokio::test]
    async fn the_address_comes_back_from_a_server_on_the_loopback() {
        let srv = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let at = srv.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let mut b = [0u8; 512];
            let (n, from) = srv.recv_from(&mut b).await.unwrap();
            let id = u16::from_be_bytes([b[0], b[1]]);
            assert_eq!(&b[2..n], &query(id)[2..]);
            let stale = answer(id ^ 0xffff, 0x8400, &[b"192.0.2.1"]);
            srv.send_to(&stale, from).await.unwrap();
            let ok = answer(id, 0x8400, &[b"198.51.100.23"]);
            srv.send_to(&ok, from).await.unwrap();
        });
        assert_eq!(
            public_ipv4(&at, Duration::from_secs(5)).await,
            Ok(Ipv4Addr::new(198, 51, 100, 23))
        );
    }

    /// SECURITY: an answer from another address than the server's, its id
    /// right, is not read: the socket is connected. A failure is a rule that
    /// names an address anyone could send without forging the server's.
    #[tokio::test]
    async fn an_answer_from_another_address_is_not_read() {
        let srv = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let other = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let at = srv.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let mut b = [0u8; 512];
            let (_, from) = srv.recv_from(&mut b).await.unwrap();
            let id = u16::from_be_bytes([b[0], b[1]]);
            let forged = answer(id, 0x8400, &[b"198.51.100.66"]);
            other.send_to(&forged, from).await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let ok = answer(id, 0x8400, &[b"198.51.100.23"]);
            srv.send_to(&ok, from).await.unwrap();
        });
        assert_eq!(
            public_ipv4(&at, Duration::from_secs(5)).await,
            Ok(Ipv4Addr::new(198, 51, 100, 23))
        );
    }

    /// A silent server: two queries, then a refusal that names the server.
    #[tokio::test]
    async fn a_silent_server_is_asked_twice_then_refused() {
        let srv = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let at = srv.local_addr().unwrap().to_string();
        let t0 = std::time::Instant::now();
        let e = public_ipv4(&at, Duration::from_millis(300))
            .await
            .unwrap_err();
        assert!(e.contains("no answer from") && e.contains(&at), "{e}");
        assert!(t0.elapsed() >= Duration::from_millis(600));
        let mut b = [0u8; 512];
        let mut seen = 0;
        while let Ok(Ok(_)) =
            tokio::time::timeout(Duration::from_millis(50), srv.recv_from(&mut b)).await
        {
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    /// A private address answered is refused, not taken into the rule.
    #[tokio::test]
    async fn a_private_answer_is_refused() {
        let srv = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let at = srv.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let mut b = [0u8; 512];
            let (_, from) = srv.recv_from(&mut b).await.unwrap();
            let id = u16::from_be_bytes([b[0], b[1]]);
            let a = answer(id, 0x8400, &[b"172.31.1.20"]);
            srv.send_to(&a, from).await.unwrap();
        });
        let e = public_ipv4(&at, Duration::from_secs(5)).await.unwrap_err();
        assert!(e.contains("172.31.1.20 is private"), "{e}");
    }
}
