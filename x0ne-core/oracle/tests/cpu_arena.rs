//! The arena against a model of it, on random plans: where regions land, what a frozen arena
//! lets through, what a scrub zeroes. No GPU: host memory, or a backing that only counts.
//!
//! The arena has no free: regions are planned, the arena is frozen, and client regions are
//! zeroed in place. A "random alloc and free sequence" is here a random plan followed by random
//! writes, reads and scrubs.

mod common;

use common::Rng;
use x0ne_core_runtime::arena::{
    Arena, Backing, Building, Class, DeviceAddressed, Frozen, HostBacking, Slice, ALIGN,
};
use x0ne_core_runtime::{Error, Result};

const POISON: u8 = 0xA5;

/// What the model keeps of one planned region.
#[derive(Debug, Clone, Copy)]
struct Planned {
    slice: Slice,
    offset: u64,
    len: u64,
    class: Class,
}

fn random_len(rng: &mut Rng, big: usize) -> u64 {
    match rng.below(6) {
        0 => 1,
        1 => *rng.pick(&[ALIGN - 1, ALIGN, ALIGN + 1, 2 * ALIGN - 1, 2 * ALIGN]),
        2 => rng.range(1, 64) as u64,
        3 => rng.range(1, 1024) as u64,
        4 => rng.range(1, 3) as u64 * ALIGN,
        _ => rng.range(1, big) as u64,
    }
}

/// A random plan with regions up to 70 000 bytes.
fn random_plan(rng: &mut Rng, regions: usize) -> (Arena<Building>, Vec<Planned>) {
    random_plan_of(rng, regions, 70_000)
}

/// A random plan: the arena and the model of where each region must be.
fn random_plan_of(rng: &mut Rng, regions: usize, big: usize) -> (Arena<Building>, Vec<Planned>) {
    let mut a = Arena::new();
    let mut model: Vec<Planned> = Vec::new();
    let mut end = 0u64;
    for i in 0..regions {
        if rng.chance(8) {
            let before = a.planned_bytes();
            assert!(
                matches!(a.alloc("empty", 0, Class::ClientData), Err(Error::Arena(_))),
                "an empty region was planned"
            );
            assert_eq!(a.planned_bytes(), before, "a refused region moved the plan");
            assert_eq!(a.regions().len(), model.len());
        }
        let len = random_len(rng, big);
        let class = if rng.chance(2) {
            Class::Weights
        } else {
            Class::ClientData
        };
        let slice = a.alloc(&format!("r{i}"), len, class).expect("alloc");
        let offset = end.next_multiple_of(ALIGN);
        end = offset + len;
        assert_eq!((slice.offset(), slice.len()), (offset, len), "region {i}");
        assert!(!slice.is_empty());
        model.push(Planned {
            slice,
            offset,
            len,
            class,
        });
        assert_eq!(
            a.planned_bytes(),
            end,
            "the plan ends where its last region ends"
        );
    }
    (a, model)
}

/// Every region is aligned, inside the plan, apart from every other, and listed as planned.
#[test]
fn random_plans_are_aligned_disjoint_and_inside_the_allocation() {
    for seed in 0..300 {
        let mut rng = Rng::new(seed);
        let n = rng.range(1, 40);
        let (a, model) = random_plan(&mut rng, n);
        let listed = a.regions();
        assert_eq!(listed.len(), model.len());
        for (i, (r, m)) in listed.iter().zip(&model).enumerate() {
            assert_eq!(
                (r.offset, r.len, r.class),
                (m.offset, m.len, m.class),
                "seed {seed} region {i}"
            );
            assert_eq!(r.label, format!("r{i}"));
            assert_eq!(r.offset % ALIGN, 0, "seed {seed} region {i} is not aligned");
            assert!(r.offset + r.len <= a.planned_bytes());
        }
        for (i, x) in model.iter().enumerate() {
            for y in &model[i + 1..] {
                assert!(
                    x.offset + x.len <= y.offset,
                    "seed {seed}: regions overlap or are out of order"
                );
            }
        }
        let total = a.planned_bytes();
        let f = a.freeze(HostBacking::allocate).expect("freeze");
        assert_eq!(f.backing().len(), total.max(1));
        assert_eq!(f.regions().len(), model.len());
    }
}

/// A write lands exactly on its bytes and a refused write changes nothing; a read returns
/// what the model holds; a scrub zeroes the client regions and nothing else, padding included.
#[test]
fn random_writes_reads_and_scrubs_match_a_byte_model() {
    for seed in 0..120 {
        let mut rng = Rng::new(1000 + seed);
        let n = rng.range(1, 24);
        let (a, model) = random_plan_of(&mut rng, n, 3000);
        let total = a.planned_bytes() as usize;
        let mut f = a.freeze(HostBacking::allocate).expect("freeze");
        let mut bytes = vec![POISON; total];
        assert_eq!(
            f.backing().bytes(),
            &bytes[..],
            "a new host backing is poisoned"
        );
        for _ in 0..200 {
            let m = *rng.pick(&model);
            match rng.below(10) {
                0..=4 => {
                    let at = rng.below(m.len as usize + 2) as u64;
                    let len = rng.below(m.len as usize + 3);
                    let data = rng.bytes(len);
                    let fits = at + len as u64 <= m.len;
                    let r = f.write(m.slice, at, &data);
                    assert_eq!(
                        r.is_ok(),
                        fits,
                        "seed {seed}: write of {len} at {at} in {}",
                        m.len
                    );
                    if fits {
                        let o = (m.offset + at) as usize;
                        bytes[o..o + len].copy_from_slice(&data);
                    } else {
                        assert!(matches!(r, Err(Error::Arena(_))));
                    }
                }
                5..=7 => {
                    let at = rng.below(m.len as usize + 2) as u64;
                    let len = rng.below(m.len as usize + 3);
                    let mut out = vec![0x11u8; len];
                    let fits = at + len as u64 <= m.len;
                    let r = f.read(m.slice, at, &mut out);
                    assert_eq!(
                        r.is_ok(),
                        fits,
                        "seed {seed}: read of {len} at {at} in {}",
                        m.len
                    );
                    if fits {
                        let o = (m.offset + at) as usize;
                        assert_eq!(out, bytes[o..o + len], "seed {seed}: read differs");
                    } else {
                        assert!(
                            out.iter().all(|&b| b == 0x11),
                            "a refused read wrote its buffer"
                        );
                    }
                }
                8 => {
                    // The largest offsets: no wrap into another region.
                    for at in [u64::MAX, u64::MAX - 1, m.len, m.len + 1, u64::MAX - m.len] {
                        assert!(f.write(m.slice, at, &[1]).is_err(), "write at {at}");
                        assert!(f.read(m.slice, at, &mut [0]).is_err(), "read at {at}");
                    }
                    // Nothing at the end of a slice is still inside it.
                    assert!(f.write(m.slice, m.len, &[]).is_ok());
                }
                _ => {
                    let zeroed = f.scrub_client_data().expect("scrub");
                    let mut want = 0;
                    for m in model.iter().filter(|m| m.class == Class::ClientData) {
                        bytes[m.offset as usize..(m.offset + m.len) as usize].fill(0);
                        want += m.len;
                    }
                    assert_eq!(zeroed, want, "seed {seed}: bytes a scrub reports");
                }
            }
            assert_eq!(
                f.backing().bytes(),
                &bytes[..],
                "seed {seed}: the backing left the model"
            );
        }
        // After a last scrub: every client byte is zero, every weight byte and every padding
        // byte is what it was.
        f.scrub_client_data().expect("scrub");
        let mem = f.backing().bytes();
        let mut covered = vec![false; total];
        for m in &model {
            let r = m.offset as usize..(m.offset + m.len) as usize;
            covered[r.clone()].fill(true);
            if m.class == Class::ClientData {
                assert!(
                    mem[r].iter().all(|&b| b == 0),
                    "seed {seed}: a client byte survived"
                );
            } else {
                assert_eq!(
                    mem[r.clone()],
                    bytes[r],
                    "seed {seed}: a scrub touched weights"
                );
            }
        }
        for (i, c) in covered.iter().enumerate() {
            if !c {
                assert_eq!(mem[i], POISON, "seed {seed}: padding byte {i} was written");
            }
        }
    }
}

/// A backing that holds no memory: it records what it is asked, with a device address.
struct Counting {
    len: u64,
    base: u64,
    zeroed: std::cell::RefCell<Vec<(u64, u64)>>,
}

impl Counting {
    fn at(base: u64) -> impl FnOnce(u64) -> Result<Counting> {
        move |len| {
            Ok(Counting {
                len,
                base,
                zeroed: Default::default(),
            })
        }
    }
    fn inside(&self, offset: u64, len: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len => Ok(()),
            _ => Err(Error::Arena(format!("{offset}+{len} outside {}", self.len))),
        }
    }
}

impl Backing for Counting {
    fn len(&self) -> u64 {
        self.len
    }
    fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.inside(offset, data.len() as u64)
    }
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        self.inside(offset, out.len() as u64)
    }
    fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        self.inside(offset, len)?;
        self.zeroed.borrow_mut().push((offset, len));
        Ok(())
    }
}

impl DeviceAddressed for Counting {
    fn base_addr(&self) -> u64 {
        self.base
    }
}

/// The precondition of every kernel launch: an address the arena gives lies inside the slice
/// it was asked for, whole spans included. `addr`, `ptr` and `span` are how every `unsafe`
/// launch of the runtime gets its pointers.
#[test]
fn device_addresses_and_spans_stay_inside_their_slice() {
    const BASE: u64 = 0x7f00_0000_0000;
    for seed in 0..200 {
        let mut rng = Rng::new(5000 + seed);
        let n = rng.range(1, 30);
        let (a, model) = random_plan(&mut rng, n);
        let f: Arena<Frozen<Counting>> = a.freeze(Counting::at(BASE)).expect("freeze");
        for m in &model {
            let (lo, hi) = (BASE + m.offset, BASE + m.offset + m.len);
            assert_eq!(f.ptr(m.slice).expect("ptr"), lo);
            for _ in 0..40 {
                let at = match rng.below(4) {
                    0 => m.len - 1,
                    1 => m.len,
                    2 => rng.below(m.len as usize) as u64,
                    _ => rng.next_u64(),
                };
                match f.addr(m.slice, at) {
                    Ok(p) => {
                        assert!(at < m.len, "seed {seed}: address {at} past {} given", m.len);
                        assert_eq!(p, lo + at);
                    }
                    Err(e) => {
                        assert!(
                            at >= m.len,
                            "seed {seed}: address {at} of {} refused",
                            m.len
                        );
                        assert!(matches!(e, Error::Arena(_)));
                    }
                }
                let len = match rng.below(4) {
                    0 => 0,
                    1 => m.len,
                    2 => rng.below(m.len as usize + 2) as u64,
                    _ => rng.next_u64(),
                };
                match f.span(m.slice, at, len) {
                    Ok(p) => {
                        assert!(len > 0, "an empty span has no address");
                        assert_eq!(p, lo + at);
                        assert!(
                            p >= lo && p + len <= hi,
                            "seed {seed}: span {at}+{len} leaves its slice"
                        );
                    }
                    Err(_) => {
                        let fits = len > 0 && at.checked_add(len).is_some_and(|e| e <= m.len);
                        assert!(!fits, "seed {seed}: span {at}+{len} of {} refused", m.len);
                    }
                }
            }
        }
    }
}

/// A slice is good for the arena that planned it only: another arena, even with the same plan,
/// refuses it on every entry point.
#[test]
fn a_slice_of_another_arena_is_refused_on_every_entry_point() {
    let build = || {
        let mut a = Arena::new();
        let s = a.alloc("x", 512, Class::ClientData).expect("alloc");
        (a.freeze(Counting::at(0x1000)).expect("freeze"), s)
    };
    // Arenas made on several threads at once still tell their slices apart.
    let made: Vec<(Arena<Frozen<Counting>>, Slice)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..8)
            .map(|_| s.spawn(|| (0..16).map(|_| build()).collect::<Vec<_>>()))
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().expect("thread"))
            .collect()
    });
    for (i, (arena, own)) in made.iter().enumerate() {
        assert!(arena.ptr(*own).is_ok());
        for (j, (_, foreign)) in made.iter().enumerate() {
            if i == j {
                continue;
            }
            assert_eq!((foreign.offset(), foreign.len()), (own.offset(), own.len()));
            assert_ne!(foreign, own, "two arenas issued the same slice");
            assert!(
                arena.ptr(*foreign).is_err(),
                "ptr of arena {j} on arena {i}"
            );
            assert!(arena.addr(*foreign, 1).is_err());
            assert!(arena.span(*foreign, 0, 8).is_err());
            assert!(arena.read(*foreign, 0, &mut [0; 4]).is_err());
        }
    }
    let (mut a, _) = build();
    let (_, foreign) = build();
    assert!(matches!(a.write(foreign, 0, &[0; 4]), Err(Error::Arena(_))));
}

/// Sizes at the edge of u64 are refused, never wrapped into a small plan, and a refusal leaves
/// the plan as it was.
#[test]
fn plans_at_the_edge_of_u64_are_refused_not_wrapped() {
    let mut a = Arena::new();
    let s = a
        .alloc("huge", u64::MAX - 10, Class::Weights)
        .expect("the first region fits");
    assert_eq!((s.offset(), s.len()), (0, u64::MAX - 10));
    // The next aligned offset does not exist.
    assert!(matches!(
        a.alloc("more", 1, Class::Weights),
        Err(Error::Arena(_))
    ));
    assert_eq!(a.planned_bytes(), u64::MAX - 10);
    assert_eq!(a.regions().len(), 1);

    let mut a = Arena::new();
    a.alloc("a", 1, Class::Weights).expect("alloc");
    // Offset 256 exists; 256 + len does not.
    for len in [u64::MAX, u64::MAX - 255, u64::MAX - ALIGN + 1] {
        assert!(
            matches!(a.alloc("b", len, Class::ClientData), Err(Error::Arena(_))),
            "{len}"
        );
        assert_eq!(a.planned_bytes(), 1);
    }
    let s = a
        .alloc("b", u64::MAX - ALIGN, Class::ClientData)
        .expect("the largest that fits");
    assert_eq!(s.offset(), ALIGN);
    assert_eq!(a.planned_bytes(), u64::MAX);
    // Frozen on a backing that only counts: the bounds still hold at the top of u64.
    let mut f = a.freeze(Counting::at(0)).expect("freeze");
    assert!(f.write(s, s.len() - 1, &[0]).is_ok());
    assert!(f.write(s, s.len(), &[0]).is_err());
    assert!(f.write(s, u64::MAX, &[0]).is_err());
    assert_eq!(f.span(s, 0, s.len()).expect("the whole slice"), ALIGN);
    assert!(f.span(s, 1, s.len()).is_err());
    assert_eq!(f.scrub_client_data().expect("scrub"), u64::MAX - ALIGN);
    assert_eq!(f.backing().zeroed.borrow()[..], [(ALIGN, u64::MAX - ALIGN)]);
}

/// The single allocation: asked for the planned size (one byte for an empty plan), refused
/// when the backing is shorter, and its error is the caller's.
#[test]
fn freeze_asks_for_the_plan_and_refuses_a_short_backing() {
    let asked = std::cell::Cell::new(0);
    let f = Arena::new()
        .freeze(|len| {
            asked.set(len);
            Counting::at(0)(len)
        })
        .expect("an empty plan freezes");
    assert_eq!(asked.get(), 1);
    assert!(f.regions().is_empty());

    for seed in 0..50 {
        let mut rng = Rng::new(9000 + seed);
        let (a, _) = random_plan(&mut rng, 5);
        let want = a.planned_bytes();
        let short = rng.below(want as usize) as u64;
        let r = a.freeze(|len| {
            assert_eq!(len, want);
            Counting::at(0)(short)
        });
        assert!(
            matches!(r, Err(Error::Arena(_))),
            "seed {seed}: {short} bytes for {want}"
        );
    }
    let (a, _) = random_plan(&mut Rng::new(1), 3);
    let r = a.freeze(|_| -> Result<Counting> { Err(Error::Gpu("no memory".into())) });
    assert!(matches!(r, Err(Error::Gpu(m)) if m == "no memory"));
}

/// A scrub that fails on a region reports it, and zeroes each client region once, in plan
/// order, with exactly its bounds.
#[test]
fn a_scrub_zeroes_each_client_region_once_with_its_bounds() {
    for seed in 0..100 {
        let mut rng = Rng::new(7000 + seed);
        let n = rng.range(1, 30);
        let (a, model) = random_plan(&mut rng, n);
        let mut f = a.freeze(Counting::at(0)).expect("freeze");
        let want: Vec<(u64, u64)> = model
            .iter()
            .filter(|m| m.class == Class::ClientData)
            .map(|m| (m.offset, m.len))
            .collect();
        let total = f.scrub_client_data().expect("scrub");
        assert_eq!(total, want.iter().map(|w| w.1).sum::<u64>());
        assert_eq!(f.backing().zeroed.borrow()[..], want[..], "seed {seed}");
    }
    struct Failing;
    impl Backing for Failing {
        fn len(&self) -> u64 {
            1 << 20
        }
        fn write(&mut self, _: u64, _: &[u8]) -> Result<()> {
            Ok(())
        }
        fn read(&self, _: u64, _: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn zero(&mut self, _: u64, _: u64) -> Result<()> {
            Err(Error::Gpu("memset failed".into()))
        }
    }
    let mut a = Arena::new();
    a.alloc("w", 64, Class::Weights).expect("alloc");
    let mut only_weights = a.freeze(|_| Ok(Failing)).expect("freeze");
    assert_eq!(
        only_weights.scrub_client_data().expect("nothing to zero"),
        0
    );
    let mut a = Arena::new();
    a.alloc("c", 64, Class::ClientData).expect("alloc");
    let mut f = a.freeze(|_| Ok(Failing)).expect("freeze");
    assert!(matches!(f.scrub_client_data(), Err(Error::Gpu(_))));
}
