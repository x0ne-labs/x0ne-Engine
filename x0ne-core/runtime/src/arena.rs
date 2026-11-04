//! The device arena. Every buffer is planned while the arena is `Building`;
//! `freeze` makes the single allocation; `Frozen` has no allocation method at all.
//!
//! A late allocation does not compile:
//!
//! ```compile_fail
//! use x0ne_core_runtime::arena::{Arena, Class, HostBacking};
//! let mut a = Arena::new();
//! a.alloc("weights", 64, Class::Weights).unwrap();
//! let mut f = a.freeze(HostBacking::allocate).unwrap();
//! f.alloc("late", 64, Class::ClientData).unwrap();
//! ```
//!
//! Nor does reusing the `Building` arena after `freeze`:
//!
//! ```compile_fail
//! use x0ne_core_runtime::arena::{Arena, Class, HostBacking};
//! let mut a = Arena::new();
//! let _f = a.freeze(HostBacking::allocate).unwrap();
//! a.alloc("late", 64, Class::ClientData).unwrap();
//! ```
//!
//! The same code without the offending line compiles and runs:
//!
//! ```
//! use x0ne_core_runtime::arena::{Arena, Class, HostBacking};
//! let mut a = Arena::new();
//! let w = a.alloc("weights", 64, Class::Weights).unwrap();
//! let mut f = a.freeze(HostBacking::allocate).unwrap();
//! f.write(w, 0, &[1, 2, 3]).unwrap();
//! ```

use crate::error::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

/// Offsets are multiples of this. cudaMalloc gives 256-byte alignment; keep it per slice.
pub const ALIGN: u64 = 256;

/// What a region holds. `ClientData` is everything derived from a request:
/// KV cache, activations, logits, scratch, cuBLAS workspace. `scrub` zeroes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Weights,
    ClientData,
}

/// A planned region. Only valid for the arena that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    arena: u64,
    offset: u64,
    len: u64,
}

impl Slice {
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, Clone)]
pub struct Region {
    pub label: String,
    pub class: Class,
    pub offset: u64,
    pub len: u64,
}

/// The memory behind a frozen arena: host memory in tests, device memory in E1.
pub trait Backing: Sized {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn write(&mut self, offset: u64, data: &[u8]) -> Result<()>;
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()>;
    fn zero(&mut self, offset: u64, len: u64) -> Result<()>;
}

/// A backing whose bytes have a device address (one contiguous allocation).
pub trait DeviceAddressed: Backing {
    fn base_addr(&self) -> u64;
}

/// Host memory, poisoned with 0xA5 so that bytes never written stand out.
#[derive(Debug)]
pub struct HostBacking(Vec<u8>);

impl HostBacking {
    pub fn allocate(len: u64) -> Result<Self> {
        let len =
            usize::try_from(len).map_err(|_| Error::Arena("size exceeds host usize".into()))?;
        Ok(HostBacking(vec![0xA5; len]))
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }
}

fn host_range(total: usize, offset: u64, len: usize) -> Result<std::ops::Range<usize>> {
    let start = usize::try_from(offset).map_err(|_| Error::Arena("offset exceeds usize".into()))?;
    match start.checked_add(len) {
        Some(end) if end <= total => Ok(start..end),
        _ => Err(Error::Arena(format!(
            "range {offset}+{len} outside {total} bytes"
        ))),
    }
}

impl Backing for HostBacking {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let r = host_range(self.0.len(), offset, data.len())?;
        self.0[r].copy_from_slice(data);
        Ok(())
    }
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let r = host_range(self.0.len(), offset, out.len())?;
        out.copy_from_slice(&self.0[r]);
        Ok(())
    }
    fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        let len = usize::try_from(len).map_err(|_| Error::Arena("length exceeds usize".into()))?;
        let r = host_range(self.0.len(), offset, len)?;
        self.0[r].fill(0);
        Ok(())
    }
}

pub struct Building {
    id: u64,
    regions: Vec<Region>,
    size: u64,
}

pub struct Frozen<B: Backing> {
    id: u64,
    regions: Vec<Region>,
    mem: B,
}

pub struct Arena<S> {
    state: S,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Default for Arena<Building> {
    fn default() -> Self {
        Self::new()
    }
}

impl Arena<Building> {
    pub fn new() -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Arena {
            state: Building {
                id,
                regions: Vec::new(),
                size: 0,
            },
        }
    }

    pub fn alloc(&mut self, label: &str, len: u64, class: Class) -> Result<Slice> {
        if len == 0 {
            return Err(Error::Arena(format!("{label}: empty region")));
        }
        let s = &mut self.state;
        let offset = s
            .size
            .checked_next_multiple_of(ALIGN)
            .ok_or_else(|| Error::Arena("size overflow".into()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Arena("size overflow".into()))?;
        s.regions.push(Region {
            label: label.to_owned(),
            class,
            offset,
            len,
        });
        s.size = end;
        Ok(Slice {
            arena: s.id,
            offset,
            len,
        })
    }

    /// Bytes the single allocation will take.
    pub fn planned_bytes(&self) -> u64 {
        self.state.size
    }

    pub fn regions(&self) -> &[Region] {
        &self.state.regions
    }

    /// Makes the one allocation. Consumes the builder: nothing can be added afterwards.
    pub fn freeze<B: Backing>(
        self,
        allocate: impl FnOnce(u64) -> Result<B>,
    ) -> Result<Arena<Frozen<B>>> {
        let Building { id, regions, size } = self.state;
        let mem = allocate(size.max(1))?;
        if mem.len() < size {
            return Err(Error::Arena(format!(
                "backing gave {} bytes for {size}",
                mem.len()
            )));
        }
        Ok(Arena {
            state: Frozen { id, regions, mem },
        })
    }
}

impl<B: Backing> Arena<Frozen<B>> {
    fn check(&self, s: Slice, at: u64, len: usize) -> Result<u64> {
        if s.arena != self.state.id {
            return Err(Error::Arena("slice from another arena".into()));
        }
        match at.checked_add(len as u64) {
            Some(end) if end <= s.len => Ok(s.offset + at),
            _ => Err(Error::Arena(format!(
                "{len} bytes at {at} exceed a {}-byte slice",
                s.len
            ))),
        }
    }

    pub fn write(&mut self, s: Slice, at: u64, data: &[u8]) -> Result<()> {
        let off = self.check(s, at, data.len())?;
        self.state.mem.write(off, data)
    }

    pub fn read(&self, s: Slice, at: u64, out: &mut [u8]) -> Result<()> {
        let off = self.check(s, at, out.len())?;
        self.state.mem.read(off, out)
    }

    /// Zeroes every `ClientData` region; returns the bytes zeroed. Weights are untouched.
    pub fn scrub_client_data(&mut self) -> Result<u64> {
        let mut total = 0;
        for r in self
            .state
            .regions
            .iter()
            .filter(|r| r.class == Class::ClientData)
        {
            self.state.mem.zero(r.offset, r.len)?;
            total += r.len;
        }
        Ok(total)
    }

    pub fn regions(&self) -> &[Region] {
        &self.state.regions
    }

    pub fn backing(&self) -> &B {
        &self.state.mem
    }
}

impl<B: DeviceAddressed> Arena<Frozen<B>> {
    /// Device address of byte `at` of slice `s`, checked against the slice.
    pub fn addr(&self, s: Slice, at: u64) -> Result<u64> {
        let off = self.check(s, at, 0)?;
        if at >= s.len {
            return Err(Error::Arena(format!(
                "address {at} past a {}-byte slice",
                s.len
            )));
        }
        Ok(self.state.mem.base_addr() + off)
    }

    /// Device address of slice `s`.
    pub fn ptr(&self, s: Slice) -> Result<u64> {
        self.addr(s, 0)
    }

    /// Device address of the `len` bytes at byte `at` of slice `s`, all of them inside it.
    pub fn span(&self, s: Slice, at: u64, len: u64) -> Result<u64> {
        let n = usize::try_from(len).map_err(|_| Error::Arena(format!("span of {len} bytes")))?;
        if n == 0 {
            return Err(Error::Arena("empty span".into()));
        }
        let off = self.check(s, at, n)?;
        Ok(self.state.mem.base_addr() + off)
    }
}
