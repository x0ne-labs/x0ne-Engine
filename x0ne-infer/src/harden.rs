//! The TD process and the keys it holds.
//!
//! [`process`](crate::harden::process) makes the process non-dumpable and
//! drops its core limit: processes of the same user can neither ptrace it
//! nor open `/proc/<pid>/mem`, and a crash writes no core file (ssh-agent
//! does the same). [`SecretBytes`](crate::harden::SecretBytes) holds a key in
//! `memfd_secret(2)` memory where the kernel offers it (Linux 5.14+,
//! unmapped from the kernel's direct map), else in `mlock`ed anonymous
//! pages; both stay out of swap and core dumps, and both are zeroed before
//! the pages are returned.
//!
//! Not covered: copies made by the libraries we hand a key to (ring's
//! parsed TLS key, the DER rustls takes by value) live in the ordinary heap;
//! nothing a process does stands against root or the kernel inside the
//! guest (`THREAT_MODEL.md`, the attested channel table).

use std::io;
use zeroize::Zeroize;

/// The variables through which the dynamic loader brings a shared object
/// into the process before `main` (`LD_PRELOAD`, `LD_AUDIT`): code the image
/// did not measure, with the process's privileges; and `GLIBC_TUNABLES`, which
/// the loader parses before `main` too (CVE-2023-4911) and which sets glibc's
/// behaviour from outside the image. Refused by every build, whatever their
/// value.
pub const LOADER_VARIABLES: [&str; 3] = ["LD_PRELOAD", "LD_AUDIT", "GLIBC_TUNABLES"];

/// The variables that choose the file a library name opens, refused by a
/// release build whatever their value. `LD_LIBRARY_PATH`: searched before
/// the loader's cache for every name (ld.so(8)), those x0ne Core and NVML
/// hand `dlopen` after `main` included (`libcuda.so.1`, `libcublas.so.13`,
/// `libnvidia-ml.so.1`); a `libnvidia-ml.so.1` put there ran its constructor
/// in a release `x0ne-infer` (2026-11-04). `CUDA_INJECTION64_PATH`: a
/// library the CUDA driver loads and calls at initialisation (CUPTI's
/// `cupti_trace_injection` README, CUDA 13.3.1). `NVTX_INJECTION64_PATH`:
/// the one NVTX loads (`nvtx3/nvtxDetail/nvtxInit.h`), named in cuBLAS and
/// cuBLASLt 13.6. A build with debug assertions takes them: cargo sets
/// `LD_LIBRARY_PATH` for every test and every `cargo run`.
pub const RELEASE_LOADER_VARIABLES: [&str; 3] = [
    "LD_LIBRARY_PATH",
    "CUDA_INJECTION64_PATH",
    "NVTX_INJECTION64_PATH",
];

/// Whether this build refuses [`RELEASE_LOADER_VARIABLES`] too: one built
/// without debug assertions.
pub const RELEASE: bool = !cfg!(debug_assertions);

/// The variables a build refuses for which `is_set` holds: those of
/// [`LOADER_VARIABLES`], then, when `release`, those of
/// [`RELEASE_LOADER_VARIABLES`], in that order.
pub fn loader_variables_among(is_set: impl Fn(&str) -> bool, release: bool) -> Vec<&'static str> {
    let extra: &[&'static str] = if release {
        &RELEASE_LOADER_VARIABLES
    } else {
        &[]
    };
    LOADER_VARIABLES
        .iter()
        .chain(extra)
        .copied()
        .filter(|v| is_set(v))
        .collect()
}

/// `Err` naming the variables this build refuses that are set in this
/// process's environment. Call first thing in `main`, before any `dlopen`: a
/// preloaded object, or one the loader found on `LD_LIBRARY_PATH` for a
/// library the binary links, has run its constructors by then, so for those
/// this stops a launcher's mistake from serving, it does not undo it.
pub fn no_loader_variables() -> Result<(), String> {
    let found = loader_variables_among(|v| std::env::var_os(v).is_some(), RELEASE);
    if found.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} set: the process would load code or take settings the image did not measure",
        found.join(", ")
    ))
}

/// `PR_SET_DUMPABLE 0` and `RLIMIT_CORE 0`. Call before reading a key or
/// starting a thread.
pub fn process() -> io::Result<()> {
    #[cfg(target_os = "linux")]
    // SAFETY: integer arguments only; no memory is shared with the kernel.
    unsafe {
        let zero: libc::c_ulong = 0;
        if libc::prctl(libc::PR_SET_DUMPABLE, zero, zero, zero, zero) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(unix)]
    // SAFETY: `none` outlives the call; setrlimit only reads it.
    unsafe {
        let none = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::setrlimit(libc::RLIMIT_CORE, &none) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// What `prctl(PR_GET_DUMPABLE)` says; `true` where there is no such call.
pub fn is_dumpable() -> bool {
    #[cfg(target_os = "linux")]
    // SAFETY: read-only query.
    unsafe {
        libc::prctl(libc::PR_GET_DUMPABLE) != 0
    }
    #[cfg(not(target_os = "linux"))]
    true
}

/// Where a [`SecretBytes`] lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// `memfd_secret(2)`: unevictable, so never swapped; not dumped, absent
    /// from the kernel's direct map, not shared with a forked child.
    SecretMem,
    /// Anonymous pages: `mlock`, `MADV_DONTDUMP`, `MADV_WIPEONFORK`.
    Locked,
    /// Not Linux: the heap, zeroed on drop and nothing more.
    Heap,
}

/// A key in memory kept out of swap and core dumps, and zeroed on drop. On
/// Linux it is never on the heap: if neither `memfd_secret` nor `mlock`
/// works, construction fails. `memfd_secret` pages are written at
/// construction: a kernel with Linux commit 97d34aa65c29 counts them against
/// `RLIMIT_MEMLOCK` then, and past it construction fails, not a later write.
pub struct SecretBytes {
    region: Region,
    len: usize,
}

enum Region {
    #[cfg(target_os = "linux")]
    Mapped {
        ptr: std::ptr::NonNull<u8>,
        map_len: usize,
        backing: Backing,
    },
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    Heap(Box<[u8]>),
}

// SAFETY: the mapping is owned by this value alone; `&self` only reads it.
unsafe impl Send for SecretBytes {}
// SAFETY: as above; mutation needs `&mut self`.
unsafe impl Sync for SecretBytes {}

impl SecretBytes {
    /// `len` zero bytes.
    pub fn new(len: usize) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            linux::map(len, Backing::SecretMem).or_else(|secret| {
                linux::map(len, Backing::Locked).map_err(|locked| {
                    io::Error::new(
                        locked.kind(),
                        format!(
                            "neither memfd_secret ({secret}) nor mlock ({locked}) holds {len} \
                             bytes; {}",
                            linux::memlock_limit()
                        ),
                    )
                })
            })
        }
        #[cfg(not(target_os = "linux"))]
        Ok(Self {
            region: Region::Heap(vec![0; len].into_boxed_slice()),
            len,
        })
    }

    pub fn from_slice(bytes: &[u8]) -> io::Result<Self> {
        let mut s = Self::new(bytes.len())?;
        s.expose_mut().copy_from_slice(bytes);
        Ok(s)
    }

    /// One backing, no fallback: what the tests compare.
    #[cfg(all(test, target_os = "linux"))]
    fn with_backing(len: usize, want: Backing) -> io::Result<Self> {
        linux::map(len, want)
    }

    pub fn backing(&self) -> Backing {
        match &self.region {
            #[cfg(target_os = "linux")]
            Region::Mapped { backing, .. } => *backing,
            Region::Heap(_) => Backing::Heap,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn expose(&self) -> &[u8] {
        match &self.region {
            #[cfg(target_os = "linux")]
            // SAFETY: `ptr` maps `map_len >= len` readable bytes owned by self.
            Region::Mapped { ptr, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), self.len)
            },
            Region::Heap(b) => &b[..self.len],
        }
    }

    pub fn expose_mut(&mut self) -> &mut [u8] {
        match &mut self.region {
            #[cfg(target_os = "linux")]
            // SAFETY: as in `expose`, and `&mut self` makes it exclusive.
            Region::Mapped { ptr, .. } => unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr(), self.len)
            },
            Region::Heap(b) => &mut b[..self.len],
        }
    }

    /// The whole region, padding included: what the wipe covers.
    fn region_mut(&mut self) -> &mut [u8] {
        match &mut self.region {
            #[cfg(target_os = "linux")]
            // SAFETY: the full mapping, owned by self.
            Region::Mapped { ptr, map_len, .. } => unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr(), *map_len)
            },
            Region::Heap(b) => b,
        }
    }
}

impl std::fmt::Debug for SecretBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretBytes({} bytes, {:?})", self.len, self.backing())
    }
}

#[cfg(test)]
thread_local! {
    /// Per drop: non-zero bytes before the wipe, and after it.
    static WIPES: std::cell::RefCell<Vec<(usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for SecretBytes {
    fn drop(&mut self) {
        let region = self.region_mut();
        #[cfg(test)]
        let before = region.iter().filter(|b| **b != 0).count();
        region.zeroize();
        #[cfg(test)]
        {
            let after = region.iter().filter(|b| **b != 0).count();
            WIPES.with(|w| w.borrow_mut().push((before, after)));
        }
        #[cfg(target_os = "linux")]
        if let Region::Mapped {
            ptr,
            map_len,
            backing,
        } = &self.region
        {
            linux::unmap(*ptr, *map_len, *backing);
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{Backing, Region, SecretBytes};
    use std::io;
    use std::ptr::NonNull;

    fn page() -> usize {
        // SAFETY: read-only query.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(p).ok().filter(|p| *p > 0).unwrap_or(4096)
    }

    fn last() -> io::Error {
        io::Error::last_os_error()
    }

    pub(super) fn map(len: usize, backing: Backing) -> io::Result<SecretBytes> {
        let page = page();
        let map_len = len.max(1).div_ceil(page) * page;
        let ptr = match backing {
            Backing::SecretMem => secretmem(map_len)?,
            Backing::Locked => locked(map_len)?,
            Backing::Heap => return Err(io::ErrorKind::Unsupported.into()),
        };
        Ok(SecretBytes {
            region: Region::Mapped {
                ptr,
                map_len,
                backing,
            },
            len,
        })
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn secretmem(map_len: usize) -> io::Result<NonNull<u8>> {
        // SAFETY: fd syscalls on a descriptor created and closed here; the
        // mapping is checked before use and unmapped on every error path.
        unsafe {
            let fd = libc::syscall(libc::SYS_memfd_secret, libc::O_CLOEXEC);
            if fd < 0 {
                return Err(last());
            }
            let fd = fd as libc::c_int;
            let mapped = (|| {
                if libc::ftruncate(fd, map_len as libc::off_t) != 0 {
                    return Err(last());
                }
                let p = libc::mmap(
                    std::ptr::null_mut(),
                    map_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                );
                if p == libc::MAP_FAILED {
                    return Err(last());
                }
                // A shared mapping would follow a fork.
                if libc::madvise(p, map_len, libc::MADV_DONTFORK) != 0 {
                    let e = last();
                    libc::munmap(p, map_len);
                    return Err(e);
                }
                // Over RLIMIT_MEMLOCK the mmap fails (EAGAIN, 6.12, 2026-10-13); since
                // 97d34aa65c29 the first write does, by SIGBUS (6.12.115, 2026-11-05).
                // A kernel copy into the pages gets EFAULT there instead: done now.
                if let Err(e) = fault_in(p, map_len) {
                    libc::munmap(p, map_len);
                    return Err(e);
                }
                Ok(NonNull::new(p.cast::<u8>()).expect("mmap returned null"))
            })();
            libc::close(fd);
            mapped
        }
    }

    /// Writes zeros over `p..p + len` by `read(2)` from `/dev/zero`: a fault
    /// the kernel refuses is EFAULT here, where a store from this process
    /// would get SIGBUS.
    ///
    /// # Safety
    /// `p..p + len` is a writable mapping that nothing else refers to.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe fn fault_in(p: *mut libc::c_void, len: usize) -> io::Result<()> {
        // SAFETY: an fd opened and closed here; `read` writes inside the
        // mapping only, at most `len - done` bytes from `p + done`.
        unsafe {
            let zero = libc::open(c"/dev/zero".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
            if zero < 0 {
                let e = last();
                return Err(io::Error::new(e.kind(), format!("/dev/zero: {e}")));
            }
            let mut done = 0;
            let out = loop {
                if done == len {
                    break Ok(());
                }
                let n = libc::read(zero, p.cast::<u8>().add(done).cast(), len - done);
                match usize::try_from(n) {
                    Ok(0) => break Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
                    Ok(n) => done += n,
                    Err(_) => {
                        let e = last();
                        if e.kind() != io::ErrorKind::Interrupted {
                            break Err(e);
                        }
                    }
                }
            };
            libc::close(zero);
            out.map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!(
                        "its pages could not be written ({e}); a kernel with Linux commit \
                         97d34aa65c29 refuses them past the user's RLIMIT_MEMLOCK"
                    ),
                )
            })
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn secretmem(_map_len: usize) -> io::Result<NonNull<u8>> {
        Err(io::ErrorKind::Unsupported.into())
    }

    /// `RLIMIT_MEMLOCK`'s soft limit, for a message.
    pub(super) fn memlock_limit() -> String {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `lim` outlives the call; getrlimit only writes it.
        if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) } != 0 {
            return format!("RLIMIT_MEMLOCK unread ({})", last());
        }
        if lim.rlim_cur == libc::RLIM_INFINITY {
            return "RLIMIT_MEMLOCK unlimited".into();
        }
        format!("RLIMIT_MEMLOCK {} bytes", lim.rlim_cur)
    }

    fn locked(map_len: usize) -> io::Result<NonNull<u8>> {
        // SAFETY: an anonymous private mapping created, advised and, on any
        // error, unmapped here.
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                return Err(last());
            }
            let fail = |e: io::Error| {
                libc::munmap(p, map_len);
                Err(e)
            };
            if libc::mlock(p, map_len) != 0 {
                return fail(last());
            }
            for advice in [libc::MADV_DONTDUMP, libc::MADV_WIPEONFORK] {
                if libc::madvise(p, map_len, advice) != 0 {
                    return fail(last());
                }
            }
            Ok(NonNull::new(p.cast::<u8>()).expect("mmap returned null"))
        }
    }

    pub(super) fn unmap(ptr: NonNull<u8>, map_len: usize, backing: Backing) {
        // SAFETY: the mapping this value created; nothing refers to it after drop.
        unsafe {
            if backing == Backing::Locked {
                libc::munlock(ptr.as_ptr().cast(), map_len);
            }
            libc::munmap(ptr.as_ptr().cast(), map_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SECURITY: each loader variable is found alone and with the others, by
    /// every build; the variables that choose a library's file by a release
    /// build only. `LD_LIBRARY_PATH`, which cargo sets for this very test, is
    /// found by none of the debug lists.
    #[test]
    fn the_loader_variables_are_found_and_only_them() {
        for release in [false, true] {
            for var in ["LD_PRELOAD", "LD_AUDIT", "GLIBC_TUNABLES"] {
                assert_eq!(loader_variables_among(|v| v == var, release), [var]);
            }
            assert!(loader_variables_among(|_| false, release).is_empty());
        }
        for var in [
            "LD_LIBRARY_PATH",
            "CUDA_INJECTION64_PATH",
            "NVTX_INJECTION64_PATH",
        ] {
            assert_eq!(loader_variables_among(|v| v == var, true), [var]);
            assert!(loader_variables_among(|v| v == var, false).is_empty());
        }
        assert_eq!(
            loader_variables_among(|_| true, false),
            LOADER_VARIABLES,
            "the three of every build, in the documented order"
        );
        let all: Vec<&str> = LOADER_VARIABLES
            .iter()
            .chain(&RELEASE_LOADER_VARIABLES)
            .copied()
            .collect();
        assert_eq!(loader_variables_among(|_| true, true), all);
        assert_eq!(RELEASE, !cfg!(debug_assertions));
    }

    fn last_wipe() -> (usize, usize) {
        WIPES.with(|w| *w.borrow().last().expect("a wipe was recorded"))
    }

    /// Instrumented drop: the secret is in the region until the wipe, and
    /// not one byte of it after, on every backing this machine offers.
    #[test]
    fn a_dropped_secret_is_wiped_before_its_pages_are_returned() {
        #[allow(unused_mut)]
        let mut made = vec![SecretBytes::new(32).unwrap()];
        #[cfg(target_os = "linux")]
        made.push(SecretBytes::with_backing(32, Backing::Locked).unwrap());
        for mut s in made {
            let backing = s.backing();
            s.expose_mut().copy_from_slice(&[0xa5; 32]);
            assert_eq!(s.expose(), &[0xa5; 32]);
            drop(s);
            assert_eq!(last_wipe(), (32, 0), "{backing:?}");
        }
    }

    #[test]
    fn debug_names_the_length_not_the_bytes() {
        let s = SecretBytes::from_slice(b"\x01\x02\x03\x04").unwrap();
        assert_eq!(
            format!("{s:?}"),
            format!("SecretBytes(4 bytes, {:?})", s.backing())
        );
        assert_eq!(s.len(), 4);
    }

    /// The `/proc/self/smaps` block of the mapping that holds `addr`.
    #[cfg(target_os = "linux")]
    fn vma_of(addr: usize) -> String {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mut block = String::new();
        let mut inside = false;
        for line in smaps.lines() {
            let head = line
                .split_whitespace()
                .next()
                .and_then(|r| r.split_once('-'))
                .and_then(|(a, b)| {
                    Some((
                        usize::from_str_radix(a, 16).ok()?,
                        usize::from_str_radix(b, 16).ok()?,
                    ))
                });
            if let Some((start, end)) = head {
                if inside {
                    break;
                }
                inside = (start..end).contains(&addr);
            }
            if inside {
                block.push_str(line);
                block.push('\n');
            }
        }
        assert!(!block.is_empty(), "no mapping holds {addr:#x}");
        block
    }

    fn vm_flags(block: &str) -> Vec<String> {
        block
            .lines()
            .find_map(|l| l.strip_prefix("VmFlags:"))
            .expect("VmFlags")
            .split_whitespace()
            .map(str::to_string)
            .collect()
    }

    /// The file a smaps block maps: what follows the inode on its first line.
    /// A `memfd_secret` mapping is `/secretmem (deleted)`; an ordinary memfd
    /// is `/memfd:<name> (deleted)`, whatever name it was given.
    #[cfg(target_os = "linux")]
    fn mapped_file(block: &str) -> String {
        let head = block.lines().next().unwrap_or_default();
        head.split_whitespace()
            .skip(5)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Whether the kernel gives this process a `memfd_secret` descriptor,
    /// asked without `SecretBytes`: where it does, a key in `mlock` memory is
    /// a regression, not a fallback.
    #[cfg(target_os = "linux")]
    fn kernel_offers_memfd_secret() -> io::Result<()> {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        // SAFETY: a descriptor created and closed here.
        unsafe {
            let fd = libc::syscall(libc::SYS_memfd_secret, libc::O_CLOEXEC);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            libc::close(fd as libc::c_int);
            Ok(())
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Err(io::ErrorKind::Unsupported.into())
    }

    /// The flags every `mlock` key carries: locked, not dumped, wiped in a fork.
    #[cfg(target_os = "linux")]
    fn assert_locked_flags(block: &str) {
        let flags = vm_flags(block);
        for f in ["lo", "dd", "wf"] {
            assert!(flags.iter().any(|x| x == f), "Locked lacks {f}: {block}");
        }
    }

    /// This process's cgroup v2 `memory.stat`, when it has one with an
    /// `unevictable` line: other cgroups' pages do not move it.
    #[cfg(target_os = "linux")]
    fn own_memory_stat() -> Option<std::path::PathBuf> {
        let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
        let dir = cgroup.lines().find_map(|l| l.strip_prefix("0::"))?;
        let stat = std::path::Path::new("/sys/fs/cgroup")
            .join(dir.trim_start_matches('/'))
            .join("memory.stat");
        let text = std::fs::read_to_string(&stat).ok()?;
        text.lines()
            .any(|l| l.starts_with("unevictable "))
            .then_some(stat)
    }

    /// `Unevictable` in kB: pages reclaim never takes, so never swapped
    /// (`mlock`ed, or of an unevictable mapping). From `memory_stat` when
    /// given, else from the machine's `/proc/meminfo`.
    #[cfg(target_os = "linux")]
    fn unevictable_kb(memory_stat: Option<&std::path::Path>) -> u64 {
        let (path, key, unit) = match memory_stat {
            Some(p) => (p, "unevictable ", 1024),
            None => (std::path::Path::new("/proc/meminfo"), "Unevictable:", 1),
        };
        let text = std::fs::read_to_string(path).unwrap();
        let value = text
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .map(|v| v.trim().trim_end_matches("kB").trim())
            .and_then(|v| v.parse::<u64>().ok());
        value.unwrap_or_else(|| panic!("{key} in {}", path.display())) / unit
    }

    /// `region` written from /dev/zero.
    #[cfg(target_os = "linux")]
    fn write_zeros(region: &mut [u8]) {
        use std::io::Read;
        // A kernel copy: past the limit a fault is EFAULT here, SIGBUS in user code.
        let mut zero = std::fs::File::open("/dev/zero").unwrap();
        zero.read_exact(region).expect("fault the region in");
    }

    /// kB of `Unevictable` gained by `fault`, while what it returns lives,
    /// read until it reaches `want` kB or 2 s pass: per-CPU counts reach the
    /// counter up to a second late.
    #[cfg(target_os = "linux")]
    fn unevictable_gain<T>(
        memory_stat: Option<&std::path::Path>,
        want: u64,
        fault: impl FnOnce() -> T,
    ) -> u64 {
        let before = unevictable_kb(memory_stat);
        let _held = fault();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let gain = unevictable_kb(memory_stat).saturating_sub(before);
            if gain >= want || std::time::Instant::now() >= deadline {
                return gain;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// (region, its gain, the control's gain) in kB, and the counter read:
    /// `memfd_secret` pages against as many ordinary anonymous ones. `None`
    /// when RLIMIT_MEMLOCK, which bounds `memfd_secret` memory, leaves no room
    /// for 1 MiB.
    #[cfg(target_os = "linux")]
    fn secretmem_and_ordinary_gains() -> Option<(u64, u64, u64, String)> {
        // make ci's three test profiles share one container and may run this test at
        // once: copies started 1 s apart failed their control (2026-11-06). One at a time.
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(std::env::temp_dir().join("x0ne-infer-unevictable.lock"))
            .expect("the lock file of the Unevictable measure");
        lock.lock().expect("lock the Unevictable measure");
        let stat = own_memory_stat();
        let counter = stat
            .as_deref()
            .map_or("/proc/meminfo".into(), |p| p.display().to_string());
        let stat = stat.as_deref();
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `lim` outlives the call; getrlimit only writes it.
        let got = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) };
        assert_eq!(got, 0);
        // Half the limit at most: the rest stays for the other tests' keys.
        let half = usize::try_from(lim.rlim_cur / 2).unwrap_or(usize::MAX);
        let len = half.min(4 << 20) & !0xfffff;
        if len == 0 {
            return None;
        }
        let want = (len / 1024) as u64;
        // SAFETY: a private anonymous mapping, written through the slice
        // below while it lives, unmapped here.
        let control = unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert_ne!(p, libc::MAP_FAILED);
            let region = std::slice::from_raw_parts_mut(p.cast::<u8>(), len);
            let gain = unevictable_gain(stat, want, || write_zeros(region));
            libc::munmap(p, len);
            gain
        };
        // Its pages are written when it is made (`fault_in`), then once more.
        let secret = unevictable_gain(stat, want, || {
            let mut s = SecretBytes::with_backing(len, Backing::SecretMem).expect("memfd_secret");
            write_zeros(s.expose_mut());
            s
        });
        drop(lock);
        Some((want, secret, control, counter))
    }

    /// Linux: out of swap and core dumps, not inherited by a fork; in
    /// `memfd_secret` memory wherever the kernel gives this process some.
    #[cfg(target_os = "linux")]
    #[test]
    fn on_linux_the_secret_is_unevictable_undumped_and_not_inherited() {
        let s = SecretBytes::from_slice(&[1; 32]).unwrap();
        let block = vma_of(s.expose().as_ptr() as usize);
        let flags = vm_flags(&block);
        let has = |f: &str| flags.iter().any(|x| x == f);
        match kernel_offers_memfd_secret() {
            Ok(()) => {
                let why = SecretBytes::with_backing(32, Backing::SecretMem).err();
                assert_eq!(s.backing(), Backing::SecretMem, "memfd_secret: {why:?}");
                assert_eq!(mapped_file(&block), "/secretmem (deleted)", "{block}");
                // dd: the kernel's VM_DONTDUMP; dc: our MADV_DONTFORK.
                assert!(has("dd") && has("dc"), "{block}");
                // No VM_LOCKED ("lo") since 97d34aa65c29 (stable 6.12.113; 6.12.115, 2026-11-05):
                // the pages stay unevictable, which is what keeps them out of swap.
                match secretmem_and_ordinary_gains() {
                    Some((len, secret, control, counter)) => {
                        assert!(
                            control < len / 4,
                            "control: {control} of {len} kB, {counter}"
                        );
                        assert!(
                            secret >= len * 3 / 4,
                            "{secret} of {len} kB, {counter}: {block}"
                        );
                    }
                    None => assert!(
                        has("lo"),
                        "RLIMIT_MEMLOCK leaves no room to measure that the pages are \
                         unevictable, and this kernel does not mark them VM_LOCKED: {block}"
                    ),
                }
            }
            Err(e) => {
                eprintln!("memfd_secret refused here ({e}): the key must be in mlock memory");
                assert_eq!(s.backing(), Backing::Locked, "{block}");
                assert_locked_flags(&block);
            }
        }
        let locked = SecretBytes::with_backing(32, Backing::Locked).unwrap();
        assert_locked_flags(&vma_of(locked.expose().as_ptr() as usize));
    }

    /// `memfd_secret` pages are out of the kernel's direct map: not even this
    /// process reads them through `/proc/self/mem`; a heap buffer, the
    /// control, reads back.
    #[cfg(target_os = "linux")]
    #[test]
    fn secretmem_is_out_of_reach_of_proc_mem() {
        use std::os::unix::fs::FileExt;
        let mut s = match SecretBytes::with_backing(32, Backing::SecretMem) {
            Ok(s) => s,
            Err(e) => {
                if let Err(refused) = kernel_offers_memfd_secret() {
                    eprintln!("memfd_secret refused here ({refused}): nothing to read");
                    return;
                }
                panic!("the kernel offers memfd_secret, SecretBytes failed with it: {e}");
            }
        };
        s.expose_mut().copy_from_slice(&[0x5a; 32]);
        let mem = std::fs::File::open("/proc/self/mem").unwrap();
        let control = Box::new([0x5au8; 32]);
        let mut buf = [0u8; 32];
        mem.read_exact_at(&mut buf, control.as_ptr() as u64)
            .unwrap();
        assert_eq!(buf, [0x5a; 32], "control: the heap reads back");
        let mut buf = [0u8; 32];
        assert!(
            mem.read_exact_at(&mut buf, s.expose().as_ptr() as u64)
                .is_err(),
            "secretmem read through /proc/self/mem: {buf:?}"
        );
    }

    /// 32 bytes at `addr` of this process, read with `process_vm_readv(2)`.
    #[cfg(target_os = "linux")]
    fn process_vm_read(addr: *const u8) -> io::Result<[u8; 32]> {
        let mut buf = [0u8; 32];
        let local = libc::iovec {
            iov_base: buf.as_mut_ptr().cast(),
            iov_len: buf.len(),
        };
        let remote = libc::iovec {
            iov_base: addr.cast_mut().cast(),
            iov_len: buf.len(),
        };
        // SAFETY: `local` points at `buf`, which outlives the call; the kernel
        // checks `remote` against this process's mappings itself.
        let n = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        match usize::try_from(n) {
            Ok(32) => Ok(buf),
            Ok(_) => Err(io::ErrorKind::UnexpectedEof.into()),
            Err(_) => Err(io::Error::last_os_error()),
        }
    }

    /// `memfd_secret` pages refuse `process_vm_readv(2)`, the call a process
    /// of the same user would make, even from this process (EFAULT); a heap
    /// buffer, the control, reads back.
    #[cfg(target_os = "linux")]
    #[test]
    fn secretmem_is_out_of_reach_of_process_vm_readv() {
        let mut s = match SecretBytes::with_backing(32, Backing::SecretMem) {
            Ok(s) => s,
            Err(e) => {
                if let Err(refused) = kernel_offers_memfd_secret() {
                    eprintln!("memfd_secret refused here ({refused}): nothing to read");
                    return;
                }
                panic!("the kernel offers memfd_secret, SecretBytes failed with it: {e}");
            }
        };
        s.expose_mut().copy_from_slice(&[0x5a; 32]);
        let control = Box::new([0x5au8; 32]);
        let back = process_vm_read(control.as_ptr());
        assert_eq!(back.ok(), Some([0x5a; 32]), "control: the heap reads back");
        let got = process_vm_read(s.expose().as_ptr());
        assert_eq!(
            got.as_ref().map_err(io::Error::raw_os_error),
            Err(Some(libc::EFAULT)),
            "secretmem read through process_vm_readv: {got:?}"
        );
    }
}
