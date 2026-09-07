//! Stream P: where a chunk's `build_secs` goes.
//!
//! `build_secs` (`chunks.rs`) is the wall of turning one chunk's witness
//! directory into a `Layer`. Four different kinds of work happen in there --
//! the files are read, their bytes are decoded into arrays, some of what they
//! carry is recomputed and compared against them, and the objects every region
//! reads are built -- and on the sequence that one number is 86% of the
//! generation thread's own time, a term no boundary this program reports has
//! ever counted (`briefs/o-overlap.md` section 5). Which of the four dominates
//! decides which fix is worth building, so the split comes first.
//!
//! A timed place is a `Site`; every site belongs to one `Kind`, and a kind's
//! total is the sum over its sites. Both views are published: the four
//! `*_secs` the brief names, and the per-site map underneath them, because
//! "construct" is a dozen separate loops and a fix has to be aimed at one of
//! them rather than at the word.
//!
//! Time is charged **exclusively**: a scope charges its own elapsed time less
//! whatever the scopes that opened and closed inside it charged, so a site may
//! wrap a region that contains other sites and still report only its own part.
//! The bookkeeping for that is per thread, which is what the loaders are
//! today: one thread builds one chunk's layer, and the parallel loops inside
//! `Attn::build` open no scopes of their own. A region that is *made* parallel
//! has to keep its scope on the thread that opens it and open none inside the
//! workers, or the outer scope will count the workers' wall and the workers
//! will charge the same time again.
//!
//! The counters are process-wide and a caller reads them by difference:
//! `snapshot()` before a region and `since(&snap)` after it. That is how
//! `chunks.rs` attributes the work to a chunk, and it is why nothing here
//! needs to know what a chunk is.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use serde_json::{Value, json};

/// The four kinds of work the brief splits `build_secs` into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// `std::fs::read` and its like: bytes off disk, or out of the page cache,
    /// before anything looks at them.
    Read,
    /// Turning those bytes into the arrays the prover holds: the element-wise
    /// widenings, the truncations, the tiling gathers, one fresh `Vec` per
    /// column.
    Decode,
    /// Recomputation the loader does on purpose, to compare against what the
    /// dump claims. These are load-bearing -- they are what catches an
    /// extractor/AIR disagreement -- and they may only get faster.
    Check,
    /// Building the objects the regions read: the scores, the exp table, the
    /// numerators, the denominators, the A.V outputs, the per-key-tile tables.
    Construct,
    /// Stream W: getting and giving back the memory the four above run in.
    /// A chunk builds and frees several GB of `Vec` and stream P could not see
    /// any of it -- every allocation sat between two scopes rather than inside
    /// one. This kind exists so that the seconds either show up here or are
    /// shown not to be here, which is what decides whether an arena is the fix.
    Alloc,
}

impl Kind {
    pub fn key(self) -> &'static str {
        match self {
            Kind::Read => "read_secs",
            Kind::Decode => "decode_secs",
            Kind::Check => "check_secs",
            Kind::Construct => "construct_secs",
            Kind::Alloc => "alloc_secs",
        }
    }
}

/// Every timed place in the loaders. The name is what a run's JSON prints, and
/// it is `kind/what` so a reader can add the parts back up to their kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Site {
    /// one whole witness file, read and counted
    ReadFile,
    /// `manifest.json`, which is read once per key tile today
    ReadManifest,
    /// the per-element widenings in the read helpers
    DecodeU16,
    DecodeI32,
    DecodeI8,
    DecodeI16,
    /// a column read whole and then cut to `tokens x d`
    DecodeTrunc,
    /// a second copy of an array that was just decoded, to change its width
    /// again or to pair it up
    DecodeRecast,
    /// the head-cycling, S-and-C-tiling copies that put a dump's heads into a
    /// key tile's layout
    DecodeGather,
    /// rule 21: both diagonal block halves' denominators against the dump's `zz.i32`
    CheckRule21,
    /// every key tile reads the same master exp table, asserted by comparing them
    CheckTableEq,
    /// torch's own numerators against the ones the prover just derived
    CheckTorchNum,
    /// parsing the manifest text
    ManifestParse,
    /// the exact integer scores, `Q . K` per (instance, token)
    Scores,
    /// the row maxima, the shifts, the per-instance exponents and the scans
    /// over them
    Maxima,
    /// the exp table itself
    Table,
    /// the committed index and the numerator the table gives for it
    Numerators,
    /// the per-token denominators
    Denominator,
    /// the A.V accumulators
    Av,
    /// a prefix key tile's total denominator over the pair
    ZzTotal,
    /// Stream W: `--tight-table`'s two passes over every key tile's shifts -- the
    /// largest exp-table key the layer looks up, and then the count of keys the
    /// shrunk table can no longer hold. Both run in `main.rs`'s `tighten_of`,
    /// after `Layer::load` has returned and inside the same `build_secs`, which
    /// is why stream P's split could not see them.
    Tighten,
    /// Stream W: one big buffer got, or given back.
    Alloc,
    /// the per-key-tile loop's own work, outside the sites above
    KeyTiles,
    /// the seven matmuls and the head, assembled
    Mats,
    /// the attnout region's accumulation over the key tiles
    AttnOut,
}

impl Site {
    pub const ALL: [Site; 25] = [
        Site::ReadFile, Site::ReadManifest,
        Site::DecodeU16, Site::DecodeI32, Site::DecodeI8, Site::DecodeI16,
        Site::DecodeTrunc, Site::DecodeRecast, Site::DecodeGather,
        Site::CheckRule21, Site::CheckTableEq, Site::CheckTorchNum,
        Site::ManifestParse, Site::Scores, Site::Maxima, Site::Table,
        Site::Numerators, Site::Denominator, Site::Av, Site::ZzTotal,
        Site::Tighten, Site::Alloc,
        Site::KeyTiles, Site::Mats, Site::AttnOut,
    ];

    pub fn kind(self) -> Kind {
        match self {
            Site::ReadFile | Site::ReadManifest => Kind::Read,
            Site::DecodeU16 | Site::DecodeI32 | Site::DecodeI8 | Site::DecodeI16
            | Site::DecodeTrunc | Site::DecodeRecast | Site::DecodeGather => Kind::Decode,
            Site::CheckRule21 | Site::CheckTableEq | Site::CheckTorchNum => Kind::Check,
            Site::Alloc => Kind::Alloc,
            _ => Kind::Construct,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Site::ReadFile => "read/file",
            Site::ReadManifest => "read/manifest",
            Site::DecodeU16 => "decode/u16",
            Site::DecodeI32 => "decode/i32",
            Site::DecodeI8 => "decode/i8",
            Site::DecodeI16 => "decode/i16",
            Site::DecodeTrunc => "decode/trunc",
            Site::DecodeRecast => "decode/recast",
            Site::DecodeGather => "decode/gather",
            Site::CheckRule21 => "check/rule21",
            Site::CheckTableEq => "check/table_eq",
            Site::CheckTorchNum => "check/torch_num",
            Site::ManifestParse => "construct/manifest_parse",
            Site::Scores => "construct/scores",
            Site::Maxima => "construct/maxima",
            Site::Table => "construct/table",
            Site::Numerators => "construct/numerators",
            Site::Denominator => "construct/denominator",
            Site::Av => "construct/av",
            Site::ZzTotal => "construct/zz_total",
            Site::Tighten => "construct/tighten",
            Site::Alloc => "alloc/pool",
            Site::KeyTiles => "construct/key_tiles",
            Site::Mats => "construct/mats",
            Site::AttnOut => "construct/attnout",
        }
    }

    fn idx(self) -> usize {
        Site::ALL.iter().position(|s| *s == self).expect("every site is in ALL")
    }
}

static NANOS: [AtomicU64; Site::ALL.len()] = [const { AtomicU64::new(0) }; Site::ALL.len()];
static BYTES: AtomicU64 = AtomicU64::new(0);
static FILES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// What the scopes that have closed inside the scope now open on this
    /// thread cost, in nanoseconds. A scope hands its own total up to its
    /// parent here and each subtracts this from its elapsed time, which is
    /// what makes the charge exclusive.
    static CHILD: Cell<u64> = const { Cell::new(0) };
}

/// A timed region. Charges its site on drop.
pub struct Scope {
    site: Site,
    t: Instant,
    outer: u64,
}

pub fn scope(site: Site) -> Scope {
    Scope { site, t: Instant::now(), outer: CHILD.replace(0) }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let total = self.t.elapsed().as_nanos() as u64;
        let inner = CHILD.replace(self.outer + total);
        NANOS[self.site.idx()].fetch_add(total.saturating_sub(inner), Relaxed);
    }
}

/// Read a witness file whole: the read is timed as `read/file` and its bytes
/// are counted, so the read row can carry a rate. Every raw read in the
/// loaders goes through here.
pub fn read_file(path: &str) -> Result<Vec<u8>, String> {
    let raw = {
        let _s = scope(Site::ReadFile);
        std::fs::read(path).map_err(|e| format!("{path}: {e}"))?
    };
    BYTES.fetch_add(raw.len() as u64, Relaxed);
    FILES.fetch_add(1, Relaxed);
    Ok(raw)
}

/// `manifest.json`, read and parsed, each half timed under its own site. Every
/// key tile does this today, which is why the two are visible separately.
pub fn read_json(path: &str) -> Result<Value, String> {
    let text = {
        let _s = scope(Site::ReadManifest);
        std::fs::read_to_string(path).map_err(|e| e.to_string())?
    };
    BYTES.fetch_add(text.len() as u64, Relaxed);
    FILES.fetch_add(1, Relaxed);
    let _s = scope(Site::ManifestParse);
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

/// The counters as they stand. A caller times a region by taking one of these
/// before it and calling `since` after it.
#[derive(Clone)]
pub struct Snapshot {
    nanos: [u64; Site::ALL.len()],
    bytes: u64,
    files: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        nanos: std::array::from_fn(|i| NANOS[i].load(Relaxed)),
        bytes: BYTES.load(Relaxed),
        files: FILES.load(Relaxed),
    }
}

/// What one region cost, split by site and summed by kind.
#[derive(Clone)]
pub struct Split {
    secs: [f64; Site::ALL.len()],
    pub bytes: u64,
    pub files: u64,
}

pub fn since(s: &Snapshot) -> Split {
    Split {
        secs: std::array::from_fn(|i| {
            NANOS[i].load(Relaxed).saturating_sub(s.nanos[i]) as f64 / 1e9
        }),
        bytes: BYTES.load(Relaxed).saturating_sub(s.bytes),
        files: FILES.load(Relaxed).saturating_sub(s.files),
    }
}

impl Split {
    /// One kind's seconds: its sites, added up.
    pub fn kind(&self, k: Kind) -> f64 {
        Site::ALL.iter().filter(|s| s.kind() == k).map(|s| self.secs[s.idx()]).sum()
    }

    pub fn total(&self) -> f64 {
        self.secs.iter().sum()
    }

    /// The sites that cost something, by name. A site at zero is left out, so
    /// the map says where the time went rather than which sites exist.
    pub fn parts(&self) -> Value {
        let mut o = serde_json::Map::new();
        for s in Site::ALL {
            let v = self.secs[s.idx()];
            if v > 0.0 {
                o.insert(s.name().to_string(), json!(v));
            }
        }
        Value::Object(o)
    }

    /// Put the four kinds, the byte count and the per-site map into a row that
    /// already carries `build_secs`. The keys sit beside it rather than under
    /// it, which is how every other phase in this program is reported.
    pub fn merge_into(&self, j: &mut Value) {
        for k in [Kind::Read, Kind::Decode, Kind::Check, Kind::Construct, Kind::Alloc] {
            j[k.key()] = json!(self.kind(k));
        }
        j["build_attributed_secs"] = json!(self.total());
        j["build_bytes_read"] = json!(self.bytes);
        j["build_files_read"] = json!(self.files);
        j["build_parts"] = self.parts();
    }

    /// The same as an object of its own, for a summary that carries several,
    /// with the rate the brief asks the read row to carry.
    pub fn json(&self) -> Value {
        let mut v = json!({});
        self.merge_into(&mut v);
        v["read_mb_per_s"] = json!(if self.kind(Kind::Read) > 0.0 {
            self.bytes as f64 / 1e6 / self.kind(Kind::Read)
        } else {
            0.0
        });
        v
    }
}
