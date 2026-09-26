//! The key tiles' witness generation, on the device.
//!
//! One kernel per sub-component, one thread per committed row, with the witness
//! uploaded instead of the trace: what the host would have built with
//! `attention::trace` and then permuted and repacked, the kernel writes straight
//! into the committed column at its coset index.
//!
//! `ZK_DEVICE_WITNESS_GEN_KEY_TILE` says which path runs, on the three-value
//! pattern the other two families use, and falls back to
//! `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` when it is not set. The uploads,
//! the packing helpers and the column comparison are `crate::device_witness`'s,
//! shared with the attnout fold and the Freivalds check.
//!
//! **The one thing that is not a transcription.** The grid component's numerator
//! column commits `num[g]`, which on an honest witness equals `table[key(g)]`;
//! a kernel that recomputed it from the table would be smaller and would
//! silently repair `Corrupt::Numerator`, turning a rung-3 rejection into an
//! acceptance. So `num` is uploaded, narrowed to u16 and checked.

use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::device_witness::{FGrid, FMult, Fam, add_into, pack_i32, pack_i32_from_i32,
                            pack_i32_from_i32_into, pack_i32_into, pack_i64, pack_u16,
                            pack_u16_into, pack_u16_raw, ptrs, timed_up, timed_up_into,
                            uninit, witness_gen_mut, zeroes};
use crate::gadgets::attention::trace::{Cfg, Layouts};
use crate::gadgets::attention::witness::{Attn, Variant};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FKeyTileCfg {
    pub gc: FGrid,
    pub gk: FGrid,
    pub go: FGrid,
    pub pk: u32,
    pub pq: u32,
    pub kq: u32,
    pub pv: u32,
    pub oq: u32,
    pub log_row: u32,
    pub rows_tok: u32,
    pub rw: u32,
    pub s: u32,
    pub c: u32,
    pub hd: u32,
    pub insts: u32,
    pub toks: u32,
    pub variant: u32,
    pub split_k: u32,
    pub off_k: i32,
    pub off_q: i32,
    pub range_check: u32,
    pub z_lo_bits: u32,
    pub z_hi_bits: u32,
    pub z_min: i64,
    /// B11 stream N, lever 2: the diagonal block's mask, which the grid kernel
    /// needs for the exp multiplicity and for nothing else.
    pub diagonal_block: u32,
    pub mutate: u32,
}

unsafe extern "C" {
    fn zk_witness_gen_key_tile_chan(
        kk: *const i32,
        qq: *const i32,
        cols: *const *mut u32,
        m_k: FMult,
        m_q: FMult,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
    fn zk_witness_gen_key_tile_grid(
        dd: *const i32,
        num: *const u16,
        xx: *const u16,
        sc: *const u32,
        delta: *const u32,
        vv: *const i32,
        table0: i64,
        cols: *const *mut u32,
        n_aux: u32,
        m_exp: FMult,
        m_aux: FMult,
        m_v: FMult,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
    fn zk_witness_gen_key_tile_out(
        out: *const u32,
        cols: *const *mut u32,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
    fn zk_witness_gen_key_tile_tok(
        mm: *const u32,
        zz: *const u32,
        cols: *const *mut u32,
        with_max: u32,
        m_lo: FMult,
        m_hi: FMult,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
}

/// B11 stream L: the key tile witness generation's own switch, defaulting to `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS`.
///
/// `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` turns on three device generation paths at once -- the Freivalds check
/// components, the attnout, and every non-diagonal block key tile. Stream L measured them
/// apart on the sequence's own witness and they are not in the same state: the
/// Freivalds check and attnout witness generations reproduce the host cell for cell, while the key tile witness generation
/// disagrees with the host on the grid component and the proof is then rejected
/// (`off_table` bumps that the host never makes). One switch cannot express
/// that, so the key tile gate becomes its own three-value variable which falls back
/// to `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` when it is not set. `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1 ZK_DEVICE_WITNESS_GEN_KEY_TILE=0` is
/// the configuration that takes the Freivalds checks' win and leaves the key tiles on the host
/// until the witness generation is fixed.
pub fn key_tile_mode() -> u32 {
    match std::env::var("ZK_DEVICE_WITNESS_GEN_KEY_TILE") {
        Ok(v) => v.parse().unwrap_or(0),
        Err(_) => crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS"),
    }
}

/// B11 stream M: which witness generation *this* key tile takes, switch and layout together.
///
/// A diagonal block key tile commits no per-token component and `commit_base_dev` has no
/// notion of that layout, so it generates its witness on the host whatever the switch says
/// (`attn_trace::commit_base_any`). That override used to live inline at the
/// one place that picked the path, and the consequence -- that a diagonal-block-bearing
/// witness therefore leaves multiplicity bumps on the host -- had to be
/// rediscovered by whoever else needed it. `devpre::witness_gen_dev` needs it, so the
/// decision lives here now and both callers read the same function: this one
/// picks the path, and `compose::generate` asks it of every key tile before the
/// first witness generation runs.
pub fn key_tile_witness_gen_mode(a: &Attn) -> u32 {
    if a.diagonal_block { diagonal_block_mode() } else { key_tile_mode() }
}

/// `ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK`: the diagonal block key tiles' own witness generation switch, defaulting **off**.
///
/// B11 stream N, lever 2. Until this stream a diagonal block key tile built on the host
/// whatever any switch said, because `commit_base_dev` had no notion of a
/// layout with no per-token component and would have committed columns the AIR
/// does not expect. It has that notion now (`commit_base_dev` takes
/// `with_rows`, `devpre::key_tile_dev` with it), but a path is born off and
/// graduates on its gate: this is the switch that opens it, and the mirror
/// runs under `ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK=2` on a diagonal-block-bearing witness.
///
/// It falls back to `key_tile_mode` only once it has gated -- until then, not even
/// `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1` reaches the diagonal block.
pub fn diagonal_block_mode() -> u32 {
    std::env::var("ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// B11 stream L: must the exp table's multiplicity column be the host's?
///
/// `ZK_DEVICE_TABLES=1` builds `m_exp` on the card, by summing the per-key-tile exp
/// counters `commit_base_dev` left there. That is right exactly when *every*
/// key tile's bumps are on the card -- and a diagonal block key tile always generates its witness on the host,
/// because it commits no per-token component and the device witness generation has no notion of
/// its layout. So on a diagonal-block-bearing witness the bumps are split across the
/// two sides and neither half alone is the column.
///
/// Only the two uniform states were ever run. All key tiles on the host is the
/// sequence as stream K left it; all key tiles on the device is a rectangle, which
/// has no diagonal block. The sequence's own chunks are the mixed state, and taking the
/// device half alone there silently loses the diagonal block key tiles' exp lookups -- a
/// LogUp sum that does not cancel, with every other check passing.
///
/// `a.zz_total` is the local signal that a diagonal block rides with this key tile: it is
/// set on the prefix half of a pair and on nothing else (`layer.rs:352`). When
/// it is, the device key tile merges its counters into the host array as it does for
/// every other table, and the host column is the one committed. `compose.rs`
/// asks `layer.key_tiles.iter().any(|b| b.diagonal_block)`, which is the same question
/// about the same layer, and the two answers have to agree or the column and
/// its counters come from different places.
pub fn exp_mult_on_host(a: &Attn) -> bool {
    // B11 stream N, lever 2: `a.zz_total.is_some()` says a diagonal block key tile rides
    // with this one, and the reason that forced this key tile's counter back to the
    // host was that the diagonal block's own bumps were made *there*. Once the diagonal block
    // generates its witness on the card that reason is gone and both halves are device
    // counters, so the condition gains the diagonal block's mode.
    //
    // This predicate and `compose::generate`'s `host_diagonal_block` are twins and must
    // say the same thing. Drift between them is not loud where it happens: the
    // proof is rejected later by `sums_cancel` with every other check green.
    crate::devtab::mode() != 1 || (a.zz_total.is_some() && diagonal_block_mode() != 1)
}

/// One key tile array, packed and uploaded, with its own byte line.
fn key_tile_up(byte_tag: &'static str, f: impl FnOnce() -> Vec<u32>) -> Uint32Vec {
    timed_up("gen/witness_gen_key_tile_pack", "gen/witness_gen_key_tile_h2d", byte_tag, f)
}

/// The same, with the pack able to target the pinned pool.
fn key_tile_up_into(
    byte_tag: &'static str,
    words: usize,
    into: impl FnOnce(&mut [u32]),
    f: impl FnOnce() -> Vec<u32>,
) -> Uint32Vec {
    timed_up_into("gen/witness_gen_key_tile_pack", "gen/witness_gen_key_tile_h2d", byte_tag, words, into, f)
}

// ------------------------------------------------ the key tile dump (stream F) --

static KEY_TILE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn raw<T>(v: &[T]) -> &[u8] {
    // Safety: writing the exact bytes of a POD slice; the reader is told the
    // element type and count by the shape JSON beside it.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// B11 stream F: this key tile's witness arrays, raw, for the standalone probe.
///
/// `ZK_KEY_TILE_DUMP=DIR` writes one subdirectory per key tile -- the arrays
/// `commit_base_dev` uploads, plus the exp table and a shape JSON -- so
/// `experiments/b11/probe/key_tile_up.cu` can price a device-side generation path
/// against the *real* values and not only against the right sizes: it
/// recomputes `dd`, `num`, `zz` and `out` on the card and compares them with
/// what is here. Off unless the variable is set (~2.4 GB for a 16-instance
/// key tile at C=16K), and it never runs in a timed loop.
fn key_tile_dump(a: &Attn, with_out: bool) {
    let Ok(dir) = std::env::var("ZK_KEY_TILE_DUMP") else { return };
    let b = KEY_TILE_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::path::Path::new(&dir).join(format!("key_tile{b}"));
    std::fs::create_dir_all(&d).expect("key_tile_dump directory");
    let put = |name: &str, bytes: &[u8]| {
        std::fs::write(d.join(name), bytes).unwrap_or_else(|e| panic!("key_tile_dump {name}: {e}"));
    };
    let sh = a.shape;
    put("kk.i32", raw(&a.kk));
    put("qq.i32", raw(&a.qq));
    put("vv.i32", raw(&a.vv));
    put("mm.i64", raw(&a.mm));
    put("dd.i64", raw(&a.dd));
    put("num.i64", raw(&a.num));
    put("zz.i64", raw(&a.zz));
    put("out.i64", raw(&a.out));
    put("delta.i64", raw(&a.delta));
    put("table.u32", raw(&a.table));
    let j = serde_json::json!({
        "s": sh.s, "c": sh.c, "hd": sh.hd, "insts": sh.insts,
        "log_table": a.log_table, "table_len": a.table.len(),
        "nmax": a.nmax, "diagonal_block": a.diagonal_block, "with_out": with_out,
        "variant": format!("{:?}", a.variant),
        "off_table": a.off_table,
        "delta": a.delta,
        "bytes": {
            "kk": std::mem::size_of_val(&a.kk[..]), "qq": std::mem::size_of_val(&a.qq[..]),
            "vv": std::mem::size_of_val(&a.vv[..]), "dd": std::mem::size_of_val(&a.dd[..]),
            "num": std::mem::size_of_val(&a.num[..]), "out": std::mem::size_of_val(&a.out[..]),
        },
    });
    put("shape.json", serde_json::to_string_pretty(&j).unwrap().as_bytes());
    eprintln!("key_tile_dump: key tile {b} -> {}", d.display());
}

// ================================================================= the key tiles ==

/// One key tile's committed columns, on the device, with the log size of each --
/// a key tile is four sub-components on three grids plus a token block, so unlike the
/// other two families its columns are not all the same height.
pub struct DevKeyTile {
    pub base: Vec<BaseFieldVec>,
    pub logs: Vec<u32>,
    pub at: Vec<usize>,
    /// B7a milestone 4 (section 3 row 1): this key tile's exp-table counter, left
    /// where the scatter-add put it. `zk_table_witness_gen` sums one per key tile into
    /// the `m_exp` column, so with `ZK_DEVICE_TABLES=1` the 33.5 MB download and
    /// the parallel merge below never happen.
    pub m_exp: Uint32Vec,
    /// B7a milestone 4 (section 3 row 1): the five range counters this key tile's
    /// three witness generations bumped -- the key, query, value and the two `z` limbs -- and
    /// `intsplit`'s auxiliary one where the variant has it. `zk_pre_witness_gen` sums
    /// one per key tile into each range table's multiplicity column.
    pub m_k: Uint32Vec,
    pub m_q: Uint32Vec,
    pub m_v: Uint32Vec,
    pub m_aux: Option<Uint32Vec>,
    pub m_lo: Uint32Vec,
    pub m_hi: Uint32Vec,
}

/// Build one key tile's committed columns on the device.
///
/// `mult_exp` and `mult_r` are the caller's, exactly as `commit_base` takes
/// them: 20.6 called this the awkward part, and it is a counter array, which is
/// what the scatter-add already handles.
/// One key tile's committed base columns, built on the device.
///
/// B11 stream N, lever 2 (a): `with_rows` is `!a.diagonal_block`, and it says whether
/// this key tile commits the **per-token component** -- the row maxima, the
/// denominator and its low limb. A diagonal block key tile does not: there is one maximum
/// and one denominator per *token*, not per key tile, so the diagonal block rides on the
/// prefix key tile's. That distinction was the whole of why a diagonal block key tile could
/// not generate its witness on the card. The witness generation had no notion of it and would have committed
/// a per-token component the AIR does not expect there, so `commit_base_any` sent
/// every diagonal block to the host whatever the switch said; now the witness generation knows, and
/// `diagonal_block_mode` is what opens the path.
pub fn commit_base_dev(
    a: &Attn,
    l: &Layouts,
    cfg: &Cfg,
    widths: &[u32],
    mult_exp: &mut [u32],
    mult_r: &mut [Vec<u32>],
    off_table: &mut usize,
    with_out: bool,
    merge_m_exp: bool,
    with_rows: bool,
) -> DevKeyTile {
    let sh = a.shape;
    // the host's two layout assertions, kept here because they are about the
    // layout and not about where the witness generation runs
    assert_eq!(
        l.key.q * l.key.g.chunks,
        sh.s,
        "grid layout leaves padded token slots: q={} chunks={} S={}",
        l.key.q,
        l.key.g.chunks,
        sh.s
    );
    assert_eq!(l.out.q * l.out.g.chunks, sh.s, "output layout leaves padded token slots");
    let (lc, lk, lo) = (&l.chan, &l.key, &l.out);
    let ridx = |b: u32| widths.iter().position(|w| *w == b).unwrap();
    let rrows = 1usize << l.log_row;
    let with_max = a.variant != Variant::Bf16x;
    let n_aux = if a.variant == Variant::IntExp { 0 } else { lk.q };

    key_tile_dump(a, with_out);

    // B11 stream N, lever 2: `notok` is this lever's own negative control -- a
    // diagonal block key tile that builds the per-token component anyway, which is precisely
    // what `commit_base_any` used to send every diagonal block to the host to avoid.
    // It has to be decided here, before the uploads, because the two arrays
    // that component reads are skipped with it.
    let with_rows = with_rows || witness_gen_mut(Fam::KeyTile) == 8;

    let t_up = std::time::Instant::now();
    // The four arrays that are 99.99% of the key tile's bytes (stream F's table:
    // dd 1,073.74 MB, num 536.87, kk 268.44, vv 268.44, qq 8.39) go through the
    // pinned pool when it is on. `mm`/`zz`/`delta` are 0.26 MB together and are
    // left on the pageable path: they would each grow the pool to no measurable
    // end.
    let d_kk = key_tile_up_into("key_tile/kk", a.kk.len(), |d| pack_i32_from_i32_into(d, &a.kk),
                            || pack_i32_from_i32(&a.kk));
    let d_qq = key_tile_up_into("key_tile/qq", a.qq.len(), |d| pack_i32_from_i32_into(d, &a.qq),
                            || pack_i32_from_i32(&a.qq));
    let d_vv = key_tile_up_into("key_tile/vv", a.vv.len(), |d| pack_i32_from_i32_into(d, &a.vv),
                            || pack_i32_from_i32(&a.vv));
    let d_dd = key_tile_up_into("key_tile/dd", a.dd.len(), |d| pack_i32_into(d, &a.dd, "key tile dd"),
                            || pack_i32(&a.dd, "key tile dd"));
    let d_num = key_tile_up_into("key_tile/num", a.num.len().div_ceil(2),
                             |d| pack_u16_into(d, &a.num, "key tile num"),
                             || pack_u16(&a.num, "key tile num"));
    let d_delta = key_tile_up("key_tile/delta", || pack_i64(&a.delta));
    let d_mm = with_rows.then(|| key_tile_up("key_tile/mm", || pack_i64(&a.mm)));
    // B11 stream L: `zz_row()`, not `zz`.
    //
    // This is the whole of what b-port called the prefix key tiles' witness-generation kernels
    // "predating the diagonal block's key discipline", and it is on this side of the
    // link rather than in the kernel. When a diagonal block key tile rides with this one the
    // per-token component belongs to the *pair*, and the denominator it commits is
    // the pair's total -- `Z >= nmax` holds of the sum and of neither half,
    // because the key whose shift is zero may be in either (`layer.rs:352`).
    // `commit_base` has always read `a.zz_row()` for it; `commit_base_dev`
    // uploaded `a.zz`, so on any diagonal-block-bearing witness the device built a
    // denominator missing the diagonal block's keys -- 48 of 352 columns wrong on a
    // C=4K chunk (the 3 x rw per-token columns; the 16 maxima were right), and
    // `off_table` bumps the host never makes, hence the rejection. Rectangles
    // have no diagonal block, `zz_row()` is `zz` there, and that is why stream G's
    // switch matrix and stream F's whole loop saw nothing.
    // B11 stream N: both of these feed the per-token component alone, so a diagonal block
    // key tile uploads neither.
    let d_zz = with_rows.then(|| key_tile_up("key_tile/zz", || pack_i64(a.zz_row())));
    // only the bf16x variant reads these two, and only it pays for them
    let bf16x = a.variant == Variant::Bf16x;
    let d_xx = bf16x.then(|| key_tile_up("key_tile/xx", || pack_u16_raw(&a.xx)));
    let d_sc = bf16x.then(|| key_tile_up("key_tile/sc", || pack_i64(&a.sc)));
    let d_out = with_out.then(|| key_tile_up("key_tile/out", || pack_i64(&a.out)));
    crate::util::add_phase("gen/witness_gen_key_tile_up", t_up.elapsed().as_secs_f64());

    let (variant, split_k) = match a.variant {
        Variant::Bf16x => (0u32, 0u32),
        Variant::IntExp => (1u32, 0u32),
        Variant::IntSplit(k) => (2u32, k),
    };
    let bcfg = FKeyTileCfg {
        gc: FGrid::of(&lc.g),
        gk: FGrid::of(&lk.g),
        go: FGrid::of(&lo.g),
        pk: lc.pk as u32,
        pq: lc.pq as u32,
        kq: lk.q as u32,
        pv: lk.pv as u32,
        oq: lo.q as u32,
        log_row: l.log_row,
        rows_tok: rrows as u32,
        rw: l.rw as u32,
        s: sh.s as u32,
        c: sh.c as u32,
        hd: sh.hd as u32,
        insts: sh.insts as u32,
        toks: sh.toks() as u32,
        variant,
        split_k,
        off_k: a.off_k(),
        off_q: a.off_q(),
        range_check: u32::from(cfg.range_check),
        z_lo_bits: cfg.z_lo_bits,
        z_hi_bits: cfg.z_hi_bits,
        z_min: cfg.z_min as i64,
        diagonal_block: u32::from(a.diagonal_block),
        mutate: witness_gen_mut(Fam::KeyTile),
    };

    let t_alloc = std::time::Instant::now();
    let mut base: Vec<BaseFieldVec> = vec![];
    let mut logs: Vec<u32> = vec![];
    let mut at = vec![];
    let counts = zeroes(3);

    // ---- the channel component ------------------------------------------------
    at.push(base.len());
    let chan = uninit(lc.pk + lc.pq, lc.g.rows);
    let chan_p = ptrs(&chan);
    let (mk, mq) = if cfg.range_check {
        (zeroes(mult_r[ridx(a.k_bits)].len()), zeroes(mult_r[ridx(a.q_bits)].len()))
    } else {
        (zeroes(1), zeroes(1))
    };
    crate::util::add_phase("gen/witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_key_tile_chan(
            d_kk.device_ptr as *const i32,
            d_qq.device_ptr as *const i32,
            chan_p.as_ptr(),
            if cfg.range_check { FMult::of(&mk) } else { FMult::none() },
            if cfg.range_check { FMult::of(&mq) } else { FMult::none() },
            counts.device_ptr as *mut u32,
            bcfg,
        );
    }
    logs.extend(vec![lc.g.log_rows; chan.len()]);
    base.extend(chan);

    // ---- the grid component ---------------------------------------------------
    at.push(base.len());
    let grid = uninit(2 * lk.q + n_aux + lk.pv, lk.g.rows);
    let grid_p = ptrs(&grid);
    let m_exp = zeroes(mult_exp.len());
    let m_aux = match a.variant {
        Variant::IntSplit(k) => zeroes(mult_r[ridx(k)].len()),
        _ => zeroes(1),
    };
    let m_v = zeroes(mult_r[ridx(8)].len());
    unsafe {
        zk_witness_gen_key_tile_grid(
            d_dd.device_ptr as *const i32,
            d_num.device_ptr as *const u16,
            d_xx.as_ref().map(|d| d.device_ptr as *const u16).unwrap_or(std::ptr::null()),
            d_sc.as_ref().map(|d| d.device_ptr).unwrap_or(std::ptr::null()),
            d_delta.device_ptr,
            d_vv.device_ptr as *const i32,
            a.table[0] as i64,
            grid_p.as_ptr(),
            n_aux as u32,
            FMult::of(&m_exp),
            if matches!(a.variant, Variant::IntSplit(_)) { FMult::of(&m_aux) } else { FMult::none() },
            if cfg.range_check { FMult::of(&m_v) } else { FMult::none() },
            counts.device_ptr as *mut u32,
            bcfg,
        );
    }
    logs.extend(vec![lk.g.log_rows; grid.len()]);
    base.extend(grid);

    // ---- the A.V outputs ---------------------------------------------------
    at.push(base.len());
    if with_out {
        let outc = uninit(lo.q, lo.g.rows);
        let out_p = ptrs(&outc);
        unsafe {
            zk_witness_gen_key_tile_out(
                d_out.as_ref().unwrap().device_ptr,
                out_p.as_ptr(),
                counts.device_ptr as *mut u32,
                bcfg,
            );
        }
        logs.extend(vec![lo.g.log_rows; outc.len()]);
        base.extend(outc);
    }

    // ---- the per-token component ----------------------------------------------
    // B11 stream N, lever 2 (a): absent on a diagonal block key tile. `at` still gains both
    // ends of it, so `at[3] == at[4]` there and every index after the outputs
    // keeps its meaning -- the host's `commit_base` pushes the same empty span
    // for the same reason.
    at.push(base.len());
    let (m_lo, m_hi) = if with_rows {
        let n_tok = if with_max { l.rw } else { 0 } + 3 * l.rw;
        let tok = uninit(n_tok, rrows);
        let tok_p = ptrs(&tok);
        let m_lo = zeroes(mult_r[ridx(cfg.z_lo_bits)].len());
        let m_hi = zeroes(mult_r[ridx(cfg.z_hi_bits)].len());
        unsafe {
            zk_witness_gen_key_tile_tok(
                d_mm.as_ref().unwrap().device_ptr,
                d_zz.as_ref().unwrap().device_ptr,
                tok_p.as_ptr(),
                u32::from(with_max),
                FMult::of(&m_lo),
                FMult::of(&m_hi),
                counts.device_ptr as *mut u32,
                bcfg,
            );
        }
        logs.extend(vec![l.log_row; tok.len()]);
        base.extend(tok);
        (m_lo, m_hi)
    } else {
        // The two `z` range counters this component would have bumped, at their
        // real width and all zero. A diagonal block key tile makes no `z` lookups at all,
        // but `zk_pre_witness_gen` sums one counter *per key tile* into each range
        // table's multiplicity column and asserts every one of them is the
        // table's width -- so the shape has to be there even when the content
        // is nothing.
        (zeroes(mult_r[ridx(cfg.z_lo_bits)].len()), zeroes(mult_r[ridx(cfg.z_hi_bits)].len()))
    };
    at.push(base.len());
    crate::util::add_phase("gen/witness_gen_key_tile_kernel", t_k.elapsed().as_secs_f64());

    // ---- the counters, back to the host ------------------------------------
    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    // 21.4's "counters: D2H and merge" line, for the widest counter there is:
    // 33.5 MB back and a parallel add, per key tile, for an array whose only reader
    // was `base_mults[0]`. The table witness generation sums the device copies instead.
    if merge_m_exp {
        add_into(mult_exp, &m_exp.to_vec());
    }
    if crate::devpre::merge_counters() {
        if cfg.range_check {
            add_into(&mut mult_r[ridx(a.k_bits)], &mk.to_vec());
            add_into(&mut mult_r[ridx(a.q_bits)], &mq.to_vec());
            add_into(&mut mult_r[ridx(8)], &m_v.to_vec());
        }
        if let Variant::IntSplit(k) = a.variant {
            add_into(&mut mult_r[ridx(k)], &m_aux.to_vec());
        }
        // B11 stream N: a diagonal block key tile has no per-token component, so it made no
        // `z` range bumps to merge.
        if with_rows {
            add_into(&mut mult_r[ridx(cfg.z_lo_bits)], &m_lo.to_vec());
            add_into(&mut mult_r[ridx(cfg.z_hi_bits)], &m_hi.to_vec());
        }
    }
    *off_table += got[0] as usize;
    crate::util::add_phase("gen/witness_gen_counters", t_d.elapsed().as_secs_f64());

    DevKeyTile { base, logs, at, m_exp, m_k: mk, m_q: mq, m_v,
              m_aux: matches!(a.variant, Variant::IntSplit(_)).then_some(m_aux),
              m_lo, m_hi }
}
