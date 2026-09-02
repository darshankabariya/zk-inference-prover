#![allow(unused_imports)]

pub mod base_field_vec;
pub mod bindings;
pub mod bindings_airs;
pub mod blake_2s_hash_vec;
pub mod poseidon252;
pub mod secure_field_vec;

pub use base_field_vec::BaseFieldVec;
pub use blake_2s_hash_vec::Blake2sHashVec;
pub use secure_field_vec::SecureFieldVec;

/// Get CUDA memory info (free and total memory in bytes)
/// Returns (free_bytes, total_bytes)
pub fn get_cuda_memory_info() -> (usize, usize) {
    let mut free_mem: usize = 0;
    let mut total_mem: usize = 0;
    unsafe {
        bindings::cuda_get_memory_info(&mut free_mem, &mut total_mem);
    }
    (free_mem, total_mem)
}

/// Stream GG: what the stream-ordered pool holds, as the pool reports it.
///
/// `[used_current, used_high, reserved_current, reserved_high]` in bytes.
/// `used` is what the allocations outstanding add up to; `reserved` is what
/// the pool holds from the driver, which is the number `cudaMemGetInfo` and
/// `nvidia-smi` see.  The `high` pair is since the last `mem_pool_reset_high`.
/// `None` when the pool was never created (the fallback `cudaMalloc` path).
pub fn mem_pool_stats() -> Option<[u64; 4]> {
    let mut out = [0u64; 4];
    let rc = unsafe { bindings::cuda_mem_pool_stats(out.as_mut_ptr()) };
    (rc == 0).then_some(out)
}

/// Reset the pool's two high-water marks.
pub fn mem_pool_reset_high() {
    unsafe {
        bindings::cuda_mem_pool_reset_high();
    }
}

/// Print CUDA memory usage
pub fn print_cuda_memory(label: &str) {
    let (free, total) = get_cuda_memory_info();
    let used = total - free;
    println!(
        "[CUDA MEM] {}: used={:.2} MB, free={:.2} MB, total={:.2} MB",
        label,
        used as f64 / 1024.0 / 1024.0,
        free as f64 / 1024.0 / 1024.0,
        total as f64 / 1024.0 / 1024.0
    );
}
