#![allow(deprecated)]

// mod lib;

extern crate bindgen;
extern crate cc;

fn main() {
    if cfg!(target_os = "linux") {
        cc::Build::new()
            .cuda(true)
            .cpp(true)
            .flag("-cudart=shared")
            .flag("-arch=all-major")
            .files(&["src/cuda/kernels/miner.cu"])
            .compile("miner.a");

        println!("cargo:rustc-link-search=native=/user/local/cuda-12.6/lib");
        println!("cargo:rustc-link-search=/user/local/cuda-12.6/lib");
        println!("cargo:rustc-env=LD_LIBRARY_PATH=/user/local/cuda-12.6/lib");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
