#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{CStr, CString};

extern "C" {
    pub fn singleRun(
        threads: ::std::os::raw::c_int,
        blocks: ::std::os::raw::c_int,
        gpu: ::std::os::raw::c_int,
        jobSize: ::std::os::raw::c_ulong,
        inputJson: *const ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_char;

    pub fn getGpus() -> ::std::os::raw::c_int;

    pub fn whichBusy() -> ::std::os::raw::c_int;

    // fn freeString(s: *mut c_char);
}

pub struct GpuSingleJob {
    pub json: String,
    pub threads: u16,
    pub blocks: u16,
    pub gpu: u16,
    pub job_size: u64,
}


pub async fn run_gpu_single(params: GpuSingleJob) -> Result<String, String> {
    let c_str = CString::new(params.json).unwrap();
    let c_json: *const ::std::os::raw::c_char = c_str.as_ptr() as *const ::std::os::raw::c_char;
    let res = unsafe {
        let result_ptr = singleRun(
            ::std::os::raw::c_int::from(params.threads),
            ::std::os::raw::c_int::from(params.blocks),
            ::std::os::raw::c_int::from(params.gpu),
            ::std::os::raw::c_ulong::from(params.job_size),
            c_json,
        );

        if !result_ptr.is_null() {
            let c_str = CStr::from_ptr(result_ptr);
            Ok(format!("{}", c_str.to_string_lossy()))
        } else {
            Err("result_ptr is null".to_string())
        }
    };
    res
}

pub async fn get_gpus() -> u32 {
    let res = unsafe {
        let result = getGpus();
        result as u32
    };
    res
}

pub async fn which_busy() -> u32 {
    let res = unsafe {
        let result = whichBusy();
        result as u32
    };
    res
}
