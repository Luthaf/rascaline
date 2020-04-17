use std::os::raw::c_char;

pub unsafe fn copy_str_to_c(string: &str, name: *mut c_char, bufflen: usize) {
    let size = std::cmp::min(string.len(), bufflen - 1);
    std::ptr::copy(string.as_ptr(), name as *mut u8, size);
    // NULL-terminate the string
    name.add(size).write(0);
}
