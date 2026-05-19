#[cfg(target_os = "macos")]
fn main() {
    let Ok(output) = std::process::Command::new("clang")
        .args(["--print-file-name", "libclang_rt.osx.a"])
        .output()
    else {
        return;
    };
    let Ok(path) = String::from_utf8(output.stdout) else {
        return;
    };
    let path = path.trim();
    if std::path::Path::new(path).exists() {
        println!("cargo:rustc-link-arg={path}");
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {}
