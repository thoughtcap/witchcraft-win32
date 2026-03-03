//! assets.rs — Zstd asset loader for compressed configuration files.

use once_cell::sync::OnceCell;
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
};

/// A Zstandard-compressed asset that is decompressed at most once and
/// cached for the program's lifetime.
pub struct Asset {
    path: &'static str,
    decompressed: OnceCell<Result<Vec<u8>, ()>>,
}

impl Asset {
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            decompressed: OnceCell::new(),
        }
    }

    /// Decompress directly into a file without buffering the entire payload in RAM.
    pub fn decompress_to_file(&'static self, assets: &PathBuf, dest: &Path) -> Result<(), ()> {
        let mut out = std::fs::File::create(dest)
            .map_err(|e| log::warn!("failed to create {}: {}", dest.display(), e))?;

        let src = assets.join(self.path);
        let mut reader = std::fs::File::open(&src)
            .map_err(|e| log::warn!("failed to open {}: {}", src.display(), e))?;
        zstd::stream::copy_decode(&mut reader, &mut out)
            .map_err(|e| log::warn!("failed to decompress asset: {}", e))?;

        Ok(())
    }

    /// Return the decompressed bytes; performs work only on first call.
    pub fn bytes(&'static self, assets: &PathBuf) -> Result<&'static [u8], ()> {
        self.decompressed
            .get_or_init(|| {
                let compressed: Vec<u8> = match fs::read(Path::new(&assets.join(self.path))) {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("failed to read warp asset {}: {}", self.path, e);
                        return Err(());
                    }
                };

                let mut cursor = Cursor::new(compressed);
                match zstd::stream::decode_all(&mut cursor) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        log::warn!("failed to decompress warp asset {}", e);
                        Err(())
                    }
                }
            })
            .as_deref()
            .map_err(|_| ())
    }
}

#[macro_export]
macro_rules! embed_zst_asset {
    ($vis:vis $name:ident, $path:literal) => {
        $vis static $name: $crate::assets::Asset =
            $crate::assets::Asset::new_file($path);
    };
}
