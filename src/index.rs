use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use thiserror::Error;

const VERSION: u32 = 1;
const ALIGN: usize = 8;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("unsupported version {version}, expected {expected}")]
    UnsupportedVersion { version: u32, expected: u32 },
}

pub struct Index {
    mmap: Mmap,
    num_records: u32,
    dim: u32,
}

#[repr(C)]
struct Header {
    magic: [u8; 4],
    version: u32,
    num_records: u32,
    dim: u32,
}

#[repr(C)]
struct Entry {
    offset: u64,
    len: u32,
    _pad: u32,
}

fn pad(n: usize) -> usize {
    (n + ALIGN - 1) & !(ALIGN - 1)
}

impl Index {
    pub fn save(path: impl AsRef<Path>, dim: u32, vectors: &[Vec<f32>], paths: &[String]) -> Result<()> {
        assert_eq!(vectors.len(), paths.len());
        let num_records = vectors.len() as u32;

        let header_size = std::mem::size_of::<Header>();
        let vectors_size = num_records as usize * dim as usize * std::mem::size_of::<f32>();
        let vectors_offset = pad(header_size);
        let entry_table_offset = pad(vectors_offset + vectors_size);

        let mut file = std::fs::File::create(path.as_ref())?;

        // Write header
        let header = Header {
            magic: *b"LMDB",
            version: VERSION,
            num_records,
            dim,
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const Header as *const u8,
                std::mem::size_of::<Header>(),
            )
        };
        file.write_all(bytes)?;
        // Pad to alignment
        let padding = pad(header_size) - header_size;
        for _ in 0..padding {
            file.write_all(&[0u8])?;
        }

        // Write vectors
        for v in vectors {
            assert_eq!(v.len(), dim as usize);
            let bytes = unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
            };
            file.write_all(bytes)?;
        }
        // Pad vectors block
        let vectors_end = vectors_offset + vectors_size;
        let vectors_padding = pad(vectors_end) - vectors_end;
        for _ in 0..vectors_padding {
            file.write_all(&[0u8])?;
        }

        // Write entry table + string data
        let mut string_offset = 0u64;
        let mut string_data = Vec::new();
        let mut entries: Vec<Entry> = Vec::with_capacity(num_records as usize);
        for p in paths {
            entries.push(Entry {
                offset: entry_table_offset as u64
                    + (num_records as u64 * std::mem::size_of::<Entry>() as u64)
                    + string_offset,
                len: p.len() as u32,
                _pad: 0,
            });
            string_data.extend_from_slice(p.as_bytes());
            string_offset += p.len() as u64;
        }
        let entry_bytes = unsafe {
            std::slice::from_raw_parts(
                entries.as_ptr() as *const u8,
                entries.len() * std::mem::size_of::<Entry>(),
            )
        };
        file.write_all(entry_bytes)?;
        file.write_all(&string_data)?;

        Ok(())
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header_size = std::mem::size_of::<Header>();
        let header: &Header = unsafe { &*(mmap[..header_size].as_ptr() as *const Header) };

        if &header.magic != b"LMDB" {
            anyhow::bail!(IndexError::InvalidMagic);
        }
        if header.version != VERSION {
            anyhow::bail!(IndexError::UnsupportedVersion {
                version: header.version,
                expected: VERSION,
            });
        }

        Ok(Self {
            mmap,
            num_records: header.num_records,
            dim: header.dim,
        })
    }

    pub fn len(&self) -> u32 {
        self.num_records
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn vector(&self, idx: u32) -> &[f32] {
        let header_size = std::mem::size_of::<Header>();
        let vectors_offset = pad(header_size);
        let start = vectors_offset + idx as usize * self.dim as usize * std::mem::size_of::<f32>();
        let end = start + self.dim as usize * std::mem::size_of::<f32>();
        let bytes = &self.mmap[start..end];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim as usize) }
    }

    pub fn path(&self, idx: u32) -> &str {
        let header_size = std::mem::size_of::<Header>();
        let vectors_size = self.num_records as usize * self.dim as usize * std::mem::size_of::<f32>();
        let vectors_offset = pad(header_size);
        let entry_table_offset = pad(vectors_offset + vectors_size);
        let entry_size = std::mem::size_of::<Entry>();

        let entry_offset = entry_table_offset + idx as usize * entry_size;
        let entry: &Entry = unsafe { &*(self.mmap[entry_offset..entry_offset + entry_size].as_ptr() as *const Entry) };

        let start = entry.offset as usize;
        let end = start + entry.len as usize;
        let bytes = &self.mmap[start..end];
        std::str::from_utf8(bytes).unwrap()
    }
}
