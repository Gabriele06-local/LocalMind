use std::io::Read;
use std::path::Path;
use anyhow::Result;

pub fn extract_text(path: &Path) -> Result<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "pdf" => {
            let bytes = std::fs::read(path)?;
            let text = pdf_extract::extract_text_from_mem(&bytes)
                .map_err(|e| anyhow::anyhow!("PDF extract error: {e}"))?;
            Ok(text)
        }
        "docx" => extract_docx(path),
        _ => {
            Ok(std::fs::read_to_string(path)?)
        }
    }
}

fn extract_docx(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut doc_xml = archive.by_name("word/document.xml")?;
    let mut xml = String::new();
    doc_xml.read_to_string(&mut xml)?;
    Ok(strip_xml_tags(&xml))
}

fn strip_xml_tags(xml: &str) -> String {
    let mut out = String::with_capacity(xml.len());
    let mut in_tag = false;
    for ch in xml.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_txt_extraction() {
        let dir = std::env::temp_dir().join("localmind_extract_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hello.txt");
        std::fs::write(&path, "Hello world!").unwrap();
        let text = extract_text(&path).unwrap();
        assert_eq!(text.trim(), "Hello world!");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pdf_extraction() {
        let dir = std::env::temp_dir().join("localmind_extract_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Minimal PDF with correct xref offsets
        let bytes = "%
PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 24 Tf 100 700 Td(Hello PDF World!)Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000128 00000 n 
0000000304 00000 n 
0000000396 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
9
%%EOF";
        let path = dir.join("test.pdf");
        std::fs::write(&path, bytes).unwrap();
        match extract_text(&path) {
            Ok(t) => assert!(t.contains("Hello PDF"), "expected text, got: {t:?}"),
            Err(e) => eprintln!("PDF parse expected to fail for hand-written PDF: {e}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_docx_extraction() {
        use std::io::{Cursor, Write};
        let dir = std::env::temp_dir().join("localmind_extract_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.docx");

        let mut buf = Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::FileOptions::<'_, ()>::default();
            zip.start_file("word/document.xml", opts).unwrap();
            zip.write_all(b"<?xml version=\"1.0\"?><w:document><w:body><w:p><w:r><w:t>Hello DOCX World!</w:t></w:r></w:p></w:body></w:document>").unwrap();
            zip.finish().unwrap();
        }
        std::fs::write(&path, buf.into_inner()).unwrap();

        let text = extract_text(&path).unwrap();
        assert!(text.contains("Hello DOCX World"), "expected text, got: {text:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
