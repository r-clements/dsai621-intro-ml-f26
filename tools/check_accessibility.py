#!/usr/bin/env python3
"""
Check that a PDF carries the structures PDF/UA and WCAG rely on.

    python3 check_accessibility.py HW1-accessible.pdf HW1_sol-accessible.pdf

This is a structural smoke test, not a conformance certification. It catches
the failures that actually happen in practice: tagging silently not enabled,
missing document title, missing language, math emitted with no MathML.

For a real verdict run veraPDF (https://verapdf.org):
    verapdf -f ua2 --format text HW1-accessible.pdf

Requires: pypdf   (pip install pypdf)
"""

import sys
import zlib
import re

try:
    import pypdf
except ImportError:
    sys.exit("pypdf is required:  pip install pypdf")


GREEN, RED, YELL, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELL = OFF = ""


def result(ok, label, detail=""):
    """ok: True / False / None (advisory)."""
    mark = f"{GREEN}PASS{OFF}" if ok else (f"{RED}FAIL{OFF}" if ok is False else f"{YELL}NOTE{OFF}")
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok is not False


def raw_bytes(path):
    """All stream data, decompressed where possible, for substring probes."""
    blob = open(path, "rb").read()
    out = [blob]
    for m in re.finditer(rb"stream\r?\n", blob):
        start = m.end()
        end = blob.find(b"endstream", start)
        if end == -1:
            continue
        try:
            out.append(zlib.decompress(blob[start:end]))
        except Exception:
            pass
    return b"".join(out)


def check(path):
    print(f"\n{path}")
    print("-" * len(path))
    ok = True

    try:
        r = pypdf.PdfReader(path)
    except Exception as e:
        return result(False, "file opens", str(e)[:70])

    root = r.trailer["/Root"]

    # --- 1. Tagged and structured -------------------------------------
    marked = False
    try:
        marked = bool(root["/MarkInfo"]["/Marked"])
    except Exception:
        pass
    ok &= result(marked, "Tagged (/MarkInfo /Marked true)",
                 "" if marked else "tagging did not take effect")

    has_tree = "/StructTreeRoot" in root
    ok &= result(has_tree, "Structure tree present (/StructTreeRoot)")

    # --- 2. Document metadata -----------------------------------------
    title = (r.metadata or {}).get("/Title") if r.metadata else None
    if not title:
        # PDF 2.0 deprecates the Info dictionary; the title lives in XMP dc:title.
        try:
            xmp = r.xmp_metadata
            title = (xmp.dc_title or {}).get("x-default") if xmp else None
        except Exception:
            title = None
    ok &= result(bool(title), "Document title set",
                 f'"{title}"' if title else "PDF/UA requires a title")

    disp = False
    try:
        disp = bool(root["/ViewerPreferences"]["/DisplayDocTitle"])
    except Exception:
        pass
    ok &= result(disp, "DisplayDocTitle true",
                 "" if disp else "reader will show the filename instead of the title")

    lang = root.get("/Lang")
    ok &= result(bool(lang), "Natural language set (/Lang)", str(lang) if lang else "")

    # --- 3. Standards claim -------------------------------------------
    blob = raw_bytes(path)
    ua = re.search(rb"pdfuaid[:/](?:part|schema)?[^0-9]{0,20}([12])", blob)
    ua2 = b"pdfuaid" in blob
    result(ua2 or None, "PDF/UA identifier in XMP metadata",
           f"part {ua.group(1).decode()}" if ua else ("found" if ua2 else "absent"))

    ver = r.pdf_header
    result(None, "PDF version", ver)

    # --- 4. Math ------------------------------------------------------
    n_mathml = blob.count(b"<math")
    n_af = blob.count(b"/AF")
    if n_mathml:
        result(True, "MathML present", f"{n_mathml} <math> element(s)")
    else:
        ok &= result(False, "MathML present",
                     "no MathML found — equations will read as garbled glyphs")
    result(bool(n_af) or None, "Associated Files referenced (/AF)", f"{n_af} reference(s)")

    # --- 5. Images ----------------------------------------------------
    n_fig = blob.count(b"/Figure")
    n_alt = blob.count(b"/Alt")
    if n_fig:
        good = n_alt >= n_fig
        ok &= result(good, "Every /Figure has /Alt text", f"{n_fig} figures, {n_alt} alt entries")
    else:
        result(None, "Figures", "none in this document")

    # --- 6. Tables ----------------------------------------------------
    n_table = blob.count(b"/Table")
    if n_table:
        n_th = blob.count(b"/TH")
        ok &= result(n_th > 0, "Table header cells tagged (/TH)",
                     f"{n_table} table(s), {n_th} header cell(s)")
    else:
        result(None, "Tables", "none in this document")

    # --- 7. Text extraction sanity ------------------------------------
    try:
        txt = "".join((p.extract_text() or "") for p in r.pages)
        ok &= result(len(txt.strip()) > 200, "Text extracts",
                     f"{len(txt.strip())} characters over {len(r.pages)} page(s)")
    except Exception as e:
        ok &= result(False, "Text extracts", str(e)[:60])

    return ok


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    results = [check(p) for p in paths]
    print()
    if all(results):
        print(f"{GREEN}All structural checks passed.{OFF} "
              "Confirm with veraPDF before relying on this for compliance.")
    else:
        print(f"{RED}Some checks failed.{OFF} See above.")
        sys.exit(1)
