# ============================================================
# CELL 1 - run once per Colab session (installs the OCR tools)
# ============================================================
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr poppler-utils
!pip -q install pytesseract pdf2image pandas openpyxl


# ============================================================
# CELL 2 - paste this whole cell, run it once (loads the code)
# ============================================================
import re
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps

INVOICE_PAGE = 2                      # invoice is normally page 2; other pages are only a fallback
FALLBACK_PAGES = [3, 4, 1, 5, 6]
KNOWN_CONTRACTS = ["2025_6750", "2024_0617"]
DPI = 300

MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"


# ---------------------------------------------------------------- OCR helpers
def render(pdf_path, page_no):
    try:
        return convert_from_path(pdf_path, dpi=DPI, first_page=page_no, last_page=page_no)[0]
    except Exception:
        return None


def ocr(img, psm=6, scale=1):
    if scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img.convert("L"))
    return pytesseract.image_to_string(img, config=f"--psm {psm}")


def ocr_data(img, psm=6):
    """One OCR pass that returns (text, word-positions). Used for text AND for locating the RECEIVED stamp."""
    gray = ImageOps.autocontrast(img.convert("L"))
    d = pytesseract.image_to_data(gray, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    lines = {}
    for i, w in enumerate(d["text"]):
        if w.strip():
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(w)
    return "\n".join(" ".join(ws) for ws in lines.values()), d


def page_text(img):
    """Full-page OCR with two layout modes so text next to stamps is not lost."""
    return ocr(img, 6) + "\n" + ocr(img, 4)


def header_text(img):
    """Zoomed OCR of the top-right invoice header (INVOICE / DATE / TERMS block)."""
    w, h = img.size
    crop = img.crop((int(w * 0.55), int(h * 0.10), w, int(h * 0.34)))
    return ocr(crop, 6, 2) + "\n" + ocr(crop, 11, 2)


def looks_like_invoice(text):
    t = text.upper()
    return "ENERGY" in t and ("AMOUNT DUE" in t or "BALANCE DUE" in t or "DESCRIPTION OF SERVICES" in t)


# ---------------------------------------------------------------- parsing helpers
def _d(s):
    """Parse a date string into datetime (None if it can't)."""
    s = s.strip().replace(".", "")
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%b %d, %Y", "%b %d %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _fmt(dt):
    return f"{dt.month}/{dt.day}/{dt.year}" if dt else None


def _six(s):
    """'060126' -> datetime(2026, 6, 1)"""
    try:
        return datetime.strptime(s, "%m%d%y")
    except ValueError:
        return None


def get_contract(text):
    flat = re.sub(r"[\s\-]+", "_", text)
    for c in KNOWN_CONTRACTS:
        if c in flat:
            return c
    m = re.search(r"CN[:\s]*?(20\d{2})[_\s\-]*(\d{4})", text)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def get_amount(text):
    for pat in (r"TOTAL\s+AMOUNT\s+DUE[^\d]{0,12}([\d,]+\.\d{2})",
                r"Balance\s+Due[^\d]{0,30}([\d,]+\.\d{2})",
                r"Total\s*\$\s*([\d,]+\.\d{2})"):
        m = re.search(pat, text, re.I)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def get_doc_number(text):
    """Document number on the page-1 transmittal form, e.g. K0VP2762398 / J0VP2742403."""
    cands = []
    for tok in re.findall(r"\b[A-Z0-9]{11}\b", text.upper()):
        if re.fullmatch(r"[A-Z][0O]V[A-Z][0-9OIL]{7}", tok):
            fixed = tok[0] + "0" + tok[2:4] + tok[4:].replace("O", "0").replace("I", "1").replace("L", "1")
            cands.append(fixed)
    return max(set(cands), key=cands.count) if cands else None


def get_transmittal_amount(text):
    """Page 1 'Payment Total: $ 1150.00' - used as a fallback / cross-check."""
    m = re.search(r"Payment\s+Total:?\s*\$?\s*([\d,]+\.\d{2})", text, re.I)
    return float(m.group(1).replace(",", "")) if m else None


def get_invoice_number(*texts):
    pat = r"INVOICE\s*(?:NO\.?|#)?\s*[:#]?\s*(\d[\d ]{2,7}\d)(?!\d)"
    for t in texts:
        for m in re.finditer(pat, t, re.I):
            digits = m.group(1).replace(" ", "")
            if 4 <= len(digits) <= 6:
                return digits
    # DocuSign subject line, e.g. "Invoice #28171 for Contract"
    for t in texts:
        m = re.search(r"Invoice\s*#?\s*(\d{5})\s+for\s+Contract", t, re.I)
        if m:
            return m.group(1)
    return None


def get_invoice_date(*texts):
    for t in texts:
        m = re.search(r"\bDATE\s+(\d{1,2}/\d{1,2}/\d{4})", t, re.I)
        if m and _d(m.group(1)):
            return _d(m.group(1))
    return None


def get_service_date(text):
    """Value after 'Date goods/service received:' -> M/D/YYYY or M/D/YYYY-M/D/YYYY, else 0."""
    m = re.search(r"goods\s*/?\s*service\s+received\s*:?([^\n]*)\n?([^\n]*)", text, re.I)
    if not m:
        return 0
    cand = m.group(1).strip() or m.group(2).strip()

    r = re.search(r"(\d{6})\s*[-–—]\s*(\d{6})", cand)                      # 060126-063026
    if r and _six(r.group(1)) and _six(r.group(2)):
        return f"{_fmt(_six(r.group(1)))}-{_fmt(_six(r.group(2)))}"

    r = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-–—]\s*(\d{1,2}/\d{1,2}/\d{2,4})", cand)   # 4/1/26-4/30/26
    if r and _d(r.group(1)) and _d(r.group(2)):
        return f"{_fmt(_d(r.group(1)))}-{_fmt(_d(r.group(2)))}"

    r = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", cand)                       # single date
    if r and _d(r.group(1)):
        return _fmt(_d(r.group(1)))
    return 0


# ---------------------------------------------------------------- RECEIVED stamp
STAMP_BY = (r"\bBy\s+[A-Za-z .'\-]{2,30}?\s+at\s+\d{1,2}\s*:\s*\d{2}\s*[ap]\.?m\.?,?\s*"
            rf"((?:{MONTHS})[a-z]*\.?\s+\d{{1,2}},?\s*\d{{4}})")
DATE_RX = r"(?<![\d/])(\d{1,2}/\d{1,2}/\d{4})(?![\d/\-])"


def _stamp_by(t):
    m = re.search(STAMP_BY, t, re.I)
    return _d(m.group(1)) if m and _d(m.group(1)) else None


def _stamp_bfs(t, inv_date=None):
    """'Invoice Received  mm/dd/yyyy  Business and Financial Services' stamp.
    Skips the invoice's own DATE line. If only the invoice date is found, it is used
    (the stamp and invoice can legitimately share a date)."""
    cands, prev = [], ""
    for line in t.splitlines():
        if not line.strip():
            continue
        if not re.search(r"\bDATE\b", line) and prev.strip().upper() != "DATE":
            for m in re.finditer(DATE_RX, line):
                dt = _d(m.group(1))
                if dt:
                    cands.append(dt)
        prev = line
    for dt in cands:
        if dt != inv_date:
            return dt
    return cands[0] if cands else None


def get_received(img, text, inv_date=None, data=None):
    """Date from the RECEIVED stamp:
       'By <name> at <time>, <Mon dd, yyyy>'   or   'Invoice Received mm/dd/yyyy'."""
    hit = _stamp_by(text)
    if hit:
        return hit
    if data is None:
        _, data = ocr_data(img)
    for i, w in enumerate(data["text"]):
        w = w.strip()
        if w.endswith(":") or not w.upper().startswith("RECEIVED"):   # skip 'received:' of the service-date line
            continue
        x, y = data["left"][i], data["top"][i]
        crop = img.crop((max(x - 520, 0), max(y - 60, 0), min(x + 1100, img.width), y + 260))
        for psm in (6, 4, 11):
            for scale in (3, 2):
                t = ocr(crop, psm, scale)
                hit = _stamp_by(t) or _stamp_bfs(t, inv_date)
                if hit:
                    return hit
    top = img.crop((0, 0, img.width, int(img.height * 0.36)))        # last resort: sweep top third
    for psm in (11, 6):
        hit = _stamp_by(ocr(top, psm, 2))
        if hit:
            return hit
    return None


# ---------------------------------------------------------------- main per-PDF routine
def extract_invoice(pdf_path):
    name = pdf_path.split("/")[-1]
    page_used = INVOICE_PAGE
    img = render(pdf_path, INVOICE_PAGE)
    text, data = ocr_data(img) if img else ("", None)

    if not looks_like_invoice(text):                    # page 2 isn't the invoice -> look at nearby pages
        for p in FALLBACK_PAGES:
            alt = render(pdf_path, p)
            if alt is None:
                continue
            alt_text, alt_data = ocr_data(alt)
            if looks_like_invoice(alt_text):
                img, text, data, page_used = alt, alt_text, alt_data, p
                break

    # extra (slower) passes only when something is missing
    amount = get_amount(text)
    inv_no = get_invoice_number(text)
    inv_dt = get_invoice_date(text)
    if (amount is None or not inv_no or not inv_dt) and img:
        text += "\n" + ocr(img, 4)
        hdr = header_text(img)
        amount = amount if amount is not None else get_amount(text)
        inv_no = inv_no or get_invoice_number(text, hdr)
        inv_dt = inv_dt or get_invoice_date(text, hdr)
    if not inv_no:                                      # DocuSign certificate page has the number in its subject
        for p in (3, 4):
            c = render(pdf_path, p)
            if c:
                inv_no = get_invoice_number(ocr(c, 6))
                if inv_no:
                    break

    # page 1 transmittal amount (cheap: low-res, cropped) used as fallback / cross-check
    p1_amt, doc_no = None, None
    try:
        p1 = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)[0]
        p1 = p1.crop((0, int(p1.height * 0.08), p1.width, int(p1.height * 0.55)))
        p1_text = ocr(p1, 6)
        p1_amt = get_transmittal_amount(p1_text)
        doc_no = get_doc_number(p1_text)
    except Exception:
        pass

    rec = get_received(img, text, inv_dt, data) if img else None
    contract = get_contract(text)
    amount_final = amount if amount is not None else p1_amt

    issues = []
    if amount_final is None:
        issues.append("amount")
    elif amount is not None and p1_amt is not None and abs(amount - p1_amt) > 0.005:
        issues.append(f"amount differs from page-1 transmittal ({p1_amt:,.2f})")
    if not inv_no:
        issues.append("invoice #")
    if not inv_dt:
        issues.append("invoice date")
    if not contract:
        issues.append("contract")
    if not doc_no:
        issues.append("document #")
    if page_used != INVOICE_PAGE:
        issues.append(f"invoice found on page {page_used}")

    return {
        "File": name,
        "Document Number": doc_no,
        "Contract Number": contract,
        "Invoice Amount": amount_final,
        "Service Date": get_service_date(text),
        "Invoice Date": inv_dt.strftime("%m/%d/%Y") if inv_dt else None,
        "Invoice Number": inv_no,
        "Received to BFS": rec.strftime("%m/%d/%Y") if rec else None,
        "Needs Review": "; ".join(issues),
    }


def to_excel(df, out_path="invoice_log.xlsx"):
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Invoices")
        ws = xw.sheets["Invoices"]
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(len(str(c.value or "")) for c in col) + 3
        amt_col = list(df.columns).index("Invoice Amount") + 1
        for r in range(2, len(df) + 2):
            ws.cell(row=r, column=amt_col).number_format = '"$"#,##0.00'
    return out_path


def sort_key(p):
    m = re.search(r"(\d+)", p.split("/")[-1])
    return int(m.group(1)) if m else 0


def flag_dates(df):
    """Adds a note when the Received date looks like an OCR misread (far from the invoice date)."""
    inv = pd.to_datetime(df["Invoice Date"], errors="coerce")
    rec = pd.to_datetime(df["Received to BFS"], errors="coerce")
    bad = rec.notna() & inv.notna() & ((rec < inv - pd.Timedelta(days=5)) | (rec > inv + pd.Timedelta(days=90)))
    df.loc[bad, "Needs Review"] = (df.loc[bad, "Needs Review"] + "; received date looks wrong").str.strip("; ")
    return df


def run_all(paths, workers=None):
    workers = workers or max(2, os.cpu_count() or 2)
    results, t0 = {}, time.time()

    def job(p):
        try:
            return extract_invoice(p)
        except Exception as e:
            return {"File": p.split("/")[-1], "Contract Number": f"ERROR: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, p): p for p in paths}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results[futs[f]] = r
            print(f"[{n}/{len(paths)}] {r['File']}  ({time.time() - t0:.0f}s elapsed)", flush=True)
    return flag_dates(pd.DataFrame([results[p] for p in paths]))


print("Extractor loaded. Now run Cell 3.")


# ============================================================
# CELL 3 - run this every time you have new PDFs
# (deletes old PDFs, asks you to upload, processes, downloads the Excel)
# ============================================================
import glob, os
from google.colab import files

for f in glob.glob("*.pdf"):              # clear old PDFs so nothing is counted twice
    os.remove(f)

files.upload()                            # click "Choose files" and select your PDFs
paths = sorted(glob.glob("*.pdf"), key=sort_key)
print(f"Processing {len(paths)} PDF(s) ...", flush=True)

df = run_all(paths)                       # prints [n/total] progress
display(df)
files.download(to_excel(df))              # downloads invoice_log.xlsx
