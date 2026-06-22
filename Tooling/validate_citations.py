"""
validate_citations.py  (run on YOUR machine; checks reference METADATA only)
----------------------------------------------------------------------------
For each reference line it finds the DOI, asks Crossref AND DataCite (so arXiv
'10.48550/arXiv...' and figshare DOIs are handled too), and reports whether the
DOI resolves and whether the registered title actually appears in your reference
line. It does NOT check whether the in-text CLAIM matches the source (Claude does
that from your numbered-papers zip).

Prep: references.txt, ONE reference per line.
Run:  pip install requests ; python validate_citations.py references.txt
Out:  citation_check.csv  (focus on: resolves != ok, title_in_ref = NO, year mismatch)
"""
import sys, re, csv, time
import requests

DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', re.I)
YEAR_RE = re.compile(r'(?:19|20)\d{2}')
UA = {"User-Agent": "AIRC-Fusion-citation-check/2.0 (mailto:you@example.com)"}


def _norm(s):
    return re.sub(r'[^a-z0-9 ]', '', (s or '').lower())


def lookup(doi):
    """Return (title, year, source) from Crossref, falling back to DataCite."""
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=20)
        if r.status_code == 200:
            m = r.json()["message"]
            yr = None
            for k in ("published-print", "published-online", "issued"):
                if m.get(k, {}).get("date-parts"):
                    yr = m[k]["date-parts"][0][0]; break
            return (m.get("title") or [""])[0], yr, "crossref"
    except Exception:
        pass
    try:  # DataCite: arXiv, figshare, Zenodo, etc.
        r = requests.get(f"https://api.datacite.org/dois/{doi}", headers=UA, timeout=20)
        if r.status_code == 200:
            a = r.json()["data"]["attributes"]
            title = (a.get("titles") or [{}])[0].get("title", "")
            return title, a.get("publicationYear"), "datacite"
    except Exception:
        pass
    return None, None, None


def check(line):
    m = DOI_RE.search(line)
    if not m:
        return dict(doi="", resolves="NO-DOI", title_in_ref="", year="", flags="website/no DOI - verify link by hand")
    doi = m.group(0).rstrip('.')
    title, year, src = lookup(doi)
    if title is None:
        return dict(doi=doi, resolves="FAIL", title_in_ref="", year="",
                    flags="DOI not found in Crossref OR DataCite - SUSPECT, verify at doi.org")
    tin = "YES" if _norm(title)[:40] in _norm(line) else "NO  <-- check"
    yrs = YEAR_RE.findall(line)
    ymatch = "ok" if (year and str(year) in line) else (f"ref={yrs} reg={year}" if year else "")
    flags = []
    if tin.startswith("NO"):
        flags.append("title may not match this DOI")
    if year and str(year) not in line and yrs:
        flags.append("year mismatch")
    return dict(doi=doi, resolves=f"ok({src})", title_in_ref=tin, year=ymatch,
                flags="; ".join(flags) or "ok", crossref_title=title)


def main():
    if len(sys.argv) < 2:
        print("usage: python validate_citations.py references.txt"); return
    rows = []
    for i, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        r = check(line); r["line"] = i; rows.append(r)
        print(f"[{i:>3}] {r['resolves']:<12} title_in_ref={r.get('title_in_ref',''):<10} {r['flags']}")
        time.sleep(0.25)
    cols = ["line", "doi", "resolves", "title_in_ref", "year", "flags", "crossref_title"]
    with open("citation_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("\nWrote citation_check.csv. Investigate any row where resolves=FAIL, "
          "title_in_ref=NO, or flags mention a mismatch.")


if __name__ == "__main__":
    main()
