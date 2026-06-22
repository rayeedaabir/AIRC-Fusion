"""
renumber_references.py  (NEW 21 Jun 2026)
-----------------------------------------------------------------------------------
Cureus numbers references in order of FIRST appearance in the text. This script does
that renumber automatically once the manuscript is assembled, and it also audits the
reference list. It does NOT invent or alter any reference - it only reorders/renumbers.

INPUTS
  --manuscript  the assembled manuscript (.docx or .txt) whose in-text citations use
                the SAME numbering as the reference list below (i.e. the CJCS_References499
                scheme). Citations may be [3], [3,5], [3-6], [3, 5-7].
  --refs        the reference list (.docx or .txt). Each reference is one line/paragraph
                beginning "N. " or "N " (N = its current number in that same scheme).

OUTPUTS (written next to the manuscript)
  manuscript_renumbered.txt   the body text with every citation remapped to appearance order
  references_in_order.txt     the reference list reordered + renumbered 1..K by appearance
  renumber_report.txt         old->new map, plus two integrity checks:
                                - CITED BUT NOT IN LIST  (broken citation)
                                - IN LIST BUT NEVER CITED (Cureus will reject these)

USAGE
  python renumber_references.py --manuscript manuscript.docx --refs CJCS_References499.docx
"""
import re, os, argparse


def read_text(path):
    if path.lower().endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_refs(text):
    """Return dict {oldnum: reference_text} from lines beginning 'N.' or 'N '."""
    refs = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.\)]?\s+(.*\S)\s*$", line)
        if m:
            refs[int(m.group(1))] = m.group(2).strip()
    return refs


def expand(token_body):
    """'3, 5-7' -> [3,5,6,7] preserving order of appearance within the bracket."""
    out = []
    for part in token_body.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                out.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


CITE = re.compile(r"\[(\d[\d,\s\-]*)\]")


def first_appearance_order(text):
    order = []
    for m in CITE.finditer(text):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
    return order


def fmt_citation(nums, old2new):
    """Render a remapped citation, compressing consecutive runs to a-b (Cureus: hyphen, no spaces)."""
    new = sorted(old2new[n] for n in nums if n in old2new)
    if not new:
        return "[?]"
    parts, i = [], 0
    while i < len(new):
        j = i
        while j + 1 < len(new) and new[j + 1] == new[j] + 1:
            j += 1
        parts.append(str(new[i]) if i == j else f"{new[i]}-{new[j]}")
        i = j + 1
    return "[" + ",".join(parts) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", required=True)
    ap.add_argument("--refs", required=True)
    a = ap.parse_args()
    body = read_text(a.manuscript)
    refs = parse_refs(read_text(a.refs))

    order = first_appearance_order(body)
    old2new = {old: i + 1 for i, old in enumerate(order)}

    cited = set(order)
    listed = set(refs)
    missing = sorted(cited - listed)       # cited but not in the reference list
    uncited = sorted(listed - cited)       # in the list but never cited

    new_body = CITE.sub(lambda m: fmt_citation(expand(m.group(1)), old2new), body)

    outdir = os.path.dirname(os.path.abspath(a.manuscript))
    with open(os.path.join(outdir, "manuscript_renumbered.txt"), "w", encoding="utf-8") as f:
        f.write(new_body)
    with open(os.path.join(outdir, "references_in_order.txt"), "w", encoding="utf-8") as f:
        for old in order:
            if old in refs:
                f.write(f"{old2new[old]}. {refs[old]}\n")
    with open(os.path.join(outdir, "renumber_report.txt"), "w", encoding="utf-8") as f:
        f.write("OLD -> NEW (order of first appearance)\n")
        for old in order:
            f.write(f"  [{old}] -> [{old2new[old]}]\n")
        f.write(f"\nCITED BUT NOT IN LIST (fix these): {missing or 'none'}\n")
        f.write(f"IN LIST BUT NEVER CITED (remove or cite): {uncited or 'none'}\n")

    print(f"Cited references: {len(cited)} | Listed: {len(listed)}")
    print(f"CITED BUT NOT IN LIST: {missing or 'none'}")
    print(f"IN LIST BUT NEVER CITED: {uncited or 'none'}")
    print(f"Wrote manuscript_renumbered.txt, references_in_order.txt, renumber_report.txt in {outdir}")


if __name__ == "__main__":
    main()
# --- end of file (sentinel) ---
