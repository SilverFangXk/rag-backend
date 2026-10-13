"""Step 2 of RAG: CHUNK the data.

An embedding model turns a piece of text into ONE vector. If you embed a whole
portfolio as one vector, every question matches "a bit of everything" and nothing
well. If you embed single words, there is no meaning. So we cut the portfolio into
small, self-contained pieces ("chunks"), each about one idea.

Our strategy (it follows how the data is shaped):
  * profile / availability / contact  -> one chunk each
  * each project                       -> 1 overview chunk + 1 chunk per walkthrough step
  * each experience                    -> 1 chunk (split by paragraph if the text is long)
  * each education / certification     -> 1 chunk each
  * skills                             -> ONE chunk PER CATEGORY (individual skills are
                                          2-word documents: useless to embed alone)
  * "overview" chunk                   -> a table of contents, for broad questions like
                                          "tell me about her"
  * knowledge/extra_facts.md           -> facts that live in your JSX, not Firestore

Every chunk starts with a header line ("Project: X", "Experience: role at company").
That context travels inside the embedding, so "what did she do at Nexteer" matches
even when the paragraph itself never repeats the company name.

This file is PURE (no network), so it is easy to unit-test.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
MAX_CHARS = 1000  # ~250 tokens: small enough to stay focused, big enough to keep context


@dataclass(frozen=True)
class Chunk:
    id: str  # stable and deterministic -> re-indexing updates instead of duplicating
    section: str  # human label shown in the UI ("Projects", "Experience", ...)
    title: str
    text: str

    def fingerprint(self, salt: str = "") -> str:
        """Hash of everything that affects the vector. Unchanged hash => skip re-embedding."""
        raw = f"{salt}|{self.section}|{self.title}|{self.text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- small helpers
def _s(v) -> str:
    return "" if v is None else str(v).strip()


def clean(v) -> str:
    return _s(v).replace("**", "")  # the admin uses **bold** markup in some fields


def paragraphs(v) -> list[str]:
    """description can be a string (new) or a list of lines (old data)."""
    parts = [clean(x) for x in v] if isinstance(v, list) else re.split(r"\n\s*\n", clean(v))
    return [p.strip() for p in parts if p and p.strip()]


def fmt_date(v) -> str:
    """'2026-07-15' -> 'Jul 2026'. Anything else is returned as typed."""
    s = _s(v)
    m = re.fullmatch(r"(\d{4})-(\d{2})(?:-\d{2})?", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{MONTHS[int(m.group(2)) - 1]} {m.group(1)}"
    return s


def fmt_month_long(v) -> str:
    """'2027-01' -> 'January 2027'."""
    s = _s(v)
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{MONTHS_LONG[int(m.group(2)) - 1]} {m.group(1)}"
    return s


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Pack paragraphs into pieces <= max_chars; split oversized ones by sentence."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > max_chars:  # no punctuation at all: hard cut
                units.append(sent[:max_chars])
                sent = sent[max_chars:]
            if sent:
                units.append(sent)

    pieces, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 2 > max_chars:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


def slug(*parts) -> str:
    raw = "--".join(re.sub(r"[^a-zA-Z0-9_-]+", "-", _s(p)).strip("-").lower() or "x" for p in parts)
    return raw[:150]


def _visible(item: dict) -> bool:
    return item.get("visible") is not False  # missing value counts as visible


def _lines(*pairs) -> list[str]:
    """('Label', value) pairs -> 'Label: value' lines, skipping empty values."""
    out = []
    for label, value in pairs:
        if isinstance(value, (list, tuple)):
            value = ", ".join(clean(x) for x in value if clean(x))
        value = clean(value)
        if value:
            out.append(f"{label}: {value}")
    return out


def _multi(base_id: str, section: str, title: str, header: str, body: str) -> list[Chunk]:
    """One chunk if the body is short; otherwise several, each repeating the header."""
    pieces = split_text(body) or [""]
    out = []
    for n, piece in enumerate(pieces):
        h = header if n == 0 else f"{header} (part {n + 1})"
        out.append(Chunk(f"{base_id}--{n}", section, title, f"{h}\n{piece}".strip()))
    return out


# --------------------------------------------------------------- per-section builders
def _profile(site: dict, nxt: dict, contact: dict, owner: str) -> list[Chunk]:
    chunks = []
    lines = [f"Profile of {owner}"] + _lines(
        ("Headline", site.get("heroTitle")),
        ("About", site.get("heroText")),
        ("Current status badge", site.get("badgeText")),
        ("Resume / CV (French)", site.get("cvUrl")),
        ("Resume / CV (English)", site.get("cvUrl2")),
        ("Experience summary", site.get("expStatus")),
    )
    if len(lines) > 1:
        chunks.append(Chunk("profile--main", "Profile", owner, "\n".join(lines)))

    lines = [f"Availability and goals of {owner}"] + _lines(
        ("Heading", nxt.get("heading")),
        ("Details", nxt.get("text")),
        ("Currently seeking", nxt.get("seeking")),
        ("Focus areas", nxt.get("focusAreas")),
        ("Available from", fmt_month_long(nxt.get("availableFrom"))),
        ("Duration (months)", nxt.get("duration")),
        ("Note", nxt.get("footerNote")),
    )
    if len(lines) > 1:
        chunks.append(Chunk("profile--availability", "Availability", "What's next", "\n".join(lines)))

    linkedin = _s(contact.get("linkedin"))
    if linkedin and not linkedin.startswith("http"):
        linkedin = f"https://linkedin.com/in/{linkedin}"
    lines = [f"Contact details of {owner}"] + _lines(
        ("Email", contact.get("email")),
        ("Phone", contact.get("phone")),
        ("LinkedIn", linkedin),
        ("Based in", contact.get("location")),
    )
    if len(lines) > 1:
        chunks.append(Chunk("profile--contact", "Contact", "Contact", "\n".join(lines)))
    return chunks


def _projects(items: list[dict]) -> list[Chunk]:
    chunks = []
    for p in items:
        if not _visible(p):
            continue
        title = clean(p.get("title")) or clean(p.get("slug"))
        if not title:
            continue
        pid = slug(p.get("slug") or title)
        fields = p.get("fields") or ([p["field"]] if p.get("field") else [])
        details = "; ".join(
            f"{clean(d.get('label'))}: {clean(d.get('value'))}"
            for d in (p.get("details") or [])
            if isinstance(d, dict) and clean(d.get("label"))
        )
        body = "\n".join(
            _lines(
                ("Summary", p.get("summary")),
                ("Technologies", p.get("tags")),
                ("Domain", fields),
                ("Details", details),
                ("Source code", p.get("githubUrl")),
                ("Live demo", p.get("demoUrl")),
                ("Report (PDF)", p.get("pdfUrl")),
            )
        )
        chunks.append(Chunk(f"projects--{pid}--overview", "Projects", title, f"Project: {title}\n{body}".strip()))

        for i, st in enumerate(p.get("steps") or []):
            if not isinstance(st, dict):
                continue
            step_title = clean(st.get("title")) or clean(st.get("label")) or f"Step {i + 1}"
            text = clean(st.get("text"))
            if not text:
                continue
            chunks += _multi(
                f"projects--{pid}--step{i + 1}", "Projects", title,
                f"Project: {title} - {step_title}", text,
            )
    return chunks


def _experience(items: list[dict], project_titles: dict[str, str]) -> list[Chunk]:
    chunks = []
    for e in items:
        if not _visible(e):
            continue
        role, company = clean(e.get("role")), clean(e.get("company"))
        if not (role or company):
            continue
        title = f"{role} at {company}" if role and company else role or company
        start, end = fmt_date(e.get("startDate")), fmt_date(e.get("endDate"))
        if start:
            dates = f"{start} - {end}" if end else f"{start} - Present"
        else:
            dates = clean(e.get("duration"))  # old free-text format
        related = [project_titles.get(s, s) for s in (e.get("projectSlugs") or [])]
        # NOTE: no "status" (ongoing/past) is stored here on purpose. It would go stale.
        # The LLM gets today's date at question time and works it out from the dates.
        head = "\n".join(
            [f"Experience: {title}"]
            + _lines(("Location", e.get("location")), ("Dates", dates),
                     ("Technologies", e.get("tags")), ("Projects done here", related))
        )
        body_title = clean(e.get("descriptionTitle"))
        body = "\n\n".join(paragraphs(e.get("description")))
        if body_title and body:
            body = f"{body_title}:\n{body}"
        chunks += _multi(f"experience--{slug(e.get('id') or title)}", "Experience", title, head, body)
    return chunks


def _education(items: list[dict]) -> list[Chunk]:
    chunks = []
    for ed in items:
        degree, school = clean(ed.get("degree")), clean(ed.get("school"))
        if not (degree or school):
            continue
        title = f"{degree} - {school}" if degree and school else degree or school
        start, end = clean(ed.get("startYear")), clean(ed.get("endYear"))
        period = f"{start} - {end}" if start and end else f"{start} - Present" if start else (
            f"Graduated {end}" if end else clean(ed.get("duration")))
        text = "\n".join([f"Education: {title}"] + _lines(
            ("Location", ed.get("location")), ("Period", period), ("Description", ed.get("description"))))
        chunks.append(Chunk(f"education--{slug(ed.get('id') or title)}", "Education", title, text))
    return chunks


def _certifications(items: list[dict]) -> list[Chunk]:
    chunks = []
    for c in items:
        name = clean(c.get("name"))
        if not name:
            continue
        text = "\n".join([f"Certification: {name}"] + _lines(
            ("Issued by", c.get("issuer")), ("Date", c.get("date")),
            ("Description", c.get("description")), ("Verification link", c.get("credentialUrl"))))
        chunks.append(Chunk(f"certifications--{slug(c.get('id') or name)}", "Certifications", name, text))
    return chunks


def _skills(items: list[dict], owner: str) -> list[Chunk]:
    groups: dict[str, list[str]] = {}
    for s in items:
        name = clean(s.get("name"))
        if name:
            groups.setdefault(clean(s.get("category")) or "Other", []).append(name)
    chunks = [
        Chunk(f"skills--{slug(cat)}", "Skills", cat, f"Skills in {cat}: {', '.join(names)}")
        for cat, names in groups.items()
    ]
    if groups:
        listing = "\n".join(f"{cat}: {', '.join(names)}" for cat, names in groups.items())
        chunks += _multi("skills--overview", "Skills", "All skills",
                         f"All technical skills of {owner}, by category", listing)
    return chunks


def _overview(data: dict, owner: str) -> list[Chunk]:
    """A table of contents. Broad questions ('tell me about her') match this chunk."""
    lines = []
    projs = [p for p in data.get("projects", []) if _visible(p) and clean(p.get("title"))]
    if projs:
        lines.append("Projects: " + "; ".join(
            f"{clean(p['title'])}" + (f" ({clean(p.get('summary'))[:110]})" if clean(p.get("summary")) else "")
            for p in projs))
    exps = [e for e in data.get("experience", []) if _visible(e) and (clean(e.get("role")) or clean(e.get("company")))]
    if exps:
        lines.append("Experience: " + "; ".join(
            f"{clean(e.get('role'))} at {clean(e.get('company'))}"
            + (f" ({fmt_date(e.get('startDate'))} - {fmt_date(e.get('endDate')) or 'Present'})" if e.get("startDate") else "")
            for e in exps))
    eds = [e for e in data.get("education", []) if clean(e.get("degree")) or clean(e.get("school"))]
    if eds:
        lines.append("Education: " + "; ".join(f"{clean(e.get('degree'))}, {clean(e.get('school'))}" for e in eds))
    certs = [c for c in data.get("certifications", []) if clean(c.get("name"))]
    if certs:
        lines.append("Certifications: " + "; ".join(
            clean(c["name"]) + (f" ({clean(c.get('issuer'))})" if clean(c.get("issuer")) else "") for c in certs))
    if not lines:
        return []
    return _multi("overview--main", "Overview", "Portfolio overview",
                  f"Overview of {owner}'s portfolio (projects, experience, education, certifications)",
                  "\n".join(lines))


def _extra_facts(markdown: str) -> list[Chunk]:
    """Each '## Heading' section of extra_facts.md becomes one chunk."""
    if not markdown:
        return []
    md = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    chunks = []
    for part in re.split(r"(?m)^##\s+", md)[1:]:  # [0] is whatever precedes the first '##'
        head, _, body = part.partition("\n")
        head, body = head.strip(), body.strip()
        if head and body:
            chunks += _multi(f"facts--{slug(head)}", "About", head, head, body)
    return chunks


# ----------------------------------------------------------------------- entry point
def build_chunks(data: dict, owner_full_name: str, extra_facts_md: str = "") -> list[Chunk]:
    st = data.get("settings", {})
    project_titles = {clean(p.get("slug")): clean(p.get("title")) for p in data.get("projects", [])}

    chunks: list[Chunk] = []
    chunks += _profile(st.get("site", {}), st.get("next", {}), st.get("contact", {}), owner_full_name)
    chunks += _overview(data, owner_full_name)
    chunks += _projects(data.get("projects", []))
    chunks += _experience(data.get("experience", []), project_titles)
    chunks += _education(data.get("education", []))
    chunks += _certifications(data.get("certifications", []))
    chunks += _skills(data.get("skills", []), owner_full_name)
    chunks += _extra_facts(extra_facts_md)

    # ids must be unique; if two items collide, keep both by suffixing
    seen: dict[str, int] = {}
    unique = []
    for c in chunks:
        n = seen.get(c.id, 0)
        seen[c.id] = n + 1
        unique.append(c if n == 0 else Chunk(f"{c.id}-dup{n}", c.section, c.title, c.text))
    return unique
