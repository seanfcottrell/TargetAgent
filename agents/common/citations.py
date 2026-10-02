#!/usr/bin/env python
"""
Machine verification of the literature citations agents return.

Recall from memory is not evidence. A citation counts as support only if a
machine can trace it, and two things are checked:

  SESSION  the citation was among the results of a web search an agent ran
           during this panel: its URL, its PMID (read from a PubMed result URL)
           or its title matches a logged search result
  PUBMED   its PMID resolves in PubMed, and the title, first author and year
           the agent wrote agree with the record

One status per citation:

  verified             PMID resolves and agrees, and a search returned the paper
  verified_session     no PMID, but its URL or title is a search result of this panel
  exists_not_searched  PMID resolves and agrees, but no search returned it: recalled
  pmid_mismatch        PMID resolves to a different paper       (fabrication signal)
  pmid_not_found       PMID does not exist in PubMed            (fabrication signal)
  untraceable          no identifier, no matching search result: memory or unsourced

SUPPORTING = {verified, verified_session}. Everything else is unverifiable, and
the two fabrication signals are reported as such. What a paper actually says
is not checked -- only that the reference is real and was found, not recalled;
the critic still judges whether it supports the claim.

Title agreement is a fuzzy match (normalised containment, or token Jaccard >=
0.6, since search-result titles carry site suffixes such as "- PubMed"); the
score is reported so a reader can check it. Years may differ by one (epub vs
print). PubMed summaries are cached in cache/pubmed_cache.json.
"""
from __future__ import annotations

import json, os, re, time, unicodedata, urllib.request

SUPPORTING = {"verified", "verified_session"}
FABRICATION = {"pmid_mismatch", "pmid_not_found"}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
PMID_URL = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/)(\d{4,9})")


# ------------------------------------------------------------------ search logs
def _get(b, k, default=None):
    return b.get(k, default) if isinstance(b, dict) else getattr(b, k, default)


def search_log_from_content(blocks) -> list[dict]:
    """Queries and results of server-side web searches in one response's content
    (SDK objects or dicts). Result content is a list on success and a single
    error object on failure."""
    queries, out = {}, []
    for b in blocks or []:
        t = _get(b, "type")
        if t == "server_tool_use" and _get(b, "name") == "web_search":
            queries[_get(b, "id")] = (_get(b, "input") or {}).get("query")
        elif t == "web_search_tool_result":
            content = _get(b, "content")
            entry = dict(tool_use_id=_get(b, "tool_use_id"), query=queries.get(_get(b, "tool_use_id")))
            if isinstance(content, list):
                entry["results"] = [dict(url=_get(r, "url"), title=_get(r, "title"),
                                         page_age=_get(r, "page_age")) for r in content]
            else:
                entry["error"] = _get(content, "error_code")
            out.append(entry)
    return out


def norm_url(u: str | None) -> str:
    if not u:
        return ""
    u = re.sub(r"^https?://", "", u.strip().lower())
    u = re.sub(r"^www\.", "", u)
    return re.split(r"[?#]", u)[0].rstrip("/")


def norm_title(t: str | None) -> str:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()


def title_similarity(a: str | None, b: str | None) -> float:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / len(ta | tb)


def surname(name: str | None) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower().strip()
    n = n.split(",")[0].strip()
    parts = n.split()
    if not parts:
        return ""
    # PubMed gives "Chapin JC"; agents may write "Chapin", "J. C. Chapin" or "Chapin, J"
    return parts[0] if len(parts) == 1 or len(parts[-1]) <= 3 else parts[-1]


# ------------------------------------------------------------------ PubMed
class PubMed:
    def __init__(self, cache_path: str):
        self.path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}

    def get(self, pmids) -> dict:
        want = sorted({str(p) for p in pmids if p and str(p).isdigit()} - set(self.cache))
        for i in range(0, len(want), 100):
            chunk = want[i:i + 100]
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(EUTILS + ",".join(chunk), timeout=30) as r:
                        res = json.load(r)["result"]
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            for u in chunk:
                rec = res.get(u, {})
                if "error" in rec or not rec.get("title"):
                    self.cache[u] = dict(found=False)
                else:
                    self.cache[u] = dict(found=True, title=rec.get("title"),
                                         first_author=(rec.get("authors") or [{}])[0].get("name"),
                                         year=(re.findall(r"\d{4}", rec.get("pubdate", "")) or [None])[0])
            time.sleep(0.4)                      # NCBI: <= 3 requests/s without a key
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        json.dump(self.cache, open(self.path, "w"))
        return {str(p): self.cache.get(str(p)) for p in pmids if p}


# ------------------------------------------------------------------ verification
def verify(items: list[dict], session: list[dict], pubmed: PubMed) -> dict:
    """items: [{"agent", "where", "citation": {...}}]. session: search-log entries."""
    urls, titles, pmids_seen = set(), [], set()
    for e in session:
        for r in e.get("results", []) or []:
            urls.add(norm_url(r.get("url")))
            titles.append(r.get("title") or "")
            m = PMID_URL.search(r.get("url") or "")
            if m:
                pmids_seen.add(m.group(1))
    pm = pubmed.get([str((it["citation"] or {}).get("pmid") or "").strip() for it in items])
    out = []
    for it in items:
        c = it["citation"] or {}
        pmid = str(c.get("pmid") or "").strip()
        pmid = pmid if pmid.isdigit() else ""
        via = None
        if c.get("url") and norm_url(c["url"]) in urls:
            via = "url"
        elif pmid and pmid in pmids_seen:
            via = "pmid"
        elif c.get("title") and titles and max(title_similarity(c["title"], t) for t in titles) >= 0.6:
            via = "title"
        rec = pm.get(pmid) if pmid else None
        check = None
        if pmid:
            if not rec or not rec.get("found"):
                status = "pmid_not_found"
            else:
                ts = title_similarity(c.get("title"), rec.get("title")) if c.get("title") else None
                au = (surname(c.get("first_author")) == surname(rec.get("first_author"))
                      if c.get("first_author") else None)
                yr = (abs(int(c["year"]) - int(rec["year"])) <= 1
                      if c.get("year") and rec.get("year") else None)
                check = dict(pubmed_title=rec.get("title"), pubmed_first_author=rec.get("first_author"),
                             pubmed_year=rec.get("year"), title_similarity=None if ts is None else round(ts, 2),
                             author_agrees=au, year_agrees=yr)
                agrees = (ts is None or ts >= 0.6) and au is not False and yr is not False
                status = ("verified" if via else "exists_not_searched") if agrees else "pmid_mismatch"
        else:
            status = "verified_session" if via else "untraceable"
        out.append(dict(agent=it["agent"], where=it["where"], claim=(c.get("claim") or "")[:240],
                        title=c.get("title"), first_author=c.get("first_author"), year=c.get("year"),
                        pmid=pmid or None, url=c.get("url"), declared_source=c.get("source"),
                        status=status, supports=status in SUPPORTING, session_match=via, pubmed=check))
    counts = {}
    for o in out:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    n_q = sum(1 for e in session if e.get("query"))
    return dict(method=("session = the panel's logged web-search results; pubmed = esummary "
                        "title/first-author/year agreement; only 'verified' and 'verified_session' "
                        "count as support"),
                n_citations=len(out), n_searches=n_q,
                n_search_results=sum(len(e.get("results", []) or []) for e in session),
                status_counts=counts, citations=out)


def load_session(paths) -> list[dict]:
    """Search-log entries from <tag>.tools.json / <tag>.searches.json files."""
    out = []
    for p in paths:
        for e in json.load(open(p)):
            if e.get("tool") == "web_search" or "results" in e or "error" in e:
                out.append(e)
    return out
