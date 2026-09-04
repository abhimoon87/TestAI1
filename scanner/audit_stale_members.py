"""Maintenance: find universe members whose data is stale but NOT annotated.

``universes.SUSPENDED_OR_DELISTED`` is a hand-maintained list of names that
still sit in the universe lists but no longer trade (GSPL — suspended May
2026, TATAMETALI — merged into Tata Steel 2024).  The scan engine skips those
names so they are never re-fetched.  This script re-checks reality against
that list, so the annotation stays current:

    1) STALE — NOT ANNOTATED    -> candidates to ADD to SUSPENDED_OR_DELISTED
    2) STALE — ALREADY ANNOTATED -> ok (engine already skips them)
    3) ANNOTATED BUT FRESH       -> candidates to REMOVE (trading resumed)
    4) NO DATA IN WINDOW         -> not fetched at all (delisted long ago?)
    5) RENAME SUGGESTED         -> no-data symbol that maps to a live NSE
                                   symbol (ASTER -> ASTERDM, AVALONLABS -> AVALON)

Two extra safety passes run on the section-4 "missing" names:
  * live-probe — each missing name is re-attempted through the per-ticker
    provider chain (bypasses the negative-cache skip that gates the batch
    fallback), so a symbol wrongly marked dead gets a second chance;
  * rename search — the name is matched against the live NSE mainboard list
    (prefix rules + fuzzy match) and the best candidate that actually has
    data is suggested as a rename.

Usage:
    python -m scanner.audit_stale_members                          # ALL (Combined), 3y, 45d
    python -m scanner.audit_stale_members --universe "FnO STOCKS" --days 30
    python -m scanner.audit_stale_members --tickers GSPL RELIANCE --json
    python -m scanner.audit_stale_members --all                    # every static universe (one union fetch)
    python -m scanner.audit_stale_members --all --no-probe --no-renames   # report-only, no extra network
    python -m scanner.audit_stale_members --all --fix             # + apply renames/annotation edits to universes.py
    python -m scanner.audit_stale_members --all --fix --dry-run   # preview the exact edits, write nothing

Data is served from the disk cache when warm (see scanner.cache), so repeat
runs are fast and offline.  A 3y window is the default: names suspended more
than a year ago vanish from 1y fetches entirely and would be misreported as
missing rather than stale.
"""

from __future__ import annotations

import argparse
import json
import sys

from .data_fetcher import (
    _negative_cache_load,
    fetch_batch_yfinance,
    fetch_stock_data,
)
from .scanner_engine import _find_stale_members
from .universes import UNIVERSES

# Dynamic full-market entries resolve live via symbol_fetcher (and default to
# NIFTY_BROAD copies before that); auditing them would just re-list the same
# names.  Only the static universe map is audited.
_STATIC_UNIVERSES = {
    k: v for k, v in UNIVERSES.items()
    if "Live" not in k and not k.startswith("FULL MARKET")
}

# Probe concurrency cap for the missing-name live re-check (mirrors
# data_fetcher.FALLBACK_WORKERS).
_PROBE_WORKERS = 8


def _membership_index(tickers: list) -> dict:
    """ticker -> sorted static universe names that contain it."""
    index: dict = {}
    for name, members in _STATIC_UNIVERSES.items():
        for t in members:
            if t in tickers:
                index.setdefault(t, []).append(name)
    for t in tickers:
        index.setdefault(t, [])
    return {t: sorted(names) for t, names in index.items()}


def _nse_mainboard() -> list:
    """Current NSE mainboard symbols (upper-cased), or [] when unavailable.

    Uses symbol_fetcher's 4h-cached nselib list — the same list that builds
    the FULL MARKET universe.  Empty on failure, so callers degrade to
    "no rename suggestions" instead of guessing.
    """
    try:
        from .symbol_fetcher import fetch_nse_mainboard

        symbols = fetch_nse_mainboard()
        if symbols and len(symbols) > 500:
            return [str(s).strip().upper() for s in symbols if str(s).strip()]
    except Exception:
        pass
    return []


def _rank_rename_candidates(missing: list, mainboard: list) -> dict:
    """Rank candidate NSE symbols for each data-less ticker (unverified).

    Scoring rules, in priority order:
      * 2.0  mainboard symbol starts with the bad ticker  (ASTER -> ASTERDM)
      * 1.5  bad ticker starts with the mainboard symbol (AVALONLABS -> AVALON)
      * 1.0+ratio  fuzzy match (difflib)                 (BIRLASOFT -> BSOFT)
    Returns {bad_ticker: [candidates best-first]} — empty list when the
    ticker is itself a valid NSE symbol (just no data in the window) or no
    plausible candidate exists.
    """
    import difflib

    mb = [s for s in mainboard if s]
    mb_set = set(mb)
    out: dict = {}
    for t in missing:
        if t in mb_set:
            out[t] = []  # valid symbol — suspended/new listing, not a rename
            continue
        scored: dict = {}
        for s in mb:
            if s == t:
                continue
            if s.startswith(t):
                scored[s] = max(scored.get(s, 0.0), 2.0)
            elif t.startswith(s):
                scored[s] = max(scored.get(s, 0.0), 1.5)
        for s in difflib.get_close_matches(t, mb, n=10, cutoff=0.55):
            ratio = difflib.SequenceMatcher(None, t, s).ratio()
            scored[s] = max(scored.get(s, 0.0), 1.0 + ratio)
        out[t] = [s for s, _ in sorted(scored.items(),
                                       key=lambda kv: (-kv[1], kv[0]))][:6]
    return out


def _candidate_budget(ranked: dict, budget: int = 60) -> list:
    """Fairly pick up to ``budget`` unique candidates across all tickers.

    Round-robins over the per-ticker rankings so every missing ticker keeps
    its top candidate before any ticker gets a second one — truncating the
    union alphabetically would silently drop every candidate for some
    tickers once the pool exceeds the budget.
    """
    picked: list = []
    seen: set = set()
    rounds = max((len(cs) for cs in ranked.values()), default=0)
    for r in range(rounds):
        for candidates in ranked.values():
            if len(picked) >= budget:
                return picked
            if r < len(candidates):
                c = candidates[r]
                if c not in seen:
                    seen.add(c)
                    picked.append(c)
    return picked


def _suggest_renames(missing: list, period: str = "3y") -> dict:
    """bad ticker -> best live-verified NSE symbol (or {} when none).

    Candidates come from ``_rank_rename_candidates`` against the live NSE
    mainboard list; the best-ranked candidate that actually returns data is
    verified with one batched download (yfinance + NSE fallback), so a
    suggestion is never a guess.
    """
    mainboard = _nse_mainboard()
    if not mainboard:
        return {}
    ranked = _rank_rename_candidates(missing, mainboard)
    cands = _candidate_budget(ranked)  # bounds the verification download
    if not cands:
        return {}
    fetched = fetch_batch_yfinance(cands, period=period, timeframe="D") or {}
    verified: dict = {}
    for bad, candidates in ranked.items():
        for c in candidates:
            if c in fetched:
                verified[bad] = c
                break
    return verified


def audit_stale_members(tickers: list, period: str = "3y",
                        max_age_days: float = 45.0,
                        probe_missing: bool = True,
                        suggest_renames: bool = True) -> dict:
    """Fetch ``tickers`` and split them by staleness vs the annotation list.

    Uses the same ``_find_stale_members`` the scan engine relies on, so the
    audit agrees with scan-time warnings.  ``SUSPENDED_OR_DELISTED`` is read
    at call time so tests (and the script itself) can refresh it without an
    import-order dance.

    Every name the batch fetch missed is then live-probed per-ticker (the
    batch fallback skips negative-cache entries; the per-ticker chain does
    not) and matched against the NSE mainboard for verified rename
    suggestions.
    """
    from .universes import SUSPENDED_OR_DELISTED

    raw = fetch_batch_yfinance(tickers, period=period, timeframe="D") or {}
    missing = sorted(t for t in tickers if t not in raw)

    # 1) Live-probe: re-attempt genuinely-missing names through the per-ticker
    # provider chain (jugaad -> yfinance -> nselib), which bypasses the
    # negative-cache skip that gates the batch fallback — a symbol wrongly
    # marked dead gets a second chance before being reported missing.  Each
    # probe can burn provider timeouts, so run them on a bounded pool
    # (mirrors FALLBACK_WORKERS in data_fetcher).
    if probe_missing and missing:
        from concurrent.futures import ThreadPoolExecutor

        def _probe(t):
            try:
                return t, fetch_stock_data(t, period=period,
                                           timeframe="D", retries=1)
            except Exception:
                return t, None

        workers = min(_PROBE_WORKERS, len(missing))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for t, df in pool.map(_probe, missing):
                if df is not None and not df.empty and len(df) >= 50:
                    raw[t] = df
        missing = sorted(t for t in tickers if t not in raw)

    stale = _find_stale_members(raw, max_age_days=max_age_days)
    stale_dates = dict(stale)
    stale_set = set(stale_dates)

    annotated = set(SUSPENDED_OR_DELISTED)
    unannotated = [(t, stale_dates[t]) for t in stale_dates if t not in annotated]
    annotated_stale = [(t, stale_dates[t]) for t in stale_dates if t in annotated]
    annotated_fresh = sorted(t for t in annotated
                             if t in raw and t not in stale_set)
    # Names skipped by the dead-symbol cache never even reach the providers;
    # report them distinctly from genuine "no data in window" findings.
    neg_keys = set(_negative_cache_load())
    neg_skipped = sorted(t for t in missing if t in neg_keys)

    # 2) Rename search on what is still missing.
    rename_suggestions: dict = {}
    if suggest_renames and missing:
        rename_suggestions = _suggest_renames(missing, period=period)

    return {
        "period": period,
        "max_age_days": max_age_days,
        "tickers": len(tickers),
        "fetched": len(raw),
        "stale_total": len(stale),
        "unannotated_stale": unannotated,     # [(ticker, 'YYYY-MM-DD')] oldest first
        "annotated_stale": annotated_stale,
        "annotated_fresh": annotated_fresh,
        "missing": missing,
        "neg_cache_skipped": neg_skipped,
        "rename_suggestions": rename_suggestions,  # bad -> verified good symbol
        "membership": _membership_index(tickers),
        "annotated": sorted(annotated),
    }


def audit_all_universes(period: str = "3y", max_age_days: float = 45.0,
                        probe_missing: bool = True,
                        suggest_renames: bool = True) -> dict:
    """Audit every static universe in one union fetch (no repeat downloads).

    Returns the same report as ``audit_stale_members`` plus a
    ``per_universe`` breakdown: {universe: {members, fetched, missing,
    stale_unannotated}} so symbol problems can be attributed to the exact
    list that carries them.
    """
    tickers = sorted({t for lst in _STATIC_UNIVERSES.values() for t in lst})
    res = audit_stale_members(
        tickers, period=period, max_age_days=max_age_days,
        probe_missing=probe_missing, suggest_renames=suggest_renames,
    )
    res["universe"] = "ALL STATIC UNIVERSES"
    missing_set = set(res["missing"])
    per_universe: dict = {}
    for name, lst in _STATIC_UNIVERSES.items():
        miss = sorted(t for t in lst if t in missing_set)
        per_universe[name] = {
            "members": len(lst),
            "fetched": len(lst) - len(miss),
            "missing": miss,
            "stale_unannotated": sorted(
                t for t, _d in res["unannotated_stale"] if t in lst
            ),
        }
    res["per_universe"] = per_universe
    return res


def _days_stale(last: str) -> int:
    from datetime import date
    return (date.today() - date.fromisoformat(last)).days


def _paste_ready(unannotated: list) -> str:
    lines = [f'        "{t}": "no trades since {d}",' for t, d in unannotated]
    return "SUSPENDED_OR_DELISTED.update({\n" + "\n".join(lines) + "\n    })"


def format_report(res: dict) -> str:
    """Human-readable report (also printed by the CLI)."""
    lines = [
        f"=== Stale-member audit (period {res['period']}, "
        f"cutoff {res['max_age_days']:.0f}d) ===",
        f"Tickers: {res['tickers']} | fetched: {res['fetched']} | "
        f"stale: {res['stale_total']} | missing: {len(res['missing'])}",
    ]
    per_universe = res.get("per_universe")
    if per_universe:
        width = max((len(u) for u in per_universe), default=12)
        lines.append("Per-universe (missing / members):")
        for u, info in sorted(per_universe.items()):
            missing_n = len(info["missing"])
            mark = "  <--" if missing_n else ""
            lines.append(
                f"   {u:<{width}}  {len(info['missing'])}/{info['members']} missing"
                + mark
            )
    lines.append("")
    lines.append(f"1) STALE — NOT ANNOTATED ({len(res['unannotated_stale'])}) "
                 "-> add to SUSPENDED_OR_DELISTED:")
    if res["unannotated_stale"]:
        for t, d in res["unannotated_stale"]:
            lines.append(
                f"   {t:<14} last {d}  ({_days_stale(d)}d)  "
                f"in: {', '.join(res['membership'].get(t, [])) or '(direct list)'}"
            )
        lines.append("   paste-ready:")
        lines.append("   " + _paste_ready(res["unannotated_stale"]).replace("\n", "\n   "))
    else:
        lines.append("   (none — annotation list is current)")

    lines.append("")
    lines.append(f"2) STALE — ALREADY ANNOTATED ({len(res['annotated_stale'])}) "
                 "(engine skips these):")
    for t, d in res["annotated_stale"]:
        lines.append(f"   {t:<14} last {d}  ({_days_stale(d)}d)")
    if not res["annotated_stale"]:
        lines.append("   (none)")

    lines.append("")
    lines.append(f"3) ANNOTATED BUT FRESH ({len(res['annotated_fresh'])}) "
                 "-> candidates to REMOVE (trading resumed?):")
    for t in res["annotated_fresh"]:
        lines.append(f"   {t}")
    if not res["annotated_fresh"]:
        lines.append("   (none)")

    lines.append("")
    neg = set(res["neg_cache_skipped"])
    lines.append(f"4) NO DATA IN WINDOW ({len(res['missing'])}) "
                 "-> possibly delisted before the window / bad symbol:")
    for t in res["missing"]:
        suffix = "  [dead-symbol cache — skipped, try clearing it]" if t in neg else ""
        lines.append(f"   {t}{suffix}")
    if not res["missing"]:
        lines.append("   (none)")

    lines.append("")
    renames = res.get("rename_suggestions", {})
    lines.append(f"5) MISSING — RENAME SUGGESTED ({len(renames)}) "
                 "-> live-verified against the NSE mainboard:")
    for bad, good in sorted(renames.items()):
        lines.append(f"   {bad} -> {good}   (candidate has data on {res['period']})")
    if not renames:
        lines.append("   (none)")
    return "\n".join(lines)


def apply_fixes(res: dict, path: str | None = None,
                dry_run: bool = False) -> dict:
    """Apply audit findings directly to universes.py (in place).

    * renames      — every ``"BAD"`` occurrence (universe lists, SECTOR_MAP)
                     becomes ``"GOOD"``; quoted replacement cannot touch
                     longer symbols (ASTER -> ASTERDM never hits ASTERMINDS).
    * section 1    — stale-unannotated names get inserted into
                     ``SUSPENDED_OR_DELISTED`` with the same style as the
                     hand-maintained entries ("no trades since YYYY-MM-DD").
    * section 3    — annotated-but-fresh names are removed from the dict
                     (trading resumed).

    With ``dry_run=True`` the same edits are computed and validated
    (ast.parse) but nothing is written; the summary carries a ``diff`` field
    — a unified diff showing exactly which universes.py lines would change.
    Otherwise the file is rewritten atomically (tmp + os.replace) and a
    ``.bak`` copy of the pre-fix file is kept.  Report-only when nothing is
    to be fixed.

    Returns a summary dict: {"renamed": [(bad, good, n)],
    "annotated_added": [(ticker, date)], "annotated_removed": [ticker],
    "not_found": [ticker], "backup": path|None, "changed": bool,
    "dry_run": bool, "diff": str|None}.
    """
    import ast
    import os
    import re
    import shutil

    if path is None:
        from . import universes as _u

        path = _u.__file__
    with open(path, encoding="utf-8") as f:
        text = f.read()
    orig = text

    summary: dict = {"renamed": [], "annotated_added": [],
                     "annotated_removed": [], "not_found": []}

    # 1) Verified renames — quoted-string replacement everywhere.
    for bad, good in sorted(res.get("rename_suggestions", {}).items()):
        needle = f'"{bad}"'
        n = text.count(needle)
        if n == 0:
            summary["not_found"].append(bad)
            continue
        text = text.replace(needle, f'"{good}"')
        summary["renamed"].append((bad, good, n))

    # 1b) A rename that lands on a ticker SECTOR_MAP already knows leaves two
    # entries under one key (last wins silently).  Drop the earlier duplicates.
    if summary["renamed"]:
        text, deduped = _dedupe_sector_map_keys(text)
        if deduped:
            summary["sector_map_deduped"] = deduped

    # 2) Annotation adds (section 1) and removes (section 3) inside the
    #    SUSPENDED_OR_DELISTED dict block.  Existing entry lines are kept
    #    verbatim; new entries are merged in and the whole block is rewritten
    #    with keys in alphabetical order (matching the hand-maintained style).
    adds = list(res.get("unannotated_stale", []))       # [(ticker, last_date)]
    removes = list(res.get("annotated_fresh", []))      # [ticker]
    entry_re = re.compile(r'^[ \t]*"([A-Za-z0-9._-]+)":')
    lines = text.splitlines(keepends=True)
    block_i = next((i for i, ln in enumerate(lines)
                    if ln.lstrip().startswith("SUSPENDED_OR_DELISTED")
                    and "= {" in ln), None)
    close_i = None
    if block_i is not None:
        close_i = next((j for j in range(block_i + 1, len(lines))
                        if lines[j].strip() == "}"), None)
    if (adds or removes) and (block_i is None or close_i is None):
        summary["skipped"] = "SUSPENDED_OR_DELISTED block not found/closed — " \
                              "annotation edits skipped (renames still applied)"
    elif adds or removes:
        existing = []  # (key or None, original line) for every inner line
        for ln in lines[block_i + 1:close_i]:
            m = entry_re.match(ln)
            existing.append((m.group(1) if m else None, ln))
        existing_keys = {k for k, _ln in existing if k is not None}
        removed = [t for t in removes if t in existing_keys]
        # Survivors (kept verbatim) + new entries, together key-sorted.
        entries = sorted(
            [(k, ln) for k, ln in existing
             if k is not None and k not in removes]
            + [(t, f'    "{t}": "no trades since {d}",\n')
               for t, d in adds if t not in existing_keys],
            key=lambda kv: kv[0],
        )
        tails = [ln for k, ln in existing if k is None]  # non-entry lines
        text = ("".join(lines[:block_i + 1])
                + "".join(ln for _k, ln in entries)
                + "".join(tails)
                + "".join(lines[close_i:]))
        summary["annotated_added"] = [
            (t, d) for t, d in adds if t not in existing_keys
        ]
        summary["annotated_removed"] = removed

    changed = text != orig
    summary["changed"] = changed
    summary["backup"] = None
    summary["dry_run"] = bool(dry_run)
    summary["diff"] = None
    if not changed:
        return summary
    ast.parse(text)  # hard-fail on a broken edit before touching disk
    if dry_run:
        import difflib

        summary["diff"] = "".join(difflib.unified_diff(
            orig.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="universes.py (current)",
            tofile="universes.py (after --fix)",
        )).rstrip("\n")
        return summary
    shutil.copy2(path, path + ".bak")
    tmp = path + ".fix.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    summary["backup"] = path + ".bak"
    return summary


def _dedupe_sector_map_keys(text: str):
    """Collapse duplicate keys inside ``SECTOR_MAP``, keeping each key's LAST.

    A ``--fix`` rename whose target is already a known sector ticker turns
    two entries into two entries with the same key (e.g. ``BIRLASOFT`` and
    ``BSOFT`` both mapping to ``IT``).  Python dict literals silently keep
    the last duplicate at runtime, so removing every earlier occurrence
    keeps the file honest without changing behavior.  Returns
    ``(new_text, [deduped keys])``; ``text`` is returned unchanged when the
    block is absent or has no duplicates.
    """
    import re

    start_m = re.search(r"SECTOR_MAP = \{", text)
    if not start_m:
        return text, []
    close_m = re.search(r"^\}", text[start_m.end():], re.MULTILINE)
    if not close_m:
        return text, []
    body_start = start_m.end()
    body_end = body_start + close_m.start()
    body = text[body_start:body_end]

    pair_re = re.compile(r'"([A-Za-z0-9._&%-]+)":\s*"[^"\n]*"')
    spans: dict = {}
    for m in pair_re.finditer(body):
        spans.setdefault(m.group(1), []).append((m.start(), m.end()))

    removals = []
    deduped = []
    for key, occ in spans.items():
        if len(occ) > 1:
            removals.extend(occ[:-1])  # keep the last occurrence
            deduped.append(key)
    if not removals:
        return text, []

    # Splice right-to-left, swallowing an adjacent comma (a following one
    # first, otherwise the preceding one) so the line stays well-formed.
    for s, e in sorted(removals, reverse=True):
        after = body[e:]
        m_after = re.match(r"\s*,\s*", after)
        if m_after:
            e = e + m_after.end()
        else:
            m_before = re.search(r",\s*$", body[:s])
            if m_before:
                s = m_before.start()
        body = body[:s] + body[e:]
    return text[:body_start] + body + text[body_end:], sorted(deduped)


def format_fix_summary(summary: dict) -> str:
    """Human-readable summary of what apply_fixes() changed (or didn't)."""
    if not summary.get("changed"):
        if summary.get("dry_run"):
            return ("--- --fix --dry-run: nothing would change "
                    "(annotation list and symbols current)")
        return "--- --fix: nothing to apply (annotation list and symbols current)"
    if summary.get("dry_run"):
        lines = ["--- --fix DRY RUN — would change scanner/universes.py ---"]
    else:
        lines = ["--- Applied fixes to scanner/universes.py ---"]
    for bad, good, n in summary.get("renamed", []):
        lines.append(f"Renamed  {bad} -> {good} ({n} occurrence(s))")
    for t, d in summary.get("annotated_added", []):
        lines.append(f"Added    {t} to SUSPENDED_OR_DELISTED (no trades since {d})")
    for t in summary.get("annotated_removed", []):
        lines.append(f"Removed  {t} from SUSPENDED_OR_DELISTED (trading resumed)")
    for t in summary.get("not_found", []):
        lines.append(f"Skipped  {t} -> not present in universes.py (already fixed?)")
    for t in summary.get("sector_map_deduped", []):
        lines.append(f"Sector map: {t} kept its existing entry "
                     "(duplicate from the rename removed)")
    if summary.get("skipped"):
        lines.append(f"Warning: {summary['skipped']}")
    if summary.get("backup"):
        lines.append(f"Backup:  {summary['backup']}")
    if summary.get("dry_run") and summary.get("diff"):
        lines.append("")
        lines.append("Lines that would change:")
        lines.append(summary["diff"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Audit universe members for stale (unannotated) data")
    ap.add_argument("--universe", default="ALL (Combined)",
                    help="Static universe key (default: ALL (Combined))")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Explicit ticker list (overrides --universe)")
    ap.add_argument("--period", default="3y",
                    help="Data window; 3y catches suspensions up to ~3y old")
    ap.add_argument("--days", type=float, default=45.0,
                    help="Stale cutoff in days (default 45, matches settings)")
    ap.add_argument("--all", action="store_true",
                    help="Audit every static universe in one union fetch")
    ap.add_argument("--no-probe", action="store_true",
                    help="Skip the per-ticker live re-probe of missing names")
    ap.add_argument("--no-renames", action="store_true",
                    help="Skip rename suggestions for missing names")
    ap.add_argument("--fix", action="store_true",
                    help="Apply verified renames + annotation updates directly "
                         "to universes.py (report-only without it; .bak kept)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview the exact universes.py edits --fix would "
                         "make (unified diff), without writing anything")
    ap.add_argument("--json", action="store_true",
                    help="Print the raw result dict instead of the table")
    args = ap.parse_args()

    probe_missing = not args.no_probe
    suggest_renames = not args.no_renames
    if args.all:
        res = audit_all_universes(period=args.period, max_age_days=args.days,
                                  probe_missing=probe_missing,
                                  suggest_renames=suggest_renames)
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
        res = audit_stale_members(tickers, period=args.period,
                                  max_age_days=args.days,
                                  probe_missing=probe_missing,
                                  suggest_renames=suggest_renames)
    else:
        tickers = list(_STATIC_UNIVERSES.get(args.universe, []))
        if not tickers:
            ap.error(f"unknown static universe: {args.universe} "
                     f"(choices: {sorted(_STATIC_UNIVERSES)})")
        res = audit_stale_members(tickers, period=args.period,
                                  max_age_days=args.days,
                                  probe_missing=probe_missing,
                                  suggest_renames=suggest_renames)
    fix_mode = args.fix or args.dry_run
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        if fix_mode:
            summary = apply_fixes(res, dry_run=args.dry_run)
            print(json.dumps(summary, indent=2, default=str), file=sys.stderr)
    else:
        print(format_report(res))
        if fix_mode:
            print()
            print(format_fix_summary(apply_fixes(res, dry_run=args.dry_run)))
        elif res["unannotated_stale"] or res.get("rename_suggestions"):
            print("\nNext step: re-run with --fix to apply renames and "
                  "annotation updates directly to universes.py.")


if __name__ == "__main__":
    main()