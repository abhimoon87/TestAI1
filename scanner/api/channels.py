"""
Optional external CLI channels (Agent-Reach ecosystem).

Subprocess wrappers for third-party tools: twitter-cli (Twitter/X search),
curl + Jina Reader (article text), mcporter + Exa (web search). Every call
degrades to None on any failure — channels are optional extras, a scan
never depends on them.

One-time setup (never auto-run by the app):
    uv tool install twitter-cli          # + TWITTER_AUTH_TOKEN/TWITTER_CT0
    curl + Node.js ship with Windows 10+ # Jina + Exa need no config
"""

import json
import logging
import os
import re
import shutil
import subprocess

from ..shared.cache import TTLCache

logger = logging.getLogger(__name__)

_ARTICLE_CACHE = TTLCache[str](ttl=24 * 3600, namespace="jina_article")
_EXA_CACHE = TTLCache[list](ttl=3600, namespace="exa_search")


def run_channel(args: list, timeout: int = 20, env: dict | None = None) -> str | None:
    """Run an external CLI and return stdout, or None on any failure.

    Binary is resolved via shutil.which so Windows .cmd shims (npx.cmd)
    work without a shell. Mirrors the request-layer convention: log and
    return None, never raise.
    """
    exe = shutil.which(args[0])
    if exe is None:
        logger.debug("Channel tool not installed: %s", args[0])
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.info("Channel command failed (%s): %s", args[0], e)
        return None
    if proc.returncode != 0:
        logger.info(
            "Channel %s exited %d: %s",
            args[0],
            proc.returncode,
            (proc.stderr or "").strip()[:300],
        )
        return None
    return proc.stdout


def fetch_article(url: str) -> str | None:
    """Full article text for a URL via Jina Reader (curl https://r.jina.ai/)."""
    if not url or not url.startswith(("http://", "https://")):
        return None
    cached = _ARTICLE_CACHE.get(url)
    if cached:
        return cached
    out = run_channel(
        ["curl", "-s", "--max-time", "15", f"https://r.jina.ai/{url}"],
        timeout=20,
    )
    if not out or not out.strip():
        return None
    text = out.strip()[:50000]
    _ARTICLE_CACHE.set(url, text)
    return text


def twitter_search(
    query: str,
    max_results: int = 20,
    cookies: tuple | None = None,
) -> list[dict] | None:
    """twitter-cli search. cookies = (TWITTER_AUTH_TOKEN, TWITTER_CT0).

    Returns a list of tweet dicts (possibly empty), or None when the tool
    is unavailable/failed/unauthenticated — callers treat None as "skip".
    """
    if not cookies or not cookies[0] or not cookies[1]:
        return None
    env = {
        **os.environ,
        "TWITTER_AUTH_TOKEN": cookies[0],
        "TWITTER_CT0": cookies[1],
    }
    out = run_channel(
        ["twitter", "search", query, "--max", str(max_results), "--json"],
        timeout=30,
        env=env,
    )
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        logger.info("twitter-cli returned non-JSON output")
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("tweets", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    return None


def _extract_exa_list(data, depth: int = 0):
    """Unwrap common envelopes down to a result list (2 levels max)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and depth < 2:
        for key in ("results", "data", "content"):
            inner = data.get(key)
            if isinstance(inner, str) and inner.strip().startswith(("[", "{")):
                try:
                    inner = json.loads(inner)
                except ValueError:
                    continue
            if isinstance(inner, list):
                # MCP content blocks: [{"type": "text", "text": "<json>"}]
                if (
                    inner
                    and isinstance(inner[0], dict)
                    and inner[0].get("type") == "text"
                    and str(inner[0].get("text") or "").lstrip().startswith(("[", "{"))
                ):
                    try:
                        parsed = json.loads(inner[0]["text"])
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, (list, dict)):
                        return _extract_exa_list(parsed, depth + 1)
                return _extract_exa_list(inner, depth + 1)
    return None


def _parse_exa_text(text: str) -> list[dict]:
    """Parse exa-mcp-server's plain-text blocks (Title:/URL:/Highlights:).

    Verified against the live endpoint: --output json wraps one text block
    whose results are separated by '---' lines, not structured JSON.
    """
    results = []
    for block in re.split(r"^\s*---\s*$", text, flags=re.MULTILINE):
        title_m = re.search(r"^Title:\s*(.+)$", block, flags=re.MULTILINE)
        url_m = re.search(r"^URL:\s*(\S+)\s*$", block, flags=re.MULTILINE)
        if not (title_m or url_m):
            continue
        snippet = ""
        hl = re.search(r"^Highlights:\s*\n(.*)$", block, flags=re.MULTILINE | re.DOTALL)
        if hl:
            snippet = "\n".join(
                ln for ln in hl.group(1).splitlines() if ln.strip() != "..."
            ).strip()
        results.append(
            {
                "title": title_m.group(1).strip() if title_m else "",
                "url": url_m.group(1).strip() if url_m else "",
                "snippet": snippet,
            }
        )
    return results


def _normalize_exa_items(items: list) -> list[dict]:
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or ""
        url = item.get("url") or item.get("link") or ""
        snippet = (
            item.get("text") or item.get("snippet") or item.get("description") or ""
        )
        if title or url:
            results.append(
                {"title": str(title), "url": str(url), "snippet": str(snippet)[:400]}
            )
    return results


def exa_search(query: str, n: int = 5) -> list[dict] | None:
    """Exa web search via mcporter ad-hoc URL. Returns [{title, url,
    snippet}] or None. Free, keyless, zero config.

    On-demand only: npx startup costs ~1-3s per call, so never run this
    per-ticker in a scan loop.
    """
    cache_k = _EXA_CACHE.make_key(query, str(n))
    cached = _EXA_CACHE.get(cache_k)
    if cached is not None:
        return cached
    out = run_channel(
        [
            "npx",
            "-y",
            "mcporter",
            "call",
            "https://mcp.exa.ai/mcp.web_search_exa",
            f"query={query}",
            f"numResults={n}",
            "--output",
            "json",
        ],
        timeout=70,
    )
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        logger.info("mcporter returned non-JSON output")
        return None

    results: list[dict] = []
    blocks = data.get("content") if isinstance(data, dict) else None
    if isinstance(blocks, list) and blocks:
        for block in blocks:
            t = block.get("text") if isinstance(block, dict) else None
            if not isinstance(t, str) or not t:
                continue
            if t.lstrip().startswith(("[", "{")):
                try:
                    parsed = json.loads(t)
                except ValueError:
                    parsed = None
                if isinstance(parsed, (list, dict)):
                    items = _extract_exa_list(parsed)
                    if items is not None:
                        results.extend(_normalize_exa_items(items))
                        continue
            results.extend(_parse_exa_text(t))
    else:
        items = _extract_exa_list(data)
        if items is None:
            logger.debug("Unrecognized Exa response envelope")
            return None
        results.extend(_normalize_exa_items(items))

    _EXA_CACHE.set(cache_k, results)
    return results
