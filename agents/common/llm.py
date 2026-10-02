"""
What every agent driver shares: the ONE model-call seam, the usage ledger, the
.env loader, the schema cleaner and the dataset-notes reader.

    agents/Panel1/run_agents.py         domain panel   -> call(fallbacks=True, ...)
    agents/Panel2/run_target_agents.py  target panel   -> call()
    Report/run_report.py                report writer  -> call()

THE SEAM. `call()` is the only code that talks to the model, and it does so
through LangChain (`langchain_anthropic.ChatAnthropic`). The drivers are plain
functions over JSON packets and files and know nothing about the framework, so
a different model provider, tracing or a retry policy is a change to this one
function. Keep it that way.

What the seam must keep, because the rest of the pipeline relies on it:
  * the request is exactly (system, user, schema): the drivers write it to
    <agent>.request.json before calling
  * the reply comes back as the raw JSON text, which the drivers write to
    <agent>.raw.txt before validating it against their pydantic contract
  * every server-side web search (query, result URLs and titles) is appended
    to `log`, which is what citations are machine-verified against
  * token usage per call, for the ledger

Client-side tools: a driver that wants the model to call a local function
registers it in HANDLERS (name -> callable) before calling; `call()` services
those tool calls. No current driver registers any.
"""
from __future__ import annotations

import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from agents.common.citations import search_log_from_content   # noqa: E402

ENV_FILE = os.path.join(HERE, ".env")                     # ANTHROPIC_API_KEY=...
PUBMED_CACHE = os.path.join(REPO, "cache", "pubmed_cache.json")
PRICE = {"claude-opus-5": (5.0, 25.0, 0.5), "claude-sonnet-5": (2.0, 10.0, 0.2)}   # $/M input, output, cache read

HANDLERS = {}     # name -> callable; client-side tools a driver registers before call()

FALLBACK_BETA = "server-side-fallback-2026-07-01"     # server-side refusal fallbacks
WEB_SEARCH = "web_search_20260209"
# Force gzip. httpx2 2.12 in the agents environment fails decoding zstd (with
# zstandard 0.25) and brotli (with Brotli 1.1) responses, which surfaces as a
# TypeError mid-stream rather than as a transport error.
HEADERS = {"accept-encoding": "gzip"}


def load_env_file(path):
    """Minimal .env loader (KEY=value lines, # comments, optional quotes); does
    not override variables already set in the environment."""
    if not path or not os.path.isfile(path):
        return False
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
    return True




def clean_schema(s):
    """Drop JSON-Schema keywords the structured-output validator may not accept."""
    if isinstance(s, dict):
        return {k: clean_schema(v) for k, v in s.items()
                if k not in ("title", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "default")}
    if isinstance(s, list):
        return [clean_schema(x) for x in s]
    return s




def dataset_notes(path, agent):
    """Dataset-specific notes for the USER turn, '## common' plus '## <agent>'."""
    if not path or not os.path.isfile(path):
        return ""
    out, keep = [], False
    for line in open(path):
        if line.startswith("## "):
            keep = line[3:].strip() in ("common", agent)
            continue
        if keep:
            out.append(line)
    return "".join(out).strip()




def chat_model(model, max_tokens):
    """The LangChain chat model every call uses. Streams: a high-effort structured
    reply over many candidates can run past the ten minutes a non-streaming
    request is allowed. The key is read from ANTHROPIC_API_KEY."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, max_tokens=max_tokens, streaming=True,
                         default_headers=HEADERS)


def content_blocks(msg) -> list[dict]:
    """The content of a LangChain AIMessage as Anthropic content blocks (dicts).
    A streamed tool call carries its arguments as accumulated `partial_json`;
    they are parsed back into `input` here."""
    if isinstance(msg.content, str):
        return [{"type": "text", "text": msg.content}] if msg.content else []
    out = []
    for b in msg.content:
        b = {"type": "text", "text": b} if isinstance(b, str) else dict(b)
        if b.get("type") in ("server_tool_use", "tool_use") and not b.get("input") \
                and b.get("partial_json"):
            try:
                b["input"] = json.loads(b["partial_json"])
            except json.JSONDecodeError:
                pass
        out.append(b)
    return out


def call(model, system, user, schema, max_tokens, effort, tools=None, web_search=0,
         log=None, fallbacks=False, text="join", check_stop=False, turns=12):
    """One model call. Returns (text, usage_dict).

    Structured output: `schema` is sent as output_config.format, so the reply
    text is one JSON object. The system prompt is cached (cache_control).
    Continues on `pause_turn` (a server-side search still in flight) and
    services any client-side tool the caller registered in HANDLERS.

    web_search   max uses of the server-side web search tool; 0 = no search
    fallbacks    enable server-side refusal fallbacks (retried without them if
                 the API does not accept the beta)
    text         "join": all text blocks of the reply; "last": the last one
    check_stop   raise on a refusal or a reply truncated at max_tokens instead
                 of returning its text
    turns        bound on continuations: the loop must terminate
    """
    import anthropic
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    llm = chat_model(model, max_tokens)
    tool_defs = list(tools or [])
    if web_search:
        tool_defs.append({"type": WEB_SEARCH, "name": "web_search", "max_uses": web_search})
    if tool_defs:
        llm = llm.bind_tools(tool_defs)
    kw = dict(output_config={"effort": effort,
                             "format": {"type": "json_schema", "schema": schema}})
    if fallbacks:
        kw.update(betas=[FALLBACK_BETA], fallbacks="default")

    messages = [SystemMessage(content=[{"type": "text", "text": system,
                                        "cache_control": {"type": "ephemeral"}}]),
                HumanMessage(content=user)]
    # model_served: the model that answered, which differs from `model` when a
    # server-side fallback took the request
    usage = dict(input=0, output=0, cache_read=0, cache_write=0, tool_calls=0, searches=0,
                 model_served=None)
    blocks, stop = [], None
    for _ in range(turns):
        try:
            msg = llm.invoke(messages, **kw)
        except anthropic.BadRequestError as e:
            if fallbacks and ("fallback" in str(e).lower() or "beta" in str(e).lower()):
                print("    (fallbacks not accepted; retrying without)", flush=True)
                return call(model, system, user, schema, max_tokens, effort, tools=tools,
                            web_search=web_search, log=log, fallbacks=False, text=text,
                            check_stop=check_stop, turns=turns)
            raise
        blocks = content_blocks(msg)
        stop = msg.response_metadata.get("stop_reason")
        if log is not None:     # every search, so citations can be machine-verified
            log.extend(dict(tool="web_search", **e) for e in search_log_from_content(blocks))
        um = msg.usage_metadata or {}
        det = um.get("input_token_details") or {}
        cache_read = det.get("cache_read") or 0
        cache_write = sum(det.get(k) or 0 for k in ("cache_creation", "ephemeral_5m_input_tokens",
                                                    "ephemeral_1h_input_tokens"))
        # LangChain's input_tokens includes cached tokens; the ledger keeps them apart
        usage["input"] += (um.get("input_tokens") or 0) - cache_read - cache_write
        usage["output"] += um.get("output_tokens") or 0
        usage["cache_read"] += cache_read
        usage["cache_write"] += cache_write
        usage["searches"] += sum(1 for b in blocks if b.get("type") == "server_tool_use"
                                 and b.get("name") == "web_search")
        usage["model_served"] = msg.response_metadata.get("model_name") or usage["model_served"]

        if stop == "pause_turn":                     # server-side search in flight
            messages.append(msg)
            continue

        local = [tc for tc in (msg.tool_calls or []) if tc["name"] in HANDLERS]
        if stop == "tool_use" and local:
            messages.append(msg)
            for tc in local:
                usage["tool_calls"] += 1
                try:
                    out = HANDLERS[tc["name"]](**tc["args"])
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
                if log is not None:
                    log.append(dict(tool=tc["name"], input=tc["args"], output=out))
                messages.append(ToolMessage(content=json.dumps(out), tool_call_id=tc["id"]))
            continue
        break
    else:
        raise RuntimeError(f"call did not terminate within {turns} turns")

    if check_stop and stop == "refusal":
        det = msg.response_metadata.get("stop_details") or {}
        raise RuntimeError(f"model refused: {det.get('category')} {det.get('explanation', '')}")
    if check_stop and stop == "max_tokens":
        raise RuntimeError("output truncated at max_tokens; raise --max-tokens")
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    if text == "last":
        if not texts:
            raise RuntimeError(f"no text block in response (stop_reason={stop})")
        return texts[-1], usage
    return "".join(texts), usage




def cost(usage, model):
    pi, po, pc = PRICE.get(model, (5.0, 25.0, 0.5))
    searches = usage.get("web_searches", usage.get("searches", 0)) or 0
    return (usage.get("input", 0) * pi + usage.get("output", 0) * po
            + usage.get("cache_read", 0) * pc + usage.get("cache_write", 0) * pi * 1.25) / 1e6 \
        + 0.01 * searches




def append_usage(out_dir, row):
    """Record one call's usage the moment it returns. usage.jsonl is append-only,
    so a resumed, repaired or crashed run keeps the ledger of every call already
    paid for; usage.csv is regenerated from it. (Writing usage.csv once at the end
    of a run lost every earlier call whenever a run was resumed.)"""
    import pandas as pd
    path = os.path.join(out_dir, "usage.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "usage.csv"), index=False)
