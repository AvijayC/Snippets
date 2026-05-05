from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .documents import DocumentChunk, chunk_document, iter_supported_files, load_document
from .tooling import ToolContext, ToolRegistry, ToolResult, ToolSpec


TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


class RagIndex:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.keyword_chunks: list[DocumentChunk] = []
        self.collection: Any = None
        self.chroma_error: str | None = None

    @property
    def config(self) -> Any:
        return self.runtime.config.rag

    def reindex(self, use_chroma: bool = True) -> dict[str, Any]:
        docs_path = resolve_project_path(self.config.docs_path, self.runtime.project_root)
        chroma_path = resolve_project_path(self.config.chroma_path, self.runtime.project_root)
        documents = []
        chunks: list[DocumentChunk] = []
        errors: list[str] = []
        active_paths: list[str] = []
        for path in iter_supported_files(docs_path):
            try:
                document = load_document(path)
                source_path = _display_path(path, self.runtime.project_root)
                active_paths.append(source_path)
                doc_chunks = chunk_document(
                    document,
                    self.config.chunk_chars,
                    self.config.chunk_overlap_chars,
                )
                for chunk in doc_chunks:
                    chunk.source_path = source_path
                documents.append(document)
                chunks.extend(doc_chunks)
                self.runtime.state.upsert_ingested_doc(
                    doc_id=document.id,
                    path=source_path,
                    content_hash=document.metadata["sha256"],
                    mtime=path.stat().st_mtime,
                    chunks=len(doc_chunks),
                    metadata=document.metadata,
                )
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        pruned_docs = self.runtime.state.prune_ingested_docs(active_paths)
        self.keyword_chunks = chunks
        if use_chroma:
            chroma_status = self._reindex_chroma(chroma_path, chunks)
        else:
            self.collection = None
            self.chroma_error = "Chroma indexing skipped for fast startup."
            chroma_status = {"enabled": False, "skipped": True, "reason": self.chroma_error}
        payload = {
            "docs_path": str(docs_path),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "pruned_document_records": pruned_docs,
            "errors": errors,
            "chroma": chroma_status,
        }
        self.runtime.state.add_debug_event("rag_reindex", payload)
        return payload

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        search_mode: str = "auto",
        include_terms: list[str] | None = None,
        exclude_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        top_k = top_k or self.config.top_k
        min_score = self.config.min_score if min_score is None else min_score
        search_mode = search_mode if search_mode in {"auto", "chroma", "keyword"} else "auto"
        include_terms = [term.strip() for term in include_terms or [] if str(term).strip()]
        exclude_terms = [term.strip() for term in exclude_terms or [] if str(term).strip()]
        results: list[dict[str, Any]]
        source = "keyword"
        warnings: list[str] = []
        if self.collection is not None and search_mode in {"auto", "chroma"}:
            try:
                raw = self.collection.query(query_texts=[query], n_results=top_k)
                results = self._convert_chroma_results(raw)
                source = "chroma"
            except Exception as exc:
                warnings.append(f"Chroma query failed: {exc}")
                if search_mode == "chroma" or not self.config.fallback_keyword_search:
                    results = []
                else:
                    warnings.append("Using keyword fallback.")
                    results = self._keyword_search(query, top_k, include_terms=include_terms, exclude_terms=exclude_terms)
        else:
            if search_mode == "chroma" and self.collection is None:
                warnings.append(f"Chroma unavailable: {self.chroma_error or 'collection is not loaded'}")
                results = []
            else:
                if self.chroma_error:
                    warnings.append(f"Chroma unavailable: {self.chroma_error}")
                results = self._keyword_search(query, top_k, include_terms=include_terms, exclude_terms=exclude_terms)
        if source == "chroma":
            results = _filter_terms(results, include_terms=include_terms, exclude_terms=exclude_terms)
        unfiltered_count = len(results)
        if min_score > 0:
            results = [item for item in results if item.get("score") is None or float(item.get("score") or 0.0) >= min_score]
            if len(results) < unfiltered_count:
                warnings.append(
                    f"Filtered out {unfiltered_count - len(results)} chunks below min_score {min_score}."
                )
        return {
            "results": results,
            "source": source,
            "warnings": warnings,
            "query": query,
            "settings": {
                "top_k": top_k,
                "min_score": min_score,
                "search_mode": search_mode,
                "include_terms": include_terms,
                "exclude_terms": exclude_terms,
            },
            "unfiltered_count": unfiltered_count,
        }

    def _reindex_chroma(self, chroma_path: Path, chunks: list[DocumentChunk]) -> dict[str, Any]:
        self.collection = None
        self.chroma_error = None
        try:
            import chromadb
        except Exception as exc:
            self.chroma_error = str(exc)
            return {"enabled": False, "error": str(exc)}
        try:
            chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(chroma_path))
            try:
                client.delete_collection(self.config.collection_name)
            except Exception:
                pass
            collection = client.get_or_create_collection(self.config.collection_name)
            if chunks:
                collection.add(
                    ids=[chunk.id for chunk in chunks],
                    documents=[chunk.text for chunk in chunks],
                    metadatas=[
                        {
                            "document_id": chunk.document_id,
                            "source_path": chunk.source_path,
                            "title": chunk.title,
                            "chunk_index": chunk.chunk_index,
                            **chunk.metadata,
                        }
                        for chunk in chunks
                    ],
                )
            self.collection = collection
            return {"enabled": True, "path": str(chroma_path), "chunks": len(chunks)}
        except Exception as exc:
            self.chroma_error = str(exc)
            return {"enabled": False, "error": str(exc)}

    def _convert_chroma_results(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        docs = raw.get("documents", [[]])[0] or []
        metadatas = raw.get("metadatas", [[]])[0] or []
        ids = raw.get("ids", [[]])[0] or []
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else [None] * len(docs)
        results = []
        for index, text in enumerate(docs):
            metadata = metadatas[index] or {}
            results.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "title": metadata.get("title") or "Untitled",
                    "source_path": metadata.get("source_path") or "",
                    "chunk_index": metadata.get("chunk_index"),
                    "text": text,
                    "score": None if distances[index] is None else 1.0 / (1.0 + float(distances[index])),
                    "metadata": metadata,
                }
            )
        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        include_terms: list[str] | None = None,
        exclude_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        include_terms = include_terms or []
        exclude_terms = exclude_terms or []
        query_counts = Counter(_tokens(" ".join([query, *include_terms])))
        include_tokens = set(_tokens(" ".join(include_terms)))
        exclude_tokens = set(_tokens(" ".join(exclude_terms)))
        scored = []
        for chunk in self.keyword_chunks:
            chunk_counts = Counter(_tokens(chunk.text))
            if not chunk_counts:
                continue
            chunk_tokens = set(chunk_counts)
            if include_tokens and not include_tokens.issubset(chunk_tokens):
                continue
            if exclude_tokens and exclude_tokens.intersection(chunk_tokens):
                continue
            overlap = sum(min(count, chunk_counts[token]) for token, count in query_counts.items())
            if overlap == 0:
                score = 0.0
            else:
                norm = math.sqrt(sum(v * v for v in chunk_counts.values())) or 1.0
                score = overlap / norm
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            results.append(
                {
                    "id": chunk.id,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "score": score,
                    "metadata": chunk.metadata,
                }
            )
        return results

    def keyword_search_docs(
        self,
        query: str = "",
        keywords: list[str] | None = None,
        fuzziness: float = 0.0,
        top_n: int = 10,
        match_mode: str = "any",
    ) -> dict[str, Any]:
        keywords = [item.strip() for item in keywords or [] if item.strip()]
        if not keywords and query.strip():
            keywords = [query.strip()]
        fuzziness = max(0.0, min(float(fuzziness), 1.0))
        threshold = max(0.0, min(1.0, 1.0 - fuzziness))
        top_n = max(1, min(top_n, 50))
        match_mode = match_mode if match_mode in {"any", "all"} else "any"
        results = []
        for chunk in self.keyword_chunks:
            matches = [_best_keyword_match(chunk.text, keyword) for keyword in keywords]
            accepted_matches = [match for match in matches if match["score"] >= threshold]
            if keywords and match_mode == "all" and len(accepted_matches) != len(keywords):
                continue
            if keywords and match_mode == "any" and not accepted_matches:
                continue
            if not keywords:
                continue
            score = sum(match["score"] for match in accepted_matches) / max(1, len(keywords))
            results.append(
                {
                    "id": chunk.id,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "score": score,
                    "text": chunk.text[:1600],
                    "matches": accepted_matches,
                    "metadata": chunk.metadata,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return {
            "query": query,
            "keywords": keywords,
            "fuzziness": fuzziness,
            "threshold": threshold,
            "match_mode": match_mode,
            "results": results[:top_n],
            "unfiltered_count": len(results),
        }

    def lookup_doc_source(self, source_reference: str, query: str = "", max_chunks: int = 5) -> dict[str, Any]:
        reference = " ".join(str(source_reference or "").split())
        max_chunks = max(1, min(int(max_chunks or 5), 20))
        sources: dict[tuple[str, str], dict[str, Any]] = {}
        for chunk in self.keyword_chunks:
            key = (chunk.source_path, chunk.title)
            sources.setdefault(
                key,
                {"source_path": chunk.source_path, "title": chunk.title, "chunks": []},
            )["chunks"].append(chunk)
        ranked_sources = []
        for source in sources.values():
            score = _source_match_score(reference, source["title"], source["source_path"])
            if score > 0:
                ranked_sources.append((score, source))
        ranked_sources.sort(key=lambda item: item[0], reverse=True)
        selected_sources = _select_source_matches(ranked_sources)
        chunks = []
        for score, source in selected_sources:
            source_chunks = list(source["chunks"])
            if query.strip():
                query_counts = Counter(_tokens(query))

                def chunk_score(chunk: DocumentChunk) -> float:
                    chunk_counts = Counter(_tokens(chunk.text))
                    return sum(min(count, chunk_counts[token]) for token, count in query_counts.items())

                source_chunks.sort(key=chunk_score, reverse=True)
            for chunk in source_chunks:
                chunks.append(
                    {
                        "id": chunk.id,
                        "title": chunk.title,
                        "source_path": chunk.source_path,
                        "chunk_index": chunk.chunk_index,
                        "score": score,
                        "text": chunk.text[:1800],
                        "metadata": chunk.metadata,
                    }
                )
                if len(chunks) >= max_chunks:
                    break
            if len(chunks) >= max_chunks:
                break
        return {
            "source_reference": reference,
            "query": query,
            "chunks": chunks,
            "matches": [
                {
                    "title": source["title"],
                    "source_path": source["source_path"],
                    "score": score,
                    "chunk_count": len(source["chunks"]),
                }
                for score, source in selected_sources[:5]
            ],
        }


def register_rag_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="retrieve_docs",
            description=(
                "Retrieve relevant documentation chunks and optional subagent summaries for a question. "
                "The model may retry with broader top_k, min_score=0, search_mode=keyword, or different query terms "
                "when the first retrieval is too narrow. Use coverage controls when later matches may add new facts; "
                "keep candidate_k and max_chunks modest, normally 6-8 or less, unless the user asks for exhaustive coverage. "
                "When a retrieved chunk names a source to follow, call lookup_doc_source with that exact name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or search query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum chunks to retrieve.",
                        "minimum": 1,
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Optional score floor from 0.0 to 1.0. Use 0 or omit for broad recall.",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "search_mode": {
                        "type": "string",
                        "description": "Search backend preference. Use auto normally, keyword for lexical retry, chroma for vector-only search.",
                        "enum": ["auto", "keyword", "chroma"],
                    },
                    "include_terms": {
                        "type": "array",
                        "description": "Optional terms that must appear in matched chunks. Use for narrowing after broad searches.",
                        "items": {"type": "string"},
                    },
                    "exclude_terms": {
                        "type": "array",
                        "description": "Optional terms that should not appear in matched chunks.",
                        "items": {"type": "string"},
                    },
                    "summarizer_prompt": {
                        "type": "string",
                        "description": (
                            "Optional extra guidance for document summarizer subagents. Use this often to tell "
                            "summarizers what details, entities, metrics, dates, caveats, or evidence to preserve "
                            "for the final answer."
                        ),
                    },
                    "coverage_mode": {
                        "type": "string",
                        "description": (
                            "Use on when the question needs broad coverage, comparisons, caveats, exceptions, "
                            "or distinct facts from more than the first few matches. Use off for narrow lookups."
                        ),
                        "enum": ["auto", "off", "on"],
                    },
                    "coverage_goal": {
                        "type": "string",
                        "description": (
                            "Specific guidance for coverage-aware subagents about what distinct facts, caveats, "
                            "conflicts, exceptions, or comparison points to collect."
                        ),
                    },
                    "candidate_k": {
                        "type": "integer",
                        "description": "Candidate chunks to retrieve before wave-based coverage summarization. Prefer 6-8 for normal coverage.",
                        "minimum": 1,
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum candidate chunks to examine with coverage summarization. Prefer 4-6 for normal coverage.",
                        "minimum": 1,
                    },
                    "follow_references": {
                        "type": "boolean",
                        "description": (
                            "When true in coverage mode, follow obvious named-source references found inside "
                            "retrieved chunks, even if those sources did not rank in the initial search."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=retrieve_docs_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="keyword_search_docs",
            description=(
                "Keyword-search the documentation with optional fuzzy matching. "
                "Use this for literal terms, names, units, exact phrases, or when semantic retrieval needs verification."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional phrase to search when keywords are not provided.",
                    },
                    "keywords": {
                        "type": "array",
                        "description": "Keywords or exact phrases to search for.",
                        "items": {"type": "string"},
                    },
                    "fuzziness": {
                        "type": "number",
                        "description": "0.0 requires exact or near-exact matches; 1.0 is very loose.",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Maximum ranked matches to return.",
                        "minimum": 1,
                    },
                    "match_mode": {
                        "type": "string",
                        "description": "Use any to match at least one keyword, or all to require every keyword.",
                        "enum": ["any", "all"],
                    },
                },
                "additionalProperties": False,
            },
            handler=keyword_search_docs_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="lookup_doc_source",
            description=(
                "Open documentation chunks by a referenced source title, source path, appendix, guide, runbook, "
                "or document name. Use this when retrieved text says to check another named source."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_reference": {
                        "type": "string",
                        "description": "Source title, source path, appendix name, guide name, or document reference.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional focus query for selecting chunks from the matched source.",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum chunks to return from matched source documents.",
                        "minimum": 1,
                    },
                },
                "required": ["source_reference"],
                "additionalProperties": False,
            },
            handler=lookup_doc_source_tool,
        )
    )


async def retrieve_docs_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query", ""))
    top_k = _coerce_int(args.get("top_k"), ctx.config.rag.top_k)
    top_k = max(1, min(top_k, ctx.config.rag.max_tool_top_k))
    min_score = _coerce_min_score(args.get("min_score"), ctx.config.rag.min_score)
    search_mode = str(args.get("search_mode") or "auto")
    include_terms = _coerce_terms(args.get("include_terms"))
    exclude_terms = _coerce_terms(args.get("exclude_terms"))
    summarizer_prompt = _coerce_summarizer_prompt(args.get("summarizer_prompt"))
    coverage_mode = _coerce_coverage_mode(args.get("coverage_mode"), ctx.config.rag.coverage_mode)
    coverage_goal = _coerce_summarizer_prompt(args.get("coverage_goal"))
    coverage_enabled = _coverage_enabled(coverage_mode, coverage_goal, args, query=query)
    follow_references = _coerce_bool(args.get("follow_references"), ctx.config.rag.coverage_follow_references)
    candidate_k = top_k
    if coverage_enabled:
        candidate_k = _coerce_int(args.get("candidate_k"), ctx.config.rag.coverage_candidate_k)
        candidate_k = max(1, min(candidate_k, ctx.config.rag.max_tool_top_k))
    retrieval = ctx.runtime.rag.retrieve(
        query=query,
        top_k=candidate_k,
        min_score=min_score,
        search_mode=search_mode,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
    )
    results = retrieval["results"]
    summaries = []
    coverage = {"enabled": False, "mode": coverage_mode}
    chunks_for_output = results
    coverage_candidate_filter: dict[str, Any] = {}
    if coverage_enabled:
        results, coverage_candidate_filter = _filter_coverage_candidates(
            results,
            min_score=ctx.config.rag.coverage_min_candidate_score,
            skip_zero_score=ctx.config.rag.coverage_skip_zero_score_candidates,
        )
        coverage, summaries, chunks_for_output = await _run_coverage_retrieval(
            ctx=ctx,
            query=query,
            results=results,
            summarizer_prompt=summarizer_prompt,
            coverage_goal=coverage_goal,
            requested_max_chunks=args.get("max_chunks"),
            candidate_k=candidate_k,
            follow_references=follow_references,
            candidate_filter=coverage_candidate_filter,
        )
    elif ctx.config.rag.use_subagent_summaries and results:
        summarizer = getattr(ctx.runtime, "summarize_chunks", None)
        if callable(summarizer):
            summary_chunks = _select_summary_chunks(
                results,
                max_count=ctx.config.rag.summarize_top_k,
                min_relative_score=ctx.config.rag.summarize_min_relative_score,
            )
            summaries = await summarizer(
                query,
                summary_chunks,
                ctx,
                summarizer_prompt=summarizer_prompt,
            )
    compact_chunks = []
    for item in chunks_for_output:
        compact_chunks.append(
            {
                "id": item["id"],
                "title": item["title"],
                "source_path": item["source_path"],
                "chunk_index": item["chunk_index"],
                "score": item["score"],
                "text": item["text"][:1600],
            }
        )
    return ToolResult(
        ok=True,
        data={
            "query": query,
            "source": retrieval["source"],
            "settings": retrieval["settings"],
            "summarizer_prompt": summarizer_prompt,
            "unfiltered_count": retrieval["unfiltered_count"],
            "chunks": compact_chunks,
            "summaries": summaries,
            "coverage": coverage,
        },
        summary=_retrieve_summary(retrieval, results, coverage),
        warnings=retrieval["warnings"],
        debug_messages=[
            f"Retrieved {len(results)} chunks for query: {query}",
            f"Retrieval settings: {retrieval['settings']}",
            f"Summarizer prompt: {summarizer_prompt or '(none)'}",
            f"Coverage mode: {coverage_mode}; enabled={coverage.get('enabled')}",
            f"Coverage candidate filter: {coverage_candidate_filter or '(not applied)'}",
            f"Coverage goal: {coverage_goal or '(none)'}",
            f"Subagent summaries generated: {len(summaries)}",
        ],
    )


def _select_summary_chunks(
    results: list[dict[str, Any]],
    max_count: int,
    min_relative_score: float,
) -> list[dict[str, Any]]:
    if not results or max_count <= 0:
        return []
    max_count = max(1, max_count)
    relative_floor = max(0.0, min(float(min_relative_score or 0.0), 1.0))
    top_score = _numeric_score(results[0])
    if top_score is None:
        return results[:max_count]
    if top_score <= 0:
        return results[:1]
    threshold = top_score * relative_floor
    selected = [item for item in results if _numeric_score(item) is None or _numeric_score(item) >= threshold]
    if not selected:
        selected = results[:1]
    return selected[:max_count]


def _filter_coverage_candidates(
    results: list[dict[str, Any]],
    min_score: float,
    skip_zero_score: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not results:
        return [], {
            "applied": False,
            "input_count": 0,
            "output_count": 0,
            "dropped_count": 0,
            "min_score": min_score,
            "skip_zero_score": skip_zero_score,
        }
    threshold = max(0.0, float(min_score or 0.0))
    kept = []
    dropped = []
    for item in results:
        score = _numeric_score(item)
        if score is None:
            kept.append(item)
            continue
        if skip_zero_score and score <= 0:
            dropped.append(item)
            continue
        if threshold > 0 and score < threshold:
            dropped.append(item)
            continue
        kept.append(item)
    retained_best_candidate = False
    if not kept:
        kept = results[:1]
        retained_best_candidate = True
    return kept, {
        "applied": bool(dropped),
        "input_count": len(results),
        "output_count": len(kept),
        "dropped_count": len(results) - len(kept),
        "min_score": threshold,
        "skip_zero_score": skip_zero_score,
        "retained_best_candidate": retained_best_candidate,
    }


def _numeric_score(item: dict[str, Any]) -> float | None:
    score = item.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


async def _run_coverage_retrieval(
    ctx: ToolContext,
    query: str,
    results: list[dict[str, Any]],
    summarizer_prompt: str,
    coverage_goal: str,
    requested_max_chunks: Any,
    candidate_k: int,
    follow_references: bool,
    candidate_filter: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = ctx.config.rag
    max_chunks = _coerce_int(requested_max_chunks, config.coverage_max_chunks)
    max_chunks = max(1, min(max_chunks, config.coverage_max_chunks, config.max_tool_top_k))
    if ctx.config.rag.use_subagent_summaries:
        max_chunks = max(1, min(max_chunks, config.coverage_subagent_max_chunks))
    wave_size = max(1, min(config.coverage_wave_size, max_chunks))
    stop_after_empty = max(1, config.coverage_stop_after_empty_waves)
    min_new_facts = max(1, config.coverage_min_new_facts)
    coverage: dict[str, Any] = {
        "enabled": True,
        "mode": "on",
        "goal": coverage_goal,
        "candidate_k": candidate_k,
        "candidate_count": len(results),
        "candidate_filter": candidate_filter or {},
        "max_chunks": max_chunks,
        "wave_size": wave_size,
        "chunks_examined": 0,
        "stop_reason": "",
        "waves": [],
        "reference_followups": [],
        "follow_references": follow_references,
        "reference_max_chunks": max(0, min(config.coverage_reference_max_chunks, config.max_tool_top_k)),
        "reference_max_depth": max(0, config.coverage_reference_max_depth),
        "distinct_facts": [],
        "duplicate_fact_count": 0,
    }
    if not results:
        coverage["stop_reason"] = "no_candidates"
        return coverage, [], []

    summarizer = getattr(ctx.runtime, "summarize_chunks", None)
    summaries: list[dict[str, Any]] = []
    examined: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    empty_waves = 0
    stop_reason = "all_candidates_examined"

    for wave_index, start in enumerate(range(0, max_chunks, wave_size), start=1):
        wave_chunks = results[start : min(start + wave_size, max_chunks)]
        if not wave_chunks:
            break
        examined.extend(wave_chunks)
        wave_prompt = _coverage_summarizer_prompt(
            base_prompt=summarizer_prompt,
            coverage_goal=coverage_goal,
            known_facts=facts,
        )
        if callable(summarizer) and ctx.config.rag.use_subagent_summaries:
            wave_summaries = await summarizer(query, wave_chunks, ctx, summarizer_prompt=wave_prompt)
        else:
            wave_summaries = [_fallback_coverage_summary(chunk, wave_prompt) for chunk in wave_chunks]
        summaries.extend(wave_summaries)

        wave_new_count = 0
        wave_duplicate_count = 0
        sources = []
        for item_index, chunk in enumerate(wave_chunks):
            summary = wave_summaries[item_index] if item_index < len(wave_summaries) else {}
            fact_texts = _extract_coverage_fact_texts(summary, chunk, query=query, coverage_goal=coverage_goal)
            new_fact_texts = []
            duplicate_fact_texts = []
            for fact_text in fact_texts:
                duplicate = _find_similar_fact(fact_text, facts)
                if duplicate is None:
                    fact = _coverage_fact(fact_text, chunk, wave_index)
                    facts.append(fact)
                    new_fact_texts.append(fact_text)
                    wave_new_count += 1
                else:
                    duplicate_fact_texts.append(fact_text)
                    duplicate.setdefault("duplicate_count", 0)
                    duplicate["duplicate_count"] += 1
                    wave_duplicate_count += 1
            sources.append(
                {
                    "title": chunk.get("title"),
                    "source_path": chunk.get("source_path"),
                    "chunk_index": chunk.get("chunk_index"),
                    "score": chunk.get("score"),
                    "adds_new_information": bool(new_fact_texts),
                    "new_fact_count": len(new_fact_texts),
                    "duplicate_fact_count": len(duplicate_fact_texts),
                    "new_facts": new_fact_texts[:4],
                }
            )

        coverage["waves"].append(
            {
                "index": wave_index,
                "chunk_count": len(wave_chunks),
                "summary_count": len(wave_summaries),
                "new_fact_count": wave_new_count,
                "duplicate_fact_count": wave_duplicate_count,
                "sources": sources,
            }
        )
        coverage["chunks_examined"] += len(wave_chunks)
        coverage["duplicate_fact_count"] += wave_duplicate_count
        if wave_new_count < min_new_facts:
            empty_waves += 1
        else:
            empty_waves = 0
        if empty_waves >= stop_after_empty:
            stop_reason = "no_new_information"
            break

    if follow_references and coverage["reference_max_chunks"] > 0 and coverage["reference_max_depth"] > 0:
        followed_summaries, followed_chunks = await _follow_coverage_references(
            ctx=ctx,
            query=query,
            coverage_goal=coverage_goal,
            summarizer_prompt=summarizer_prompt,
            coverage=coverage,
            summaries=summaries,
            examined=examined,
            facts=facts,
            remaining=coverage["reference_max_chunks"],
            max_depth=coverage["reference_max_depth"],
        )
        summaries.extend(followed_summaries)
        examined.extend(followed_chunks)

    if coverage["chunks_examined"] >= min(max_chunks, len(results)) and len(results) > coverage["chunks_examined"]:
        stop_reason = "max_chunks_examined"
    coverage["stop_reason"] = stop_reason
    coverage["distinct_facts"] = facts[:32]
    coverage["distinct_fact_count"] = len(facts)
    return coverage, summaries, examined


def _coverage_summarizer_prompt(
    base_prompt: str,
    coverage_goal: str,
    known_facts: list[dict[str, Any]],
) -> str:
    known = "\n".join(f"- {fact['text']}" for fact in known_facts[-16:]) or "- None yet."
    coverage_prompt = (
        "Coverage mode is active. Do not only summarize the first relevant-looking detail. "
        "Compare this chunk against the facts already found and preserve any materially new facts, "
        "conflicts, caveats, exceptions, thresholds, names, dates, codes, units, or comparison points.\n\n"
        f"Coverage goal:\n{coverage_goal or 'Find distinct information relevant to the user question.'}\n\n"
        f"Facts already found:\n{known}\n\n"
        "Return concise bullets under these labels when possible: New facts, Already covered, Conflicts or caveats. "
        "Only use facts present in the chunk."
    )
    return "\n\n".join(part for part in [base_prompt, coverage_prompt] if part).strip()[:2200]


async def _follow_coverage_references(
    ctx: ToolContext,
    query: str,
    coverage_goal: str,
    summarizer_prompt: str,
    coverage: dict[str, Any],
    summaries: list[dict[str, Any]],
    examined: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    remaining: int,
    max_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if remaining <= 0 or max_depth <= 0:
        return [], []
    working_chunks = list(examined)
    working_summaries = list(summaries)
    existing_sources = {_chunk_identity(chunk) for chunk in working_chunks}
    seen_references: set[str] = set()
    summarizer = getattr(ctx.runtime, "summarize_chunks", None)
    followed_summaries: list[dict[str, Any]] = []
    followed_chunks: list[dict[str, Any]] = []
    for depth in range(1, max_depth + 1):
        if remaining <= 0:
            break
        references = _detect_source_references("\n".join(_reference_texts(working_chunks, working_summaries)))
        references = [reference for reference in references if _normalize_reference(reference) not in seen_references]
        if not references:
            break
        progressed = False
        for reference in references:
            if remaining <= 0:
                break
            seen_references.add(_normalize_reference(reference))
            lookup = ctx.runtime.rag.lookup_doc_source(
                source_reference=reference,
                query=coverage_goal or query,
                max_chunks=remaining,
            )
            chunks = [chunk for chunk in lookup["chunks"] if _chunk_identity(chunk) not in existing_sources]
            if not chunks:
                coverage["reference_followups"].append(
                    {
                        "reference": reference,
                        "depth": depth,
                        "matched": False,
                        "match_count": len(lookup["matches"]),
                        "chunks_examined": 0,
                        "new_fact_count": 0,
                    }
                )
                continue
            chunks = chunks[:remaining]
            wave_prompt = _coverage_summarizer_prompt(
                base_prompt=summarizer_prompt,
                coverage_goal=(
                    f"{coverage_goal or 'Find distinct information relevant to the user question.'} "
                    f"Followed reference at depth {depth}: {reference}."
                ),
                known_facts=facts,
            )
            if callable(summarizer) and ctx.config.rag.use_subagent_summaries:
                new_summaries = await summarizer(query, chunks, ctx, summarizer_prompt=wave_prompt)
            else:
                new_summaries = [_fallback_coverage_summary(chunk, wave_prompt) for chunk in chunks]
            wave_index = len(coverage["waves"]) + 1
            new_fact_count = 0
            duplicate_fact_count = 0
            sources = []
            for item_index, chunk in enumerate(chunks):
                existing_sources.add(_chunk_identity(chunk))
                summary = new_summaries[item_index] if item_index < len(new_summaries) else {}
                fact_texts = _extract_coverage_fact_texts(summary, chunk, query=query, coverage_goal=coverage_goal)
                new_fact_texts = []
                duplicate_fact_texts = []
                for fact_text in fact_texts:
                    duplicate = _find_similar_fact(fact_text, facts)
                    if duplicate is None:
                        fact = _coverage_fact(fact_text, chunk, wave_index)
                        facts.append(fact)
                        new_fact_texts.append(fact_text)
                        new_fact_count += 1
                    else:
                        duplicate_fact_texts.append(fact_text)
                        duplicate.setdefault("duplicate_count", 0)
                        duplicate["duplicate_count"] += 1
                        duplicate_fact_count += 1
                sources.append(
                    {
                        "title": chunk.get("title"),
                        "source_path": chunk.get("source_path"),
                        "chunk_index": chunk.get("chunk_index"),
                        "score": chunk.get("score"),
                        "adds_new_information": bool(new_fact_texts),
                        "new_fact_count": len(new_fact_texts),
                        "duplicate_fact_count": len(duplicate_fact_texts),
                        "new_facts": new_fact_texts[:4],
                    }
                )
            followed_summaries.extend(new_summaries)
            followed_chunks.extend(chunks)
            working_chunks.extend(chunks)
            working_summaries.extend(new_summaries)
            coverage["chunks_examined"] += len(chunks)
            coverage["duplicate_fact_count"] += duplicate_fact_count
            coverage["waves"].append(
                {
                    "index": wave_index,
                    "chunk_count": len(chunks),
                    "summary_count": len(new_summaries),
                    "new_fact_count": new_fact_count,
                    "duplicate_fact_count": duplicate_fact_count,
                    "reference": reference,
                    "depth": depth,
                    "sources": sources,
                }
            )
            coverage["reference_followups"].append(
                {
                    "reference": reference,
                    "depth": depth,
                    "matched": True,
                    "match_count": len(lookup["matches"]),
                    "chunks_examined": len(chunks),
                    "new_fact_count": new_fact_count,
                    "matches": lookup["matches"][:3],
                }
            )
            remaining -= len(chunks)
            progressed = True
        if not progressed:
            break
    return followed_summaries, followed_chunks


def _reference_texts(chunks: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[str]:
    texts = [str(chunk.get("text") or "") for chunk in chunks]
    texts.extend(str(summary.get("summary") or "") for summary in summaries)
    return texts


def _detect_source_references(text: str) -> list[str]:
    patterns = [
        r"\b(?:check|see|open|consult|look at|refer to|points? to|points? reviewers to|refers? reviewers to|lives in|read from)\s+(?:the\s+)?(?:\*\*)?([A-Z][A-Za-z0-9 _./-]{2,90}?(?:Addendum|Appendix|Guide|Runbook|Notes?|Ledger|Challenge|Source|Rule|Document))(?:\*\*)?",
        r"\b(sample_docs/[A-Za-z0-9_./-]+\.(?:md|markdown|txt|html|htm|pdf|docx))\b",
    ]
    references = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            reference = " ".join(match.group(1).strip(" .,:;*`\"'").split())
            if reference and not _is_generic_source_reference(reference) and reference not in references:
                references.append(reference)
    return references[:8]


def _is_generic_source_reference(reference: str) -> bool:
    normalized = " ".join(_tokens(reference))
    return normalized in {
        "this addendum",
        "that addendum",
        "the addendum",
        "this appendix",
        "that appendix",
        "the appendix",
        "this guide",
        "that guide",
        "the guide",
        "this ledger",
        "that ledger",
        "the ledger",
        "this note",
        "that note",
        "the note",
        "this document",
        "that document",
        "the document",
        "this source",
        "that source",
        "the source",
    }


def _chunk_identity(chunk: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (chunk.get("id"), chunk.get("source_path"), str(chunk.get("chunk_index")))


def _normalize_reference(reference: str) -> str:
    return " ".join(_tokens(reference))


def _fallback_coverage_summary(chunk: dict[str, Any], prompt: str) -> dict[str, Any]:
    del prompt
    return {
        "ok": True,
        "id": chunk.get("id"),
        "title": chunk.get("title"),
        "source_path": chunk.get("source_path"),
        "summary": chunk.get("text", "")[:1200],
        "summarizer_prompt": "",
        "coverage_fallback": True,
    }


def _extract_coverage_fact_texts(
    summary: dict[str, Any],
    chunk: dict[str, Any],
    query: str,
    coverage_goal: str,
) -> list[str]:
    source_text = str(summary.get("summary") or "").strip() or str(chunk.get("text") or "")
    if not source_text.strip():
        source_text = str(chunk.get("text") or "")
    query_tokens = {token for token in _tokens(" ".join([query, coverage_goal])) if len(token) >= 4}
    fact_words = {
        "threshold",
        "owner",
        "label",
        "referenced",
        "reviewer",
        "rule",
        "phrase",
        "token",
        "period",
        "window",
        "status",
        "group",
        "escalate",
        "calibration",
        "exception",
        "caveat",
        "conflict",
        "goal",
        "value",
        "code",
    }
    candidates = []
    for sentence in _split_fact_sentences(source_text):
        cleaned = _clean_fact_text(sentence)
        if len(cleaned) < 18 or len(cleaned) > 320:
            continue
        lowered = cleaned.lower()
        if any(
            marker in lowered
            for marker in (
                "does not mention",
                "does not contain",
                "do not mention",
                "not contain",
                "not provide",
                "not related",
                "not mentioned",
                "not listed",
                "not specified",
                "not present",
                "unrelated",
                "no numeric",
                "no mention",
                "no information",
                "only describes",
                "cannot extract",
                "unavailable in this source",
            )
        ):
            continue
        tokens = set(_tokens(cleaned))
        has_query_overlap = bool(tokens.intersection(query_tokens))
        has_fact_word = bool(tokens.intersection(fact_words))
        has_number = bool(re.search(r"\d", cleaned))
        has_code = bool(re.search(r"\b[A-Z]{2,}[-A-Z0-9]{2,}\b", cleaned))
        if has_query_overlap or has_fact_word or has_number or has_code:
            candidates.append(cleaned)
    if not candidates:
        candidates = [_clean_fact_text(source_text[:280])] if source_text.strip() else []
    return _dedupe_texts(candidates)[:5]


def _split_fact_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[*`>#]+", " ", text)
    pieces = re.split(r"(?:\n+|(?<=[.!?])\s+|;\s+)", normalized)
    return [piece.strip(" -:\t") for piece in pieces if piece.strip(" -:\t")]


def _clean_fact_text(text: str) -> str:
    return " ".join(text.replace("\u202f", " ").split())


def _coverage_fact(text: str, chunk: dict[str, Any], wave_index: int) -> dict[str, Any]:
    return {
        "text": text,
        "first_seen_wave": wave_index,
        "sources": [
            {
                "title": chunk.get("title"),
                "source_path": chunk.get("source_path"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            }
        ],
    }


def _find_similar_fact(text: str, facts: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = _normalize_fact(text)
    for fact in facts:
        existing = _normalize_fact(fact.get("text") or "")
        if not existing:
            continue
        if normalized == existing or SequenceMatcher(None, normalized, existing).ratio() >= 0.86:
            return fact
    return None


def _normalize_fact(text: str) -> str:
    return " ".join(_tokens(text))


def _dedupe_texts(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if _find_similar_text(value, result) is None:
            result.append(value)
    return result


def _find_similar_text(value: str, existing: list[str]) -> str | None:
    normalized = _normalize_fact(value)
    for item in existing:
        candidate = _normalize_fact(item)
        if normalized == candidate or SequenceMatcher(None, normalized, candidate).ratio() >= 0.9:
            return item
    return None


def _retrieve_summary(retrieval: dict[str, Any], results: list[dict[str, Any]], coverage: dict[str, Any]) -> str:
    if coverage.get("enabled"):
        return (
            f"Retrieved {len(results)} documentation chunks via {retrieval['source']}. "
            f"Coverage examined {coverage.get('chunks_examined', 0)} of {coverage.get('candidate_count', 0)} "
            f"candidates and found {coverage.get('distinct_fact_count', 0)} distinct facts."
        )
    return f"Retrieved {len(results)} documentation chunks via {retrieval['source']}."


async def keyword_search_docs_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "")
    keywords = _coerce_terms(args.get("keywords"))
    fuzziness = _coerce_min_score(args.get("fuzziness"), 0.0)
    top_n = max(1, min(_coerce_int(args.get("top_n"), 10), 50))
    match_mode = str(args.get("match_mode") or "any")
    search = ctx.runtime.rag.keyword_search_docs(
        query=query,
        keywords=keywords,
        fuzziness=fuzziness,
        top_n=top_n,
        match_mode=match_mode,
    )
    chunks = [
        {
            "id": item["id"],
            "title": item["title"],
            "source_path": item["source_path"],
            "chunk_index": item["chunk_index"],
            "score": item["score"],
            "text": item["text"],
            "matches": item["matches"],
        }
        for item in search["results"]
    ]
    return ToolResult(
        ok=True,
        data={**search, "chunks": chunks, "results": chunks},
        summary=f"Found {len(chunks)} keyword documentation matches.",
        debug_messages=[
            f"Keyword search terms: {search['keywords']}",
            f"Fuzziness: {search['fuzziness']}; threshold: {search['threshold']}; match_mode: {search['match_mode']}",
        ],
    )


async def lookup_doc_source_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    source_reference = str(args.get("source_reference") or "")
    query = str(args.get("query") or "")
    max_chunks = max(1, min(_coerce_int(args.get("max_chunks"), 5), 20))
    lookup = ctx.runtime.rag.lookup_doc_source(
        source_reference=source_reference,
        query=query,
        max_chunks=max_chunks,
    )
    chunks = lookup["chunks"]
    return ToolResult(
        ok=True,
        data=lookup,
        summary=f"Opened {len(chunks)} documentation chunks for source reference: {source_reference}.",
        debug_messages=[
            f"Source reference lookup: {source_reference}",
            f"Matched sources: {lookup['matches'][:3]}",
        ],
        warnings=[] if chunks else [f"No documentation source matched: {source_reference}"],
    )


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value)]


def _source_match_score(reference: str, title: str, source_path: str) -> float:
    clean_reference = " ".join(reference.lower().split())
    if not clean_reference:
        return 0.0
    candidates = {
        " ".join(str(title or "").lower().split()),
        " ".join(str(source_path or "").lower().split()),
        Path(str(source_path or "")).stem.replace("_", " ").replace("-", " ").lower(),
    }
    best = 0.0
    reference_tokens = set(_tokens(clean_reference))
    for candidate in candidates:
        if not candidate:
            continue
        candidate_tokens = set(_tokens(candidate))
        if clean_reference == candidate:
            best = max(best, 1.0)
        elif clean_reference in candidate or candidate in clean_reference:
            best = max(best, 0.95)
        elif reference_tokens and reference_tokens.issubset(candidate_tokens):
            best = max(best, 0.9)
        else:
            best = max(best, SequenceMatcher(None, clean_reference, candidate).ratio())
    return best if best >= 0.55 else 0.0


def _select_source_matches(
    ranked_sources: list[tuple[float, dict[str, Any]]],
    max_sources: int = 3,
) -> list[tuple[float, dict[str, Any]]]:
    if not ranked_sources:
        return []
    best_score = ranked_sources[0][0]
    if best_score >= 0.9:
        floor = max(0.9, best_score - 0.03)
        return [item for item in ranked_sources if item[0] >= floor][:max_sources]
    return ranked_sources[:max_sources]


def _best_keyword_match(text: str, keyword: str) -> dict[str, Any]:
    clean_keyword = " ".join(keyword.lower().split())
    clean_text = " ".join(text.lower().split())
    if not clean_keyword:
        return {"keyword": keyword, "score": 0.0, "matched_text": "", "snippet": ""}
    exact_index = clean_text.find(clean_keyword)
    if exact_index >= 0:
        return {
            "keyword": keyword,
            "score": 1.0,
            "matched_text": keyword,
            "snippet": _snippet_around(text, keyword),
            "match_type": "exact",
        }
    keyword_tokens = _tokens(clean_keyword)
    text_tokens = _tokens(text)
    if not keyword_tokens or not text_tokens:
        return {"keyword": keyword, "score": 0.0, "matched_text": "", "snippet": ""}
    window_size = max(1, len(keyword_tokens))
    best_score = 0.0
    best_text = ""
    for size in range(max(1, window_size - 1), min(len(text_tokens), window_size + 2) + 1):
        for start in range(0, max(1, len(text_tokens) - size + 1)):
            candidate = " ".join(text_tokens[start : start + size])
            score = SequenceMatcher(None, clean_keyword, candidate).ratio()
            if score > best_score:
                best_score = score
                best_text = candidate
    return {
        "keyword": keyword,
        "score": best_score,
        "matched_text": best_text,
        "snippet": _snippet_around(text, best_text),
        "match_type": "fuzzy",
    }


def _snippet_around(text: str, needle: str, width: int = 280) -> str:
    if not needle:
        return " ".join(text.split())[:width]
    lowered = text.lower()
    index = lowered.find(str(needle).lower())
    if index < 0:
        return " ".join(text.split())[:width]
    start = max(0, index - width // 2)
    end = min(len(text), index + len(str(needle)) + width // 2)
    return " ".join(text[start:end].split())


def _coerce_min_score(value: Any, default: float) -> float:
    try:
        score = float(value if value is not None else default)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(score, 1.0))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _coerce_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:20]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_summarizer_prompt(value: Any) -> str:
    return " ".join(str(value or "").split())[:1200]


def _coerce_coverage_mode(value: Any, default: str) -> str:
    mode = str(value or default or "auto").lower()
    return mode if mode in {"auto", "off", "on"} else "auto"


def _coverage_enabled(mode: str, coverage_goal: str, args: dict[str, Any], query: str = "") -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    query_tokens = set(_tokens(query))
    return bool(
        coverage_goal
        or args.get("candidate_k") is not None
        or args.get("max_chunks") is not None
        or args.get("follow_references") is True
        or {"follow", "reference", "references", "chain", "full", "complete", "all", "every"}.intersection(
            query_tokens
        )
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _filter_terms(
    results: list[dict[str, Any]],
    include_terms: list[str],
    exclude_terms: list[str],
) -> list[dict[str, Any]]:
    include_tokens = set(_tokens(" ".join(include_terms)))
    exclude_tokens = set(_tokens(" ".join(exclude_terms)))
    if not include_tokens and not exclude_tokens:
        return results
    filtered = []
    for item in results:
        tokens = set(_tokens(item.get("text") or ""))
        if include_tokens and not include_tokens.issubset(tokens):
            continue
        if exclude_tokens and exclude_tokens.intersection(tokens):
            continue
        filtered.append(item)
    return filtered


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
