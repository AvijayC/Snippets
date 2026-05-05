from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_demo.config import load_settings, project_root
from rag_demo.rag import RagIndex, _detect_source_references, keyword_search_docs_tool, register_rag_tools, retrieve_docs_tool
from rag_demo.sample_data import ensure_sample_docs
from rag_demo.state import AppState
from rag_demo.tooling import ToolContext, ToolRegistry


def make_rag(tmp_path: Path) -> RagIndex:
    docs = tmp_path / "docs"
    ensure_sample_docs(docs)
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)
    return rag


def test_retrieve_accepts_breadth_and_term_filters(tmp_path: Path) -> None:
    rag = make_rag(tmp_path)

    result = rag.retrieve(
        query="hydration",
        top_k=4,
        min_score=0.0,
        search_mode="keyword",
        include_terms=["ounces"],
    )

    assert result["settings"]["top_k"] == 4
    assert result["settings"]["search_mode"] == "keyword"
    assert result["settings"]["include_terms"] == ["ounces"]
    assert result["results"]
    assert all("ounces" in item["text"].lower() for item in result["results"])


def test_retrieve_min_score_can_filter_results(tmp_path: Path) -> None:
    rag = make_rag(tmp_path)

    result = rag.retrieve(query="hydration", top_k=4, min_score=1.0, search_mode="keyword")

    assert result["results"] == []
    assert result["unfiltered_count"] > 0
    assert any("min_score" in warning for warning in result["warnings"])


def test_retrieve_docs_schema_exposes_retry_controls() -> None:
    registry = ToolRegistry()

    register_rag_tools(registry)

    schema = registry.get("retrieve_docs").openai_schema()["function"]["parameters"]["properties"]
    assert {
        "query",
        "top_k",
        "min_score",
        "search_mode",
        "include_terms",
        "exclude_terms",
        "summarizer_prompt",
        "coverage_mode",
        "coverage_goal",
        "candidate_k",
        "max_chunks",
        "follow_references",
    }.issubset(schema)


def test_keyword_search_docs_ranks_literal_matches(tmp_path: Path) -> None:
    rag = make_rag(tmp_path)

    result = rag.keyword_search_docs(keywords=["72 ounces"], fuzziness=0.0, top_n=3)

    assert result["results"]
    assert result["results"][0]["score"] == 1.0
    assert result["results"][0]["matches"][0]["match_type"] == "exact"


def test_keyword_search_docs_schema_is_registered() -> None:
    registry = ToolRegistry()

    register_rag_tools(registry)

    schema = registry.get("keyword_search_docs").openai_schema()["function"]["parameters"]["properties"]
    assert {"query", "keywords", "fuzziness", "top_n", "match_mode"}.issubset(schema)


def test_lookup_doc_source_finds_source_by_title(tmp_path: Path) -> None:
    rag = make_rag(tmp_path)

    result = rag.lookup_doc_source("Hydration Guide", query="ounces", max_chunks=2)

    assert result["chunks"]
    assert result["chunks"][0]["title"] == "Hydration Guide"


def test_lookup_doc_source_exact_title_does_not_return_fuzzy_neighbors(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "amber.md").write_text("# Amber Quorum Addendum\nAmber phrase.", encoding="utf-8")
    (docs / "meridian.md").write_text("# Meridian Exception Addendum\nMeridian label.", encoding="utf-8")
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)

    result = rag.lookup_doc_source("Amber Quorum Addendum", max_chunks=3)

    assert [chunk["title"] for chunk in result["chunks"]] == ["Amber Quorum Addendum"]
    assert [match["title"] for match in result["matches"]] == ["Amber Quorum Addendum"]


def test_lookup_doc_source_schema_is_registered() -> None:
    registry = ToolRegistry()

    register_rag_tools(registry)

    schema = registry.get("lookup_doc_source").openai_schema()["function"]["parameters"]["properties"]
    assert {"source_reference", "query", "max_chunks"}.issubset(schema)


def test_detect_source_references_ignores_generic_this_addendum() -> None:
    references = _detect_source_references(
        "The Slate Relay Ledger points to the Amber Quorum Addendum. "
        "The clerk later points to this addendum when filing the packet."
    )

    assert references == ["Amber Quorum Addendum"]


async def test_keyword_search_docs_tool_returns_ranked_chunks(tmp_path: Path) -> None:
    rag = make_rag(tmp_path)
    runtime = rag.runtime
    runtime.rag = rag

    result = await keyword_search_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {"keywords": ["ounces"], "fuzziness": 0.0, "top_n": 2},
    )

    assert result.ok is True
    assert result.data["chunks"]
    assert result.data["chunks"][0]["matches"][0]["keyword"] == "ounces"


async def test_retrieve_docs_tool_passes_summarizer_prompt(tmp_path: Path) -> None:
    rag = make_rag(tmp_path)
    seen = {}

    async def summarize_chunks(query, chunks, context, summarizer_prompt=""):
        del query, context
        seen["summarizer_prompt"] = summarizer_prompt
        return [
            {
                "ok": True,
                "id": chunks[0]["id"],
                "title": chunks[0]["title"],
                "source_path": chunks[0]["source_path"],
                "summary": "Focused summary.",
                "summarizer_prompt": summarizer_prompt,
            }
        ]

    runtime = rag.runtime
    runtime.summarize_chunks = summarize_chunks
    runtime.rag = rag

    result = await retrieve_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {
            "query": "hydration",
            "top_k": 2,
            "search_mode": "keyword",
            "summarizer_prompt": "Keep targets and units.",
        },
    )

    assert result.ok is True
    assert seen["summarizer_prompt"] == "Keep targets and units."
    assert result.data["summarizer_prompt"] == "Keep targets and units."
    assert result.data["summaries"][0]["summarizer_prompt"] == "Keep targets and units."


async def test_retrieve_docs_tool_coverage_reports_distinct_facts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("# One\nAlpha threshold is 11 psi. See the Beta Addendum.", encoding="utf-8")
    (docs / "beta.md").write_text("# Beta Addendum\nThe referenced label is beta-latch.", encoding="utf-8")
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    settings.rag.use_subagent_summaries = False
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)
    runtime.rag = rag

    result = await retrieve_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {
            "query": "Alpha threshold",
            "top_k": 1,
            "coverage_mode": "on",
            "coverage_goal": "Find thresholds and referenced labels.",
            "candidate_k": 1,
            "max_chunks": 2,
            "follow_references": True,
        },
    )

    assert result.ok is True
    assert result.data["coverage"]["enabled"] is True
    facts = " ".join(fact["text"] for fact in result.data["coverage"]["distinct_facts"])
    assert "11 psi" in facts
    assert "beta-latch" in facts


async def test_retrieve_docs_tool_auto_coverage_for_reference_chain_queries(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text("# Alpha Runbook\nAlpha gate is 11 psi. The handoff points to the Beta Addendum.", encoding="utf-8")
    (docs / "beta.md").write_text("# Beta Addendum\nBeta label is beta-latch.", encoding="utf-8")
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    settings.rag.use_subagent_summaries = False
    settings.rag.coverage_mode = "auto"
    settings.rag.coverage_max_chunks = 1
    settings.rag.coverage_reference_max_chunks = 2
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)
    runtime.rag = rag

    result = await retrieve_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {
            "query": "follow the Alpha reference chain",
            "top_k": 1,
            "search_mode": "keyword",
        },
    )

    assert result.ok is True
    coverage = result.data["coverage"]
    assert coverage["enabled"] is True
    assert [item["reference"] for item in coverage["reference_followups"] if item["matched"]] == ["Beta Addendum"]
    facts = " ".join(fact["text"] for fact in coverage["distinct_facts"])
    assert "11 psi" in facts
    assert "beta-latch" in facts


async def test_retrieve_docs_tool_coverage_filters_zero_score_candidates(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "needle.md").write_text("# Needle\nNeedle threshold is 14 psi.", encoding="utf-8")
    for index in range(4):
        (docs / f"noise_{index}.md").write_text(
            f"# Noise {index}\nThis document is unrelated filler content.",
            encoding="utf-8",
        )
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    settings.rag.use_subagent_summaries = True
    settings.rag.coverage_subagent_max_chunks = 4
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)
    runtime.rag = rag
    seen_chunk_counts = []

    async def summarize_chunks(query, chunks, context, summarizer_prompt=""):
        del query, context, summarizer_prompt
        seen_chunk_counts.append(len(chunks))
        return [
            {
                "ok": True,
                "id": chunk["id"],
                "title": chunk["title"],
                "source_path": chunk["source_path"],
                "summary": chunk["text"],
            }
            for chunk in chunks
        ]

    runtime.summarize_chunks = summarize_chunks

    result = await retrieve_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {
            "query": "needle",
            "top_k": 5,
            "search_mode": "keyword",
            "coverage_mode": "on",
            "coverage_goal": "Find threshold values.",
            "candidate_k": 5,
            "max_chunks": 5,
            "follow_references": False,
        },
    )

    assert result.ok is True
    coverage = result.data["coverage"]
    assert coverage["candidate_filter"]["input_count"] == 5
    assert coverage["candidate_filter"]["output_count"] == 1
    assert coverage["candidate_filter"]["dropped_count"] == 4
    assert coverage["chunks_examined"] == 1
    assert seen_chunk_counts == [1]


async def test_retrieve_docs_tool_coverage_caps_subagent_chunks(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for index in range(6):
        (docs / f"fact_{index}.md").write_text(
            f"# Fact {index}\nCoverage item {index} has threshold {10 + index} psi.",
            encoding="utf-8",
        )
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    settings.rag.use_subagent_summaries = True
    settings.rag.coverage_max_chunks = 6
    settings.rag.coverage_subagent_max_chunks = 2
    settings.rag.coverage_wave_size = 3
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)
    runtime.rag = rag
    seen_chunks = []

    async def summarize_chunks(query, chunks, context, summarizer_prompt=""):
        del query, context, summarizer_prompt
        seen_chunks.extend(chunk["id"] for chunk in chunks)
        return [
            {
                "ok": True,
                "id": chunk["id"],
                "title": chunk["title"],
                "source_path": chunk["source_path"],
                "summary": chunk["text"],
            }
            for chunk in chunks
        ]

    runtime.summarize_chunks = summarize_chunks

    result = await retrieve_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {
            "query": "coverage threshold",
            "top_k": 6,
            "search_mode": "keyword",
            "coverage_mode": "on",
            "coverage_goal": "Find all coverage thresholds.",
            "candidate_k": 6,
            "max_chunks": 6,
            "follow_references": False,
        },
    )

    assert result.ok is True
    assert result.data["coverage"]["max_chunks"] == 2
    assert result.data["coverage"]["chunks_examined"] == 2
    assert len(seen_chunks) == 2


async def test_retrieve_docs_tool_recursively_follows_references_with_limits(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text("# Alpha Runbook\nAlpha gate is 11 psi. See the Beta Addendum.", encoding="utf-8")
    (docs / "beta.md").write_text("# Beta Addendum\nBeta label is beta-latch. Consult the Gamma Note.", encoding="utf-8")
    (docs / "gamma.md").write_text("# Gamma Note\nGamma owner is Casey Lin. Refer to the Delta Ledger.", encoding="utf-8")
    (docs / "delta.md").write_text("# Delta Ledger\nDelta code is DLT-9.", encoding="utf-8")
    settings = load_settings(project_root() / "config" / "default_config.json")
    settings.rag.docs_path = str(docs)
    settings.rag.use_subagent_summaries = False
    settings.rag.coverage_max_chunks = 1
    settings.rag.coverage_reference_max_chunks = 2
    settings.rag.coverage_reference_max_depth = 3
    state = AppState(tmp_path / "state.db")
    runtime = SimpleNamespace(config=settings, project_root=tmp_path, state=state)
    rag = RagIndex(runtime)
    rag.reindex(use_chroma=False)
    runtime.rag = rag

    result = await retrieve_docs_tool(
        ToolContext(runtime=runtime, chat_id="chat", run_id="run"),
        {
            "query": "Alpha gate",
            "top_k": 1,
            "search_mode": "keyword",
            "coverage_mode": "on",
            "coverage_goal": "Collect chained alpha, beta, gamma, and delta facts.",
            "candidate_k": 1,
            "max_chunks": 1,
            "follow_references": True,
        },
    )

    assert result.ok is True
    coverage = result.data["coverage"]
    followed = coverage["reference_followups"]
    assert [item["reference"] for item in followed if item["matched"]] == ["Beta Addendum", "Gamma Note"]
    assert [item["depth"] for item in followed if item["matched"]] == [1, 2]
    assert coverage["chunks_examined"] == 3
    facts = " ".join(fact["text"] for fact in coverage["distinct_facts"])
    assert "11 psi" in facts
    assert "beta-latch" in facts
    assert "Casey Lin" in facts
    assert "DLT-9" not in facts
