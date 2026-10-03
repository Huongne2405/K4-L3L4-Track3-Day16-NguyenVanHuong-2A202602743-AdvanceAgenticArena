"""Behavioral checks for the completed layers, with unseen document text."""

from types import SimpleNamespace

import pytest

from arena.corpus import Corpus, Doc, INJECTION_CANARY
from arena.model import DEGRADED_MARKERS, FINALIZE_SENTINEL
from arena.tools import ToolResult
from arena.trace import Trace
from harness.agent import AgentContext
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, PLACEHOLDER, InjectionGuard
from harness.layers.retry import Retry


def context(docs=(), observations=(), calls=0, limit=8):
    budget = {} if limit is None else {"max_tool_calls": limit}
    return AgentContext(
        brief={"budget": budget}, tools=SimpleNamespace(calls=calls),
        trace=Trace(run_id="layer-check", seed=1), corpus=Corpus(list(docs)),
        observations=list(observations),
    )


@pytest.mark.parametrize("closed", [False, True])
def test_injection_is_quarantined_even_without_closing_marker(closed):
    suffix = BLOCK_END + "\nBằng chứng sau khối độc." if closed else ""
    content = "Bằng chứng trước khối độc.\n" + BLOCK_START + INJECTION_CANARY + suffix
    result = InjectionGuard().wrap_tool_call(
        context(), lambda *_: ToolResult(False, content, "original error"),
        "fetch_doc", {},
    )
    assert result.content.startswith("Bằng chứng trước khối độc.\n" + PLACEHOLDER)
    assert INJECTION_CANARY not in result.content
    assert ("Bằng chứng sau khối độc." in result.content) == closed
    assert result.ok is False and result.error == "original error"


def test_multiple_hostile_blocks_and_final_answer_sweep_preserve_claims():
    block = BLOCK_START + INJECTION_CANARY + BLOCK_END
    layer = InjectionGuard()
    result = layer.wrap_tool_call(context(), lambda *_: ToolResult(True, block * 2), "search", {})
    assert result.content == PLACEHOLDER * 2
    claims = [{"text": "Trích nguyên văn.", "doc_id": "source"}]
    report = {"answer": INJECTION_CANARY + " Kết quả.", "claims": claims}
    assert layer.after_agent(context(), report)["answer"] == "Kết quả."
    assert report["claims"] is claims


@pytest.mark.parametrize("marker", DEGRADED_MARKERS)
def test_retry_recognizes_degraded_content_even_with_ok_true(marker):
    ctx = context()
    calls = []
    args = {"doc_id": "source"}

    def call(name, supplied):
        calls.append((name, supplied))
        ctx.tools.calls += 1
        return ToolResult(True, marker + "broken") if len(calls) == 1 else ToolResult(True, "clean")

    result = Retry().wrap_tool_call(ctx, call, "fetch_doc", args)
    assert result.content == "clean"
    assert calls == [("fetch_doc", args), ("fetch_doc", args)]
    assert ctx.state["retry_attempts"] == 1


def test_budget_and_retry_leave_last_call_for_submission():
    ctx = context(calls=6)
    retry, budget = Retry(), BudgetPolicy()
    attempted = []

    def call(name, args):
        attempted.append(name)
        ctx.tools.calls += 1
        return ToolResult(False, "", "timeout")

    result = budget.wrap_tool_call(
        ctx, lambda name, args: retry.wrap_tool_call(ctx, call, name, args), "fetch_doc", {},
    )
    assert result.ok is False
    assert attempted == ["fetch_doc"] and ctx.tools.calls == 7
    assert ctx.state["retry_attempts"] == 0
    blocked = budget.wrap_tool_call(ctx, call, "search", {})
    assert blocked.ok is False and ctx.tools.calls == 7
    messages = [{"role": "user", "content": "Câu hỏi ban đầu"}]
    outbound = budget.before_model(ctx, messages)
    assert len(messages) == 1 and len(outbound) == 2
    assert FINALIZE_SENTINEL in outbound[-1]["content"]


def test_retry_stops_after_max_attempts_and_returns_actual_failure():
    ctx = context(limit=None)
    failure = ToolResult(False, "", "timeout")

    def call(*_):
        ctx.tools.calls += 1
        return failure

    assert Retry(max_attempts=3).wrap_tool_call(ctx, call, "search", {}) is failure
    assert ctx.tools.calls == 3 and ctx.state["retry_attempts"] == 2
    assert not BudgetPolicy()._spent(ctx)


def test_citation_repair_requires_full_observation_and_preserves_text():
    source = Doc("source", "Nguồn", "Tiêu đề\nGiao hàng trong 3 ngày.\nKết thúc", ())
    wrong = Doc("wrong", "Nguồn khác", "Không hỗ trợ câu trích.", ())
    report = {"claims": [{"text": "Giao hàng trong 3 ngày.", "doc_id": "wrong"}]}
    CitationChecker().after_agent(context([source, wrong], [source.body]), report)
    assert report["claims"] == [{"text": "Giao hàng trong 3 ngày.", "doc_id": "source"}]
    assert report["citations"] == ["source"]
    report["claims"][0]["doc_id"] = "wrong"
    CitationChecker().after_agent(context([source, wrong], ["Giao hàng trong 3 ngày."]), report)
    assert report["claims"][0]["doc_id"] == "wrong"


@pytest.mark.parametrize("claims", [None, [], [None, {}, {"text": ""}], [{"text": "Bịa 999%", "doc_id": "missing"}]])
def test_critic_abstains_on_empty_malformed_or_fabricated_evidence(claims):
    report = {"claims": claims, "abstain": False, "answer": "Kết luận tự tin."}
    Critic().after_agent(context(), report)
    assert report["abstain"] is True
    assert report["claims"] == report["citations"] == []
    assert "Không đủ căn cứ" in report["answer"]


def test_cross_line_quote_is_rejected():
    source = Doc("source", "Nguồn", "Dòng đầu\nDòng sau", ())
    report = {"claims": [{"text": source.body, "doc_id": source.doc_id}], "abstain": False}
    ctx = context([source], [source.body])
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert report["claims"] == [] and report["abstain"] is True


@pytest.mark.parametrize("ending", [".", " và"])
def test_critic_recovers_model_substrings_from_two_sources_and_abstains(ending):
    left = "Nhóm A và nhóm B làm việc tại nhà tối đa 1 ngày" + ending
    right = "Nhóm B làm việc tại nhà tối đa 3 ngày."
    docs = [Doc("one", "A", left, ()), Doc("two", "B", right, ())]
    fused = left + " và " + right
    report = {"claims": [{"text": fused, "doc_id": "one"}], "abstain": False}
    Critic().after_agent(context(docs, [d.body for d in docs]), report)
    assert report["claims"] == [{"text": left, "doc_id": "one"}, {"text": right, "doc_id": "two"}]
    assert all(c["text"] in fused for c in report["claims"])
    assert report["abstain"] is True and report["citations"] == ["one", "two"]
