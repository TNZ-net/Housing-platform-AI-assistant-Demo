import pandas as pd
import pytest
from hp import assistant
from hp.tools import TOOLS

TOOL_CALL = '{"tool": "search_docs", "args": {"query": "rows"}}'


def scripted(monkeypatch, *replies):
    queue, sent = list(replies), []

    def fake_chat(messages):
        sent.append(list(messages))
        return queue.pop(0)

    monkeypatch.setattr(assistant, "chat", fake_chat)
    return sent


def test_parse_copes_with_chatter_and_code_fences():
    messy = 'Sure! ```json\n{"answer": "hi"}\n``` hope that helps'
    assert assistant.parse(messy) == {"answer": "hi"}
    assert assistant.parse("no json here") is None
    assert assistant.parse('{"broken": ') is None


def test_a_tool_call_then_an_answer(monkeypatch):
    sent = scripted(monkeypatch, TOOL_CALL, '{"answer": "20,640 rows"}')
    monkeypatch.setattr(assistant, "run_tool", lambda name, args: "THE RESULT")
    answer, trace = assistant.ask("How many rows?")
    assert answer == "20,640 rows"
    assert [t["tool"] for t in trace] == ["search_docs"]
    assert sent[1][-1] == {"role": "user", "content": "Tool result:\nTHE RESULT"}  # model saw it


def test_a_plain_text_reply_is_used_as_the_answer(monkeypatch):
    scripted(monkeypatch, "  Hello there!  ")
    assert assistant.ask("hi") == ("Hello there!", [])


def test_it_gives_up_after_max_steps(monkeypatch):
    sent = scripted(monkeypatch, *[TOOL_CALL] * assistant.MAX_STEPS)
    monkeypatch.setattr(assistant, "run_tool", lambda name, args: "again")
    answer, trace = assistant.ask("loop forever")
    assert "could not finish" in answer and len(sent) == assistant.MAX_STEPS


def test_the_prompt_lists_every_tool():
    assert all(name in assistant.system_prompt() for name in TOOLS)


def test_chat_speaks_the_openai_format(fake_server, monkeypatch):
    url, seen, reply = fake_server
    monkeypatch.setenv("LLM_BASE_URL", url + "/")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    reply["body"] = {"choices": [{"message": {"content": "hi there"}}]}
    assert assistant.chat([{"role": "user", "content": "hi"}]) == "hi there"
    assert seen[0]["path"] == "/chat/completions"
    assert seen[0]["body"]["model"] == "some-model" and seen[0]["auth"] == "Bearer secret"


def test_chat_without_settings_explains_what_to_set(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        assistant.chat([])


def test_evaluate_scores_tool_choice_and_answers(tmp_path, monkeypatch, capsys):
    csv = tmp_path / "eval.csv"
    pd.DataFrame({"question": ["q1", "q2"], "expected_tool": ["search_docs", "get_stats"],
                  "expected_contains": ["1990", ""]}).to_csv(csv, index=False)
    replies = {"q1": ("It is from 1990.", [{"tool": "search_docs", "args": {}, "result": "ok"}]),
               "q2": ("Nothing.", [{"tool": "search_docs", "args": {}, "result": "ok"}])}
    monkeypatch.setattr(assistant, "ask", lambda q: replies[q])
    monkeypatch.setattr(assistant, "ROOT", tmp_path)
    assistant.evaluate(csv)
    out = capsys.readouterr().out
    assert "tool choice correct : 50%" in out and "answer has expected text: 100%" in out
    assert (tmp_path / "reports" / "assistant_eval.csv").exists()