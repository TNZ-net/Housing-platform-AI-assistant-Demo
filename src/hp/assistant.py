import argparse
import json
import logging
import os
import pandas as pd
from hp import ROOT, setup_logging
from hp.tools import TOOLS, post_json, run_tool
log = logging.getLogger("assistant")
MAX_STEPS = 4

def system_prompt():
    tools = "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())
    return (
        "You are the assistant for a housing-price platform. Reply with ONE JSON object and "
        "nothing else.\n"
        'To use a tool: {"tool": "<name>", "args": {...}}\n'
        'To answer the user: {"answer": "<text>"}\n\n'
        f"Tools:\n{tools}\n\n"
        "Rules:\n"
        "- Questions about the dataset, columns, model, drift or API: use search_docs.\n"
        "- Questions about numbers in our database or API usage: use get_stats.\n"
        "- Use predict_price only when the user gives the house details. If details are missing, "
        "ask for them in an answer.\n"
        "- Only state facts that a tool returned. If a tool returns an error, say so.\n"
        "- Greetings and off-topic questions need no tool: just answer."
    )


def chat(messages):
    """Send the conversation to any OpenAI-compatible chat endpoint and return the reply text."""
    base = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not base or not model:
        raise RuntimeError("set LLM_BASE_URL and LLM_MODEL in .env (and LLM_API_KEY if needed)")
    headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY') or 'none'}"}
    status, body = post_json(f"{base.rstrip('/')}/chat/completions",
                             {"model": model, "messages": messages, "temperature": 0}, headers)
    if status != 200:
        raise RuntimeError(f"LLM endpoint returned {status}: {body}")
    return body["choices"][0]["message"]["content"]


def parse(text):
    """Pull the first JSON object out of a reply (models often add chatter or code fences)."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def ask(question):
    """Answer one question. Returns (answer text, list of tool calls made)."""
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": question}]
    trace = []
    for _ in range(MAX_STEPS):
        reply = chat(messages)
        action = parse(reply)
        if action is None or ("tool" not in action and "answer" not in action):
            return reply.strip(), trace  # not in our format: treat it as a plain-text answer
        if "answer" in action:
            return str(action["answer"]), trace
        tool, args = str(action["tool"]), action.get("args")
        result = run_tool(tool, args if isinstance(args, dict) else {})
        trace.append({"tool": tool, "args": args, "result": result})
        log.info("tool %s(%s) -> %s", tool, args, result[:80].replace("\n", " "))
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Tool result:\n{result}"}]
    return f"I could not finish within {MAX_STEPS} steps.", trace


def evaluate(path=ROOT / "data" / "assistant_eval.csv"):
    """Score tool choice, tool success and (where the CSV says what to expect) the answer text."""
    rows = []
    for q in pd.read_csv(path, keep_default_na=False).itertuples():
        answer, trace = ask(q.question)
        first = trace[0] if trace else None
        wanted = [w.lower() for w in q.expected_contains.split("|") if w]
        rows.append({
            "question": q.question, "expected_tool": q.expected_tool,
            "tool_used": first["tool"] if first else "none",
            "tool_ok": None if first is None else not first["result"].startswith("ERROR"),
            "answer_ok": None if not wanted else any(w in answer.lower() for w in wanted),
            "answer": answer})
    df = pd.DataFrame(rows)
    df["tool_correct"] = df["tool_used"] == df["expected_tool"]
    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports" / "assistant_eval.csv", index=False)
    print(df.groupby("expected_tool")["tool_correct"].agg(["sum", "count"]).to_string())
    print(f"\ntool choice correct : {df['tool_correct'].mean():.0%}")
    called = df[df["tool_ok"].notna()]
    print(f"tool ran without error: {called['tool_ok'].mean():.0%}  ({len(called)} calls)")
    checked = df[df["answer_ok"].notna()]
    print(f"answer has expected text: {checked['answer_ok'].mean():.0%} ({len(checked)} questions)")
    print("details saved to reports/assistant_eval.csv")


def launch_ui():
    import gradio as gr

    def respond(question):
        answer, trace = ask(question)
        return answer, "\n".join(f"{t['tool']}({t['args']})" for t in trace) or "(no tool used)"

    gr.Interface(respond, gr.Textbox(label="Question"),
                 [gr.Textbox(label="Answer"), gr.Textbox(label="Tools used")],
                 title="Housing Platform Assistant").launch()


def main():
    ap = argparse.ArgumentParser(description="Housing platform assistant")
    ap.add_argument("--ask", help="ask one question")
    ap.add_argument("--eval", action="store_true", help="score on data/assistant_eval.csv")
    ap.add_argument("--ui", action="store_true", help="launch the web page")
    args = ap.parse_args()
    setup_logging("assistant")
    if args.ask:
        answer, trace = ask(args.ask)
        for t in trace:
            print(f"[tool] {t['tool']}({t['args']})")
        print(answer)
    elif args.eval:
        evaluate()
    elif args.ui:
        launch_ui()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()