import os
import re
import json
import time
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
from openai import OpenAI


VALID_ANSWER_STATUS = {
    "correct",
    "incorrect",
    "incomplete",
    "unclear",
    "missing_field",
    "error",
}

VALID_REASONING_LABELS = {
    "over_reasoning",
    "under_reasoning",
    "adequate_reasoning",
    "missing",
}

VALID_TOKEN_IMPACTS = {
    "none",
    "hit_but_correct",
    "hit_and_incomplete",
    "hit_and_wrong",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)

    parser.add_argument("--judge_model", type=str, default="gpt-4.1")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://mmia1-8293-resource.openai.azure.com/openai/v1/",
    )
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))

    parser.add_argument("--judge_max_tokens", type=int, default=512)

    parser.add_argument(
        "--expected_final_max_tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens used by the final model.",
    )

    parser.add_argument("--sleep", type=float, default=0.0)

    return parser.parse_args()


def load_data(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")

    if ext == ".csv":
        import pandas as pd
        return pd.read_csv(path).fillna("").to_dict("records")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_json(path, data):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, path)


def get_by_alias(item, aliases, default=""):
    lower_map = {str(k).lower().strip(): k for k in item.keys()}

    for name in aliases:
        if name in item and item[name] not in [None, ""]:
            return item[name]

        key = lower_map.get(name.lower().strip())
        if key is not None and item[key] not in [None, ""]:
            return item[key]

    return default


def get_query(item):
    return str(get_by_alias(item, [
        "original_query", "problem", "prompt", "query", "user_query",
        "Question", "question", "input", "task"
    ]))


def get_reference(item):
    return str(get_by_alias(item, [
        "reference_answer", "answer", "solution", "Final answer",
        "final_answer", "gold_answer", "ground_truth", "target",
        "expected_answer"
    ]))


def get_response(item):
    return str(get_by_alias(item, [
        "final_response", "response", "model_response", "raw_response",
        "prediction", "generated_answer", "model_answer", "output",
        "final_answer_generated"
    ]))


def get_raw_response(item):
    raw = get_by_alias(item, ["raw_response"], default="")
    return str(raw) if raw else get_response(item)


def get_reasoning_text(item):
    token_report = item.get("token_report", {})
    if not isinstance(token_report, dict):
        token_report = {}

    return str(
        item.get("reasoning_text")
        or token_report.get("reasoning_text")
        or item.get("final_reasoning_text")
        or ""
    )


def safe_get(d, keys, default=None):
    cur = d

    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)

    return cur if cur is not None else default


def detect_final_token_limit_hit(item, expected_final_max_tokens=None):
    candidates = [
        item.get("hit_max_tokens"),
        safe_get(item, ["token_report", "hit_max_tokens"]),
        safe_get(item, ["final_token_report", "hit_max_tokens"]),
        safe_get(item, ["final_usage", "hit_max_tokens"]),
    ]

    if any(c is True for c in candidates):
        return True

    finish_reasons = [
        item.get("finish_reason"),
        item.get("final_finish_reason"),
        safe_get(item, ["final_usage", "finish_reason"]),
        safe_get(item, ["token_report", "finish_reason"]),
    ]

    for fr in finish_reasons:
        if isinstance(fr, str) and fr.lower() in {
            "length",
            "max_tokens",
            "token_limit",
        }:
            return True

    output_tokens = (
        item.get("output_tokens")
        or item.get("final_output_tokens")
        or safe_get(item, ["token_report", "output_tokens"])
        or safe_get(item, ["final_usage", "output_tokens"])
        or safe_get(item, ["usage", "completion_tokens"])
    )

    if expected_final_max_tokens is not None and isinstance(output_tokens, (int, float)):
        if output_tokens >= expected_final_max_tokens:
            return True

    return False


def normalize_token_limit_hit_impact(
    final_token_limit_hit,
    eval_result,
    answer_status,
    judge_impact,
):
    if judge_impact not in VALID_TOKEN_IMPACTS:
        judge_impact = "none"

    if not final_token_limit_hit:
        return "none"

    if eval_result is True:
        return "hit_but_correct"

    if judge_impact in {"hit_and_incomplete", "hit_and_wrong"}:
        return judge_impact

    if answer_status == "incomplete":
        return "hit_and_incomplete"

    return "hit_and_wrong"


def is_content_filter_error(error_msg):
    low = error_msg.lower()

    return (
        "content_filter" in low
        or "responsibleaipolicyviolation" in low
        or "filtered due to the prompt" in low
    )


def extract_json_object(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def judge_answer_and_reasoning(
    client,
    model,
    query,
    reference,
    response,
    reasoning_text,
    raw_response,
    final_token_limit_hit,
    judge_max_tokens,
):
    prompt = f"""
You are a strict answer correctness and reasoning-efficiency judge.

Evaluate the MODEL RESPONSE for the ORIGINAL QUESTION using the REFERENCE ANSWER.

Judge four things.

A) Correctness:
- Return true if the final answer is correct and semantically equivalent to the reference answer.
- Return false if the answer is incorrect, incomplete, unsupported, contradictory, missing, or unclear.
- Accept equivalent wording, formatting, units, notation, and valid paraphrases.
- Ignore formatting differences such as LaTeX, capitalization, punctuation, or whitespace.

B) Answer status:
- "correct": final answer is correct.
- "incorrect": final answer is wrong.
- "incomplete": final answer is cut off, unfinished, missing final conclusion, or truncated.
- "unclear": cannot determine.

C) Reasoning efficiency:
- "over_reasoning": substantially more reasoning than necessary, repeated ideas, unnecessary derivations, redundant analysis, or continued reasoning after enough information.
- "under_reasoning": insufficient reasoning, missing necessary justification, unsupported jumps, incomplete solution, or truncation.
- "adequate_reasoning": reasoning effort is appropriate for the task complexity.

D) Token-limit impact:
- "none": token limit did not hit.
- "hit_but_correct": token limit hit, but final answer is still correct and complete enough.
- "hit_and_incomplete": token limit hit and answer is incomplete/truncated.
- "hit_and_wrong": token limit hit and answer is wrong, not merely incomplete.

Important rules:
1. A correct answer can still be over_reasoning.
2. An incorrect or incomplete answer may be under_reasoning.
3. If TOKEN LIMIT HIT is false, token_limit_hit_impact should be "none".
4. If TOKEN LIMIT HIT is true and eval_result is true, token_limit_hit_impact should be "hit_but_correct".
5. If TOKEN LIMIT HIT is true and eval_result is false, token_limit_hit_impact should be either "hit_and_incomplete" or "hit_and_wrong".
6. Return only valid JSON.

Return exactly this JSON schema:
{{
  "eval_result": true,
  "answer_status": "correct",
  "reasoning_label": "adequate_reasoning",
  "token_limit_hit_impact": "none"
}}

Allowed answer_status:
correct, incorrect, incomplete, unclear

Allowed reasoning_label:
over_reasoning, under_reasoning, adequate_reasoning

Allowed token_limit_hit_impact:
none, hit_but_correct, hit_and_incomplete, hit_and_wrong

TOKEN LIMIT HIT:
{final_token_limit_hit}

ORIGINAL QUESTION:
{query}

REFERENCE ANSWER:
{reference}

MODEL RESPONSE:
{response}

EXTRACTED REASONING TEXT:
{reasoning_text}

RAW RESPONSE:
{raw_response}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict correctness and reasoning-efficiency judge. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=judge_max_tokens,
    )

    output = completion.choices[0].message.content.strip()
    parsed = extract_json_object(output)

    if not parsed:
        return {
            "eval_result": False,
            "answer_status": "unclear",
            "reasoning_label": "adequate_reasoning",
            "token_limit_hit_impact": "none",
        }

    eval_result = bool(parsed.get("eval_result", False))
    answer_status = parsed.get("answer_status") or ("correct" if eval_result else "incorrect")
    reasoning_label = parsed.get("reasoning_label") or "adequate_reasoning"
    token_limit_hit_impact = parsed.get("token_limit_hit_impact") or "none"

    if answer_status not in {"correct", "incorrect", "incomplete", "unclear"}:
        answer_status = "correct" if eval_result else "incorrect"

    if reasoning_label not in {
        "over_reasoning",
        "under_reasoning",
        "adequate_reasoning",
    }:
        reasoning_label = "adequate_reasoning"

    if token_limit_hit_impact not in VALID_TOKEN_IMPACTS:
        token_limit_hit_impact = "none"

    return {
        "eval_result": eval_result,
        "answer_status": answer_status,
        "reasoning_label": reasoning_label,
        "token_limit_hit_impact": token_limit_hit_impact,
    }


def compute_summary(results):
    total = len(results)

    correct = sum(1 for x in results if x.get("eval_result") is True)
    incorrect = sum(1 for x in results if x.get("eval_result") is False)
    errors = sum(1 for x in results if x.get("judge_error"))

    token_hit = [x for x in results if x.get("final_token_limit_hit") is True]
    token_not_hit = [x for x in results if x.get("final_token_limit_hit") is False]

    reasoning_counts = Counter(x.get("reasoning_label", "missing") for x in results)
    answer_status_counts = Counter(x.get("answer_status", "missing") for x in results)

    token_impact_counts = Counter()
    by_reasoning_correctness = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
        }
    )

    for x in results:
        impact = normalize_token_limit_hit_impact(
            final_token_limit_hit=x.get("final_token_limit_hit") is True,
            eval_result=x.get("eval_result"),
            answer_status=x.get("answer_status"),
            judge_impact=x.get("token_limit_hit_impact", "none"),
        )

        x["token_limit_hit_impact"] = impact
        token_impact_counts[impact] += 1

        label = x.get("reasoning_label", "missing")
        by_reasoning_correctness[label]["total"] += 1

        if x.get("eval_result") is True:
            by_reasoning_correctness[label]["correct"] += 1
        elif x.get("eval_result") is False:
            by_reasoning_correctness[label]["incorrect"] += 1

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": correct / total if total else 0.0,

        "final_token_limit_hit": {
            "total_hit": len(token_hit),
            "hit_and_true": sum(1 for x in token_hit if x.get("eval_result") is True),
            "hit_and_false": sum(1 for x in token_hit if x.get("eval_result") is False),
            "hit_accuracy": (
                sum(1 for x in token_hit if x.get("eval_result") is True) / len(token_hit)
                if token_hit
                else 0.0
            ),
        },

        "final_token_limit_not_hit": {
            "total_not_hit": len(token_not_hit),
            "not_hit_and_true": sum(1 for x in token_not_hit if x.get("eval_result") is True),
            "not_hit_and_false": sum(1 for x in token_not_hit if x.get("eval_result") is False),
            "not_hit_accuracy": (
                sum(1 for x in token_not_hit if x.get("eval_result") is True) / len(token_not_hit)
                if token_not_hit
                else 0.0
            ),
        },

        "reasoning_efficiency": {
            "over_reasoning": reasoning_counts.get("over_reasoning", 0),
            "under_reasoning": reasoning_counts.get("under_reasoning", 0),
            "adequate_reasoning": reasoning_counts.get("adequate_reasoning", 0),
            "missing": reasoning_counts.get("missing", 0),
            "over_reasoning_rate": reasoning_counts.get("over_reasoning", 0) / total if total else 0.0,
            "under_reasoning_rate": reasoning_counts.get("under_reasoning", 0) / total if total else 0.0,
            "adequate_reasoning_rate": reasoning_counts.get("adequate_reasoning", 0) / total if total else 0.0,
        },

        "answer_status_counts": dict(answer_status_counts),
        "token_limit_hit_impact_counts": dict(token_impact_counts),
        "by_reasoning_correctness": dict(by_reasoning_correctness),
    }


def validate_summary(summary):
    total = summary["total"]

    if summary["correct"] + summary["incorrect"] != total:
        raise ValueError("Invalid summary: correct + incorrect != total")

    hit = summary["final_token_limit_hit"]["total_hit"]
    not_hit = summary["final_token_limit_not_hit"]["total_not_hit"]

    if hit + not_hit != total:
        raise ValueError("Invalid summary: token_hit + token_not_hit != total")

    impact = summary["token_limit_hit_impact_counts"]

    none = impact.get("none", 0)
    hit_impacts = (
        impact.get("hit_but_correct", 0)
        + impact.get("hit_and_incomplete", 0)
        + impact.get("hit_and_wrong", 0)
    )

    if none != not_hit:
        raise ValueError("Invalid summary: impact none != token_not_hit")

    if hit_impacts != hit:
        raise ValueError("Invalid summary: token-hit impacts != token_hit")

    reasoning = summary["reasoning_efficiency"]

    reasoning_total = (
        reasoning["over_reasoning"]
        + reasoning["under_reasoning"]
        + reasoning["adequate_reasoning"]
        + reasoning["missing"]
    )

    if reasoning_total != total:
        raise ValueError("Invalid summary: reasoning counts != total")


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not found.")

    input_dir = os.path.dirname(args.input_file)
    input_stem = os.path.splitext(os.path.basename(args.input_file))[0]

    if args.output_file is None:
        args.output_file = os.path.join(input_dir, f"judge_result_{input_stem}.json")

    summary_path = os.path.join(
        os.path.dirname(args.output_file),
        f"judge_result_{input_stem}_summary.json",
    )

    print("\nGeneral Judge")
    print("Input:", args.input_file)
    print("Output:", args.output_file)
    print("Summary:", summary_path)
    print("Judge model:", args.judge_model)
    print("Expected final max tokens:", args.expected_final_max_tokens)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    data = load_data(args.input_file)
    results = []

    if os.path.exists(args.output_file):
        try:
            results = load_data(args.output_file)

            if len(results) > len(data):
                print("Existing output is longer than input. Starting fresh.")
                results = []
            else:
                print(f"\nResuming from {len(results)}/{len(data)}")

        except Exception as e:
            print(f"Could not load existing output: {e}")
            results = []

    start_idx = len(results)

    try:
        for idx in tqdm(range(start_idx, len(data)), initial=start_idx, total=len(data)):
            item = data[idx]

            query = get_query(item)
            reference = get_reference(item)
            response = get_response(item)
            raw_response = get_raw_response(item)
            reasoning_text = get_reasoning_text(item)

            final_token_limit_hit = detect_final_token_limit_hit(
                item,
                expected_final_max_tokens=args.expected_final_max_tokens,
            )

            item["final_token_limit_hit"] = final_token_limit_hit

            if not query or not reference or not response:
                item["eval_result"] = False
                item["answer_status"] = "missing_field"
                item["reasoning_label"] = "missing"
                item["token_limit_hit_impact"] = normalize_token_limit_hit_impact(
                    final_token_limit_hit=final_token_limit_hit,
                    eval_result=False,
                    answer_status="missing_field",
                    judge_impact="none",
                )
                item["judge_model"] = args.judge_model
                item["judge_label"] = "missing_required_field"
                item["judge_error"] = {
                    "missing_query": not bool(query),
                    "missing_reference": not bool(reference),
                    "missing_response": not bool(response),
                }

            else:
                try:
                    judge = judge_answer_and_reasoning(
                        client=client,
                        model=args.judge_model,
                        query=query,
                        reference=reference,
                        response=response,
                        reasoning_text=reasoning_text,
                        raw_response=raw_response,
                        final_token_limit_hit=final_token_limit_hit,
                        judge_max_tokens=args.judge_max_tokens,
                    )

                    item["eval_result"] = judge["eval_result"]
                    item["answer_status"] = judge["answer_status"]
                    item["reasoning_label"] = judge["reasoning_label"]

                    item["token_limit_hit_impact"] = normalize_token_limit_hit_impact(
                        final_token_limit_hit=final_token_limit_hit,
                        eval_result=judge["eval_result"],
                        answer_status=judge["answer_status"],
                        judge_impact=judge["token_limit_hit_impact"],
                    )

                    item["judge_model"] = args.judge_model

                except Exception as e:
                    error_msg = str(e)

                    item["eval_result"] = False
                    item["answer_status"] = "error"
                    item["reasoning_label"] = "missing"
                    item["token_limit_hit_impact"] = normalize_token_limit_hit_impact(
                        final_token_limit_hit=final_token_limit_hit,
                        eval_result=False,
                        answer_status="error",
                        judge_impact="none",
                    )
                    item["judge_model"] = args.judge_model
                    item["judge_label"] = (
                        "content_filtered"
                        if is_content_filter_error(error_msg)
                        else "error"
                    )
                    item["judge_error"] = error_msg

            results.append(item)

            summary = compute_summary(results)
            validate_summary(summary)

            save_json(args.output_file, results)
            save_json(summary_path, summary)

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")

        summary = compute_summary(results)
        validate_summary(summary)

        save_json(args.output_file, results)
        save_json(summary_path, summary)

        return

    summary = compute_summary(results)
    validate_summary(summary)

    save_json(args.output_file, results)
    save_json(summary_path, summary)

    print("\nFinished.")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved: {args.output_file}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
