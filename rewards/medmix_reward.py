import math
import re
from datetime import datetime

VALID = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_BOX_CONTENT_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)

_MEDCALC_DATE_IDS = {13, 68}
_MEDCALC_TUPLE_IDS = {69}
_MEDCALC_INTEGER_IDS = {4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51}
_MEDCALC_DECIMAL_IDS = {2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67}


def extract_choices(text: str):
    letters = re.findall(r"\b([A-Z])\b", text)
    return [x for x in letters if x in VALID]


def _extract_last_box_content(text: str):
    contents = _BOX_CONTENT_RE.findall(text)
    return contents[-1] if contents else None


def calculate_mcqa_accuracy_reward(solution_str: str, gt_choice: str) -> float:
    last = _extract_last_box_content(solution_str)
    if last is None:
        return 0.0

    choices = extract_choices(last)
    uniq = set(choices)
    if len(uniq) != 1:
        return 0.0
    pred = next(iter(uniq))
    return 1.0 if pred == gt_choice else 0.0


def _extract_boxed_answer_strict(text: str) -> str:
    last = _extract_last_box_content(text)
    return last.strip() if last is not None else ""


def _safe_eval(expr: str):
    return eval(
        expr,
        {"__builtins__": None},
        {
            "min": min,
            "pow": pow,
            "round": round,
            "abs": abs,
            "int": int,
            "float": float,
            "math": math,
        },
    )


def _extract_medcalc_answer(boxed_answer: str, calid: int) -> str:
    extracted_answer = str(boxed_answer or "").strip()

    if calid in _MEDCALC_DATE_IDS:
        match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)
            return f"{month:02}/{day:02}/{year}"
        return "N/A"

    if calid in _MEDCALC_TUPLE_IDS:
        normalized = extracted_answer.replace("[", "(").replace("]", ")").replace("'", "").replace('"', "")
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?",
            normalized,
        )
        if match:
            weeks = match.group(1)
            days = match.group(3)
            return f"({weeks}, {days})"
        return "N/A"

    if calid in _MEDCALC_INTEGER_IDS:
        match = re.search(r"(\d+) out of", extracted_answer)
        if match:
            return match.group(1)

        match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
        if match:
            return str(len(match.group(0).split(",")))

        match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
        if match:
            return match[-1][0]
        return "N/A"

    if calid in _MEDCALC_DECIMAL_IDS:
        match = re.search(r"str\((.*)\)", extracted_answer)
        if match:
            expression = (
                match.group(1)
                .replace("^", "**")
                .replace("is odd", "% 2 == 1")
                .replace("is even", "% 2 == 0")
                .replace("sqrt", "math.sqrt")
                .replace(".math", "")
                .replace("weight", "")
                .replace("height", "")
                .replace("mg/dl", "")
                .replace("g/dl", "")
                .replace("mmol/L", "")
                .replace("kg", "")
                .replace("g", "")
                .replace("mEq/L", "")
            )
            expression = expression.split("#")[0]
            if expression.count("(") > expression.count(")"):
                expression += ")" * (expression.count("(") - expression.count(")"))
            elif expression.count(")") > expression.count("("):
                expression = "(" * (expression.count(")") - expression.count("(")) + expression
            try:
                return str(_safe_eval(expression))
            except Exception:
                return "N/A"

        match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
        if match:
            try:
                return str(_safe_eval(match.group(1)))
            except Exception:
                return "N/A"

        match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
        if match:
            try:
                return str(_safe_eval(match[-1][0]) / 100)
            except Exception:
                return "N/A"

        match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
        if match:
            try:
                return str(_safe_eval(match[-1][0]))
            except Exception:
                return "N/A"

        return "N/A"

    return "N/A"


def _compute_medcalc_score(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    info = extra_info or {}
    calc_id = info.get("calculator_id", info.get("calc_id"))
    lower_bound = info.get("lower_bound")
    upper_bound = info.get("upper_bound")

    try:
        calid = int(calc_id)
    except Exception:
        return 0.0

    boxed_answer = _extract_boxed_answer_strict(solution_str)
    if not boxed_answer:
        return 0.0

    answer = _extract_medcalc_answer(boxed_answer, calid)

    # four types of calculation verification
    if calid in _MEDCALC_DATE_IDS:
        try:
            pred = datetime.strptime(answer, "%m/%d/%Y").date()
            gt = datetime.strptime(str(ground_truth), "%m/%d/%Y").date()
            return 1.0 if pred == gt else 0.0
        except Exception:
            return 0.0

    if calid in _MEDCALC_TUPLE_IDS:
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?",
            str(ground_truth),
        )
        if not match:
            return 0.0

        gt_tuple = f"({match.group(1)}, {match.group(3)})"
        match = re.search(r"\(?[\"\']?(\d+)\s*,\s*(\d+)\s*\)?", str(answer))
        if not match:
            return 0.0

        pred_tuple = f"({match.group(1)}, {match.group(2)})"
        try:
            return 1.0 if eval(pred_tuple) == eval(gt_tuple) else 0.0
        except Exception:
            return 0.0

    if calid in _MEDCALC_INTEGER_IDS:
        try:
            numeric = eval(str(answer))
            return 1.0 if numeric == eval(str(ground_truth)) else 0.0
        except Exception:
            return 0.0

    if calid in _MEDCALC_DECIMAL_IDS:
        try:
            numeric = float(eval(str(answer)))
            lo = float(eval(str(lower_bound)))
            hi = float(eval(str(upper_bound)))
            return 1.0 if lo <= numeric <= hi else 0.0
        except Exception:
            return 0.0

    return 0.0


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs):
    # compute calculation verification score
    if "medcalc-bench" in str(data_source).lower():
        return _compute_medcalc_score(solution_str, ground_truth, extra_info)

    # compute MCQA accuracy score
    assert ground_truth in VALID
    return calculate_mcqa_accuracy_reward(solution_str, ground_truth)
