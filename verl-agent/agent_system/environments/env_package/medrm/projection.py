import re

ACTIONS = ["search", "answer"]

def medrm_projection(model_responses):
    valids = [0] * len(model_responses)
    actions = [""] * len(model_responses)
    original_responses = list(model_responses)

    for i, response in enumerate(model_responses):
        action, valid = _split_response(response)
        actions[i] = action
        valids[i] = valid

    return actions, valids, original_responses

def _split_response(model_response):
    # strip <think> block to reach the action part
    action_part = re.sub(r"^\s*<think>.*?</think>\s*", "", model_response, flags=re.DOTALL).strip()
    action = _postprocess_response(action_part)
    return (action, 1) if action is not None else (None, 0)

def _postprocess_response(response):
    # reject if fabricated <information> block
    if "<information>" in response or "</information>" in response:
        return None
    m_search = re.search(r"<search>(.*?)</search>", response, flags=re.DOTALL)
    m_answer = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if m_search and m_answer:
        return None
    if m_search:
        # must be the only content
        if response.strip() != m_search.group(0).strip():
            return None
        return m_search.group(0)
    if m_answer:
        # adapted from generate_agent_search: require \boxed{} inside
        prefix = response[:m_answer.start()].strip()
        if prefix:
            return None
        return m_answer.group(0)
    return None