import random
import time
import traceback

from google.genai.types import GenerateContentConfig, ThinkingConfig
from transformers import AutoTokenizer
from threading import Lock, Barrier, Event
import queue

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(input):
    tokens = tokenizer.encode(input)
    token_length = len(tokens)
    return token_length


def _handle_retry_error(e, attempt, max_try_times, sleep=False):
    if "context" in str(e).lower():
        raise ValueError(f"Context length error: {e}")
    print(f"Error: {e}")
    print(traceback.format_exc())
    if attempt == max_try_times - 1:
        raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
    if sleep:
        err_str = str(e).lower()
        if "throttling" in err_str or "too many" in err_str or "rate" in err_str:
            wait = min(60 * (2 ** attempt), 300)  # exponential backoff, cap at 5 min
            print(f"Throttled. Waiting {wait}s before retry (attempt {attempt + 1}/{max_try_times})...")
            time.sleep(wait)
        else:
            time.sleep(random.randint(2, 5))


def query_qwen(client, model_name, prompt, max_try_times=6, temperature=0.0):
    """Query Qwen; returns (thought, response). thought is '' if no reasoning_content."""
    for attempt in range(max_try_times):
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096,
            )
            thought = r.choices[0].message.reasoning_content or ""
            response = r.choices[0].message.content or ""
            return thought, response
        except Exception as e:
            _handle_retry_error(e, attempt, max_try_times, sleep=False)
    return "", ""  # unreachable


_vllm_request_queue = queue.Queue()
_vllm_dispatcher_lock = Lock()
_vllm_dispatcher_started = False


_VLLM_BATCH_WINDOW = 0.05  # seconds to wait after first item to collect concurrent requests


def _vllm_dispatcher(llm):
    """Background dispatcher: drains the request queue in batches for maximum vLLM throughput."""
    while True:
        # Block until at least one request arrives
        first = _vllm_request_queue.get()
        batch = [first]
        # Wait briefly to let concurrent threads pile in before dispatching
        time.sleep(_VLLM_BATCH_WINDOW)
        try:
            while True:
                batch.append(_vllm_request_queue.get_nowait())
        except queue.Empty:
            pass

        prompts = [item["prompt"] for item in batch]
        sampling_params = batch[0]["sampling_params"]  # all share same params in this pipeline

        try:
            outputs = llm.generate(prompts, sampling_params)
            for item, out in zip(batch, outputs):
                item["result"].put(out.outputs[0].text)
        except Exception as e:
            for item in batch:
                item["result"].put(e)


def _ensure_vllm_dispatcher(llm):
    global _vllm_dispatcher_started
    with _vllm_dispatcher_lock:
        if not _vllm_dispatcher_started:
            import threading
            t = threading.Thread(target=_vllm_dispatcher, args=(llm,), daemon=True)
            t.start()
            _vllm_dispatcher_started = True


def _parse_vllm_output(output_text, enable_thinking):
    if enable_thinking and "<think>" in output_text and "</think>" in output_text:
        thought = output_text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        response = output_text.split("</think>", 1)[1].strip()
    else:
        thought = ""
        response = output_text.strip()
    return thought, response


def query_qwen_local(llm, local_tokenizer, prompt, max_new_tokens=4096, temperature=0.0, enable_thinking=True):
    """Query a local Qwen model via vLLM Python API; returns (thought, response).

    Concurrent calls from multiple threads are automatically batched into a
    single llm.generate() call by the background dispatcher, so increasing
    CONCURRENT_NUM improves throughput.

    Args:
        llm: vllm.LLM instance (pre-initialized by the caller).
        local_tokenizer: tokenizer loaded from the same model path.
        prompt: plain-text user prompt.
        enable_thinking: if True, the model may emit a <think>…</think> block.
    """
    from vllm import SamplingParams

    _ensure_vllm_dispatcher(llm)

    messages = [{"role": "user", "content": prompt}]
    formatted = local_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Pre-check length to avoid poisoning other prompts in the same dispatcher batch
    max_model_len = llm.llm_engine.model_config.max_model_len
    token_ids = local_tokenizer.encode(formatted)
    if len(token_ids) >= max_model_len:
        raise ValueError(
            f"Context length error: prompt has {len(token_ids)} tokens, "
            f"exceeds max_model_len {max_model_len}"
        )

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

    result_q = queue.Queue()
    _vllm_request_queue.put({"prompt": formatted, "sampling_params": sampling_params, "result": result_q})

    output_text = result_q.get()
    if isinstance(output_text, Exception):
        raise output_text
    return _parse_vllm_output(output_text, enable_thinking)


def query_gemini(client, model_name, prompt, with_thinking=True, thinking_budget=4096,
                 max_output_tokens=4096, max_try_times=5):
    """Query Gemini; returns (thought, response).

    with_thinking=False disables the thinking config (used for base generation).
    """
    thought = ""
    response = ""
    for attempt in range(max_try_times):
        try:
            config_kwargs = dict(max_output_tokens=max_output_tokens, temperature=0.0)
            if with_thinking:
                config_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_budget=thinking_budget, include_thoughts=True
                )
            gemini_response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(**config_kwargs),
            )
            if not getattr(gemini_response, "candidates", None):
                raise ValueError("Empty Gemini response: candidates is empty")
            candidate = gemini_response.candidates[0]
            parts = getattr(getattr(candidate, "content", None), "parts", None)
            if not parts:
                raise ValueError("Empty Gemini response: candidate.content.parts is None")

            thought_parts, response_parts = [], []
            for part in parts:
                txt = getattr(part, "text", None)
                if not txt:
                    continue
                if with_thinking and getattr(part, "thought", False):
                    thought_parts.append(txt)
                else:
                    response_parts.append(txt)
            thought = "\n".join(thought_parts).strip()
            response = "".join(response_parts).strip()
            return thought, response
        except Exception as e:
            _handle_retry_error(e, attempt, max_try_times, sleep=True)
    return thought, response


def query_gpt(client, model_name, prompt, max_try_times=6):
    """Query GPT; returns ('', response)."""
    for attempt in range(max_try_times):
        try:
            if hasattr(client, "responses"):
                r = client.responses.create(
                    model=model_name,
                    input=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_output_tokens=4096,
                )
                return "", r.output_text or ""
            else:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                )
                return "", r.choices[0].message.content or ""
        except Exception as e:
            _handle_retry_error(e, attempt, max_try_times, sleep=False)
    return "", ""  # unreachable


def query_bedrock(client, model_name, prompt, max_try_times=5, temperature=0.0):
    """Query AWS Bedrock; returns (thought, response)."""
    thought = ""
    response = None
    for attempt in range(max_try_times):
        try:
            bedrock_response = client.converse(
                modelId=model_name,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 16384, "temperature": temperature},
            )
            content = (
                bedrock_response.get("output", {})
                .get("message", {})
                .get("content", [])
            )
            response = None
            thought = ""
            for part in content:
                if response is None and "text" in part:
                    response = part["text"]
                reasoning = part.get("reasoningContent")
                if reasoning and not thought:
                    thought = reasoning.get("reasoningText", {}).get("text", "")
            return thought, (response or "")
        except Exception as e:
            _handle_retry_error(e, attempt, max_try_times, sleep=True)
    return thought, (response or "")
