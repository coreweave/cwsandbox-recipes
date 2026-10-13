"""Translate upstream JevBench requests to native Anthropic structured outputs."""


def native_transport(url, body, headers, timeout_s):
    from jevbench.adapters.base import http_post_json

    if url != "https://api.anthropic.com/v1/chat/completions":
        raise ValueError("Unexpected transport endpoint")
    native = {
        "model": body["model"],
        "system": "\n".join(m["content"] for m in body["messages"] if m["role"] == "system"),
        "messages": [m for m in body["messages"] if m["role"] != "system"],
        "max_tokens": body["max_tokens"],
        "temperature": body["temperature"],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": body["response_format"]["json_schema"]["schema"],
            }
        },
    }
    status, response, latency = http_post_json(
        "https://api.anthropic.com/v1/messages",
        native,
        {
            "x-api-key": headers["Authorization"].removeprefix("Bearer "),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    if status != 200 or not isinstance(response, dict):
        return status, response, latency
    content = "".join(b["text"] for b in response.get("content", []) if b.get("type") == "text")
    usage = response.get("usage", {})
    return (
        status,
        {
            "model": response.get("model", body["model"]),
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            },
            "native_request": native,
            "native_response": response,
        },
        latency,
    )
