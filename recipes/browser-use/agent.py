"""Run inside the sandbox as an unprivileged user."""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from browser_use import Agent, Browser, ChatOpenAI
from pydantic import BaseModel

from verify import verify

OUTPUT = Path("/work/output")


class PageAnswer(BaseModel):
    title: str
    information_link: str


async def main():
    OUTPUT.mkdir(exist_ok=True)
    browser = Browser(
        cdp_url="http://127.0.0.1:9222",
        allowed_domains=["https://example.com"],
        keep_alive=True,
    )
    llm = ChatOpenAI(
        model="Qwen/Qwen3.6-35B-A3B",
        base_url="https://api.inference.wandb.ai/v1",
        api_key=os.environ["MODEL_API_KEY"],
        default_headers={"OpenAI-Project": os.environ["INFERENCE_PROJECT"]},
        temperature=0,
        frequency_penalty=None,
        reasoning_models=[],
        max_completion_tokens=8192,
        max_retries=1,
        timeout=60,
    )
    agent = Agent(
        task=(
            "Open https://example.com. Read its exact page title and the destination URL "
            "of the Learn more link. Return title and information_link. Do not follow the link."
        ),
        browser=browser,
        llm=llm,
        output_model_schema=PageAnswer,
        use_vision=True,
        use_judge=False,
        enable_planning=False,
        max_failures=2,
        step_timeout=90,
        llm_timeout=60,
    )
    result = {"passed": False}
    started = time.monotonic()
    chromium = None
    try:
        # Launch explicitly: Browser Use 0.13.10 adds --no-sandbox when it detects Docker.
        chromium = await asyncio.create_subprocess_exec(
            "/usr/bin/chromium",
            "--headless=new",
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=9222",
            "--user-data-dir=/work/profile",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "about:blank",
        )
        async with httpx.AsyncClient() as client:
            for _ in range(50):
                if chromium.returncode is not None:
                    raise RuntimeError("Chromium exited before CDP became available")
                try:
                    response = await client.get("http://127.0.0.1:9222/json/version", timeout=1)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
            else:
                raise TimeoutError("Chromium CDP startup timed out")
        history = await asyncio.wait_for(agent.run(max_steps=12), timeout=240)
        history.save_to_file(str(OUTPUT / "history.json"))
        await browser.take_screenshot(path=str(OUTPUT / "final.png"))
        page = await browser.get_current_page()
        observed = json.loads(
            await page.evaluate(
                "() => JSON.stringify({url:location.href,title:document.title,"
                "links:Array.from(document.querySelectorAll('a')).map(a=>a.href)})"
            )
        )
        answer = PageAnswer.model_validate_json(history.final_result() or "{}").model_dump()
        result.update(
            answer=answer,
            observed=observed,
            agent_reported_success=history.is_successful(),
            passed=verify(answer, observed),
            steps=history.number_of_steps(),
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    finally:
        result["agent_seconds"] = round(time.monotonic() - started, 3)
        (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        try:
            await browser.kill()
        finally:
            if chromium is not None and chromium.returncode is None:
                chromium.terminate()
                try:
                    await asyncio.wait_for(chromium.wait(), timeout=10)
                except TimeoutError:
                    chromium.kill()
                    await chromium.wait()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
