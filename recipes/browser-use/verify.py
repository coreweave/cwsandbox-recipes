"""Check the agent's answer against the browser's observed page."""

from urllib.parse import urlparse


def verify(answer: dict, observed: dict) -> bool:
    link = answer.get("information_link", "")
    parsed = urlparse(link)
    return bool(
        answer.get("title") == observed.get("title") == "Example Domain"
        and observed.get("url") == "https://example.com/"
        and link in observed.get("links", [])
        and parsed.scheme == "https"
        and parsed.hostname in {"iana.org", "www.iana.org"}
    )
