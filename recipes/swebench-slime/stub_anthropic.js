// Minimal Anthropic Messages stand-in for the slime adapter: answers every
// /v1/messages call with one text turn and logs what claude-code sent.
const http = require("http");
const fs = require("fs");

const log = (o) => fs.appendFileSync("/tmp/stub.jsonl", JSON.stringify(o) + "\n");

http
  .createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      let j = {};
      try {
        j = JSON.parse(body || "{}");
      } catch (e) {}
      log({
        method: req.method,
        url: req.url,
        auth: req.headers["authorization"] || req.headers["x-api-key"] || "",
        stream: !!j.stream,
        model: j.model,
        n_messages: (j.messages || []).length,
      });
      const json = (code, o) => {
        res.writeHead(code, { "content-type": "application/json" });
        res.end(JSON.stringify(o));
      };
      if (req.url.startsWith("/v1/messages/count_tokens")) return json(200, { input_tokens: 1 });
      if (!req.url.startsWith("/v1/messages")) return json(200, {});
      const text = "Stub adapter reply: no change needed.";
      const msg = {
        id: "msg_stub",
        type: "message",
        role: "assistant",
        model: j.model || "slime-actor",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 0 },
      };
      if (!j.stream) {
        return json(200, {
          ...msg,
          content: [{ type: "text", text }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 6 },
        });
      }
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      const ev = (t, d) => res.write(`event: ${t}\ndata: ${JSON.stringify(d)}\n\n`);
      ev("message_start", { type: "message_start", message: msg });
      ev("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
      ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } });
      ev("content_block_stop", { type: "content_block_stop", index: 0 });
      ev("message_delta", {
        type: "message_delta",
        delta: { stop_reason: "end_turn", stop_sequence: null },
        usage: { output_tokens: 6 },
      });
      ev("message_stop", { type: "message_stop" });
      res.end();
    });
  })
  .listen(18001, "127.0.0.1");
