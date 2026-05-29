/**
 * Frontend for the Gesture Agent.
 * If served by the Flask backend on the same origin, API_BASE stays empty.
 * To run the frontend on a different host (e.g. opened from disk), override
 *   window.GESTURE_AGENT_API = "http://127.0.0.1:5050";
 * before this script loads.
 */
const API_BASE = (window.GESTURE_AGENT_API || "").replace(/\/$/, "");

const els = {
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  send: document.getElementById("send"),
  reset: document.getElementById("reset"),
  stream: document.getElementById("stream"),
  images: document.getElementById("images"),
  health: document.getElementById("health"),
  structure: document.getElementById("structure"),
  chunks: document.getElementById("chunks"),
};

const state = {
  sessionId: localStorage.getItem("gesture_agent_session") || null,
  busy: false,
};

async function api(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }
  return res;
}

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const res = await api("/api/session", { method: "POST" });
  const data = await res.json();
  state.sessionId = data.session_id;
  localStorage.setItem("gesture_agent_session", state.sessionId);
  return state.sessionId;
}

async function refreshHealth() {
  try {
    const res = await api("/api/health");
    const data = await res.json();
    els.health.textContent = `model: ${data.model || "default"} · docs: ${data.doc_count}`;
    els.health.classList.add("ok");
    els.health.classList.remove("bad");
  } catch (err) {
    els.health.textContent = `后端不可用：${err.message}`;
    els.health.classList.add("bad");
    els.health.classList.remove("ok");
  }
}

function renderMarkdown(text) {
  if (!text) return "";
  const normalized = normalizeAnswerMarkdown(text);
  if (typeof window.marked !== "undefined") {
    if (typeof window.marked.setOptions === "function") {
      window.marked.setOptions({ gfm: true, breaks: true });
    }
    const html = window.marked.parse(normalized);
    if (typeof window.DOMPurify !== "undefined") {
      return window.DOMPurify.sanitize(html, { ADD_TAGS: ["img"], ADD_ATTR: ["src", "alt", "loading"] });
    }
    return html;
  }
  return fallbackMarkdown(normalized);
}

/**
 * Cleans up patterns that defeat markdown rendering:
 *   1. The whole answer wrapped in ```json … ``` → unfold the JSON object
 *      into Markdown sections.
 *   2. The whole answer wrapped in ```markdown / ```md / ``` … ``` → strip
 *      the outer fence so marked renders the inner Markdown.
 * Falls back to the original text when nothing matches.
 */
function normalizeAnswerMarkdown(text) {
  const trimmed = text.trim();

  // Match a single outer fenced block that spans the entire answer.
  const fenceMatch = trimmed.match(/^```([a-zA-Z0-9_-]*)\s*\n([\s\S]*?)\n?```$/);
  if (fenceMatch) {
    const lang = (fenceMatch[1] || "").toLowerCase();
    const inner = fenceMatch[2];
    // Avoid ambiguity: only strip the fence if there's no other fence inside.
    const innerHasFence = /```/.test(inner);
    if (!innerHasFence) {
      if (lang === "json") {
        const unfolded = tryUnfoldJsonObject(inner);
        if (unfolded !== null) return unfolded;
      }
      if (lang === "" || lang === "markdown" || lang === "md" || lang === "text") {
        return inner;
      }
    }
  }

  // No fence, but the whole answer is a JSON object literal.
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    const unfolded = tryUnfoldJsonObject(trimmed);
    if (unfolded !== null) return unfolded;
  }

  return text;
}

function tryUnfoldJsonObject(candidate) {
  let parsed;
  try {
    parsed = JSON.parse(candidate);
  } catch (_) {
    return null;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

  const lines = [];
  for (const [key, value] of Object.entries(parsed)) {
    lines.push(`## ${key}`);
    lines.push(formatJsonValueAsMarkdown(value));
    lines.push("");
  }
  return lines.join("\n");
}

function formatJsonValueAsMarkdown(value, depth = 0) {
  if (value === null || value === undefined) return "_（无内容）_";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    // Try to render array-of-objects (same keys) as a table; otherwise as a list.
    if (value.length && value.every((v) => v && typeof v === "object" && !Array.isArray(v))) {
      const keys = Array.from(new Set(value.flatMap((v) => Object.keys(v))));
      const header = `| ${keys.join(" | ")} |`;
      const sep = `| ${keys.map(() => "---").join(" | ")} |`;
      const rows = value.map((row) =>
        `| ${keys.map((k) => mdCell(row[k])).join(" | ")} |`,
      );
      return [header, sep, ...rows].join("\n");
    }
    return value.map((item) => `- ${formatJsonValueAsMarkdown(item, depth + 1)}`).join("\n");
  }
  if (typeof value === "object") {
    // Detect the "columns + rows" table convention used by the model.
    const keys = Object.keys(value);
    if (keys.includes("对比维度") && Array.isArray(value["对比维度"])) {
      return renderColumnTable(value);
    }
    const lines = [];
    for (const [k, v] of Object.entries(value)) {
      if (v && typeof v === "object") {
        lines.push(`**${k}**`);
        lines.push(formatJsonValueAsMarkdown(v, depth + 1));
      } else {
        lines.push(`- **${k}**：${formatJsonValueAsMarkdown(v, depth + 1)}`);
      }
    }
    return lines.join("\n");
  }
  return String(value);
}

function renderColumnTable(value) {
  const dims = value["对比维度"];
  const otherKeys = Object.keys(value).filter((k) => k !== "对比维度");
  const header = `| 维度 | ${otherKeys.join(" | ")} |`;
  const sep = `| --- | ${otherKeys.map(() => "---").join(" | ")} |`;
  const rows = dims.map((dim, i) => {
    const cells = otherKeys.map((k) => mdCell(Array.isArray(value[k]) ? value[k][i] : value[k]));
    return `| ${mdCell(dim)} | ${cells.join(" | ")} |`;
  });
  return [header, sep, ...rows].join("\n");
}

function mdCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value.replace(/\|/g, "\\|").replace(/\n+/g, " ");
  if (Array.isArray(value)) return value.map(mdCell).join("；");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function fallbackMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  // very small subset: code fences, inline code, bold, italics, headings, lists, links
  let html = escaped.replace(/```([\s\S]*?)```/g, (_, body) => `<pre><code>${body}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/^###### (.+)$/gm, "<h6>$1</h6>")
    .replace(/^##### (.+)$/gm, "<h5>$1</h5>")
    .replace(/^#### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>");
  html = html.replace(/^(\s*)[-*] (.+)$/gm, "$1<li>$2</li>");
  html = html.replace(/(<li>[\s\S]*?<\/li>)(?=(\n[^<])|$)/g, "<ul>$1</ul>");
  html = html.replace(/\n{2,}/g, "</p><p>");
  return `<p>${html}</p>`;
}

function setBubbleContent(bubble, text, { markdown }) {
  if (markdown) {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
}

function appendMessage({ role, text, kind, persist = true, markdown }) {
  const row = document.createElement("div");
  row.className = `message ${role}${kind ? " " + kind : ""}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  // Default: render markdown only for assistant messages without an error kind.
  const useMd = markdown !== undefined ? markdown : role === "assistant" && kind !== "error";
  if (useMd) bubble.classList.add("markdown");
  setBubbleContent(bubble, text, { markdown: useMd });
  bubble.dataset.markdown = useMd ? "1" : "0";
  row.appendChild(bubble);
  if (persist === false) {
    row.dataset.transient = "1";
  }
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return { row, bubble };
}

function renderStructure(structure) {
  if (!structure) {
    els.structure.textContent = "—";
    return;
  }
  els.structure.textContent = JSON.stringify(structure, null, 2);
}

function renderChunks(chunks) {
  els.chunks.innerHTML = "";
  if (!chunks || chunks.length === 0) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "本轮没有命中资料";
    els.chunks.appendChild(li);
    return;
  }
  for (const chunk of chunks) {
    const li = document.createElement("li");
    li.className = "chunk-item";

    const details = document.createElement("details");
    details.className = "chunk-details";

    const summary = document.createElement("summary");
    summary.className = "chunk-summary";

    const title = document.createElement("span");
    title.className = "chunk-title";
    title.textContent = chunk.title;
    summary.appendChild(title);

    const meta = document.createElement("span");
    meta.className = "citation";
    const metaBits = [chunk.citation, chunk.layer];
    if (typeof chunk.score === "number") metaBits.push(`score=${chunk.score}`);
    meta.textContent = metaBits.filter(Boolean).join(" · ");
    summary.appendChild(meta);

    details.appendChild(summary);

    if (Array.isArray(chunk.terms) && chunk.terms.length) {
      const tagWrap = document.createElement("div");
      tagWrap.className = "chunk-tags";
      for (const term of chunk.terms) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = term;
        tagWrap.appendChild(tag);
      }
      details.appendChild(tagWrap);
    }

    if (chunk.text) {
      const body = document.createElement("div");
      body.className = "chunk-body markdown";
      body.innerHTML = renderMarkdown(chunk.text);
      details.appendChild(body);
    } else {
      const empty = document.createElement("div");
      empty.className = "chunk-body muted";
      empty.textContent = "（无正文）";
      details.appendChild(empty);
    }

    li.appendChild(details);
    els.chunks.appendChild(li);
  }
}

function parseImages(text) {
  if (!text) return [];
  return text
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.send.textContent = busy ? "请求中…" : "发送";
}

async function sendOnce(question, images) {
  const sessionId = await ensureSession();
  const res = await api("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, images }),
  });
  const data = await res.json();
  if (data.status === "clarify") {
    appendMessage({ role: "system", text: data.message || "需要澄清。" });
    return;
  }
  renderStructure(data.structure);
  renderChunks(data.chunks);
  if (data.error) {
    appendMessage({ role: "assistant", kind: "error", text: data.error });
    return;
  }
  appendMessage({ role: "assistant", text: data.answer || "(空响应)" });
}

async function sendStream(question, images) {
  const sessionId = await ensureSession();
  const res = await fetch(`${API_BASE}/api/ask_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, images }),
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answerEl = null;
  let answerText = "";
  let clarified = false;

  function ensureAnswerEl() {
    if (!answerEl) {
      const created = appendMessage({ role: "assistant", text: "" });
      answerEl = created.bubble;
      answerEl.classList.add("typing");
    }
    return answerEl;
  }

  function handleEvent(event, data) {
    let payload = {};
    try {
      payload = data ? JSON.parse(data) : {};
    } catch (_) {
      payload = { raw: data };
    }
    switch (event) {
      case "meta":
        renderStructure(payload.structure);
        renderChunks(payload.chunks);
        break;
      case "delta":
        ensureAnswerEl();
        answerText += payload.text || "";
        setBubbleContent(answerEl, answerText, { markdown: true });
        els.messages.scrollTop = els.messages.scrollHeight;
        break;
      case "clarify":
        clarified = true;
        appendMessage({ role: "system", text: payload.message || "需要澄清。" });
        break;
      case "error":
        appendMessage({ role: "assistant", kind: "error", text: payload.message || "请求失败" });
        break;
      case "done":
        if (answerEl) answerEl.classList.remove("typing");
        if (payload.answer) {
          ensureAnswerEl();
          setBubbleContent(answerEl, payload.answer, { markdown: true });
        }
        break;
      default:
        break;
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sepIdx;
    while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sepIdx);
      buffer = buffer.slice(sepIdx + 2);
      if (!block.trim()) continue;
      let event = "message";
      const dataLines = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      handleEvent(event, dataLines.join("\n"));
    }
  }
}

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.busy) return;
  const question = els.question.value.trim();
  if (!question) return;
  const images = parseImages(els.images.value);

  appendMessage({ role: "user", text: question });
  els.question.value = "";

  setBusy(true);
  try {
    if (els.stream.checked) {
      await sendStream(question, images);
    } else {
      await sendOnce(question, images);
    }
  } catch (err) {
    appendMessage({ role: "assistant", kind: "error", text: err.message || String(err) });
  } finally {
    setBusy(false);
  }
});

els.question.addEventListener("compositionstart", () => {
  els.question.dataset.composing = "1";
});
els.question.addEventListener("compositionend", () => {
  delete els.question.dataset.composing;
  // Some IMEs (especially on macOS) fire compositionend right before the Enter
  // keydown that committed the candidate, so isComposing reads false by then.
  // Mark a brief grace window to suppress that trailing Enter.
  els.question.dataset.justComposed = "1";
  setTimeout(() => {
    delete els.question.dataset.justComposed;
  }, 50);
});

els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    // Skip when IME is composing or just finished composing.
    if (e.isComposing || e.keyCode === 229) return;
    if (els.question.dataset.composing || els.question.dataset.justComposed) return;
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.reset.addEventListener("click", async () => {
  if (!state.sessionId) return;
  try {
    await api("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    els.messages.innerHTML = "";
    renderStructure(null);
    renderChunks([]);
    appendMessage({ role: "system", text: "已清空当前会话记忆。" });
  } catch (err) {
    appendMessage({ role: "assistant", kind: "error", text: err.message || String(err) });
  }
});

refreshHealth();
ensureSession().catch((err) => {
  appendMessage({ role: "assistant", kind: "error", text: `初始化会话失败：${err.message}` });
});
