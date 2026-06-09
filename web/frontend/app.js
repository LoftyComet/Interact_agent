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
  style: document.getElementById("style"),
  provider: document.getElementById("provider"),
  imageFiles: document.getElementById("image-files"),
  imagePreview: document.getElementById("image-preview"),
  health: document.getElementById("health"),
  structure: document.getElementById("structure"),
};

const state = {
  sessionId: localStorage.getItem("gesture_agent_session") || null,
  busy: false,
  abortController: null,
  lastChunks: [],
  pendingImages: [], // [{ name, dataUrl }]
  thinkingRow: null, // transient "正在思考" indicator
};

const MAX_IMAGE_BYTES = 10 * 1024 * 1024; // 10 MB, mirrors backend limit

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

async function loadProviders() {
  const saved = localStorage.getItem("gesture_agent_provider") || "";
  try {
    const res = await api("/api/providers");
    const data = await res.json();
    const providers = data.providers || [];
    els.provider.innerHTML = "";
    for (const p of providers) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.configured ? p.label : `${p.label}（未配置）`;
      opt.disabled = !p.configured;
      els.provider.appendChild(opt);
    }
    // Restore the saved choice if it's still available and configured.
    const match = providers.find((p) => p.id === saved && p.configured);
    if (match) {
      els.provider.value = saved;
    } else {
      const firstConfigured = providers.find((p) => p.configured);
      if (firstConfigured) els.provider.value = firstConfigured.id;
    }
  } catch (err) {
    // Leave the dropdown empty; backend default provider will be used.
  }
}

els.provider &&
  els.provider.addEventListener("change", () => {
    localStorage.setItem("gesture_agent_provider", els.provider.value);
  });

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
    linkifyCitations(bubble);
  } else {
    bubble.textContent = text;
  }
}

function linkifyCitations(container) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const citationRe = /\[(\d+)\]/g;
  const nodesToReplace = [];
  let node;
  while ((node = walker.nextNode())) {
    if (node.parentElement && node.parentElement.closest("code, pre, a")) continue;
    if (citationRe.test(node.textContent)) {
      nodesToReplace.push(node);
    }
    citationRe.lastIndex = 0;
  }
  for (const textNode of nodesToReplace) {
    const frag = document.createDocumentFragment();
    let lastIdx = 0;
    const text = textNode.textContent;
    let match;
    citationRe.lastIndex = 0;
    while ((match = citationRe.exec(text)) !== null) {
      if (match.index > lastIdx) {
        frag.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
      }
      const idx = parseInt(match[1], 10);
      const sup = document.createElement("sup");
      sup.className = "cite-ref";
      sup.textContent = `[${idx}]`;
      sup.dataset.citeIdx = idx;
      sup.addEventListener("click", (e) => {
        e.stopPropagation();
        const chunks = state.lastChunks || [];
        const chunk = chunks[idx - 1];
        if (chunk) showChunkPopover(sup, chunk, idx);
      });
      frag.appendChild(sup);
      lastIdx = citationRe.lastIndex;
    }
    if (lastIdx < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIdx)));
    }
    textNode.parentNode.replaceChild(frag, textNode);
  }
}

function buildMessageImages(images) {
  const gallery = document.createElement("div");
  gallery.className = "message-images";
  for (const src of images) {
    const picture = document.createElement("img");
    picture.src = src;
    picture.loading = "lazy";
    picture.alt = "用户上传的图片";
    gallery.appendChild(picture);
  }
  return gallery;
}

function appendMessage({ role, text, kind, persist = true, markdown, images }) {
  const row = document.createElement("div");
  row.className = `message ${role}${kind ? " " + kind : ""}`;
  if (Array.isArray(images) && images.length) {
    row.appendChild(buildMessageImages(images));
  }
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
  // Store chunks for attaching to the answer bubble later.
  state.lastChunks = chunks || [];
}

function attachChunkRefs(messageRow, chunks) {
  if (!chunks || chunks.length === 0) return;
  const refBar = document.createElement("div");
  refBar.className = "ref-bar";
  for (let i = 0; i < chunks.length; i++) {
    const chip = document.createElement("button");
    chip.className = "ref-chip";
    chip.textContent = `[${i + 1}] ${chunks[i].title || "来源"}`;
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      showChunkPopover(chip, chunks[i], i + 1);
    });
    refBar.appendChild(chip);
  }
  messageRow.appendChild(refBar);
}

function showChunkPopover(anchor, chunk, idx) {
  closeChunkPopover();
  const pop = document.createElement("div");
  pop.className = "chunk-popover";
  pop.id = "chunk-popover-active";

  const header = document.createElement("div");
  header.className = "chunk-pop-header";
  header.textContent = `[${idx}] ${chunk.title || ""}`;
  pop.appendChild(header);

  const meta = document.createElement("div");
  meta.className = "chunk-pop-meta";
  const bits = [chunk.citation, chunk.layer];
  if (typeof chunk.score === "number") bits.push(`score=${chunk.score}`);
  meta.textContent = bits.filter(Boolean).join(" · ");
  pop.appendChild(meta);

  if (Array.isArray(chunk.terms) && chunk.terms.length) {
    const tags = document.createElement("div");
    tags.className = "chunk-pop-tags";
    tags.textContent = chunk.terms.join("、");
    pop.appendChild(tags);
  }

  if (chunk.text) {
    const body = document.createElement("div");
    body.className = "chunk-pop-body markdown";
    body.innerHTML = renderMarkdown(chunk.text);
    pop.appendChild(body);
  }

  document.body.appendChild(pop);
  positionPopover(pop, anchor);

  setTimeout(() => {
    document.addEventListener("click", closeChunkPopover, { once: true });
  }, 0);
}

function positionPopover(pop, anchor) {
  const rect = anchor.getBoundingClientRect();
  pop.style.position = "fixed";
  pop.style.left = `${rect.left}px`;
  pop.style.top = `${rect.bottom + 6}px`;
  const popRect = pop.getBoundingClientRect();
  if (popRect.right > window.innerWidth - 12) {
    pop.style.left = `${window.innerWidth - popRect.width - 12}px`;
  }
  if (popRect.bottom > window.innerHeight - 12) {
    pop.style.top = `${rect.top - popRect.height - 6}px`;
  }
}

function closeChunkPopover() {
  const existing = document.getElementById("chunk-popover-active");
  if (existing) existing.remove();
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("读取文件失败"));
    reader.readAsDataURL(file);
  });
}

async function handleImageFiles(fileList) {
  const files = Array.from(fileList || []);
  for (const file of files) {
    if (!file.type.startsWith("image/")) {
      appendMessage({ role: "system", text: `已跳过非图片文件：${file.name}` });
      continue;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      appendMessage({
        role: "system",
        text: `图片过大已跳过：${file.name}（${(file.size / 1024 / 1024).toFixed(1)}MB，上限 10MB）`,
      });
      continue;
    }
    try {
      const dataUrl = await readFileAsDataUrl(file);
      state.pendingImages.push({ name: file.name, dataUrl });
    } catch (err) {
      appendMessage({ role: "system", text: `读取失败：${file.name}（${err.message}）` });
    }
  }
  renderImagePreview();
}

function renderImagePreview() {
  els.imagePreview.innerHTML = "";
  state.pendingImages.forEach((img, idx) => {
    const item = document.createElement("div");
    item.className = "image-thumb";

    const picture = document.createElement("img");
    picture.src = img.dataUrl;
    picture.alt = img.name;
    item.appendChild(picture);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "image-remove";
    remove.textContent = "×";
    remove.title = `移除 ${img.name}`;
    remove.addEventListener("click", () => {
      state.pendingImages.splice(idx, 1);
      renderImagePreview();
    });
    item.appendChild(remove);

    els.imagePreview.appendChild(item);
  });
}

function clearPendingImages() {
  state.pendingImages = [];
  els.imageFiles.value = "";
  renderImagePreview();
}

function setBusy(busy) {
  state.busy = busy;
  if (busy) {
    els.send.textContent = "停止";
    els.send.type = "button";
    els.send.classList.remove("primary");
    els.send.classList.add("stop");
    els.send.disabled = false;
  } else {
    els.send.textContent = "发送";
    els.send.type = "submit";
    els.send.classList.add("primary");
    els.send.classList.remove("stop");
    els.send.disabled = false;
  }
}

function abortInflight() {
  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
}

function showThinking() {
  if (state.thinkingRow) return;
  const row = document.createElement("div");
  row.className = "message assistant thinking";
  row.dataset.transient = "1";
  const bubble = document.createElement("div");
  bubble.className = "bubble thinking-bubble";
  bubble.innerHTML =
    '<span class="thinking-dots"><span></span><span></span><span></span></span>' +
    '<span class="thinking-label">正在思考…</span>';
  row.appendChild(bubble);
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  state.thinkingRow = row;
}

function hideThinking() {
  if (state.thinkingRow) {
    state.thinkingRow.remove();
    state.thinkingRow = null;
  }
}

async function sendOnce(question, images, style) {
  const sessionId = await ensureSession();
  state.abortController = new AbortController();
  const res = await api("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, images, style, provider: els.provider.value }),
    signal: state.abortController.signal,
  });
  const data = await res.json();
  hideThinking();
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
  const { row } = appendMessage({ role: "assistant", text: data.answer || "(空响应)" });
  attachChunkRefs(row, state.lastChunks);
}

async function sendStream(question, images, style) {
  const sessionId = await ensureSession();
  state.abortController = new AbortController();
  const res = await fetch(`${API_BASE}/api/ask_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, images, style, provider: els.provider.value }),
    signal: state.abortController.signal,
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

  function ensureAnswerEl() {
    if (!answerEl) {
      const created = appendMessage({ role: "assistant", text: "" });
      answerEl = created.bubble;
      answerEl.classList.add("typing");
    }
    return answerEl;
  }

  function handleEvent(event, data) {
    hideThinking();
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

  try {
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
  } finally {
    if (answerEl) {
      answerEl.classList.remove("typing");
      const msgRow = answerEl.closest(".message");
      if (msgRow) attachChunkRefs(msgRow, state.lastChunks);
    }
  }
}

els.imageFiles.addEventListener("change", (e) => {
  handleImageFiles(e.target.files);
});

els.send.addEventListener("click", (e) => {
  if (state.busy) {
    e.preventDefault();
    abortInflight();
  }
});

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.busy) return;
  const question = els.question.value.trim();
  if (!question) return;
  const images = state.pendingImages.map((img) => img.dataUrl);
  const style = els.style.value || "concise";

  appendMessage({ role: "user", text: question, images });
  els.question.value = "";
  clearPendingImages();

  setBusy(true);
  showThinking();
  try {
    if (els.stream.checked) {
      await sendStream(question, images, style);
    } else {
      await sendOnce(question, images, style);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // User clicked stop — keep partial content, no error shown.
    } else {
      appendMessage({ role: "assistant", kind: "error", text: err.message || String(err) });
    }
  } finally {
    state.abortController = null;
    hideThinking();
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
    appendMessage({ role: "system", text: "已清空当前会话记忆。" });
  } catch (err) {
    appendMessage({ role: "assistant", kind: "error", text: err.message || String(err) });
  }
});

refreshHealth();
loadProviders();
ensureSession().catch((err) => {
  appendMessage({ role: "assistant", kind: "error", text: `初始化会话失败：${err.message}` });
});
