let currentChatId = null;
let currentConfig = null;
let debugTimer = null;
let tokenUsageTimer = null;
let configApiTestTimer = null;
let configApiTestRequestId = 0;

const $ = (selector) => document.querySelector(selector);

document.addEventListener("DOMContentLoaded", async () => {
  bindTabs();
  bindActions();
  await loadConfig();
  await loadChats();
  await loadTools();
  await loadModels(false);
  await loadDocs();
  await loadTokenUsage();
  startDebugTimer();
  startTokenUsageTimer();
});

function bindTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $("#" + button.dataset.tab).classList.add("active");
      if (button.dataset.tab === "debug") {
        loadDebug();
      }
      if (button.dataset.tab === "tools") {
        loadTools();
      }
    });
  });
}

function bindActions() {
  $("#new-chat").addEventListener("click", createChat);
  $("#delete-history").addEventListener("click", deleteAllChats);
  $("#message-form").addEventListener("submit", sendMessage);
  $("#save-config").addEventListener("click", saveConfig);
  $("#reload-config").addEventListener("click", reloadConfigFile);
  $("#reindex-docs").addEventListener("click", reindexDocs);
  $("#refresh-models").addEventListener("click", () => loadModels(true));
  $("#refresh-tools").addEventListener("click", loadTools);
  $("#refresh-debug").addEventListener("click", loadDebug);
  $("#test-api-now").addEventListener("click", () => runConfigApiTest());
  $("#main-model-select").addEventListener("change", () => setAgentModel("main", $("#main-model-select").value));
  $("#summarizer-model-select").addEventListener("change", () => setAgentModel("doc_summarizer", $("#summarizer-model-select").value));
  $("#api-key-input").addEventListener("input", () => scheduleConfigApiTest());
  $("#config-json").addEventListener("input", () => {
    updateConfigHighlight();
    scheduleConfigApiTest();
  });
  $("#config-json").addEventListener("scroll", syncConfigHighlightScroll);
  $("#raw-data-toggle").addEventListener("change", () => {
    if (!currentConfig) return;
    currentConfig.database.raw_data_enabled = $("#raw-data-toggle").checked;
    $("#config-json").value = JSON.stringify(currentConfig, null, 2);
    $("#tools-raw-data-toggle").checked = $("#raw-data-toggle").checked;
    updateConfigHighlight();
    loadTools();
  });
  $("#tools-raw-data-toggle").addEventListener("change", async () => {
    await patchRuntimeConfig({database: {raw_data_enabled: $("#tools-raw-data-toggle").checked}});
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

function scheduleConfigApiTest(delayMs = 5000) {
  if (configApiTestTimer) clearTimeout(configApiTestTimer);
  setApiTestStatus("pending", "Waiting", "API test will run 5 seconds after the last config edit.");
  configApiTestTimer = setTimeout(() => runConfigApiTest(), delayMs);
}

async function runConfigApiTest() {
  if (configApiTestTimer) {
    clearTimeout(configApiTestTimer);
    configApiTestTimer = null;
  }
  let config;
  try {
    config = JSON.parse($("#config-json").value);
  } catch (error) {
    setApiTestStatus("error", "Invalid JSON", error.message);
    return;
  }
  const apiKey = $("#api-key-input").value.trim();
  if (apiKey) {
    config.api = config.api || {};
    config.api.api_key = apiKey;
  }
  const requestId = ++configApiTestRequestId;
  setApiTestStatus("pending", "Testing", "Calling the configured /models endpoint...");
  try {
    const result = await api("/api/config/test-api", {
      method: "POST",
      body: JSON.stringify({config}),
    });
    if (requestId !== configApiTestRequestId) return;
    renderApiTestResult(result);
  } catch (error) {
    if (requestId !== configApiTestRequestId) return;
    setApiTestStatus("error", "Failed", error.message);
  }
}

function renderApiTestResult(result) {
  const label = result.ok ? "Connected" : "Failed";
  setApiTestStatus(result.ok ? "ok" : "error", label, JSON.stringify(result, null, 2));
}

function setApiTestStatus(kind, label, details) {
  const status = $("#api-test-status");
  if (!status) return;
  status.className = `status-pill ${kind}`;
  status.textContent = label;
  $("#api-test-details").textContent = details || "";
}

async function loadConfig() {
  currentConfig = await api("/api/config");
  $("#raw-data-toggle").checked = Boolean(currentConfig.database.raw_data_enabled);
  $("#config-json").value = JSON.stringify(currentConfig, null, 2);
  updateConfigHighlight();
  setApiTestStatus("neutral", "Not tested", "Edit config to test after 5 seconds, or click Test now.");
}

async function saveConfig() {
  let config;
  try {
    config = JSON.parse($("#config-json").value);
  } catch (error) {
    alert("Config JSON is invalid: " + error.message);
    return;
  }
  const apiKey = $("#api-key-input").value.trim();
  if (apiKey) {
    config.api = config.api || {};
    config.api.api_key = apiKey;
  }
  currentConfig = await api("/api/config", {
    method: "PUT",
    body: JSON.stringify({config}),
  });
  $("#api-key-input").value = "";
  $("#config-json").value = JSON.stringify(currentConfig, null, 2);
  updateConfigHighlight();
  setApiTestStatus("neutral", "Saved", "Config saved. Edit config to test again, or click Test now.");
  await loadTools();
  await loadModels(true);
}

async function reloadConfigFile() {
  currentConfig = await api("/api/config/reload", {method: "POST", body: "{}"});
  $("#config-json").value = JSON.stringify(currentConfig, null, 2);
  $("#raw-data-toggle").checked = Boolean(currentConfig.database.raw_data_enabled);
  $("#tools-raw-data-toggle").checked = Boolean(currentConfig.database.raw_data_enabled);
  updateConfigHighlight();
  setApiTestStatus("neutral", "Reloaded", "Config reloaded. Edit config to test again, or click Test now.");
  await loadTools();
  await loadModels(true);
}

async function loadTools() {
  if (!currentConfig) {
    await loadConfig();
  }
  const data = await api("/api/tools");
  $("#tools-view").textContent = JSON.stringify(data.tools, null, 2);
  renderToolControls(data.tools);
}

async function patchRuntimeConfig(patch) {
  currentConfig = await api("/api/config", {
    method: "PATCH",
    body: JSON.stringify({patch}),
  });
  $("#raw-data-toggle").checked = Boolean(currentConfig.database.raw_data_enabled);
  $("#tools-raw-data-toggle").checked = Boolean(currentConfig.database.raw_data_enabled);
  $("#config-json").value = JSON.stringify(currentConfig, null, 2);
  updateConfigHighlight();
  await loadTools();
  await loadModels(false);
}

function renderToolControls(tools) {
  const container = $("#tool-controls");
  if (!container) return;
  container.innerHTML = "";
  $("#tools-raw-data-toggle").checked = Boolean(currentConfig && currentConfig.database.raw_data_enabled);
  const enabledTools = new Set((((currentConfig || {}).agents || {}).main || {}).enabled_tools || []);
  tools.forEach((tool) => {
    const row = document.createElement("div");
    row.className = "tool-row" + (tool.available ? " available" : " unavailable");

    const label = document.createElement("label");
    label.className = "tool-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = tool.always_enabled || enabledTools.has(tool.name);
    checkbox.disabled = Boolean(tool.always_enabled);
    checkbox.addEventListener("change", () => setToolEnabled(tool.name, checkbox.checked));
    label.appendChild(checkbox);

    const name = document.createElement("strong");
    name.textContent = tool.name;
    label.appendChild(name);
    row.appendChild(label);

    const meta = document.createElement("div");
    meta.className = "tool-meta";
    const states = [];
    states.push(tool.available ? "accessible" : "not accessible");
    if (tool.always_enabled) states.push("always enabled");
    if (tool.requires_raw_data) states.push("requires raw row access");
    meta.textContent = states.join(" | ");
    row.appendChild(meta);

    const description = document.createElement("div");
    description.className = "tool-description";
    description.textContent = tool.description || "";
    row.appendChild(description);

    container.appendChild(row);
  });
}

async function setToolEnabled(toolName, enabled) {
  const enabledTools = new Set(currentConfig.agents.main.enabled_tools || []);
  if (enabled) {
    enabledTools.add(toolName);
  } else {
    enabledTools.delete(toolName);
  }
  await patchRuntimeConfig({agents: {main: {enabled_tools: Array.from(enabledTools)}}});
}

async function loadModels(refresh) {
  try {
    const data = await api(`/api/models?refresh=${refresh ? "true" : "false"}`);
    $("#models-view").textContent = JSON.stringify(data, null, 2);
    renderModelSelectors(data.models || []);
  } catch (error) {
    $("#models-view").textContent = "Model discovery failed: " + error.message;
    renderModelSelectors([]);
  }
}

function renderModelSelectors(models) {
  if (!currentConfig) return;
  populateModelSelect($("#main-model-select"), models, currentConfig.agents.main.model);
  populateModelSelect($("#summarizer-model-select"), models, currentConfig.agents.doc_summarizer.model);
}

function populateModelSelect(select, models, selectedModel) {
  if (!select) return;
  const values = [...models];
  if (selectedModel && !values.includes(selectedModel)) values.unshift(selectedModel);
  select.innerHTML = "";
  values.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = model === selectedModel;
    select.appendChild(option);
  });
}

async function setAgentModel(agentName, modelName) {
  const patch = {agents: {[agentName]: {model: modelName}}};
  if (isThinkingModel(modelName)) {
    patch.agents[agentName].reasoning_effort = "high";
  }
  await patchRuntimeConfig(patch);
}

function isThinkingModel(modelName) {
  const value = String(modelName || "").toLowerCase();
  return value.includes("gpt-oss") || value.includes("reason") || value.includes("thinking") || value.includes("o1") || value.includes("o3") || value.includes("o4");
}

function startTokenUsageTimer() {
  if (tokenUsageTimer) clearInterval(tokenUsageTimer);
  tokenUsageTimer = setInterval(loadTokenUsage, 5000);
}

async function loadTokenUsage() {
  try {
    const data = await api("/api/token-usage?window_minutes=10&average_minutes=2");
    renderTokenUsageWindow(data);
  } catch (error) {
    $("#usage-window-counter").textContent = "10m usage unavailable: " + error.message;
  }
}

function renderTokenUsageWindow(data) {
  const totals = data.totals || {};
  const projected = data.projected_per_10m_from_average || {};
  const percent = data.percentages_of_limit || {};
  const projectedPercent = data.projected_percentages_of_limit || {};
  const totalText = formatNumber(totals.total_tokens || 0);
  const promptText = formatNumber(totals.prompt_tokens || 0);
  const completionText = formatNumber(totals.completion_tokens || 0);
  const projectedText = formatNumber(projected.total_tokens || 0);
  const percentText = Number.isFinite(percent.total_tokens) ? ` (${percent.total_tokens}% limit)` : "";
  const projectedPercentText = Number.isFinite(projectedPercent.total_tokens) ? `, avg ${projectedPercent.total_tokens}% limit` : "";
  $("#usage-window-counter").textContent = `10m usage: ${totalText} total, ${promptText} in, ${completionText} out${percentText}; 2m avg projects ${projectedText}/10m${projectedPercentText}`;
  renderUsageBuckets(data);
}

function renderUsageBuckets(data) {
  const container = $("#usage-window-bar");
  if (!container) return;
  container.innerHTML = "";
  const buckets = data.buckets || [];
  const limit = ((data.limits_per_10m || {}).total_tokens || 0) / Math.max(1, buckets.length);
  const maxTotal = Math.max(1, ...buckets.map((bucket) => bucket.total_tokens || 0), limit || 0);
  buckets.forEach((bucket) => {
    const value = bucket.total_tokens || 0;
    const bar = document.createElement("div");
    const ratio = Math.max(0.08, Math.min(1, value / maxTotal));
    const limitRatio = limit ? value / limit : 0;
    bar.className = "usage-bucket" + (limitRatio >= 1 ? " high" : limitRatio >= 0.75 ? " medium" : "");
    bar.style.height = `${Math.round(ratio * 100)}%`;
    bar.title = `${formatNumber(value)} tokens, ${bucket.call_count || 0} API calls`;
    container.appendChild(bar);
  });
}

async function loadDocs() {
  const data = await api("/api/docs");
  $("#docs-view").textContent = JSON.stringify(data.docs, null, 2);
}

async function reindexDocs() {
  const result = await api("/api/docs/reload", {method: "POST", body: "{}"});
  await loadDocs();
  await loadDebug();
  alert(`Reloaded ${result.document_count} documents and rebuilt embeddings for ${result.chunk_count} chunks.`);
}

async function loadChats() {
  const data = await api("/api/chats");
  renderChatList(data.chats);
  if (data.chats.length && !currentChatId) {
    await selectChat(data.chats[0].id);
  }
}

function renderChatList(chats) {
  const list = $("#chat-list");
  list.innerHTML = "";
  chats.forEach((chat) => {
    const button = document.createElement("button");
    button.className = "chat-item" + (chat.id === currentChatId ? " active" : "");
    button.textContent = chat.title;
    button.title = chat.title;
    button.addEventListener("click", () => selectChat(chat.id));
    list.appendChild(button);
  });
}

async function createChat() {
  const data = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({title: "New chat"}),
  });
  currentChatId = data.chat.id;
  await loadChats();
  await selectChat(currentChatId);
}

async function deleteAllChats() {
  const confirmed = window.confirm("Delete all chat history and debug events? This cannot be undone.");
  if (!confirmed) return;
  const result = await api("/api/chats", {method: "DELETE"});
  currentChatId = result.chat.id;
  $("#messages").innerHTML = "";
  updateTokenCounter([]);
  await loadChats();
  await selectChat(currentChatId);
  await loadDebug();
}

async function selectChat(chatId) {
  currentChatId = chatId;
  const data = await api(`/api/chats/${chatId}`);
  renderMessages(data.messages);
  await loadChats();
  await loadDebug();
}

function renderMessages(messages) {
  const container = $("#messages");
  container.innerHTML = "";
  messages.forEach((message) => {
    container.appendChild(buildMessageElement(message));
  });
  updateTokenCounter(messages);
  container.scrollTop = container.scrollHeight;
}

async function sendMessage(event) {
  event.preventDefault();
  if (!currentChatId) {
    await createChat();
  }
  const input = $("#message-input");
  const content = input.value.trim();
  if (!content) return;
  input.value = "";
  const sendButton = $("#message-form button");
  sendButton.disabled = true;
  setLoading(true);
  appendLocalMessage({role: "user", content});
  const pending = appendPendingMessage();
  try {
    const result = await api(`/api/chats/${currentChatId}/messages`, {
      method: "POST",
      body: JSON.stringify({content}),
    });
    pending.replaceWith(buildMessageElement(result.message));
    updateTokenCounterFromDom();
    await loadChats();
    await loadDebug();
    await loadTokenUsage();
  } catch (error) {
    pending.replaceWith(buildMessageElement({role: "assistant", content: "Error: " + error.message, metadata: {error: true}}));
  } finally {
    sendButton.disabled = false;
    setLoading(false);
  }
}

function appendLocalMessage(message) {
  const container = $("#messages");
  const item = buildMessageElement(message);
  container.appendChild(item);
  updateTokenCounterFromDom();
  container.scrollTop = container.scrollHeight;
  return item;
}

function appendPendingMessage() {
  const item = buildMessageElement({
    role: "assistant",
    content: "",
    metadata: {pending: true},
  });
  $("#messages").appendChild(item);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return item;
}

function buildMessageElement(message) {
  const item = document.createElement("div");
  const metadata = message.metadata || {};
  item.className = "message " + message.role + (metadata.pending ? " pending" : "");
  item.dataset.metadata = JSON.stringify(metadata);
  const roleEl = document.createElement("span");
  roleEl.className = "role";
  roleEl.textContent = message.role;
  const contentEl = document.createElement("div");
  contentEl.className = "message-content markdown-body";
  if (metadata.pending) {
    contentEl.innerHTML = '<span class="spinner"></span><span>Thinking...</span>';
  } else if (message.role === "assistant") {
    contentEl.innerHTML = renderMarkdown(normalizeInlineCitations(message.content || "", metadata.citations || []));
  } else {
    contentEl.textContent = message.content || "";
  }
  item.appendChild(roleEl);
  item.appendChild(contentEl);
  const metaEl = buildMessageMetadata(metadata);
  if (metaEl) {
    item.appendChild(metaEl);
  }
  return item;
}

function buildMessageMetadata(metadata) {
  const hasUsage = metadata.token_usage && Object.keys(metadata.token_usage).length;
  const contextUsage = metadata.context_usage || null;
  const citations = metadata.citations || [];
  const retrievalDetails = metadata.retrieval_details || [];
  const subagentTrace = metadata.subagent_trace || [];
  const loopTrace = metadata.loop_trace || [];
  const reasoningSummary = metadata.reasoning_summary || [];
  if (
    !hasUsage &&
    !contextUsage &&
    !citations.length &&
    !retrievalDetails.length &&
    !subagentTrace.length &&
    !loopTrace.length &&
    !reasoningSummary.length
  ) return null;
  const wrapper = document.createElement("div");
  wrapper.className = "message-meta";
  if (hasUsage) {
    const usage = document.createElement("div");
    usage.className = "token-usage";
    usage.textContent = formatUsage(metadata.token_usage);
    wrapper.appendChild(usage);
  }
  if (contextUsage) {
    wrapper.appendChild(buildContextUsage(contextUsage));
  }
  if (reasoningSummary.length) {
    wrapper.appendChild(buildReasoningSummary(reasoningSummary));
  }
  if (retrievalDetails.length) {
    wrapper.appendChild(buildRetrievalDetails(retrievalDetails));
  }
  if (subagentTrace.length) {
    wrapper.appendChild(buildSubagentTrace(subagentTrace));
  }
  if (loopTrace.length) {
    wrapper.appendChild(buildLoopTrace(loopTrace));
  }
  if (citations.length) {
    const sources = document.createElement("details");
    sources.className = "citations";
    sources.open = true;
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${citations.length})`;
    sources.appendChild(summary);
    const list = document.createElement("ol");
    citations.forEach((citation) => {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = citation.title || "Untitled source";
      const path = document.createElement("span");
      path.className = "citation-path";
      const chunk = citation.chunk_index === null || citation.chunk_index === undefined ? "" : ` chunk ${citation.chunk_index}`;
      path.textContent = ` ${citation.source_path || ""}${chunk}`;
      item.appendChild(title);
      item.appendChild(path);
      if (citation.snippet) {
        const snippet = document.createElement("div");
        snippet.className = "citation-snippet";
        snippet.textContent = citation.snippet;
        item.appendChild(snippet);
      }
      list.appendChild(item);
    });
    sources.appendChild(list);
    wrapper.appendChild(sources);
  }
  return wrapper;
}

function buildSubagentTrace(subagentTrace) {
  const details = document.createElement("details");
  details.className = "subagent-trace";
  const summary = document.createElement("summary");
  summary.textContent = `Subagents (${subagentTrace.length}${formatStatusCounts(subagentTrace)})`;
  details.appendChild(summary);
  subagentTrace.forEach((agent, index) => {
    const block = document.createElement("div");
    block.className = `subagent-block ${agent.status || "unknown"}`;

    const header = document.createElement("div");
    header.className = "subagent-header";
    const title = document.createElement("strong");
    title.textContent = `${agent.agent || "subagent"} #${index + 1}`;
    const status = document.createElement("span");
    status.className = `subagent-status ${agent.status || "unknown"}`;
    status.textContent = agent.status || "unknown";
    header.appendChild(title);
    header.appendChild(status);
    block.appendChild(header);

    const task = document.createElement("div");
    task.textContent = agent.task || "No task recorded.";
    block.appendChild(task);

    const source = document.createElement("div");
    source.className = "subagent-source";
    const chunk = agent.chunk_index === null || agent.chunk_index === undefined ? "" : ` chunk ${agent.chunk_index}`;
    source.textContent = `Source: ${agent.source_title || agent.source_id || "unknown"} ${agent.source_path || ""}${chunk}`;
    block.appendChild(source);

    const model = document.createElement("div");
    model.className = "subagent-model";
    model.textContent = `Model: ${agent.model || "unknown"}${agent.reasoning_effort ? `, reasoning=${agent.reasoning_effort}` : ""}`;
    block.appendChild(model);

    if (agent.summarizer_prompt) {
      const prompt = document.createElement("div");
      prompt.className = "subagent-prompt";
      prompt.textContent = `Focus: ${agent.summarizer_prompt}`;
      block.appendChild(prompt);
    }

    const tools = document.createElement("div");
    tools.className = "subagent-tools";
    const enabled = agent.tools_enabled || [];
    const calls = agent.tool_calls || [];
    tools.textContent = `Tools enabled: ${enabled.length ? enabled.join(", ") : "none"}; tool calls: ${calls.length ? calls.map((call) => call.name || "unknown").join(", ") : "none"}`;
    block.appendChild(tools);

    if (Number.isFinite(agent.duration_ms) || agent.token_usage) {
      const metrics = document.createElement("div");
      metrics.className = "subagent-metrics";
      const duration = Number.isFinite(agent.duration_ms) ? `Duration: ${agent.duration_ms} ms` : "";
      const usage = agent.token_usage && Object.keys(agent.token_usage).length ? formatUsage(agent.token_usage) : "";
      metrics.textContent = [duration, usage].filter(Boolean).join("; ");
      block.appendChild(metrics);
    }

    if (agent.error) {
      const error = document.createElement("div");
      error.className = "subagent-error";
      error.textContent = `Error: ${agent.error}`;
      block.appendChild(error);
    }

    if (agent.summary) {
      const result = document.createElement("div");
      result.className = "subagent-summary markdown-body";
      result.innerHTML = renderMarkdown(agent.summary);
      block.appendChild(result);
    }

    if ((agent.conversation || []).length) {
      block.appendChild(buildSubagentConversation(agent.conversation));
    }

    details.appendChild(block);
  });
  return details;
}

function buildSubagentConversation(conversation) {
  const details = document.createElement("details");
  details.className = "subagent-conversation";
  const summary = document.createElement("summary");
  summary.textContent = `Conversation (${conversation.length})`;
  details.appendChild(summary);
  conversation.forEach((message) => {
    const item = document.createElement("div");
    item.className = `subagent-conversation-message ${message.role || "unknown"}`;
    const role = document.createElement("strong");
    role.textContent = message.role || "unknown";
    item.appendChild(role);
    const content = document.createElement("pre");
    content.textContent = message.content || "";
    item.appendChild(content);
    details.appendChild(item);
  });
  return details;
}

function formatStatusCounts(items) {
  const counts = {};
  items.forEach((item) => {
    const status = item.status || "unknown";
    counts[status] = (counts[status] || 0) + 1;
  });
  const text = Object.entries(counts)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([status, count]) => `${count} ${status}`)
    .join(", ");
  return text ? `: ${text}` : "";
}

function buildRetrievalDetails(retrievalDetails) {
  const details = document.createElement("details");
  details.className = "retrieval-details";
  const summary = document.createElement("summary");
  summary.textContent = `Retrieval details (${retrievalDetails.length})`;
  details.appendChild(summary);
  retrievalDetails.forEach((retrieval, index) => {
    const block = document.createElement("div");
    block.className = "retrieval-block";
    const title = document.createElement("strong");
    title.textContent = `Retrieval ${index + 1}: ${retrieval.source || "unknown"} search`;
    block.appendChild(title);
    const query = document.createElement("div");
    query.textContent = `Query: ${retrieval.query || ""}`;
    block.appendChild(query);
    if (retrieval.settings && Object.keys(retrieval.settings).length) {
      const settings = document.createElement("div");
      settings.className = "retrieval-settings";
      settings.textContent = `Settings: ${formatRetrievalSettings(retrieval.settings)}${Number.isFinite(retrieval.unfiltered_count) ? `; matched before score filter=${retrieval.unfiltered_count}` : ""}`;
      block.appendChild(settings);
    }
    if (retrieval.summarizer_prompt) {
      const prompt = document.createElement("div");
      prompt.className = "retrieval-summarizer-prompt";
      prompt.textContent = `Summarizer focus: ${retrieval.summarizer_prompt}`;
      block.appendChild(prompt);
    }
    if (retrieval.coverage && retrieval.coverage.enabled) {
      block.appendChild(buildCoverageDetails(retrieval.coverage));
    }
    if ((retrieval.warnings || []).length) {
      const warnings = document.createElement("div");
      warnings.className = "retrieval-warning";
      warnings.textContent = `Warnings: ${retrieval.warnings.join("; ")}`;
      block.appendChild(warnings);
    }
    const list = document.createElement("ol");
    (retrieval.chunks || []).forEach((chunk) => {
      const item = document.createElement("li");
      const score = Number.isFinite(chunk.score) ? ` score=${chunk.score.toFixed(3)}` : "";
      const chunkIndex = chunk.chunk_index === null || chunk.chunk_index === undefined ? "" : ` chunk ${chunk.chunk_index}`;
      item.innerHTML = `<strong>${escapeHtml(chunk.title || "Untitled")}</strong><span class="citation-path"> ${escapeHtml(chunk.source_path || "")}${chunkIndex}${score}</span>`;
      if (chunk.snippet) {
        const snippet = document.createElement("div");
        snippet.className = "citation-snippet";
        snippet.textContent = chunk.snippet;
        item.appendChild(snippet);
      }
      list.appendChild(item);
    });
    block.appendChild(list);
    if ((retrieval.summaries || []).length) {
      const summaries = document.createElement("div");
      summaries.className = "retrieval-summaries";
      summaries.textContent = `Subagent summaries: ${retrieval.summaries.length}`;
      block.appendChild(summaries);
    }
    details.appendChild(block);
  });
  return details;
}

function buildCoverageDetails(coverage) {
  const details = document.createElement("details");
  details.className = "coverage-details";
  details.open = true;
  const summary = document.createElement("summary");
  summary.textContent = `Coverage: ${coverage.chunks_examined || 0}/${coverage.candidate_count || 0} chunks, ${coverage.distinct_fact_count || 0} distinct facts, ${coverage.stop_reason || "unknown"}`;
  details.appendChild(summary);
  if (coverage.goal) {
    const goal = document.createElement("div");
    goal.className = "coverage-goal";
    goal.textContent = `Goal: ${coverage.goal}`;
    details.appendChild(goal);
  }
  if ((coverage.reference_followups || []).length) {
    const refs = document.createElement("div");
    refs.className = "coverage-references";
    refs.textContent = `Reference follow-ups: ${coverage.reference_followups.map((item) => `${item.reference} (${item.chunks_examined || 0} chunks, ${item.new_fact_count || 0} new facts)`).join("; ")}`;
    details.appendChild(refs);
  }
  if ((coverage.distinct_facts || []).length) {
    const list = document.createElement("ol");
    list.className = "coverage-facts";
    coverage.distinct_facts.slice(0, 12).forEach((fact) => {
      const item = document.createElement("li");
      item.textContent = fact.text || "";
      list.appendChild(item);
    });
    details.appendChild(list);
  }
  (coverage.waves || []).forEach((wave) => {
    const waveEl = document.createElement("div");
    waveEl.className = "coverage-wave";
    waveEl.textContent = `Wave ${wave.index}${wave.reference ? ` (${wave.reference})` : ""}: ${wave.chunk_count || 0} chunks, ${wave.new_fact_count || 0} new facts, ${wave.duplicate_fact_count || 0} duplicates.`;
    details.appendChild(waveEl);
  });
  return details;
}

function formatRetrievalSettings(settings) {
  const parts = [];
  if (settings.search_mode) parts.push(`mode=${settings.search_mode}`);
  if (Number.isFinite(settings.top_k)) parts.push(`top_k=${settings.top_k}`);
  if (Number.isFinite(settings.min_score)) parts.push(`min_score=${settings.min_score}`);
  if ((settings.include_terms || []).length) parts.push(`include=${settings.include_terms.join(", ")}`);
  if ((settings.exclude_terms || []).length) parts.push(`exclude=${settings.exclude_terms.join(", ")}`);
  return parts.join("; ");
}

function buildReasoningSummary(reasoningSummary) {
  const details = document.createElement("details");
  details.className = "reasoning-summary";
  const summary = document.createElement("summary");
  summary.textContent = `Reasoning summary (${reasoningSummary.length})`;
  details.appendChild(summary);
  const note = document.createElement("div");
  note.className = "reasoning-note";
  note.textContent = "Safe summary of observable decisions; hidden chain-of-thought is not exposed.";
  details.appendChild(note);
  const list = document.createElement("ol");
  reasoningSummary.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
  details.appendChild(list);
  return details;
}

function buildLoopTrace(loopTrace) {
  const trace = document.createElement("details");
  trace.className = "loop-trace";
  const summary = document.createElement("summary");
  summary.textContent = `Agent loop (${loopTrace.length})`;
  trace.appendChild(summary);
  loopTrace.forEach((step) => {
    const item = document.createElement("div");
    item.className = "loop-step";
    const title = document.createElement("strong");
    title.textContent = `Iteration ${step.iteration}`;
    item.appendChild(title);
    if (step.assistant_content) {
      const content = document.createElement("div");
      content.className = "loop-content markdown-body";
      content.innerHTML = renderMarkdown(step.assistant_content);
      item.appendChild(content);
    }
    (step.tool_calls || []).forEach((call) => {
      const callEl = document.createElement("div");
      callEl.className = "loop-tool";
      callEl.textContent = `Tool call: ${call.name || "unknown"} ${JSON.stringify(call.arguments || {})}`;
      item.appendChild(callEl);
    });
    (step.tool_results || []).forEach((result) => {
      const resultEl = document.createElement("div");
      resultEl.className = result.ok === false ? "loop-result error" : "loop-result";
      resultEl.textContent = `Tool result: ${result.name || "unknown"} ${result.ok === false ? "failed" : "ok"} - ${result.summary || result.error || ""}`;
      item.appendChild(resultEl);
    });
    (step.notices || []).forEach((notice) => {
      const noticeEl = document.createElement("div");
      noticeEl.className = "loop-notice";
      noticeEl.textContent = notice;
      item.appendChild(noticeEl);
    });
    trace.appendChild(item);
  });
  return trace;
}

function formatUsage(usage) {
  const prompt = usage.prompt_tokens || 0;
  const completion = usage.completion_tokens || 0;
  const total = usage.total_tokens || prompt + completion;
  const reasoning = usage.reasoning_tokens ? `, reasoning ${usage.reasoning_tokens}` : "";
  return `Tokens: prompt ${prompt}, completion ${completion}, total ${total}${reasoning}`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function buildContextUsage(contextUsage) {
  const item = document.createElement("div");
  item.className = "context-usage";
  item.textContent = formatContextUsage(contextUsage);
  return item;
}

function formatContextUsage(contextUsage) {
  const used = contextUsage.estimated_tokens || 0;
  const budget = contextUsage.context_budget || 0;
  const windowSize = contextUsage.context_window || 0;
  const percent = Number.isFinite(contextUsage.percent_used) ? contextUsage.percent_used : 0;
  const dropped = contextUsage.history_messages_dropped || 0;
  const droppedText = dropped ? `, dropped ${dropped} history messages` : "";
  return `Context: ${used} / ${budget} input-budget tokens (${percent}%) of ${windowSize} window${droppedText}`;
}

function updateTokenCounter(messages) {
  const aggregate = {};
  let latestContext = null;
  messages.forEach((message) => {
    mergeUsage(aggregate, message.metadata && message.metadata.token_usage);
    if (message.metadata && message.metadata.context_usage) {
      latestContext = message.metadata.context_usage;
    }
  });
  $("#token-counter").textContent = formatUsage(aggregate);
  $("#context-counter").textContent = latestContext ? formatContextUsage(latestContext) : "Context: 0 / 0 tokens";
}

function updateTokenCounterFromDom() {
  const aggregate = {};
  let latestContext = null;
  document.querySelectorAll(".message").forEach((item) => {
    try {
      const metadata = JSON.parse(item.dataset.metadata || "{}");
      mergeUsage(aggregate, metadata.token_usage);
      if (metadata.context_usage) {
        latestContext = metadata.context_usage;
      }
    } catch {
      // Ignore malformed local metadata.
    }
  });
  $("#token-counter").textContent = formatUsage(aggregate);
  $("#context-counter").textContent = latestContext ? formatContextUsage(latestContext) : "Context: 0 / 0 tokens";
}

function mergeUsage(target, usage) {
  if (!usage) return;
  Object.entries(usage).forEach(([key, value]) => {
    if (Number.isFinite(value)) {
      target[key] = (target[key] || 0) + value;
    }
  });
}

function setLoading(isLoading) {
  $("#loading-indicator").classList.toggle("hidden", !isLoading);
}

function renderMarkdown(text) {
  const codeBlocks = [];
  let escaped = escapeHtml(text || "");
  escaped = escaped.replace(/```([\s\S]*?)```/g, (_, code) => {
    const token = `@@CODE_BLOCK_${codeBlocks.length}@@`;
    codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return token;
  });
  escaped = escaped.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  escaped = escaped.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  escaped = escaped.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  escaped = escaped.replace(/^&gt; (.*)$/gm, "<blockquote>$1</blockquote>");
  escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  escaped = escaped.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  escaped = escaped.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  escaped = renderLists(escaped);
  escaped = escaped
    .split(/\n{2,}/)
    .map((block) => {
      if (block.trim().startsWith("@@CODE_BLOCK_")) return block;
      if (/^<(h\d|ul|ol|blockquote|pre)/.test(block.trim())) return block;
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
  codeBlocks.forEach((html, index) => {
    escaped = escaped.replace(`@@CODE_BLOCK_${index}@@`, html);
  });
  return escaped;
}

function renderLists(html) {
  const lines = html.split("\n");
  const rendered = [];
  let listType = null;
  for (const line of lines) {
    const unordered = line.match(/^[-*] (.+)$/);
    const ordered = line.match(/^\d+\. (.+)$/);
    const nextType = unordered ? "ul" : ordered ? "ol" : null;
    if (nextType && listType !== nextType) {
      if (listType) rendered.push(`</${listType}>`);
      rendered.push(`<${nextType}>`);
      listType = nextType;
    }
    if (!nextType && listType) {
      rendered.push(`</${listType}>`);
      listType = null;
    }
    if (unordered || ordered) {
      rendered.push(`<li>${(unordered || ordered)[1]}</li>`);
    } else {
      rendered.push(line);
    }
  }
  if (listType) rendered.push(`</${listType}>`);
  return rendered.join("\n");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function normalizeInlineCitations(text, citations) {
  if (!text || !citations.length) return text;
  let index = 0;
  const nextCitationMarker = "__next_citation__";
  const titleLookup = new Map(
    citations
      .map((citation, citationIndex) => [normalizeCitationLabel(citation.title), citationIndex + 1])
      .filter(([title]) => title)
  );
  return text.replace(/[ \t]*【\s*([^】]+?)\s*】/g, (match, label, offset, fullText) => {
    let marker = citationMarkerForLabel(label, citations.length, titleLookup);
    if (marker === nextCitationMarker) {
      index += 1;
      marker = `[${Math.min(index, citations.length)}]`;
    } else if (!marker) {
      return match;
    }
    const previous = offset > 0 ? fullText[offset - 1] : "";
    const prefix = previous && !/\s/.test(previous) && !"([{".includes(previous) ? " " : "";
    return `${prefix}${marker}`;
  });
}

function citationMarkerForLabel(label, citationCount, titleLookup) {
  const normalized = normalizeCitationLabel(label);
  if (/^(source|sources|citation|citations|cite)(\s*[:#]?\s*\d+)?$/.test(normalized) || normalized.includes("source")) {
    return "__next_citation__";
  }
  if (/^\d+(\s*,\s*\d+)*$/.test(normalized)) {
    const values = normalized
      .split(",")
      .map((value) => Number.parseInt(value.trim(), 10))
      .filter((value) => Number.isFinite(value) && value >= 1 && value <= citationCount);
    const unique = [...new Set(values)];
    return unique.length ? `[${unique.join(", ")}]` : "__next_citation__";
  }
  const numericPrefix = normalized.match(/^(\d+)(?:\D.*)?$/);
  if (numericPrefix) {
    const value = Number.parseInt(numericPrefix[1], 10);
    if (Number.isFinite(value) && value >= 1 && value <= citationCount) {
      return `[${value}]`;
    }
  }
  if (titleLookup.has(normalized)) {
    return `[${titleLookup.get(normalized)}]`;
  }
  const stripped = normalized.replace(/^(source|citation|cite)\s*:\s*/, "");
  if (titleLookup.has(stripped)) {
    return `[${titleLookup.get(stripped)}]`;
  }
  return null;
}

function normalizeCitationLabel(value) {
  return String(value || "").toLowerCase().trim().replace(/\s+/g, " ");
}

function updateConfigHighlight() {
  const raw = $("#config-json").value;
  const target = $("#config-highlight");
  const editorHighlight = $("#config-editor-highlight");
  const highlighted = highlightJson(raw);
  if (editorHighlight) {
    editorHighlight.innerHTML = highlighted + "\n";
  }
  try {
    JSON.parse(raw);
    target.classList.remove("json-invalid");
    if (editorHighlight) editorHighlight.classList.remove("json-invalid");
    target.innerHTML = highlighted;
  } catch (error) {
    target.classList.add("json-invalid");
    if (editorHighlight) editorHighlight.classList.add("json-invalid");
    target.innerHTML = `<span class="json-error">Invalid JSON: ${escapeHtml(error.message)}</span>\n\n${escapeHtml(raw)}`;
  }
  syncConfigHighlightScroll();
}

function syncConfigHighlightScroll() {
  const input = $("#config-json");
  const highlight = $("#config-editor-highlight");
  if (!input || !highlight) return;
  highlight.scrollTop = input.scrollTop;
  highlight.scrollLeft = input.scrollLeft;
}

function highlightJson(jsonText) {
  const tokenPattern = /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g;
  const parts = [];
  let lastIndex = 0;
  for (const match of jsonText.matchAll(tokenPattern)) {
    parts.push(escapeHtml(jsonText.slice(lastIndex, match.index)));
    let className = "json-number";
    if (match[1]) {
      className = match[2] ? "json-key" : "json-string";
    } else if (match[3] === "true" || match[3] === "false") {
      className = "json-boolean";
    } else if (match[3] === "null") {
      className = "json-null";
    }
    parts.push(`<span class="${className}">${escapeHtml(match[0])}</span>`);
    lastIndex = match.index + match[0].length;
  }
  parts.push(escapeHtml(jsonText.slice(lastIndex)));
  return parts.join("");
}

function startDebugTimer() {
  if (debugTimer) clearInterval(debugTimer);
  debugTimer = setInterval(() => {
    if ($("#auto-debug").checked) {
      loadDebug();
    }
  }, 2000);
}

async function loadDebug() {
  const path = currentChatId ? `/api/chats/${currentChatId}/debug` : "/api/debug";
  const data = await api(path);
  renderDebug(data.events);
}

function renderDebug(events) {
  const list = $("#debug-events");
  list.innerHTML = "";
  events.slice().reverse().forEach((event) => {
    const item = document.createElement("div");
    item.className = "debug-event";
    const title = document.createElement("strong");
    title.textContent = event.event_type;
    const time = document.createElement("small");
    time.textContent = `${event.created_at} run=${event.run_id || ""}`;
    const summary = document.createElement("div");
    summary.textContent = summarizePayload(event.payload);
    item.appendChild(title);
    item.appendChild(time);
    item.appendChild(summary);
    list.appendChild(item);
  });
  $("#debug-json").textContent = JSON.stringify(events, null, 2);
}

function summarizePayload(payload) {
  if (!payload) return "";
  if (payload.error) return String(payload.error);
  if (payload.name && payload.result) return `${payload.name}: ${payload.result.summary || payload.result.error || ""}`;
  if (payload.agent && payload.status) return `${payload.agent}: ${payload.status} - ${payload.task || payload.source_title || ""}`;
  if (payload.agent && payload.usage) return `${payload.agent}: ${JSON.stringify(payload.usage)}`;
  return JSON.stringify(payload).slice(0, 240);
}
