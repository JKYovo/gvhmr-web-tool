const rootPath = window.location.pathname.endsWith("/") ? window.location.pathname : `${window.location.pathname}/`;
const api = (path) => `${rootPath}${path.replace(/^\/+/, "")}`;

const state = {
  capabilities: null,
  jobs: [],
  selectedJobId: new URLSearchParams(window.location.search).get("job"),
  selectedJob: null,
  activeMode: "single",
  polling: null,
  refreshing: false,
  submitting: { single: false, batch: false },
  activeAction: null,
  sonicSpeed: 1.0,
  thumbnailObserver: null,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message, isError = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 5200);
}

function showFeedback(id, message, isError = false) {
  const element = $(id);
  if (!element) return;
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.hidden = !message;
}

async function request(path, options = {}) {
  const response = await fetch(api(path), options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    } catch (_) {
      detail = (await response.text()) || detail;
    }
    throw new Error(detail);
  }
  return response.json();
}

function statusClass(status) {
  return String(status || "queued").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
}

function statusLabel(status) {
  return {
    queued: "等待中",
    running: "处理中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
  }[status] || status || "未知";
}

function statusPill(status) {
  return `<span class="status ${statusClass(status)}">${escapeHtml(statusLabel(status))}</span>`;
}

function fileName(job) {
  return job?.display_name || (job?.input_video || "").split("/").pop() || "-";
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function artifactUrl(job, key, inline = false) {
  const suffix = inline ? "?inline=true" : "";
  return api(`jobs/${job.job_id}/artifact/${key}${suffix}`);
}

function hasArtifact(job, key) {
  const artifacts = job?.artifacts || {};
  const map = {
    hmr4d_results: artifacts.hmr4d_results_path,
    raw_hmr4d_results: artifacts.raw_hmr4d_results_path,
    global_contact_results: artifacts.global_contact_results_path,
    flat_ground_y_results: artifacts.flat_ground_y_results_path,
    ground_constraint_metrics: artifacts.ground_constraint_metrics_path,
    sonic_reference: artifacts.sonic_reference_path,
    sonic_metadata: artifacts.sonic_metadata_path,
    incam_video: artifacts.incam_video_path,
    global_video: artifacts.global_video_path,
    preview_video: artifacts.preview_video_path,
    zip: artifacts.artifacts_zip_path,
  };
  return Boolean(map[key]);
}

function formDataFrom(form, fileInputName) {
  const data = new FormData(form);
  for (const [key, value] of [...data.entries()]) {
    if (value === "on") data.set(key, "true");
    if (key === "f_mm" && value === "") data.delete(key);
  }
  for (const checkbox of form.querySelectorAll('input[type="checkbox"]')) {
    if (!checkbox.checked && checkbox.name) data.set(checkbox.name, "false");
  }
  if (fileInputName) data.delete(fileInputName);
  return data;
}

function appendFiles(data, input, fieldName) {
  for (const file of input.files) data.append(fieldName, file);
}

function supportedExtensions() {
  return new Set(state.capabilities?.video_extensions || [".avi", ".mkv", ".mov", ".mp4", ".webm"]);
}

function fileExtension(file) {
  const index = file.name.lastIndexOf(".");
  return index >= 0 ? file.name.slice(index).toLowerCase() : "";
}

function runtimeReady() {
  const runtime = state.capabilities?.runtime;
  return runtime ? Boolean(runtime.inference_ready) : Boolean(state.capabilities);
}

function updateSubmitState(mode) {
  const button = mode === "batch" ? $("batchSubmit") : $("singleSubmit");
  const busy = state.submitting[mode];
  const ready = runtimeReady();
  button.disabled = busy || !ready;
  button.textContent = busy
    ? "正在上传..."
    : mode === "batch" ? "提交批量处理" : "提交视频处理";
  if (!ready) {
    button.title = "GPU 或模型资源尚未就绪";
  } else {
    button.removeAttribute("title");
  }
}

function renderCapabilities() {
  const cap = state.capabilities;
  const runtime = cap?.runtime || {};
  const badges = [
    `<span class="badge ready">服务正常</span>`,
    runtime.gpu_available
      ? `<span class="badge ready" title="${escapeHtml(runtime.gpu_name || "CUDA GPU")}">GPU 就绪</span>`
      : `<span class="badge error" title="${escapeHtml(runtime.gpu_error || "CUDA 不可用")}">GPU 不可用</span>`,
    runtime.assets_ready
      ? `<span class="badge ready">模型就绪</span>`
      : `<span class="badge warning" title="${escapeHtml((runtime.missing_assets || []).join("、"))}">模型缺失</span>`,
  ];
  if (cap?.gmr_bridge_available) badges.push(`<span class="badge ready">GMR 可用</span>`);
  $("capabilityBadges").innerHTML = badges.join("");
  const groundConfig = cap?.ground_constraints || {};
  const groundOptions = new Map((groundConfig.options || []).map((item) => [item.value, item]));
  document.querySelectorAll('input[name="ground_constraint"]').forEach((input) => {
    const option = groundOptions.get(input.value);
    if (option) input.disabled = !option.enabled;
  });
  const groundDefault = groundConfig.default || "none";
  document.querySelectorAll("#singleForm, #batchForm").forEach((form) => {
    const selected = form.querySelector(`input[name="ground_constraint"][value="${groundDefault}"]:not(:disabled)`)
      || form.querySelector('input[name="ground_constraint"][value="none"]');
    if (selected) selected.checked = true;
  });
  updateSubmitState("single");
  updateSubmitState("batch");

  if (!runtimeReady()) {
    const missing = (runtime.missing_assets || []).join("、");
    const reason = !runtime.gpu_available ? "CUDA GPU 不可用" : `缺少模型：${missing || "未知资源"}`;
    showFeedback("singleFormFeedback", `当前不能提交推理：${reason}`, true);
    showFeedback("batchFormFeedback", `当前不能提交推理：${reason}`, true);
  }
}

function setMode(mode) {
  state.activeMode = mode === "batch" ? "batch" : "single";
  document.querySelectorAll("[data-mode-tab]").forEach((tab) => {
    const active = tab.dataset.modeTab === state.activeMode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-mode-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.modePanel !== state.activeMode;
  });
}

function setupFileDrops() {
  document.querySelectorAll(".file-drop").forEach((zone) => {
    const input = zone.querySelector('input[type="file"]');
    if (!input) return;
    ["dragenter", "dragover"].forEach((name) => {
      zone.addEventListener(name, (event) => {
        event.preventDefault();
        zone.classList.add("drag-active");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      zone.addEventListener(name, (event) => {
        event.preventDefault();
        zone.classList.remove("drag-active");
      });
    });
    zone.addEventListener("drop", (event) => {
      const files = [...(event.dataTransfer?.files || [])];
      if (!files.length) return;
      const transfer = new DataTransfer();
      for (const file of input.multiple ? files : files.slice(0, 1)) transfer.items.add(file);
      input.files = transfer.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
  });
}

function loadVisibleJobThumbnails() {
  state.thumbnailObserver?.disconnect();
  const root = $("jobList");
  const images = [...root.querySelectorAll("img[data-thumbnail-src]")];
  const load = (image) => {
    const source = image.dataset.thumbnailSrc;
    if (!source) return;
    delete image.dataset.thumbnailSrc;
    image.addEventListener("load", () => image.closest(".job-thumb")?.classList.add("has-image"), { once: true });
    image.addEventListener("error", () => image.remove(), { once: true });
    image.src = source;
  };
  if (!("IntersectionObserver" in window)) {
    images.forEach(load);
    return;
  }
  state.thumbnailObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      state.thumbnailObserver?.unobserve(entry.target);
      load(entry.target);
    });
  }, { root, rootMargin: "80px 0px" });
  images.forEach((image) => state.thumbnailObserver.observe(image));
}

async function submitSingle(event) {
  event.preventDefault();
  if (state.submitting.single) return;
  const form = event.currentTarget;
  const input = $("singleFile");
  const file = input.files[0];
  if (!file) return showFeedback("singleFormFeedback", "请先选择一个视频。", true);
  if (!supportedExtensions().has(fileExtension(file))) {
    return showFeedback("singleFormFeedback", `不支持的视频格式：${fileExtension(file) || "无扩展名"}`, true);
  }

  const data = formDataFrom(form, "file");
  data.append("file", file);
  state.submitting.single = true;
  updateSubmitState("single");
  showFeedback("singleFormFeedback", `正在上传 ${file.name}，请不要重复点击。`);
  try {
    const job = await request("api/jobs/upload", { method: "POST", body: data });
    state.selectedJobId = job.job_id;
    showFeedback("singleFormFeedback", `任务已提交：${job.job_id}`);
    showToast(`已提交视频：${file.name}`);
    await refreshJobs();
    ensurePolling();
  } catch (error) {
    showFeedback("singleFormFeedback", `提交失败：${error.message}`, true);
    showToast(`提交失败：${error.message}`, true);
  } finally {
    state.submitting.single = false;
    updateSubmitState("single");
  }
}

async function submitBatch(event) {
  event.preventDefault();
  if (state.submitting.batch) return;
  const form = event.currentTarget;
  const input = $("batchFiles");
  if (!input.files.length) return showFeedback("batchFormFeedback", "请先选择至少一个视频。", true);

  const data = formDataFrom(form, "files");
  appendFiles(data, input, "files");
  state.submitting.batch = true;
  updateSubmitState("batch");
  showFeedback("batchFormFeedback", `正在上传 ${input.files.length} 个视频，请不要重复点击。`);
  try {
    const result = await request("api/jobs/batch-upload", { method: "POST", body: data });
    const count = result.jobs?.length || 0;
    const errorCount = result.errors?.length || 0;
    state.selectedJobId = result.jobs?.[0]?.job_id || state.selectedJobId;
    const summary = `已提交 ${count} 个任务${errorCount ? `，${errorCount} 个文件未提交` : ""}`;
    showFeedback("batchFormFeedback", summary, errorCount > 0 && count === 0);
    showToast(summary, errorCount > 0 && count === 0);
    await refreshJobs();
    ensurePolling();
  } catch (error) {
    showFeedback("batchFormFeedback", `批量提交失败：${error.message}`, true);
    showToast(`批量提交失败：${error.message}`, true);
  } finally {
    state.submitting.batch = false;
    updateSubmitState("batch");
  }
}

function jobNeedsPolling(job) {
  return ["queued", "running"].includes(job?.status)
    || ["queued", "running"].includes(job?.preview_status)
    || ["preparing", "streaming"].includes(job?.sonic_status);
}

function currentProgress(job) {
  if (!job) return null;
  if (["queued", "running"].includes(job.preview_status)) {
    return {
      percent: Number(job.preview_progress_percent || 0),
      stage: job.preview_progress_stage || (job.preview_status === "queued" ? "等待生成预览" : "生成预览"),
    };
  }
  if (["queued", "running"].includes(job.status)) {
    return {
      percent: Number(job.progress_percent || 0),
      stage: job.progress_stage || (job.status === "queued" ? "等待 GPU" : "GVHMR 处理中"),
    };
  }
  return null;
}

function normalizedProgress(progress) {
  return Math.max(0, Math.min(100, Math.round(Number(progress?.percent) || 0)));
}

function ensurePolling() {
  if (state.polling || !state.jobs.some(jobNeedsPolling)) return;
  state.polling = setInterval(async () => {
    try {
      await refreshJobs({ silent: true });
    } catch (_) {
      stopPolling();
    }
  }, 2200);
}

function stopPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = null;
}

function syncPolling() {
  if (state.jobs.some(jobNeedsPolling)) ensurePolling();
  else stopPolling();
}

async function refreshJobs({ silent = false } = {}) {
  if (state.refreshing) return;
  state.refreshing = true;
  const refreshButton = $("refreshJobs");
  if (!silent) refreshButton.disabled = true;
  try {
    const jobs = await request("jobs?limit=80");
    state.jobs = jobs;
    if (!state.selectedJobId && jobs.length) state.selectedJobId = jobs[0].job_id;

    let selected = jobs.find((job) => job.job_id === state.selectedJobId) || null;
    if (!selected && state.selectedJobId) {
      try {
        selected = await request(`jobs/${state.selectedJobId}`);
        state.jobs.unshift(selected);
      } catch (_) {
        state.selectedJobId = jobs[0]?.job_id || null;
        selected = jobs[0] || null;
      }
    }
    state.selectedJob = selected;
    renderJobs();
    renderJobDetail(selected);
    updateJobUrl(selected?.job_id || null);
    syncPolling();
  } finally {
    state.refreshing = false;
    refreshButton.disabled = false;
  }
}

function renderJobs() {
  const query = $("searchBox").value.trim().toLowerCase();
  const status = $("statusFilter").value;
  const jobs = state.jobs.filter((job) => {
    const haystack = `${job.job_id} ${fileName(job)}`.toLowerCase();
    return (!query || haystack.includes(query)) && (status === "all" || job.status === status);
  });

  const cards = jobs.map((job) => {
    const progress = currentProgress(job);
    const percent = normalizedProgress(progress);
    const initial = (fileName(job).trim()[0] || "V").toUpperCase();
    return `
      <article class="job-card ${job.job_id === state.selectedJobId ? "active" : ""}" data-job="${escapeHtml(job.job_id)}" tabindex="0">
        <div class="job-thumb" aria-hidden="true"><img data-thumbnail-src="${api(`jobs/${job.job_id}/thumbnail`)}" alt=""><span>${escapeHtml(initial)}</span></div>
        <div class="job-body">
          <div class="job-top">
            <div class="job-name" title="${escapeHtml(fileName(job))}">${escapeHtml(fileName(job))}</div>
            ${statusPill(job.status)}
          </div>
          <div class="job-meta"><span>${escapeHtml(job.job_id)}</span><time>${escapeHtml(formatTime(job.submitted_at))}</time></div>
          ${progress ? `<div class="job-progress-row"><div class="job-progress" role="progressbar" aria-label="${escapeHtml(progress.stage)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}"><span style="width:${percent}%"></span></div><strong>${percent}%</strong></div>` : ""}
        </div>
      </article>`;
  }).join("");
  $("jobList").innerHTML = cards || `<div class="job-empty">${state.jobs.length ? "没有符合筛选条件的任务" : "暂无任务"}</div>`;
  loadVisibleJobThumbnails();

  document.querySelectorAll(".job-card[data-job]").forEach((card) => {
    card.addEventListener("click", () => selectJob(card.dataset.job));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectJob(card.dataset.job);
      }
    });
  });
  if (window.matchMedia("(min-width: 821px) and (min-height: 761px)").matches) {
    requestAnimationFrame(() => {
      const activeCard = $("jobList").querySelector(".job-card.active");
      if (activeCard) activeCard.scrollIntoView({ block: "nearest" });
    });
  }
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  const cached = state.jobs.find((job) => job.job_id === jobId);
  if (cached) {
    state.selectedJob = cached;
    renderJobs();
    renderJobDetail(cached);
    updateJobUrl(jobId);
  }
  try {
    const job = await request(`jobs/${jobId}`);
    const index = state.jobs.findIndex((item) => item.job_id === jobId);
    if (index >= 0) state.jobs[index] = job;
    else state.jobs.unshift(job);
    state.selectedJob = job;
    renderJobs();
    renderJobDetail(job);
    updateJobUrl(jobId);
    syncPolling();
  } catch (error) {
    showToast(`读取任务失败：${error.message}`, true);
  }
}

function updateJobUrl(jobId) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState(null, "", url);
}

function metric(label, value) {
  const text = String(value ?? "-");
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(text)}">${escapeHtml(text)}</strong></div>`;
}

function renderDownloads(job) {
  const summary = $("downloadsSummary");
  if (!job) {
    summary.textContent = "结果下载";
    $("downloads").innerHTML = `<div class="download-empty">任务完成后显示下载项</div>`;
    return;
  }
  const items = [
    ["hmr4d_results", "当前动作结果 PT"],
    ["raw_hmr4d_results", "原始 FootMR PT"],
    ["global_contact_results", "Global V1.1 结果 PT"],
    ["ground_constraint_metrics", "地面约束指标 JSON"],
    ["sonic_reference", "SONIC Reference NPZ"],
    ["sonic_metadata", "SONIC 转换信息 JSON"],
    ["preview_video", "对比预览 MP4"],
    ["incam_video", "相机视角 MP4"],
    ["global_video", "全局视角 MP4"],
  ].filter(([key]) => hasArtifact(job, key));
  const links = items.map(([key, label]) => (
    `<a class="download" href="${artifactUrl(job, key)}">${escapeHtml(label)}</a>`
  ));
  if (hasArtifact(job, "hmr4d_results")) {
    links.push(`<a class="download" href="${api(`jobs/${job.job_id}/artifacts`)}">全部结果 ZIP</a>`);
  }
  summary.textContent = links.length ? `结果下载 (${links.length})` : "结果下载";
  $("downloads").innerHTML = links.join("") || `<div class="download-empty">当前任务还没有可下载产物</div>`;
}

function renderPreview(job) {
  const video = $("previewVideo");
  const empty = $("previewEmpty");
  if (job && hasArtifact(job, "preview_video")) {
    const artifactPath = job.artifacts.preview_video_path;
    if (video.dataset.artifactPath !== artifactPath) {
      video.dataset.artifactPath = artifactPath;
      video.src = artifactUrl(job, "preview_video", true);
      video.load();
    }
    video.hidden = false;
    empty.hidden = true;
    requestAnimationFrame(fitPreviewStage);
    return;
  }

  if (video.dataset.artifactPath) {
    video.pause();
    video.removeAttribute("src");
    video.load();
    delete video.dataset.artifactPath;
  }
  video.hidden = true;
  empty.hidden = false;
  resetPreviewStageSize();
  const title = empty.querySelector("strong");
  const note = empty.querySelector("span");
  if (job?.preview_status === "running") {
    title.textContent = "正在生成预览";
    note.textContent = "渲染完成后会自动显示";
  } else if (job?.preview_status === "queued") {
    title.textContent = "预览等待中";
    note.textContent = "当前任务已加入预览队列";
  } else if (job?.preview_status === "failed") {
    title.textContent = "预览生成失败";
    note.textContent = "推理结果仍然可用，可以重新生成预览";
  } else {
    title.textContent = "暂无预览";
    note.textContent = job?.status === "succeeded" ? "点击生成预览查看相机与全局视角" : "推理完成后可按需生成对比视频";
  }
}

function resetPreviewStageSize() {
  const stage = $("previewStage");
  stage.style.removeProperty("width");
  stage.style.removeProperty("height");
}

function fitPreviewStage() {
  const slot = $("previewSlot");
  const stage = $("previewStage");
  const video = $("previewVideo");
  if (video.hidden || !video.videoWidth || !video.videoHeight) {
    resetPreviewStageSize();
    return;
  }

  const availableWidth = slot.clientWidth;
  const availableHeight = slot.clientHeight;
  if (!availableWidth || !availableHeight) return;

  const aspectRatio = video.videoWidth / video.videoHeight;
  let width = availableWidth;
  let height = width / aspectRatio;
  if (height > availableHeight) {
    height = availableHeight;
    width = height * aspectRatio;
  }
  stage.style.width = `${Math.max(1, Math.floor(width))}px`;
  stage.style.height = `${Math.max(1, Math.floor(height))}px`;
}

function renderPaths(job) {
  if (!job) {
    $("jobPaths").innerHTML = `<div class="download-empty">请选择任务</div>`;
    return;
  }
  const paths = [
    ["任务输入", job.input_video],
    ["输出目录", job.output_dir],
  ].filter(([, value]) => value);
  $("jobPaths").innerHTML = paths.map(([label, value]) => `
    <div class="path-row">
      <span>${escapeHtml(label)}</span>
      <code>${escapeHtml(value)}</code>
      <button type="button" data-copy="${escapeHtml(value)}">复制</button>
    </div>`).join("");
}

function actionHint(job) {
  if (!job) return { text: "选择任务后显示下一步操作。", kind: "" };
  if (job.status === "queued") return { text: "任务正在等待 GPU，可在开始前取消。", kind: "warning" };
  if (job.status === "running") return { text: "GVHMR 正在处理视频，取消请求会在当前阶段结束后生效。", kind: "warning" };
  if (job.status === "failed") return { text: `处理失败：${job.error_summary || "请查看任务日志"}`, kind: "error" };
  if (job.status === "cancelled") return { text: "任务已取消，可以修改设置后重试。", kind: "warning" };
  if (job.ground_constraint_status === "fallback") {
    return {
      text: `自动平地已回退原始 FootMR：${job.ground_constraint_fallback_reason || job.ground_constraint_error || "保护条件未通过"}`,
      kind: "warning",
    };
  }
  if (job.ground_constraint_warning) {
    return {
      text: `自动平地结果已采用；诊断提示：${job.ground_constraint_warning}`,
      kind: "warning",
    };
  }
  if (job.preview_status === "running" || job.preview_status === "queued") {
    return { text: "人体动作结果已经可用，预览视频正在后台生成。", kind: "" };
  }
  if (job.preview_status === "failed") {
    return { text: `动作结果未受影响；预览失败：${job.preview_error_summary || "请查看日志"}`, kind: "error" };
  }
  if (job.sonic_status === "preparing") {
    return { text: "正在准备本地 SONIC 推流。", kind: "warning" };
  }
  if (job.sonic_status === "streaming") {
    return {
      text: `正在推流到 SONIC：${job.sonic_frame || 0} / ${job.sonic_frames || 0} 帧。`,
      kind: "warning",
    };
  }
  if (job.sonic_status === "complete") {
    return { text: "Web 端已完成该动作的 SONIC 推流。", kind: "" };
  }
  if (job.sonic_status === "stopped") {
    return { text: "该动作的 SONIC 推流已停止或被新动作替换。", kind: "warning" };
  }
  if (job.sonic_status === "paused") {
    return { text: "SONIC 已暂停，策略正在平滑回到默认姿态。", kind: "" };
  }
  if (job.sonic_status === "error") {
    return { text: `SONIC 推流失败：${job.sonic_error || "请查看任务日志"}`, kind: "error" };
  }
  if (hasArtifact(job, "preview_video")) return { text: "动作结果和预览均已生成，可以播放或下载。", kind: "" };
  return { text: "动作结果已生成，可以下载 PT，或按需生成预览视频。", kind: "" };
}

function renderActions(job) {
  const openFolderButton = $("openFolderBtn");
  const previewButton = $("previewBtn");
  const retryButton = $("retryBtn");
  const cancelButton = $("cancelBtn");
  const toGmrButton = $("toGmrBtn");
  const toSonicButton = $("toSonicBtn");
  const sonicSpeedControl = $("sonicSpeedControl");
  const sonicSpeedRange = $("sonicSpeedRange");
  const sonicSpeedValue = $("sonicSpeedValue");
  const pauseSonicButton = $("pauseSonicBtn");
  const isSucceeded = job?.status === "succeeded";
  const previewBusy = ["queued", "running"].includes(job?.preview_status);

  openFolderButton.disabled = !job || Boolean(state.activeAction);
  openFolderButton.textContent = state.activeAction === "open-folder" ? "正在打开" : "打开任务文件夹";
  openFolderButton.title = job ? "在系统文件管理器中打开当前任务目录" : "请先选择任务";

  previewButton.hidden = !isSucceeded;
  previewButton.disabled = previewBusy || state.activeAction === "preview";
  previewButton.textContent = previewBusy
    ? "正在生成预览"
    : hasArtifact(job, "preview_video") ? "播放预览" : job?.preview_status === "failed" ? "重新生成预览" : "生成预览";

  retryButton.hidden = !["failed", "cancelled"].includes(job?.status);
  retryButton.disabled = state.activeAction === "retry";
  cancelButton.hidden = !["queued", "running"].includes(job?.status);
  cancelButton.disabled = state.activeAction === "cancel";

  const canUseGmr = Boolean(state.capabilities?.gmr_bridge_available)
    && isSucceeded
    && hasArtifact(job, "hmr4d_results");
  toGmrButton.hidden = !canUseGmr;
  toGmrButton.disabled = state.activeAction === "to-gmr";

  const canUseSonic = Boolean(state.capabilities?.sonic_bridge_available)
    && isSucceeded
    && hasArtifact(job, "hmr4d_results");
  const sonicBusy = ["preparing", "streaming"].includes(job?.sonic_status);
  toSonicButton.hidden = !canUseSonic;
  toSonicButton.disabled = state.activeAction === "to-sonic";
  toSonicButton.textContent = sonicBusy ? "重新发送到 SONIC" : "发送到 SONIC";
  sonicSpeedControl.hidden = !canUseSonic;
  sonicSpeedRange.disabled = state.activeAction === "to-sonic";
  sonicSpeedRange.value = state.sonicSpeed.toFixed(2);
  sonicSpeedValue.value = `${state.sonicSpeed.toFixed(2)}×`;
  sonicSpeedValue.textContent = `${state.sonicSpeed.toFixed(2)}×`;
  sonicSpeedControl.title = "调整下一次发送给 SONIC 的动作速度；输出控制频率始终为 50 FPS";
  pauseSonicButton.hidden = !sonicBusy;
  pauseSonicButton.disabled = state.activeAction === "pause-sonic";
  pauseSonicButton.textContent = state.activeAction === "pause-sonic" ? "正在暂停" : "暂停 SONIC";
}

function renderJobProgress(job) {
  const container = $("jobProgress");
  const progress = currentProgress(job);
  container.hidden = !progress;
  if (!progress) return;

  const percent = normalizedProgress(progress);
  $("jobProgressStage").textContent = progress.stage;
  $("jobProgressPercent").textContent = `${percent}%`;
  $("jobProgressBar").style.width = `${percent}%`;
  $("jobProgressTrack").setAttribute("aria-valuenow", String(percent));
  $("jobProgressTrack").setAttribute("aria-label", progress.stage);
}

function renderJobDetail(job) {
  state.selectedJob = job || null;
  if (!job) {
    $("detailHint").textContent = "选择任务查看结果、预览和日志。";
    $("jobMetrics").innerHTML = ["状态", "视频", "任务 ID", "静态相机", "地面约束", "焦距 f_mm", "错误"].map((label) => metric(label, "-")).join("");
    $("logs").textContent = "暂无日志";
    $("jobJson").textContent = "{}";
  } else {
    const error = job.error_summary
      || job.preview_error_summary
      || job.ground_constraint_fallback_reason
      || job.ground_constraint_error
      || "-";
    $("detailHint").textContent = `${fileName(job)} · ${job.job_id}`;
    const metrics = [
      metric("状态", statusLabel(job.status)),
      metric("视频", fileName(job)),
      metric("任务 ID", job.job_id),
      metric("静态相机", job.static_cam ? "是" : "否"),
      metric("地面约束", {
        none: "不启用",
        flat_y: job.ground_constraint_status === "fallback" ? "自动平地（已回退）" : "自动平地",
        human3r: "Human3R",
      }[job.ground_constraint || "none"] || job.ground_constraint),
      metric("焦距 f_mm", job.f_mm ?? "自动"),
    ];
    if (error !== "-") metrics.push(metric("错误", error));
    $("jobMetrics").innerHTML = metrics.join("");
    $("logs").textContent = (job.logs || []).join("\n") || "暂无日志";
    $("jobJson").textContent = JSON.stringify(job, null, 2);
  }

  renderPreview(job);
  renderDownloads(job);
  renderPaths(job);
  renderActions(job);
  renderJobProgress(job);
  const hint = actionHint(job);
  $("detailActionHint").textContent = hint.text;
  $("detailActionHint").className = `detail-action-hint${hint.kind ? ` ${hint.kind}` : ""}`;
}

async function runAction(name, action) {
  if (!state.selectedJobId || state.activeAction) return;
  state.activeAction = name;
  renderActions(state.selectedJob);
  try {
    await action();
  } finally {
    state.activeAction = null;
    renderActions(state.selectedJob);
  }
}

async function previewSelected() {
  const job = state.selectedJob;
  if (!job) return showToast("请先选择任务。", true);
  if (hasArtifact(job, "preview_video")) {
    try {
      await $("previewVideo").play();
    } catch (_) {
      showToast("浏览器阻止了自动播放，请在预览画面上点击播放。", true);
    }
    return;
  }
  await runAction("preview", async () => {
    try {
      await request(`jobs/${job.job_id}/preview`, { method: "POST" });
      showToast("预览已加入队列，动作结果仍可正常下载。");
      await refreshJobs();
      ensurePolling();
    } catch (error) {
      showToast(`生成预览失败：${error.message}`, true);
    }
  });
}

async function openSelectedFolder() {
  const job = state.selectedJob;
  if (!job) return showToast("请先选择任务。", true);
  await runAction("open-folder", async () => {
    try {
      const result = await request(`jobs/${job.job_id}/open-folder`, { method: "POST" });
      showToast(`已打开任务文件夹：${result.path}`);
    } catch (error) {
      showToast(`打开任务文件夹失败：${error.message}`, true);
    }
  });
}

async function retrySelected() {
  if (!state.selectedJobId) return showToast("请先选择任务。", true);
  await runAction("retry", async () => {
    try {
      await request(`jobs/${state.selectedJobId}/retry`, { method: "POST" });
      showToast("任务已重新加入队列。");
      await refreshJobs();
      ensurePolling();
    } catch (error) {
      showToast(`重试失败：${error.message}`, true);
    }
  });
}

async function cancelSelected() {
  if (!state.selectedJobId) return showToast("请先选择任务。", true);
  await runAction("cancel", async () => {
    try {
      await request(`jobs/${state.selectedJobId}/cancel`, { method: "POST" });
      showToast("已请求取消任务。");
      await refreshJobs();
    } catch (error) {
      showToast(`取消失败：${error.message}`, true);
    }
  });
}

async function toGmrSelected() {
  if (!state.selectedJobId) return showToast("请先选择任务。", true);
  await runAction("to-gmr", async () => {
    try {
      const result = await request(`jobs/${state.selectedJobId}/to-gmr`, { method: "POST" });
      showToast(`已提交 GMR 任务：${result.job_id || result.message || "ok"}`);
    } catch (error) {
      showToast(`转 ELF3 失败：${error.message}`, true);
    }
  });
}

async function toSonicSelected() {
  if (!state.selectedJobId) return showToast("请先选择任务。", true);
  await runAction("to-sonic", async () => {
    try {
      const result = await request(
        `jobs/${state.selectedJobId}/to-sonic?speed=${encodeURIComponent(state.sonicSpeed)}`,
        { method: "POST" },
      );
      const duration = Number(result.duration_s || 0).toFixed(2);
      showToast(
        `已开始 SONIC 推流：${result.speed}× / ${result.frames} 帧 / ${duration} 秒` +
        `${result.reused ? "（复用已转换数据）" : ""}`,
      );
      await refreshJobs();
      ensurePolling();
    } catch (error) {
      showToast(`SONIC 推流失败：${error.message}`, true);
    }
  });
}

function updateSonicSpeed(event, notify = false) {
  const value = Math.round(Number(event.currentTarget.value) * 100) / 100;
  state.sonicSpeed = Math.max(0.25, Math.min(1.0, value));
  $("sonicSpeedValue").value = `${state.sonicSpeed.toFixed(2)}×`;
  $("sonicSpeedValue").textContent = `${state.sonicSpeed.toFixed(2)}×`;
  if (notify) {
    showToast(`SONIC 动作速度已设为 ${state.sonicSpeed.toFixed(2)}×；控制频率仍为 50 FPS。`);
  }
}

async function pauseSonicSelected() {
  if (!state.selectedJobId) return showToast("请先选择任务。", true);
  await runAction("pause-sonic", async () => {
    try {
      await request(`jobs/${state.selectedJobId}/sonic/pause`, { method: "POST" });
      showToast("SONIC 已暂停，正在平滑回到默认姿态。");
      await refreshJobs();
    } catch (error) {
      showToast(`暂停 SONIC 失败：${error.message}`, true);
    }
  });
}

function closeDetailPopovers(except = null) {
  document.querySelectorAll("#detailPanel .detail-secondary details[open]").forEach((details) => {
    if (details !== except) details.open = false;
  });
}

function setupDetailPopovers() {
  const detailPanel = $("detailPanel");
  detailPanel.querySelectorAll(".detail-secondary details").forEach((details) => {
    const summary = details.querySelector("summary");
    const content = Array.from(details.children).find((child) => child.tagName.toLowerCase() !== "summary");
    if (summary && content) content.dataset.popoverTitle = summary.textContent.trim();
    summary?.addEventListener("click", () => {
      if (!details.open) closeDetailPopovers(details);
    });
    details.addEventListener("toggle", () => {
      if (details.open) closeDetailPopovers(details);
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDetailPopovers();
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("#detailPanel .detail-secondary details")) closeDetailPopovers();
  });
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    showToast("路径已复制。");
  } catch (_) {
    showToast("浏览器不允许复制，请从任务 JSON 中手动复制。", true);
  }
}

async function boot() {
  setupDetailPopovers();
  setupFileDrops();
  $("singleForm").addEventListener("submit", submitSingle);
  $("batchForm").addEventListener("submit", submitBatch);
  $("refreshJobs").addEventListener("click", () => refreshJobs());
  $("searchBox").addEventListener("input", renderJobs);
  $("statusFilter").addEventListener("change", renderJobs);
  $("openFolderBtn").addEventListener("click", openSelectedFolder);
  $("previewBtn").addEventListener("click", previewSelected);
  $("retryBtn").addEventListener("click", retrySelected);
  $("cancelBtn").addEventListener("click", cancelSelected);
  $("toGmrBtn").addEventListener("click", toGmrSelected);
  $("sonicSpeedRange").addEventListener("input", (event) => updateSonicSpeed(event));
  $("sonicSpeedRange").addEventListener("change", (event) => updateSonicSpeed(event, true));
  $("toSonicBtn").addEventListener("click", toSonicSelected);
  $("pauseSonicBtn").addEventListener("click", pauseSonicSelected);
  $("previewVideo").addEventListener("loadedmetadata", () => {
    const video = $("previewVideo");
    fitPreviewStage();
    if (video.currentTime === 0 && Number.isFinite(video.duration) && video.duration > 0.02) {
      video.currentTime = Math.min(0.05, video.duration / 100);
    }
  });
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => fitPreviewStage()).observe($("previewSlot"));
  } else {
    window.addEventListener("resize", fitPreviewStage);
  }
  document.querySelectorAll("[data-mode-tab]").forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.modeTab));
  });
  document.addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy]");
    if (copyButton) copyText(copyButton.dataset.copy);
  });
  setMode(state.activeMode);
  renderJobDetail(null);

  try {
    state.capabilities = await request("api/capabilities");
    renderCapabilities();
  } catch (error) {
    $("capabilityBadges").innerHTML = `<span class="badge error">环境检查失败</span>`;
    updateSubmitState("single");
    updateSubmitState("batch");
    showToast(`环境检查失败：${error.message}`, true);
  }

  try {
    await refreshJobs();
  } catch (error) {
    showToast(`任务列表加载失败：${error.message}`, true);
  }
}

window.addEventListener("beforeunload", () => {
  state.thumbnailObserver?.disconnect();
  stopPolling();
});
boot().catch((error) => showToast(`页面初始化失败：${error.message}`, true));
