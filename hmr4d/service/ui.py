import json
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import gradio as gr

from hmr4d.service.common import guess_lan_ip, iter_video_files, terminal_job_states


LANGUAGE_CHOICES = ["中文", "English"]
DEFAULT_LANGUAGE = "中文"
_MISSING_VIDEO_INPUT = "__missing_video_input__"
_MISSING_BATCH_INPUT = "__missing_batch_input__"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
SELECTABLE_OUTPUT_CSS = """
.gradio-container { max-width: 1320px !important; }
.gradio-container, .gradio-container * {
  user-select: text !important;
  -webkit-user-select: text !important;
}
textarea, input, pre, code, .cm-content, .cm-line {
  user-select: text !important;
  -webkit-user-select: text !important;
}
.gvhmr-hero {
  border: 1px solid #d1fae5;
  border-radius: 20px;
  padding: 20px 22px;
  background:
    radial-gradient(circle at 12% 10%, rgba(16, 185, 129, 0.16), transparent 30%),
    linear-gradient(135deg, #f8fafc 0%, #ecfdf5 100%);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
  margin-bottom: 14px;
}
.gvhmr-hero h1 {
  margin: 0 0 6px;
  font-size: 30px;
  letter-spacing: -0.03em;
}
.gvhmr-card {
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 16px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}
.gvhmr-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
}
.gvhmr-item {
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.78);
  padding: 10px 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
}
.gvhmr-label {
  color: #64748b;
  font-size: 12px;
  margin-bottom: 4px;
}
.gvhmr-value {
  color: #0f172a;
  font-size: 14px;
  font-weight: 650;
  word-break: break-word;
}
.gvhmr-empty-note {
  color: #64748b;
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  padding: 12px 14px;
  background: #f8fafc;
}
.gvhmr-section-title {
  margin: 18px 0 8px;
  color: #0f172a;
  font-size: 18px;
  font-weight: 750;
}
.gvhmr-preview-video video {
  border-radius: 14px;
  background: #0f172a;
}
"""

I18N = {
    "zh": {
        "title": "# GVHMR Web 工具",
        "description": "提交单视频或批量视频任务，后台串行处理并导出 NPZ/JSON/可选预览。",
        "single_tab": "单视频处理",
        "batch_tab": "批量处理",
        "history_tab": "任务历史",
        "uploaded_video": "上传视频",
        "uploaded_videos": "批量上传视频",
        "local_video_path": "本地视频路径",
        "local_video_placeholder": "docs/example_video/tennis.mp4",
        "local_video_dir": "服务端视频目录",
        "local_video_dir_placeholder": "/data/videos",
        "static_cam": "静态相机",
        "f_mm": "焦距 (mm)",
        "save_intermediate": "保留预处理文件",
        "generate_preview": "处理完成后自动生成预览",
        "submit_job": "提交单视频任务",
        "submit_batch": "提交批量任务",
        "request_preview": "为当前任务生成预览",
        "convert_elf3": "转 ELF3",
        "gmr_status": "ELF3 转换状态",
        "status": "任务状态",
        "job_id": "任务 ID",
        "batch_id": "批次 ID",
        "output_dir": "输出目录",
        "logs": "实时日志",
        "data_file": "gvhmr_data.npz",
        "meta_file": "gvhmr_meta.json",
        "artifact_file": "artifacts.zip",
        "preview_video": "预览视频",
        "batch_summary": "批次摘要",
        "batch_jobs": "批次任务",
        "refresh_history": "刷新历史",
        "recent_jobs": "最近任务",
        "job_selector": "选择任务",
        "job_detail": "任务详情",
        "cancel_job": "取消任务",
        "retry_job": "重试任务",
        "history_status": "历史状态",
        "history_logs": "任务日志",
        "history_output_dir": "历史输出目录",
        "result_summary": "任务摘要",
        "downloads": "下载结果",
        "advanced_logs": "日志与任务详情",
        "preview_and_downloads": "预览与下载",
        "download_data": "下载 gvhmr_data.npz",
        "download_meta": "下载 gvhmr_meta.json",
        "download_artifacts": "下载 artifacts.zip",
        "language": "语言 / Language",
        "preparing": "正在提交任务...",
        "queued": "任务已入队，等待执行...",
        "running": "后台处理中...",
        "succeeded": "任务完成。",
        "failed": "任务失败：{message}",
        "cancelled": "任务已取消。",
        "batch_preparing": "正在提交批量任务...",
        "batch_running": "批量任务执行中...",
        "batch_done": "批量任务已结束。",
        "missing_video_input": "请上传视频或填写一个本地视频路径。",
        "missing_batch_input": "请上传至少一个视频，或填写一个服务端目录。",
        "video_not_found": "视频不存在：{path}",
        "dir_not_found": "目录不存在：{path}",
        "job_not_found": "任务不存在：{job_id}",
        "batch_not_found": "批次不存在：{batch_id}",
        "history_empty": "暂无任务历史。",
        "preview_not_ready": "当前任务还没有可用预览。",
        "gmr_no_job": "请先选择或完成一个 GVHMR 任务。",
        "gmr_submitted": "已提交 GMR 转 ELF3 任务：{job_id}\n状态：{status}\n输出目录：{output_dir}\n请到 GMR 页面查看进度和下载结果。",
        "history_help": "先刷新历史，再选择一个任务查看详情。",
        "lan_hint": "局域网访问地址：http://{lan_ip}:{port}",
    },
    "en": {
        "title": "# GVHMR Web Tool",
        "description": "Submit single or batch video jobs, run them in a serial GPU queue, and export NPZ/JSON with optional previews.",
        "single_tab": "Single Video",
        "batch_tab": "Batch Jobs",
        "history_tab": "Job History",
        "uploaded_video": "Upload Video",
        "uploaded_videos": "Upload Multiple Videos",
        "local_video_path": "Local Video Path",
        "local_video_placeholder": "docs/example_video/tennis.mp4",
        "local_video_dir": "Server Video Directory",
        "local_video_dir_placeholder": "/data/videos",
        "static_cam": "Static Camera",
        "f_mm": "Focal Length (mm)",
        "save_intermediate": "Keep Preprocess Artifacts",
        "generate_preview": "Generate preview after processing",
        "submit_job": "Submit Single Job",
        "submit_batch": "Submit Batch",
        "request_preview": "Generate Preview For Current Job",
        "convert_elf3": "Convert to ELF3",
        "gmr_status": "ELF3 Conversion Status",
        "status": "Task Status",
        "job_id": "Job ID",
        "batch_id": "Batch ID",
        "output_dir": "Output Directory",
        "logs": "Live Logs",
        "data_file": "gvhmr_data.npz",
        "meta_file": "gvhmr_meta.json",
        "artifact_file": "artifacts.zip",
        "preview_video": "Preview Video",
        "batch_summary": "Batch Summary",
        "batch_jobs": "Batch Jobs",
        "refresh_history": "Refresh History",
        "recent_jobs": "Recent Jobs",
        "job_selector": "Select Job",
        "job_detail": "Job Detail",
        "cancel_job": "Cancel Job",
        "retry_job": "Retry Job",
        "history_status": "History Status",
        "history_logs": "Job Logs",
        "history_output_dir": "History Output Directory",
        "result_summary": "Task Summary",
        "downloads": "Downloads",
        "advanced_logs": "Logs and Details",
        "preview_and_downloads": "Preview and Downloads",
        "download_data": "Download gvhmr_data.npz",
        "download_meta": "Download gvhmr_meta.json",
        "download_artifacts": "Download artifacts.zip",
        "language": "Language / 语言",
        "preparing": "Submitting job...",
        "queued": "Job queued and waiting...",
        "running": "Processing in the background...",
        "succeeded": "Job completed.",
        "failed": "Job failed: {message}",
        "cancelled": "Job cancelled.",
        "batch_preparing": "Submitting batch...",
        "batch_running": "Batch is running...",
        "batch_done": "Batch finished.",
        "missing_video_input": "Please upload a video or provide a local path.",
        "missing_batch_input": "Please upload at least one video or provide a server-side directory.",
        "video_not_found": "Video not found: {path}",
        "dir_not_found": "Directory not found: {path}",
        "job_not_found": "Job not found: {job_id}",
        "batch_not_found": "Batch not found: {batch_id}",
        "history_empty": "No jobs yet.",
        "preview_not_ready": "No preview is available for this job yet.",
        "gmr_no_job": "Please select or finish a GVHMR job first.",
        "gmr_submitted": "Submitted GMR ELF3 job: {job_id}\nStatus: {status}\nOutput dir: {output_dir}\nOpen the GMR page to track progress and download results.",
        "history_help": "Refresh history first, then select a job to inspect.",
        "lan_hint": "LAN URL: http://{lan_ip}:{port}",
    },
}


def _lang_code(language):
    return "en" if language in ("English", "en") else "zh"


def _text(language, key, **kwargs):
    template = I18N[_lang_code(language)][key]
    return template.format(**kwargs) if kwargs else template


def _path_name(value):
    if not value:
        return ""
    return Path(str(value)).name


def _safe_html(value):
    return escape(str(value if value is not None else "-"))


def _existing_path(value):
    if not value:
        return None
    path = Path(str(value))
    return str(path) if path.exists() else None


def _display_video_name(job):
    for key in ("display_name", "source_video_path", "input_video"):
        name = _path_name(job.get(key))
        if name:
            return name
    return "-"


def _display_time(value):
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def _display_job_detail(job):
    detail = dict(job)
    detail["display_video"] = _display_video_name(job)
    detail["submitted_at_local"] = _display_time(job.get("submitted_at"))
    detail["started_at_local"] = _display_time(job.get("started_at"))
    detail["finished_at_local"] = _display_time(job.get("finished_at"))
    detail["staged_input_video"] = job.get("input_video")
    return json.dumps(detail, indent=2, ensure_ascii=False)


def _format_job_summary(job, language):
    if not job:
        return "<div class='gvhmr-empty-note'>提交或选择任务后这里会显示摘要。</div>"
    data_path, meta_path, zip_path, preview_path = _job_files(job)
    fields = [
        (_text(language, "job_id"), job.get("job_id", "-")),
        ("Video", _display_video_name(job)),
        (_text(language, "status"), job.get("status", "-")),
        ("static_cam", job.get("static_cam", "-")),
        (_text(language, "history_output_dir"), job.get("output_dir", "-")),
        (
            _text(language, "downloads"),
            " / ".join(
                name
                for name in [_path_name(data_path), _path_name(meta_path), _path_name(zip_path), _path_name(preview_path)]
                if name
            )
            or "-",
        ),
    ]
    items = "\n".join(
        "<div class='gvhmr-item'>"
        f"<div class='gvhmr-label'>{_safe_html(label)}</div>"
        f"<div class='gvhmr-value'>{_safe_html(value)}</div>"
        "</div>"
        for label, value in fields
    )
    return f"<div class='gvhmr-card'><div class='gvhmr-grid'>{items}</div></div>"


def _empty_job_result(message):
    return (
        message,
        "",
        "",
        "<div class='gvhmr-empty-note'>暂无任务结果。</div>",
        "",
        None,
        None,
        None,
        None,
    )


def _resolve_single_input(uploaded_video, local_video_path):
    if uploaded_video:
        return Path(uploaded_video).expanduser().resolve()
    local_video_path = (local_video_path or "").strip()
    if local_video_path:
        return Path(local_video_path).expanduser().resolve()
    raise ValueError(_MISSING_VIDEO_INPUT)


def _resolve_batch_inputs(uploaded_videos, local_video_dir):
    paths = []
    for video in uploaded_videos or []:
        if video:
            paths.append(Path(video).expanduser().resolve())

    local_video_dir = (local_video_dir or "").strip()
    if local_video_dir:
        folder = Path(local_video_dir).expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found at {folder}")
        paths.extend(iter_video_files(folder))

    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError(_MISSING_BATCH_INPUT)
    return unique_paths


def _format_error(exc, language):
    message = str(exc).strip() or exc.__class__.__name__
    if message == _MISSING_VIDEO_INPUT:
        return _text(language, "missing_video_input")
    if message == _MISSING_BATCH_INPUT:
        return _text(language, "missing_batch_input")
    if isinstance(exc, FileNotFoundError):
        if message.startswith("Video not found at "):
            return _text(language, "video_not_found", path=message.removeprefix("Video not found at "))
        if message.startswith("Directory not found at "):
            return _text(language, "dir_not_found", path=message.removeprefix("Directory not found at "))
    return message


def _job_status_text(job, language):
    if job["status"] == "queued":
        return _text(language, "queued")
    if job["status"] == "running":
        return _text(language, "running")
    if job["status"] == "failed":
        return _text(language, "failed", message=job.get("error_summary") or "Unknown error")
    if job["status"] == "cancelled":
        return _text(language, "cancelled")
    return _text(language, "succeeded")


def _job_logs(job):
    return "\n".join(job.get("logs", []))


def _job_files(job):
    artifacts = job.get("artifacts", {})
    return (
        artifacts.get("data_path"),
        artifacts.get("meta_path"),
        artifacts.get("artifacts_zip_path"),
        artifacts.get("preview_video_path"),
    )


def _poll_job(manager, job_id, language):
    while True:
        job = manager.get_job(job_id)
        if job is None:
            yield (
                _text(language, "job_not_found", job_id=job_id),
                job_id,
                "",
                "",
                "",
                None,
                None,
                None,
                None,
            )
            return
        data_path, meta_path, zip_path, preview_path = _job_files(job)
        yield (
            _job_status_text(job, language),
            job["job_id"],
            job["output_dir"],
            _format_job_summary(job, language),
            _job_logs(job),
            _existing_path(preview_path),
            _existing_path(data_path),
            _existing_path(meta_path),
            _existing_path(zip_path),
        )
        if job["status"] in terminal_job_states():
            return
        time.sleep(1.0)


def _poll_batch(manager, batch_id, language):
    while True:
        batch = manager.get_batch(batch_id)
        if batch is None:
            yield (
                _text(language, "batch_not_found", batch_id=batch_id),
                batch_id,
                "",
                [],
            )
            return
        summary = json.dumps(batch, indent=2, ensure_ascii=False)
        rows = [
            [job["job_id"], _display_video_name(job), job["status"], job.get("error_summary") or ""]
            for job in manager.list_jobs(limit=max(batch["total"], 50), batch_id=batch_id)
        ]
        status = _text(language, "batch_running")
        if batch["queued"] == 0 and batch["running"] == 0:
            status = _text(language, "batch_done")
        yield (status, batch["batch_id"], summary, rows)
        if batch["queued"] == 0 and batch["running"] == 0:
            return
        time.sleep(1.0)


def build_ui(manager, settings, submit_to_gmr=None):
    lan_ip = guess_lan_ip()
    lan_hint = f"<small>{_text(DEFAULT_LANGUAGE, 'lan_hint', lan_ip=lan_ip, port=settings.port)}</small>"

    with gr.Blocks(title="GVHMR Web Tool", css=SELECTABLE_OUTPUT_CSS) as demo:
        language = gr.Radio(choices=LANGUAGE_CHOICES, value=DEFAULT_LANGUAGE, label=_text(DEFAULT_LANGUAGE, "language"))
        gr.HTML(
            f"""
            <section class="gvhmr-hero">
              <h1>{_safe_html(_text(DEFAULT_LANGUAGE, "title").lstrip("# "))}</h1>
              <p>{_safe_html(_text(DEFAULT_LANGUAGE, "description"))}</p>
            </section>
            """
        )
        gr.Markdown(lan_hint)

        with gr.Tabs():
            with gr.Tab(_text(DEFAULT_LANGUAGE, "single_tab")):
                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_video = gr.File(label=_text(DEFAULT_LANGUAGE, "uploaded_video"), file_count="single", type="filepath")
                        local_video_path = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "local_video_path"),
                            placeholder=_text(DEFAULT_LANGUAGE, "local_video_placeholder"),
                        )
                        single_static_cam = gr.Checkbox(label=_text(DEFAULT_LANGUAGE, "static_cam"), value=True)
                        single_f_mm = gr.Number(label=_text(DEFAULT_LANGUAGE, "f_mm"), value=None, precision=0)
                        single_save_intermediate = gr.Checkbox(
                            label=_text(DEFAULT_LANGUAGE, "save_intermediate"), value=False
                        )
                        single_generate_preview = gr.Checkbox(
                            label=_text(DEFAULT_LANGUAGE, "generate_preview"), value=False
                        )
                        submit_single = gr.Button(_text(DEFAULT_LANGUAGE, "submit_job"), variant="primary")
                        preview_single = gr.Button(_text(DEFAULT_LANGUAGE, "request_preview"))
                        if submit_to_gmr is not None:
                            single_to_gmr = gr.Button(_text(DEFAULT_LANGUAGE, "convert_elf3"))

                    with gr.Column(scale=1):
                        single_status = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "status"),
                            interactive=True,
                            show_copy_button=True,
                        )
                        single_job_id = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "job_id"),
                            interactive=True,
                            show_copy_button=True,
                        )
                        single_output_dir = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "output_dir"),
                            interactive=True,
                            show_copy_button=True,
                        )
                        if submit_to_gmr is not None:
                            single_gmr_status = gr.Textbox(
                                label=_text(DEFAULT_LANGUAGE, "gmr_status"),
                                interactive=True,
                                show_copy_button=True,
                            )
                        single_summary = gr.HTML("<div class='gvhmr-empty-note'>提交任务后这里会显示摘要。</div>")

                gr.HTML(f"<div class='gvhmr-section-title'>{_safe_html(_text(DEFAULT_LANGUAGE, 'preview_and_downloads'))}</div>")
                single_preview_video = gr.Video(
                    label=_text(DEFAULT_LANGUAGE, "preview_video"),
                    height=420,
                    show_download_button=True,
                    elem_classes=["gvhmr-preview-video"],
                )
                with gr.Row():
                    single_data_download = gr.DownloadButton(_text(DEFAULT_LANGUAGE, "download_data"))
                    single_meta_download = gr.DownloadButton(_text(DEFAULT_LANGUAGE, "download_meta"))
                    single_artifacts_download = gr.DownloadButton(
                        _text(DEFAULT_LANGUAGE, "download_artifacts"), variant="primary"
                    )
                with gr.Accordion(_text(DEFAULT_LANGUAGE, "advanced_logs"), open=False):
                    single_logs = gr.Code(
                        label=_text(DEFAULT_LANGUAGE, "logs"),
                        language="shell",
                        lines=12,
                    )

            with gr.Tab(_text(DEFAULT_LANGUAGE, "batch_tab")):
                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_videos = gr.File(
                            label=_text(DEFAULT_LANGUAGE, "uploaded_videos"),
                            file_count="multiple",
                            type="filepath",
                        )
                        local_video_dir = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "local_video_dir"),
                            placeholder=_text(DEFAULT_LANGUAGE, "local_video_dir_placeholder"),
                        )
                        batch_static_cam = gr.Checkbox(label=_text(DEFAULT_LANGUAGE, "static_cam"), value=True)
                        batch_f_mm = gr.Number(label=_text(DEFAULT_LANGUAGE, "f_mm"), value=None, precision=0)
                        batch_save_intermediate = gr.Checkbox(
                            label=_text(DEFAULT_LANGUAGE, "save_intermediate"), value=False
                        )
                        batch_generate_preview = gr.Checkbox(
                            label=_text(DEFAULT_LANGUAGE, "generate_preview"), value=False
                        )
                        submit_batch = gr.Button(_text(DEFAULT_LANGUAGE, "submit_batch"), variant="primary")
                    with gr.Column(scale=1):
                        batch_status = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "status"),
                            interactive=True,
                            show_copy_button=True,
                        )
                        batch_id = gr.Textbox(
                            label=_text(DEFAULT_LANGUAGE, "batch_id"),
                            interactive=True,
                            show_copy_button=True,
                        )
                        batch_jobs = gr.Dataframe(
                            headers=["job_id", "video", "status", "error"],
                            label=_text(DEFAULT_LANGUAGE, "batch_jobs"),
                            interactive=False,
                        )
                        with gr.Accordion(_text(DEFAULT_LANGUAGE, "batch_summary"), open=False):
                            batch_summary = gr.Code(label=_text(DEFAULT_LANGUAGE, "batch_summary"), language="json")

            with gr.Tab(_text(DEFAULT_LANGUAGE, "history_tab")):
                with gr.Row():
                    refresh_history = gr.Button(_text(DEFAULT_LANGUAGE, "refresh_history"), variant="primary")
                    job_selector = gr.Dropdown(label=_text(DEFAULT_LANGUAGE, "job_selector"), choices=[])
                    history_job_id = gr.Textbox(
                        label=_text(DEFAULT_LANGUAGE, "job_id"),
                        interactive=True,
                        show_copy_button=True,
                    )
                with gr.Row():
                    cancel_job = gr.Button(_text(DEFAULT_LANGUAGE, "cancel_job"))
                    retry_job = gr.Button(_text(DEFAULT_LANGUAGE, "retry_job"))
                    if submit_to_gmr is not None:
                        history_to_gmr = gr.Button(_text(DEFAULT_LANGUAGE, "convert_elf3"))

                with gr.Accordion(_text(DEFAULT_LANGUAGE, "recent_jobs"), open=False):
                    recent_jobs = gr.Dataframe(
                        headers=["job_id", "video", "status", "submitted_at"],
                        label=_text(DEFAULT_LANGUAGE, "recent_jobs"),
                        interactive=False,
                    )

                history_summary = gr.HTML("<div class='gvhmr-empty-note'>选择任务后这里会显示摘要。</div>")
                with gr.Row():
                    history_status = gr.Textbox(
                        label=_text(DEFAULT_LANGUAGE, "history_status"),
                        interactive=True,
                        show_copy_button=True,
                    )
                    history_output_dir = gr.Textbox(
                        label=_text(DEFAULT_LANGUAGE, "history_output_dir"),
                        interactive=True,
                        show_copy_button=True,
                    )
                if submit_to_gmr is not None:
                    history_gmr_status = gr.Textbox(
                        label=_text(DEFAULT_LANGUAGE, "gmr_status"),
                        interactive=True,
                        show_copy_button=True,
                    )
                gr.HTML(f"<div class='gvhmr-section-title'>{_safe_html(_text(DEFAULT_LANGUAGE, 'preview_and_downloads'))}</div>")
                history_preview = gr.Video(
                    label=_text(DEFAULT_LANGUAGE, "preview_video"),
                    height=420,
                    show_download_button=True,
                    elem_classes=["gvhmr-preview-video"],
                )
                with gr.Row():
                    history_data_download = gr.DownloadButton(_text(DEFAULT_LANGUAGE, "download_data"))
                    history_meta_download = gr.DownloadButton(_text(DEFAULT_LANGUAGE, "download_meta"))
                    history_artifacts_download = gr.DownloadButton(
                        _text(DEFAULT_LANGUAGE, "download_artifacts"), variant="primary"
                    )
                with gr.Accordion(_text(DEFAULT_LANGUAGE, "advanced_logs"), open=False):
                    history_logs = gr.Code(
                        label=_text(DEFAULT_LANGUAGE, "history_logs"),
                        language="shell",
                        lines=12,
                    )
                    history_detail = gr.Code(label=_text(DEFAULT_LANGUAGE, "job_detail"), language="json")

        def submit_single_job(uploaded, local_path, static_cam, f_mm, save_intermediate, generate_preview, lang):
            try:
                video_path = _resolve_single_input(uploaded, local_path)
                job = manager.submit_job(
                    video_source=video_path,
                    static_cam=static_cam,
                    f_mm=f_mm,
                    save_intermediate=save_intermediate,
                    generate_preview=generate_preview,
                )
            except Exception as exc:
                yield _empty_job_result(_format_error(exc, lang))
                return

            for update in _poll_job(manager, job["job_id"], lang):
                yield update

        def request_preview_job(job_id, lang):
            if not job_id:
                yield _empty_job_result(_text(lang, "preview_not_ready"))
                return
            try:
                manager.request_preview(job_id)
            except Exception as exc:
                job = manager.get_job(job_id)
                output_dir = job["output_dir"] if job else ""
                logs = _job_logs(job) if job else ""
                data_path, meta_path, zip_path, preview_path = _job_files(job or {})
                yield (
                    _format_error(exc, lang),
                    job_id,
                    output_dir,
                    _format_job_summary(job, lang) if job else "<div class='gvhmr-empty-note'>暂无任务结果。</div>",
                    logs,
                    _existing_path(preview_path),
                    _existing_path(data_path),
                    _existing_path(meta_path),
                    _existing_path(zip_path),
                )
                return
            for update in _poll_job(manager, job_id, lang):
                yield update

        def submit_gvhmr_to_gmr(job_id, lang):
            if not job_id:
                return _text(lang, "gmr_no_job")
            try:
                gmr_job = submit_to_gmr(job_id)
            except Exception as exc:
                return _format_error(exc, lang)
            return _text(
                lang,
                "gmr_submitted",
                job_id=gmr_job["job_id"],
                status=gmr_job["status"],
                output_dir=gmr_job["output_dir"],
            )

        def submit_batch_job(uploaded, local_dir, static_cam, f_mm, save_intermediate, generate_preview, lang):
            try:
                video_paths = _resolve_batch_inputs(uploaded, local_dir)
                batch = manager.submit_batch(
                    video_sources=video_paths,
                    static_cam=static_cam,
                    f_mm=f_mm,
                    save_intermediate=save_intermediate,
                    generate_preview=generate_preview,
                    input_dir=local_dir or None,
                )
            except Exception as exc:
                yield (_format_error(exc, lang), "", "", [])
                return
            for update in _poll_batch(manager, batch["batch_id"], lang):
                yield update

        def load_history():
            jobs = manager.list_jobs(limit=50)
            rows = [
                [job["job_id"], _display_video_name(job), job["status"], _display_time(job.get("submitted_at"))]
                for job in jobs
            ]
            choices = [
                (
                    f"{_display_video_name(job)} | {job['job_id']} | {job['status']} | {_display_time(job.get('submitted_at'))}",
                    job["job_id"],
                )
                for job in jobs
            ]
            first_job_id = choices[0][1] if choices else ""
            return rows, gr.update(choices=choices, value=first_job_id or None), first_job_id

        def fill_history_job_id(job_id):
            return job_id or ""

        def select_job(job_id, lang):
            if not job_id:
                return (
                    _text(lang, "history_empty"),
                    "",
                    "<div class='gvhmr-empty-note'>选择任务后这里会显示摘要。</div>",
                    "",
                    "",
                    None,
                    None,
                    None,
                    None,
                )
            job = manager.get_job(job_id)
            if job is None:
                return (
                    _text(lang, "job_not_found", job_id=job_id),
                    "",
                    "<div class='gvhmr-empty-note'>任务不存在。</div>",
                    "",
                    "",
                    None,
                    None,
                    None,
                    None,
                )
            data_path, meta_path, zip_path, preview_path = _job_files(job)
            return (
                _job_status_text(job, lang),
                job["output_dir"],
                _format_job_summary(job, lang),
                _job_logs(job),
                _display_job_detail(job),
                _existing_path(preview_path),
                _existing_path(data_path),
                _existing_path(meta_path),
                _existing_path(zip_path),
            )

        def cancel_selected(job_id, lang):
            if not job_id:
                return (
                    _text(lang, "history_help"),
                    "",
                    "<div class='gvhmr-empty-note'>先刷新历史，再选择一个任务查看详情。</div>",
                    "",
                    "",
                    None,
                    None,
                    None,
                    None,
                )
            job = manager.cancel_job(job_id)
            return select_job(job_id, lang)

        def retry_selected(job_id, lang):
            if not job_id:
                yield (
                    _text(lang, "history_help"),
                    "",
                    "<div class='gvhmr-empty-note'>先刷新历史，再选择一个任务查看详情。</div>",
                    "",
                    "",
                    None,
                    None,
                    None,
                    None,
                )
                return
            try:
                manager.retry_job(job_id)
            except Exception as exc:
                yield select_job(job_id, lang)
                return
            while True:
                job = manager.get_job(job_id)
                data_path, meta_path, zip_path, preview_path = _job_files(job or {})
                yield (
                    _job_status_text(job, lang),
                    job["output_dir"],
                    _format_job_summary(job, lang),
                    _job_logs(job),
                    _display_job_detail(job),
                    _existing_path(preview_path),
                    _existing_path(data_path),
                    _existing_path(meta_path),
                    _existing_path(zip_path),
                )
                if job["status"] in terminal_job_states():
                    return
                time.sleep(1.0)

        submit_single.click(
            fn=submit_single_job,
            inputs=[
                uploaded_video,
                local_video_path,
                single_static_cam,
                single_f_mm,
                single_save_intermediate,
                single_generate_preview,
                language,
            ],
            outputs=[
                single_status,
                single_job_id,
                single_output_dir,
                single_summary,
                single_logs,
                single_preview_video,
                single_data_download,
                single_meta_download,
                single_artifacts_download,
            ],
        )
        preview_single.click(
            fn=request_preview_job,
            inputs=[single_job_id, language],
            outputs=[
                single_status,
                single_job_id,
                single_output_dir,
                single_summary,
                single_logs,
                single_preview_video,
                single_data_download,
                single_meta_download,
                single_artifacts_download,
            ],
        )
        if submit_to_gmr is not None:
            single_to_gmr.click(
                fn=submit_gvhmr_to_gmr,
                inputs=[single_job_id, language],
                outputs=[single_gmr_status],
            )
        submit_batch.click(
            fn=submit_batch_job,
            inputs=[
                uploaded_videos,
                local_video_dir,
                batch_static_cam,
                batch_f_mm,
                batch_save_intermediate,
                batch_generate_preview,
                language,
            ],
            outputs=[batch_status, batch_id, batch_summary, batch_jobs],
        )
        refresh_history.click(fn=load_history, outputs=[recent_jobs, job_selector, history_job_id])
        job_selector.change(fn=fill_history_job_id, inputs=[job_selector], outputs=[history_job_id])
        job_selector.change(
            fn=select_job,
            inputs=[job_selector, language],
            outputs=[
                history_status,
                history_output_dir,
                history_summary,
                history_logs,
                history_detail,
                history_preview,
                history_data_download,
                history_meta_download,
                history_artifacts_download,
            ],
        )
        cancel_job.click(
            fn=cancel_selected,
            inputs=[job_selector, language],
            outputs=[
                history_status,
                history_output_dir,
                history_summary,
                history_logs,
                history_detail,
                history_preview,
                history_data_download,
                history_meta_download,
                history_artifacts_download,
            ],
        )
        retry_job.click(
            fn=retry_selected,
            inputs=[job_selector, language],
            outputs=[
                history_status,
                history_output_dir,
                history_summary,
                history_logs,
                history_detail,
                history_preview,
                history_data_download,
                history_meta_download,
                history_artifacts_download,
            ],
        )
        if submit_to_gmr is not None:
            history_to_gmr.click(
                fn=submit_gvhmr_to_gmr,
                inputs=[job_selector, language],
                outputs=[history_gmr_status],
            )

    try:
        demo.queue(default_concurrency_limit=4)
    except TypeError:
        demo.queue(concurrency_count=4)
    return demo
