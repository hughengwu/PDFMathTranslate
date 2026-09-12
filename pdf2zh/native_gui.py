"""A native desktop GUI for pdf2zh built with Tkinter (Python standard library only).

This avoids depending on the Gradio web UI. It shells out to the same `pdf2zh`
CLI entry point used everywhere else in this project, and persists translator
API keys/settings to the same ``~/.config/PDFMathTranslate/config.json`` file
that the CLI and web GUI already read, so configuration is shared across all
three interfaces.

Run with:
    .venv\\Scripts\\python.exe -m pdf2zh.native_gui
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import requests

from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    X302AITranslator,
    XinferenceTranslator,
    ZhipuTranslator,
)

# Same mapping as pdf2zh/gui.py, kept independent so this module never imports
# gradio (which pulls in the whole web stack just to build a dropdown list).
SERVICE_MAP = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "MiniMax": MiniMaxTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

LANG_MAP = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

SECRET_HINTS = ("API_KEY", "TOKEN", "SECRET", "ACCESS")


def _is_secret(env_name: str) -> bool:
    upper = env_name.upper()
    return any(h in upper for h in SECRET_HINTS)


# Translators whose chat endpoint is OpenAI-compatible, so the standard
# GET {base_url}/models call can be used to list available model ids.
OPENAI_COMPAT_SERVICES = {
    OpenAITranslator,
    OpenAIlikedTranslator,
    ModelScopeTranslator,
    GrokTranslator,
    ZhipuTranslator,
    SiliconTranslator,
    X302AITranslator,
    GeminiTranslator,
    GroqTranslator,
    DeepseekTranslator,
    MiniMaxTranslator,
    QwenMtTranslator,
}

# Base URL for services above that hard-code it (i.e. have no editable
# *_BASE_URL env of their own) rather than exposing it as a setting.
FIXED_BASE_URLS = {
    ZhipuTranslator: "https://open.bigmodel.cn/api/paas/v4",
    SiliconTranslator: "https://api.siliconflow.cn/v1",
    X302AITranslator: "https://api.302.ai/v1",
    GeminiTranslator: "https://generativelanguage.googleapis.com/v1beta/openai/",
    GroqTranslator: "https://api.groq.com/openai/v1",
    DeepseekTranslator: "https://api.deepseek.com/v1",
    MiniMaxTranslator: "https://api.minimax.io/v1",
    QwenMtTranslator: "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _find_env(translator, needle: str) -> str | None:
    """Find the env var name in translator.envs containing `needle` (e.g. "MODEL")."""
    for env_name in translator.envs:
        if needle in env_name.upper():
            return env_name
    return None


def _pdf2zh_executable() -> list[str]:
    """Locate the pdf2zh console-script installed next to the running interpreter."""
    exe_name = "pdf2zh.exe" if os.name == "nt" else "pdf2zh"
    candidate = Path(sys.executable).with_name(exe_name)
    if candidate.exists():
        return [str(candidate)]
    # Fall back to running the CLI module directly with the current interpreter.
    return [sys.executable, "-c", "from pdf2zh.pdf2zh import main; main()"]


class Pdf2zhApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PDFMathTranslate")
        self.geometry("760x680")
        self.minsize(680, 560)

        self.files: list[str] = []
        self.env_vars: dict[str, tk.StringVar] = {}
        self.model_env_name: str | None = None
        self.model_combo: ttk.Combobox | None = None
        self.fetch_models_btn: ttk.Button | None = None
        self.fetch_status_var = tk.StringVar(value="")
        self.proc: subprocess.Popen | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        self._build_widgets()
        self._on_service_change()
        self.after(100, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}

        file_frame = ttk.LabelFrame(self, text="文件")
        file_frame.pack(fill="x", **pad)

        self.file_listbox = tk.Listbox(file_frame, height=4)
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)

        file_btns = ttk.Frame(file_frame)
        file_btns.pack(side="left", padx=8, pady=8)
        ttk.Button(file_btns, text="添加文件", command=self._add_files).pack(fill="x", pady=2)
        ttk.Button(file_btns, text="移除选中", command=self._remove_selected).pack(fill="x", pady=2)
        ttk.Button(file_btns, text="清空", command=self._clear_files).pack(fill="x", pady=2)

        out_frame = ttk.Frame(self)
        out_frame.pack(fill="x", **pad)
        ttk.Label(out_frame, text="输出目录:").pack(side="left")
        self.output_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(out_frame, text="浏览", command=self._choose_output_dir).pack(side="left")

        opt_frame = ttk.LabelFrame(self, text="选项")
        opt_frame.pack(fill="x", **pad)

        row1 = ttk.Frame(opt_frame)
        row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="翻译引擎:").pack(side="left")
        self.service_var = tk.StringVar(value="Google")
        service_combo = ttk.Combobox(
            row1,
            textvariable=self.service_var,
            values=list(SERVICE_MAP.keys()),
            state="readonly",
            width=20,
        )
        service_combo.pack(side="left", padx=6)
        service_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_service_change())

        ttk.Label(row1, text="源语言:").pack(side="left", padx=(16, 0))
        self.lang_from_var = tk.StringVar(
            value=ConfigManager.get("PDF2ZH_LANG_FROM", "English")
        )
        ttk.Combobox(
            row1,
            textvariable=self.lang_from_var,
            values=list(LANG_MAP.keys()),
            state="readonly",
            width=16,
        ).pack(side="left", padx=6)

        ttk.Label(row1, text="目标语言:").pack(side="left", padx=(16, 0))
        self.lang_to_var = tk.StringVar(
            value=ConfigManager.get("PDF2ZH_LANG_TO", "Simplified Chinese")
        )
        ttk.Combobox(
            row1,
            textvariable=self.lang_to_var,
            values=list(LANG_MAP.keys()),
            state="readonly",
            width=16,
        ).pack(side="left", padx=6)

        # Filled in dynamically by _on_service_change() based on the selected
        # translator's required environment variables (API key, base URL, model...).
        self.env_frame = ttk.Frame(opt_frame)
        self.env_frame.pack(fill="x", padx=8, pady=4)

        row2 = ttk.Frame(opt_frame)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="页码范围 (留空=全部, 如 1,3-5):").pack(side="left")
        self.pages_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.pages_var, width=20).pack(side="left", padx=6)

        ttk.Label(row2, text="线程数:").pack(side="left", padx=(16, 0))
        self.threads_var = tk.StringVar(value="4")
        ttk.Entry(row2, textvariable=self.threads_var, width=6).pack(side="left", padx=6)

        self.skip_subset_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="跳过字体子集化", variable=self.skip_subset_var).pack(
            side="left", padx=(16, 0)
        )

        self.ignore_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="忽略缓存", variable=self.ignore_cache_var).pack(
            side="left", padx=(8, 0)
        )

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **pad)
        self.translate_btn = ttk.Button(
            action_frame, text="开始翻译", command=self._start_translation
        )
        self.translate_btn.pack(side="left")
        self.cancel_btn = ttk.Button(
            action_frame, text="取消", command=self._cancel_translation, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)

        log_frame = ttk.LabelFrame(self, text="日志")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text["yscrollcommand"] = scrollbar.set

    # ------------------------------------------------------------- files
    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            filetypes=[("PDF/Word files", "*.pdf *.doc *.docx"), ("All files", "*.*")]
        )
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.file_listbox.insert("end", p)
        if paths and not self.output_var.get():
            self.output_var.set(str(Path(paths[0]).parent))

    def _remove_selected(self) -> None:
        for i in reversed(self.file_listbox.curselection()):
            self.file_listbox.delete(i)
            del self.files[i]

    def _clear_files(self) -> None:
        self.file_listbox.delete(0, "end")
        self.files.clear()

    def _choose_output_dir(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.output_var.set(d)

    # ------------------------------------------------------- service envs
    def _on_service_change(self) -> None:
        for child in self.env_frame.winfo_children():
            child.destroy()
        self.env_vars.clear()
        self.model_env_name = None
        self.model_combo = None
        self.fetch_models_btn = None
        self.fetch_status_var.set("")

        translator = SERVICE_MAP[self.service_var.get()]
        if not translator.envs:
            return

        model_env = _find_env(translator, "MODEL")

        for env_name, default_value in translator.envs.items():
            value = ConfigManager.get_env_by_translatername(
                translator, env_name, default_value
            )
            row = ttk.Frame(self.env_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=env_name, width=24, anchor="w").pack(side="left")
            var = tk.StringVar(value=value or "")
            self.env_vars[env_name] = var

            if env_name == model_env:
                # Editable combobox: user can pick a fetched model or still type one.
                combo = ttk.Combobox(row, textvariable=var, state="normal")
                combo.pack(side="left", fill="x", expand=True, padx=6)
                self.model_env_name = env_name
                self.model_combo = combo
            else:
                entry = ttk.Entry(
                    row, textvariable=var, show="*" if _is_secret(env_name) else ""
                )
                entry.pack(side="left", fill="x", expand=True, padx=6)

        supports_fetch = translator in OPENAI_COMPAT_SERVICES or translator is OllamaTranslator
        if supports_fetch and self.model_combo is not None:
            fetch_row = ttk.Frame(self.env_frame)
            fetch_row.pack(fill="x", pady=(4, 2))
            self.fetch_models_btn = ttk.Button(
                fetch_row, text="获取模型列表", command=self._fetch_models
            )
            self.fetch_models_btn.pack(side="left")
            ttk.Label(fetch_row, textvariable=self.fetch_status_var, foreground="#666").pack(
                side="left", padx=8
            )

    # ------------------------------------------------------- model list
    def _fetch_models(self) -> None:
        translator = SERVICE_MAP[self.service_var.get()]
        self.fetch_status_var.set("正在获取...")
        if self.fetch_models_btn is not None:
            self.fetch_models_btn["state"] = "disabled"

        if translator is OllamaTranslator:
            host = self.env_vars.get(
                "OLLAMA_HOST", tk.StringVar(value="http://127.0.0.1:11434")
            ).get().strip() or "http://127.0.0.1:11434"
            threading.Thread(target=self._fetch_ollama_models, args=(host,), daemon=True).start()
            return

        base_url_env = _find_env(translator, "BASE_URL") or _find_env(translator, "ENDPOINT")
        api_key_env = _find_env(translator, "API_KEY")

        base_url = (
            self.env_vars[base_url_env].get().strip()
            if base_url_env and self.env_vars.get(base_url_env)
            else ""
        )
        if not base_url:
            base_url = FIXED_BASE_URLS.get(translator, "")
        api_key = (
            self.env_vars[api_key_env].get().strip()
            if api_key_env and self.env_vars.get(api_key_env)
            else ""
        )

        if not base_url:
            self._on_models_fetched([], "请先填写 Base URL。")
            return
        if not api_key:
            self._on_models_fetched([], "请先填写 API Key。")
            return

        threading.Thread(
            target=self._fetch_openai_models, args=(base_url, api_key), daemon=True
        ).start()

    def _fetch_openai_models(self, base_url: str, api_key: str) -> None:
        try:
            url = base_url.rstrip("/") + "/models"
            resp = requests.get(
                url, headers={"Authorization": f"Bearer {api_key}"}, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            models = sorted({item["id"] for item in data.get("data", []) if "id" in item})
            self.after(0, self._on_models_fetched, models, None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the status label
            self.after(0, self._on_models_fetched, [], str(exc))

    def _fetch_ollama_models(self, host: str) -> None:
        try:
            resp = requests.get(host.rstrip("/") + "/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = sorted(m["name"] for m in data.get("models", []) if "name" in m)
            self.after(0, self._on_models_fetched, models, None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the status label
            self.after(0, self._on_models_fetched, [], str(exc))

    def _on_models_fetched(self, models: list[str], error: str | None) -> None:
        if self.fetch_models_btn is not None:
            self.fetch_models_btn["state"] = "normal"
        if error:
            self.fetch_status_var.set(f"获取失败: {error}")
            return
        if not models:
            self.fetch_status_var.set("未获取到任何模型。")
            return
        self.fetch_status_var.set(f"已获取 {len(models)} 个模型")
        if self.model_combo is not None:
            self.model_combo["values"] = models
            if self.model_env_name and not self.env_vars[self.model_env_name].get():
                self.env_vars[self.model_env_name].set(models[0])

    # ------------------------------------------------------------- log
    def _log(self, line: str) -> None:
        self.log_text["state"] = "normal"
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text["state"] = "disabled"

    def _poll_log_queue(self) -> None:
        try:
            while True:
                self._log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    # ------------------------------------------------------- translation
    def _start_translation(self) -> None:
        if not self.files:
            messagebox.showwarning("提示", "请先添加要翻译的 PDF 文件。")
            return
        if self.proc is not None:
            messagebox.showwarning("提示", "已有翻译任务在运行。")
            return

        output_dir = self.output_var.get().strip()
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        translator = SERVICE_MAP[self.service_var.get()]

        # Persist entered values to the same config.json the CLI/web GUI read,
        # so a key typed here is remembered and reused everywhere.
        env_values = {k: v.get() for k, v in self.env_vars.items()}
        if env_values:
            ConfigManager.set_translator_by_name(translator.name, env_values)
        ConfigManager.set("PDF2ZH_LANG_FROM", self.lang_from_var.get())
        ConfigManager.set("PDF2ZH_LANG_TO", self.lang_to_var.get())

        cmd = [
            *_pdf2zh_executable(),
            *self.files,
            "-li",
            LANG_MAP[self.lang_from_var.get()],
            "-lo",
            LANG_MAP[self.lang_to_var.get()],
            "-s",
            translator.name,
            "-t",
            self.threads_var.get().strip() or "4",
        ]
        if output_dir:
            cmd += ["-o", output_dir]
        if self.pages_var.get().strip():
            cmd += ["-p", self.pages_var.get().strip()]
        if self.skip_subset_var.get():
            cmd.append("--skip-subset-fonts")
        if self.ignore_cache_var.get():
            cmd.append("--ignore-cache")

        self._log(f"$ {' '.join(cmd)}\n")
        self.translate_btn["state"] = "disabled"
        self.cancel_btn["state"] = "normal"
        self.progress.start(10)

        threading.Thread(target=self._run_process, args=(cmd,), daemon=True).start()

    def _run_process(self, cmd: list[str]) -> None:
        code = -1
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self.log_queue.put(line)
            code = self.proc.wait()
        except Exception as exc:  # noqa: BLE001 - surfaced to the log panel
            self.log_queue.put(f"[error] {exc}\n")
        finally:
            self.proc = None
            self.after(0, self._on_translation_done, code)

    def _on_translation_done(self, code: int) -> None:
        self.translate_btn["state"] = "normal"
        self.cancel_btn["state"] = "disabled"
        self.progress.stop()
        if code == 0:
            self._log("\n=== 翻译完成 ===\n")
            messagebox.showinfo("完成", "翻译已完成，请在输出目录查看结果。")
        else:
            self._log(f"\n=== 进程退出，代码 {code} ===\n")
            messagebox.showerror("失败", f"翻译进程异常退出（代码 {code}），请查看日志。")

    def _cancel_translation(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self._log("\n[已请求取消]\n")

    def _on_close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
        self.destroy()


def main() -> None:
    app = Pdf2zhApp()
    app.mainloop()


if __name__ == "__main__":
    main()
