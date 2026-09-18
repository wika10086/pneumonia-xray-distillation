from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback
from typing import Any

from PIL import Image, ImageTk
import torch

from dataset import IMAGE_EXTENSIONS, build_transforms
from evaluate_models import create_model_from_checkpoint
from explain_prediction import (
    build_probability_explanation,
    describe_focus_regions,
    generate_gradcam,
    safe_heatmap_name,
    save_gradcam_overlay,
)
from medical_basis import build_medical_basis, medical_basis_text
from utils import get_device


APP_TITLE = "肺炎胸片图像识别演示"
DEFAULT_THRESHOLD = 0.520043
DEFAULT_CHECKPOINT = Path("checkpoints/distilled_student_augmented_best.pt")


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def resource_path(relative_path: Path) -> Path:
    return app_base_dir() / relative_path


def writable_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def log_exception(context: str, exc: BaseException | None = None) -> None:
    log_path = writable_base_dir() / "PneumoniaPredictor_error.log"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"\n[{context}]\n")
        if exc is None:
            traceback.print_exc(file=file)
        else:
            file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if not is_image_file(path):
            raise ValueError(f"不支持的图片格式：{path.suffix}")
        return [path]
    if path.is_dir():
        images = [file_path for file_path in path.rglob("*") if is_image_file(file_path)]
        if not images:
            raise FileNotFoundError("这个文件夹里没有找到支持的图片。")
        return sorted(images)
    raise FileNotFoundError(f"找不到路径：{path}")


class Predictor:
    def __init__(self, checkpoint_path: Path, device_name: str = "auto", image_size: int = 224) -> None:
        self.device = get_device(device_name)
        self.image_size = image_size
        self.checkpoint_path = checkpoint_path
        self.model, self.checkpoint, self.class_names = self._load_model(checkpoint_path)
        if "pneumonia" not in self.class_names:
            raise ValueError(f"模型类别中没有 pneumonia：{self.class_names}")
        self.positive_index = self.class_names.index("pneumonia")
        self.negative_index = 1 - self.positive_index
        _, self.transform = build_transforms(image_size=image_size, augment=False)

    def _load_model(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        class_names = list(checkpoint.get("class_names") or [])
        if len(class_names) != 2:
            raise ValueError(f"当前界面只支持二分类模型，读到类别：{class_names}")
        model = create_model_from_checkpoint(checkpoint, num_classes=len(class_names)).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint, class_names

    @torch.no_grad()
    def predict(
        self,
        image_path: Path,
        threshold: float,
        include_heatmap: bool = False,
        heatmap_dir: Path | None = None,
    ) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pneumonia_probability = float(probabilities[self.positive_index])
        prediction_index = self.positive_index if pneumonia_probability >= threshold else self.negative_index
        heatmap_path: Path | None = None
        focus_regions: list[str] = []
        heatmap_error = ""

        if include_heatmap:
            try:
                heatmap = generate_gradcam(self.model, image_tensor, prediction_index)
                focus_regions = describe_focus_regions(heatmap)
                output_dir = heatmap_dir or (writable_base_dir() / "outputs" / "heatmaps")
                heatmap_path = save_gradcam_overlay(image, heatmap, output_dir / safe_heatmap_name(image_path))
            except Exception as exc:
                heatmap_error = f"Grad-CAM 生成失败：{exc}"

        explanation = build_probability_explanation(
            class_names=self.class_names,
            probabilities=probabilities,
            positive_index=self.positive_index,
            prediction_index=prediction_index,
            threshold=threshold,
            focus_regions=focus_regions,
            heatmap_path=heatmap_path,
        )
        if heatmap_error:
            explanation["reasons"].append(heatmap_error)
        medical_basis = build_medical_basis(
            prediction=self.class_names[prediction_index],
            positive_probability=pneumonia_probability,
            threshold=threshold,
            focus_regions=focus_regions,
        )

        return {
            "image": str(image_path),
            "prediction": self.class_names[prediction_index],
            "pneumonia_probability": pneumonia_probability,
            "threshold": threshold,
            "reason_summary": explanation["reason_summary"],
            "explanation": explanation,
            "medical_basis_summary": medical_basis["summary"],
            "medical_basis": medical_basis,
            "probabilities": {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.class_names)
            },
        }


class PredictionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x760")
        self.minsize(1080, 640)

        self.predictor: Predictor | None = None
        self.current_image_path: Path | None = None
        self.original_preview_image: ImageTk.PhotoImage | None = None
        self.heatmap_preview_image: ImageTk.PhotoImage | None = None
        self.results: list[dict[str, Any]] = []

        self.threshold_var = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.6f}")
        self.status_var = tk.StringVar(value="正在加载模型...")
        self.result_var = tk.StringVar(value="请选择图片开始预测")
        self.probability_var = tk.StringVar(value="pneumonia 概率：--")
        self.reason_var = tk.StringVar(value="评判依据：--")
        self.medical_basis_var = tk.StringVar(value="医学参考依据：--")

        self._build_ui()
        self._set_controls_state("disabled")
        threading.Thread(target=self._load_predictor_worker, daemon=True).start()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        ttk.Label(left, text="原图", font=("Microsoft YaHei UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="Grad-CAM 热力图", font=("Microsoft YaHei UI", 13, "bold")).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        self.preview_label = ttk.Label(left, text="还没有选择图片", anchor="center", relief="groove")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0), padx=(0, 6))
        self.heatmap_label = ttk.Label(
            left,
            text="单张预测后显示热力图",
            anchor="center",
            relief="groove",
            justify="center",
        )
        self.heatmap_label.grid(row=1, column=1, sticky="nsew", pady=(8, 0), padx=(6, 0))

        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(8, weight=1)

        ttk.Label(right, text="预测结果", font=("Microsoft YaHei UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(right, textvariable=self.result_var, font=("Microsoft YaHei UI", 18, "bold")).grid(
            row=1, column=0, sticky="w", pady=(12, 4)
        )
        ttk.Label(right, textvariable=self.probability_var, font=("Microsoft YaHei UI", 11)).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(
            right,
            textvariable=self.reason_var,
            font=("Microsoft YaHei UI", 10),
            justify="left",
            wraplength=430,
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            right,
            textvariable=self.medical_basis_var,
            font=("Microsoft YaHei UI", 10),
            justify="left",
            wraplength=430,
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        threshold_frame = ttk.Frame(right)
        threshold_frame.grid(row=5, column=0, sticky="ew", pady=(14, 8))
        threshold_frame.columnconfigure(1, weight=1)
        ttk.Label(threshold_frame, text="判断阈值").grid(row=0, column=0, sticky="w")
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=12)
        self.threshold_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(threshold_frame, text="pneumonia 概率 ≥ 阈值时判断为肺炎").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        button_frame = ttk.Frame(right)
        button_frame.grid(row=6, column=0, sticky="ew", pady=(8, 8))
        for index in range(4):
            button_frame.columnconfigure(index, weight=1)

        self.open_button = ttk.Button(button_frame, text="选择图片", command=self.choose_image)
        self.open_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.folder_button = ttk.Button(button_frame, text="选择文件夹", command=self.choose_folder)
        self.folder_button.grid(row=0, column=1, sticky="ew", padx=6)
        self.predict_button = ttk.Button(button_frame, text="重新预测", command=self.predict_current_image)
        self.predict_button.grid(row=0, column=2, sticky="ew", padx=6)
        self.save_button = ttk.Button(button_frame, text="保存结果", command=self.save_results)
        self.save_button.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        ttk.Label(right, text="批量结果").grid(row=7, column=0, sticky="w", pady=(10, 4))
        columns = ("prediction", "probability", "image")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=12)
        self.tree.heading("prediction", text="预测")
        self.tree.heading("probability", text="pneumonia 概率")
        self.tree.heading("image", text="图片")
        self.tree.column("prediction", width=90, anchor="center")
        self.tree.column("probability", width=120, anchor="center")
        self.tree.column("image", width=360, anchor="w")
        self.tree.grid(row=8, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 6))
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _set_controls_state(self, state: str) -> None:
        for widget in (
            self.open_button,
            self.folder_button,
            self.predict_button,
            self.save_button,
            self.threshold_entry,
        ):
            widget.configure(state=state)

    def _load_predictor_worker(self) -> None:
        try:
            checkpoint_path = resource_path(DEFAULT_CHECKPOINT)
            predictor = Predictor(checkpoint_path=checkpoint_path, device_name="auto")
        except Exception as exc:
            self.after(0, lambda: self._show_load_error(exc))
            return
        self.after(0, lambda: self._on_predictor_loaded(predictor))

    def _show_load_error(self, exc: Exception) -> None:
        log_exception("model load failed", exc)
        self.status_var.set("模型加载失败")
        messagebox.showerror("模型加载失败", f"{exc}\n\n详细信息已写入 PneumoniaPredictor_error.log")

    def _on_predictor_loaded(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self._set_controls_state("normal")
        self.status_var.set(f"模型已加载：{predictor.checkpoint_path}，设备：{predictor.device}")

    def parse_threshold(self) -> float:
        try:
            threshold = float(self.threshold_var.get())
        except ValueError as exc:
            raise ValueError("阈值必须是数字。") from exc
        if not 0 <= threshold <= 1:
            raise ValueError("阈值必须在 0 到 1 之间。")
        return threshold

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择胸片图片",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.current_image_path = Path(path)
        self.show_preview(self.current_image_path)
        self.clear_heatmap_preview("正在生成热力图...")
        self.predict_paths([self.current_image_path])

    def choose_folder(self) -> None:
        path = filedialog.askdirectory(title="选择图片文件夹")
        if not path:
            return
        try:
            image_paths = collect_images(Path(path))
        except Exception as exc:
            messagebox.showerror("读取文件夹失败", str(exc))
            return
        if image_paths:
            self.current_image_path = image_paths[0]
            self.show_preview(image_paths[0])
            self.clear_heatmap_preview("批量预测不会自动生成热力图\n选中图片后点击“重新预测”")
        self.predict_paths(image_paths)

    def predict_current_image(self) -> None:
        if self.current_image_path is None:
            messagebox.showinfo("提示", "请先选择一张图片。")
            return
        self.predict_paths([self.current_image_path])

    def predict_paths(self, image_paths: list[Path]) -> None:
        if self.predictor is None:
            messagebox.showinfo("提示", "模型还在加载，请稍等。")
            return
        try:
            threshold = self.parse_threshold()
        except ValueError as exc:
            messagebox.showerror("阈值错误", str(exc))
            return

        self._set_controls_state("disabled")
        self.status_var.set(f"正在预测 {len(image_paths)} 张图片...")
        include_heatmap = len(image_paths) == 1
        threading.Thread(
            target=self._predict_worker,
            args=(image_paths, threshold, include_heatmap),
            daemon=True,
        ).start()

    def _predict_worker(self, image_paths: list[Path], threshold: float, include_heatmap: bool) -> None:
        assert self.predictor is not None
        try:
            heatmap_dir = writable_base_dir() / "outputs" / "heatmaps" if include_heatmap else None
            results = [
                self.predictor.predict(
                    path,
                    threshold,
                    include_heatmap=include_heatmap,
                    heatmap_dir=heatmap_dir,
                )
                for path in image_paths
            ]
        except Exception as exc:
            self.after(0, lambda: self._show_predict_error(exc))
            return
        self.after(0, lambda: self._on_prediction_done(results))

    def _show_predict_error(self, exc: Exception) -> None:
        log_exception("prediction failed", exc)
        self._set_controls_state("normal")
        self.status_var.set("预测失败")
        messagebox.showerror("预测失败", f"{exc}\n\n详细信息已写入 PneumoniaPredictor_error.log")

    def _on_prediction_done(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.tree.delete(*self.tree.get_children())

        for index, result in enumerate(results):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    result["prediction"],
                    f"{result['pneumonia_probability']:.6f}",
                    result["image"],
                ),
            )

        if results:
            self.update_result_panel(results[0])
            self.current_image_path = Path(results[0]["image"])
            self.show_preview(self.current_image_path)
            self.show_heatmap_from_result(results[0])

        self._set_controls_state("normal")
        self.status_var.set(f"预测完成：{len(results)} 张图片")

    def update_result_panel(self, result: dict[str, Any]) -> None:
        prediction = result["prediction"]
        if prediction == "pneumonia":
            label = "预测：pneumonia（肺炎）"
        else:
            label = "预测：normal（正常）"
        self.result_var.set(label)
        self.probability_var.set(
            f"pneumonia 概率：{result['pneumonia_probability']:.6f}，阈值：{result['threshold']:.6f}"
        )
        explanation = result.get("explanation", {})
        reasons = explanation.get("reasons", [])
        if reasons:
            shown_reasons = reasons[:4]
            self.reason_var.set("评判依据：\n" + "\n".join(f"{index + 1}. {reason}" for index, reason in enumerate(shown_reasons)))
        else:
            self.reason_var.set("评判依据：暂无详细原因")
        medical_basis = result.get("medical_basis")
        if medical_basis:
            self.medical_basis_var.set("医学参考依据：\n" + medical_basis_text(medical_basis, max_items=3))
        else:
            self.medical_basis_var.set("医学参考依据：暂无")

    def show_preview(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((360, 520))
        self.original_preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.original_preview_image, text="")

    def clear_heatmap_preview(self, text: str = "暂无热力图") -> None:
        self.heatmap_preview_image = None
        self.heatmap_label.configure(image="", text=text)

    def show_heatmap_from_result(self, result: dict[str, Any]) -> None:
        heatmap_path = result.get("explanation", {}).get("heatmap_path", "")
        if not heatmap_path:
            self.clear_heatmap_preview("没有热力图\n单张重新预测可生成")
            return

        path = Path(heatmap_path)
        if not path.exists():
            self.clear_heatmap_preview(f"找不到热力图文件\n{path}")
            return

        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 520))
        self.heatmap_preview_image = ImageTk.PhotoImage(image)
        self.heatmap_label.configure(image=self.heatmap_preview_image, text="")

    def on_tree_select(self, _event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if index >= len(self.results):
            return
        result = self.results[index]
        self.current_image_path = Path(result["image"])
        self.update_result_panel(result)
        self.show_preview(self.current_image_path)
        self.show_heatmap_from_result(result)

    def save_results(self) -> None:
        if not self.results:
            messagebox.showinfo("提示", "还没有预测结果。")
            return
        path = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="predictions.json",
        )
        if not path:
            return
        output = {
            "checkpoint": str(resource_path(DEFAULT_CHECKPOINT)),
            "threshold": self.threshold_var.get(),
            "results": self.results,
        }
        Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"已保存结果：{path}")


def main() -> None:
    try:
        app = PredictionApp()
        app.mainloop()
    except Exception as exc:
        log_exception("unhandled app error", exc)
        raise


if __name__ == "__main__":
    main()
