import os
import sys
import json
import shutil
import requests
import tempfile
import gradio as gr
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_download_script
from rvc.lib.utils import format_title

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")

if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )

    file_name = format_title(os.path.basename(dropbox))
    model_name = file_name

    if ".pth" in model_name:
        model_name = model_name.split(".pth")[0]
    elif ".index" in model_name:
        replacements = ["nprobe_1_", "_v1", "_v2", "added_"]
        for rep in replacements:
            model_name = model_name.replace(rep, "")
        model_name = model_name.split(".index")[0]

    model_path = os.path.join(now_dir, "logs", model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(os.path.join(model_path, file_name)):
        os.remove(os.path.join(model_path, file_name))
    shutil.move(dropbox, os.path.join(model_path, file_name))
    print(f"{file_name} saved in {model_path}")
    gr.Info(f"{file_name} saved in {model_path}")

    return None


def download_tab():
    with gr.Column():
        gr.Markdown(value=i18n("## Model İndirme"))
        model_link = gr.Textbox(
            label=i18n("Model Linki"),
            placeholder=i18n("Model bağlantısını giriniz."),
            interactive=True,
        )
        model_download_output_info = gr.Textbox(
            label=i18n("Çıktı Bilgisi"),
            info=i18n("Çıktı bilgisi burada gösterilecektir."),
            value="",
            max_lines=8,
            interactive=False,
        )
        model_download_button = gr.Button(i18n("Modeli İndirin"))
        model_download_button.click(
            fn=run_download_script,
            inputs=[model_link],
            outputs=[model_download_output_info],
        )
