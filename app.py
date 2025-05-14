import gradio as gr
import sys
import os
import logging

from typing import Any

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "ParityError/Interstellar"

# Define the Citrus theme
theme = gr.themes.Citrus(
    primary_hue="red",
    secondary_hue="red",
    neutral_hue="gray",
    radius_size=gr.themes.Size(lg="24px", md="15px", sm="20px", xl="28px", xs="8px", xxl="25px", xxs="6px"),
)

# Define Gradio interface with the Citrus theme
with gr.Blocks(
    theme=theme,  # Use the Citrus theme here
    title="Voicy",

) as Applio:
    with gr.Row():
        gr.HTML(
            '<div class="logo-container"><img id="voicy-logo" src="https://raw.githubusercontent.com/ahm3texe/RVC_v2/refs/heads/main/assets/logo.png" alt="Voicy Logo"></div>'
        )

    gr.Markdown(
        i18n(
            "İstanbul Sabahattin Zaim Üniversitesi  \nYüksek Kalitede Ses Klonlama Hizmeti"
        )
    )
    gr.Markdown(
        i18n(
            "[GitHub](https://github.com/ahm3texe/RVC_v2)"
        )
    )
    with gr.Tab(i18n("Klonlama Arayüzü")):
        inference_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Eğitim")):
        train_tab()

    with gr.Tab(i18n("Model İndirme")):
        download_tab()

    gr.Markdown(
        """
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    </div>
    """
    )


def launch_gradio(server_name: str, server_port: int) -> None:
    Applio.launch(
        favicon_path="assets/favicon.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
    )


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
