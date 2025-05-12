import gradio as gr
import sys
import os
import logging

from typing import Any

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Log ayarları
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Mevcut dizini sys.path'e ekle
now_dir = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# Sekme importları
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab

# Ön gereksinimleri çalıştır
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# i18n başlat
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Discord presence'ı başlat (etkinse)
from tabs.settings.sections.presence import load_config_presence

if load_config_presence():
    from assets.discord_presence import RPCManager

    RPCManager.start_presence()

# Kurulumu kontrol et
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Temayı yükle
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "ParityError/Interstellar"

# Özel CSS: Tam ekran/indirme düğmelerini gizle ve kapsayıcı alanı 100x100 yap
custom_css = """
footer {display: none !important;}
.icon-button-wrapper {display: none !important;}
button.svelte-dpdy90 {
    width: 100px !important;
    height: 100px !important;
    padding: 0 !important;
    margin: 0 !important;
}
"""

# Gradio arayüzünü tanımla
with gr.Blocks(
    theme=my_applio, title="Voicy", css=custom_css
) as Applio:
    with gr.Row():  # Logo ve Markdown için yatay satır
        gr.Image(
            value="assets/1.jpg",
            width=100,  # Logonun boyutu zaten 100x100
            height=100,
            show_label=False,
            container=False,
            interactive=False  # İnteraktif özellikleri kapat
        )
        gr.Markdown("# Voicy")
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
                f"{port} portunda başlatılamadı, {port - 1} portunda tekrar deneniyor..."
            )
            port -= 1
        except Exception as error:
            print(f"Gradio başlatılırken bir hata oluştu: {error}")
            break
