import json
import os
import random
import sys

import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_tts_script
from tabs.inference.inference import (
    change_choices,
    create_folder_and_move_files,
    get_indexes,
    get_speakers_id,
    match_index,
    refresh_embedders_folders,
    extract_model_and_epoch,
    names,
    default_weight,
)

i18n = I18nAuto()

with open(
    os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
) as file:
    tts_voices_data = json.load(file)

short_names = [voice.get("ShortName", "") for voice in tts_voices_data]


def process_input(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file.read()
        gr.Info(f"The file has been loaded!")
        return file_path, file_path
    except UnicodeDecodeError:
        gr.Info(f"The file has to be in UTF-8 encoding.")
        return None, None

# TTS tab
def tts_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Ses Modeli"),
                info=i18n("Klonlamak istediğiniz ses modelini seçiniz."),
                choices=sorted(names, key=lambda x: extract_model_and_epoch(x)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            best_default_index_path = match_index(model_file.value)
            index_file = gr.Dropdown(
                label=i18n("Index Dosyası"),
                info=i18n("Model dosyanızın indexini seçiniz."),
                choices=get_indexes(),
                value=best_default_index_path,
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            refresh_button = gr.Button(i18n("Yenile"))

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    gr.Markdown(
        i18n(
            "Voicy, Konuşmadan Konuşmaya (Speech-to-Speech) dönüştürme yazılımıdır ve metinden konuşmaya (TTS) bileşenini çalıştırmak için EdgeTTS’i ara katman olarak kullanır."
        )
    )
    tts_voice = gr.Dropdown(
        label=i18n("TTS Sesleri"),
        info=i18n("Dönüştürme için kullanılacak TTS sesini seçin."),
        choices=short_names,
        interactive=True,
        value=random.choice(short_names),
    )

    tts_rate = gr.Slider(
        minimum=-100,
        maximum=100,
        step=1,
        label=i18n("TTS Hızı"),
        info=i18n("TTS hızını artırın veya azaltın."),
        value=0,
        interactive=True,
    )

    # Yalnızca Metinde Konuşmaya
    tts_text = gr.Textbox(
        label=i18n("Sentezlenecek Metin"),
        info=i18n("Sentezlenmesini İstediğiniz Metni Giriniz."),
        placeholder=i18n("Sentezlenmesini İstediğiniz Metni Giriniz."),
        lines=3,
    )

    with gr.Accordion(i18n("Ekstra Ayarlar"), open=False):
        with gr.Column():
            output_tts_path = gr.Textbox(
                label=i18n("TTS Sesi için Çıktı Yolu"),
                placeholder=i18n("Çıktı yolu giriniz"),
                value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
                interactive=True,
            )
            output_rvc_path = gr.Textbox(
                label=i18n("RVC Sesi için Çıktı Yolu"),
                placeholder=i18n("Çıktı yolu giriniz"),
                value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=i18n("Çıkış Formatı"),
                info=i18n("Ses dosyanızın uzantısını seçiniz."),
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="WAV",
                interactive=True,
            )
            sid = gr.Dropdown(
                label=i18n("Speaker ID"),
                info=i18n("Select the speaker ID to use for the conversion."),
                choices=get_speakers_id(model_file.value),
                value=0,
                interactive=True,
                visible=False,
            )
            split_audio = gr.Checkbox(
                label=i18n("Sesi Parçala"),
                info=i18n("Ses dosyasını belirli parçalar halinde işlem uygular."),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune = gr.Checkbox(
                label=i18n("Autotune"),
                info=i18n("Sese autotune uygular."),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Autotune Strength"),
                info=i18n("Set the autotune strength."),
                visible=False,
                value=1,
                interactive=True,
            )
            clean_audio = gr.Checkbox(
                label=i18n("Sesi Temizleyin"),
                info=i18n("Sesi gürültüden arındırır."),
                visible=True,
                value=True,
                interactive=True,
            )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Ses Temizle Gücü"),
                info=i18n("Temizleme seviyesini ayarlayın."),
                visible=True,
                value=0.5,
                interactive=True,
            )
            pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Perde Ayarı"),
                info=i18n("Sesin perdesini ayarlayın."),
                value=0,
                interactive=True,
            )
            # ... diğer ekstra ayarlar ...
            f0_file = gr.File(
                label=i18n(
                    "The f0 curve represents the variations in the base frequency..."
                ),
                visible=False,
            )

    convert_button = gr.Button(i18n("Convert"))

    with gr.Row():
        vc_output1 = gr.Textbox(
            label=i18n("Çıktı Bilgisi"),
            info=i18n("Çıktı bilgisi burada gösterilecektir."),
        )
        vc_output2 = gr.Audio(label=i18n("Oluşturulan Ses"))

    # Gerekli görünümleri toggle etmek için callback’ler
    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    autotune.change(fn=toggle_visible, inputs=[autotune], outputs=[autotune_strength])
    clean_audio.change(fn=toggle_visible, inputs=[clean_audio], outputs=[clean_strength])
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, sid, sid],
    )

    # Son olarak Convert tuşuna basıldığında yalnızca tts_text’i alıyoruz
    convert_button.click(
        fn=run_tts_script,
        inputs=[
            tts_text,
            tts_voice,
            tts_rate,
            pitch,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0_method,
            output_tts_path,
            output_rvc_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            f0_file,
            embedder_model,
            embedder_model_custom,
            sid,
        ],
        outputs=[vc_output1, vc_output2],
    )
