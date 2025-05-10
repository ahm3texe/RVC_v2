import os, sys
import gradio as gr
from core import run_infer_script
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")

# collect model files and audio files
names = [os.path.join(root, file)
         for root, _, files in os.walk(model_root)
         for file in files
         if file.endswith((".pth", ".onnx"))]
def output_path_fn(input_audio_path):
    base = os.path.splitext(os.path.basename(input_audio_path))[0]
    return os.path.join(os.path.dirname(input_audio_path), base + '_output.wav')

# build single-inference interface
def inference_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Ses Modeli"),
                choices=sorted(names),
                value=names[0] if names else None,
                interactive=True
            )
            index_file = gr.Dropdown(
                label=i18n("Index Dosyası"),
                choices=[],  # populate via refresh if needed
                interactive=True
            )

        upload_audio = gr.Audio(
            label=i18n("Ses Dosyanızı Yükleyin"), type="filepath", interactive=True
        )
        audio = gr.Dropdown(
            label=i18n("Ses Dosyanızı Seçiniz"),
            choices=[f for f in os.listdir(audio_root) if f.endswith('.wav')],
            value=None,
            interactive=True
        )

        # core sliders
        pitch = gr.Slider(minimum=-24, maximum=24, step=1,
                          label=i18n("Perde Ayarı"), value=0, interactive=True)
        index_rate = gr.Slider(minimum=0, maximum=1,
                               label=i18n("Search Feature Ratio"), value=0.75, interactive=True)
        rms_mix_rate = gr.Slider(minimum=0, maximum=1,
                                 label=i18n("Volume Envelope"), value=1, interactive=True)
        protect = gr.Slider(minimum=0, maximum=0.5,
                            label=i18n("Protect Voiceless Consonants"), value=0.5, interactive=True)

        # fixed pitch extraction method
        f0_method = "rmvpe"

        convert_button = gr.Button(i18n("Convert"))
        vc_info = gr.Textbox(label=i18n("Output Information"))
        vc_audio = gr.Audio(label=i18n("Export Audio"))

    # callbacks
    audio.change(fn=lambda path: output_path_fn(path), inputs=[audio], outputs=[vc_audio])
    convert_button.click(
        fn=run_infer_script,
        inputs=[pitch, index_rate, rms_mix_rate, protect,
                f0_method, audio, model_file, index_file],
        outputs=[vc_info, vc_audio]
    )

# launch
def main():
    with gr.Blocks(title="Voicy") as demo:
        gr.Markdown("# Voicy")
        inference_tab()
    demo.launch(server_name="127.0.0.1", server_port=6969)

if __name__ == "__main__":
    main()
