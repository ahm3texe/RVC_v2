import os
import shutil
import sys
from multiprocessing import cpu_count

import gradio as gr

from assets.i18n.i18n import I18nAuto
from core import (
    run_extract_script,
    run_index_script,
    run_preprocess_script,
    run_prerequisites_script,
    run_train_script,
)
from rvc.configs.config import get_gpu_info, get_number_of_gpus, max_vram_gpu
from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_train

i18n = I18nAuto()
now_dir = os.getcwd()
sys.path.append(now_dir)


sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

# Custom Pretraineds
pretraineds_custom_path = os.path.join(
    now_dir, "rvc", "models", "pretraineds", "custom"
)

pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)

custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)
custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)

os.makedirs(custom_embedder_root, exist_ok=True)
os.makedirs(pretraineds_custom_path_relative, exist_ok=True)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path_relative)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]


pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")


def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )


# Dataset Creator
datasets_path = os.path.join(now_dir, "assets", "datasets")

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

datasets_path_relative = os.path.relpath(datasets_path, now_dir)


def get_datasets_list():
    return [
        dirpath
        for dirpath, _, filenames in os.walk(datasets_path_relative)
        if any(filename.endswith(tuple(sup_audioext)) for filename in filenames)
    ]


def refresh_datasets():
    return {"choices": sorted(get_datasets_list()), "__type__": "update"}


# Model Names
models_path = os.path.join(now_dir, "logs")


def get_models_list():
    return [
        os.path.basename(dirpath)
        for dirpath in os.listdir(models_path)
        if os.path.isdir(os.path.join(models_path, dirpath))
        and all(excluded not in dirpath for excluded in ["zips", "mute", "reference"])
    ]


def refresh_models():
    return {"choices": sorted(get_models_list()), "__type__": "update"}


# Refresh Models and Datasets
def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )


# Refresh Custom Embedders
def get_embedder_custom_list():
    return [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]


def refresh_custom_embedder_list():
    return {"choices": sorted(get_embedder_custom_list()), "__type__": "update"}


# Drop Model
def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info(
            i18n(
                "The file you dropped is not a valid pretrained file. Please try again."
            )
        )
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path_relative, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        shutil.copy(dropbox, pretrained_path)
        gr.Info(
            i18n(
                "Click the refresh button to see the pretrained file in the dropdown menu."
            )
        )
    return None


# Drop Dataset
def save_drop_dataset_audio(dropbox, dataset_name):
    if not dataset_name:
        gr.Info("Please enter a valid dataset name. Please try again.")
        return None, None
    else:
        file_extension = os.path.splitext(dropbox)[1][1:].lower()
        if file_extension not in sup_audioext:
            gr.Info("The file you dropped is not a valid audio file. Please try again.")
        else:
            dataset_name = format_title(dataset_name)
            audio_file = format_title(os.path.basename(dropbox))
            dataset_path = os.path.join(now_dir, "assets", "datasets", dataset_name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            destination_path = os.path.join(dataset_path, audio_file)
            if os.path.exists(destination_path):
                os.remove(destination_path)
            shutil.copy(dropbox, destination_path)
            gr.Info(
                i18n(
                    "The audio file has been successfully added to the dataset. Please click the preprocess button."
                )
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path


# Drop Custom Embedder
def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)

    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))
    if config_file:
        shutil.copy(
            config_file, os.path.join(target_folder, os.path.basename(config_file))
        )

    return f"Files moved to folder {target_folder}"


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders


# Export
def get_pth_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".pth")
    ]


def get_index_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]


def refresh_pth_and_index_list():
    return (
        {"choices": sorted(get_pth_list()), "__type__": "update"},
        {"choices": sorted(get_index_list()), "__type__": "update"},
    )


# Export Pth and Index Files
def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [
        os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths
    ]
    normalized_pth_path = os.path.abspath(os.path.join(now_dir, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        print(f"Attempted to export invalid pth path: {pth_path}")
        return None


def export_index(index_path):
    allowed_paths = get_index_list()
    normalized_allowed_paths = [
        os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths
    ]
    normalized_index_path = os.path.abspath(os.path.join(now_dir, index_path))

    if normalized_index_path in normalized_allowed_paths:
        return index_path
    else:
        print(f"Attempted to export invalid index path: {index_path}")
        return None


# Upload to Google Drive
def upload_to_google_drive(pth_path, index_path):
    def upload_file(file_path):
        if file_path:
            try:
                gr.Info(f"Uploading {pth_path} to Google Drive...")
                google_drive_folder = "/content/drive/MyDrive/ApplioExported"
                if not os.path.exists(google_drive_folder):
                    os.makedirs(google_drive_folder)
                google_drive_file_path = os.path.join(
                    google_drive_folder, os.path.basename(file_path)
                )
                if os.path.exists(google_drive_file_path):
                    os.remove(google_drive_file_path)
                shutil.copy2(file_path, google_drive_file_path)
                gr.Info("File uploaded successfully.")
            except Exception as error:
                print(f"An error occurred uploading to Google Drive: {error}")
                gr.Info("Error uploading to Google Drive")

    upload_file(pth_path)
    upload_file(index_path)


# Train Tab
def train_tab():
    # Model settings section
    with gr.Accordion(i18n("Model Ayarları")):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label=i18n("Model Adı"),
                    info=i18n("Yeni modelin adı."),
                    choices=get_models_list(),
                    value="benim-projem",
                    interactive=True,
                    allow_custom_value=True,
                )
                architecture = gr.Radio( # silinebilir satır
                    label=i18n("Architecture"),
                    info=i18n(
                        "Choose the model architecture:\n- **RVC (V2)**: Default option, compatible with all clients.\n- **Applio**: Advanced quality with improved vocoders and higher sample rates, Applio-only."
                    ),
                    choices=["RVC", "Applio"],
                    value="RVC",
                    interactive=True,
                    visible=False,  # to be visible once pretraineds are ready
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label=i18n("Sampling Oranı"),
                    info=i18n("Ses dosyalarının örnekleme oranı."),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )
                vocoder = gr.Radio(
                    label=i18n("Vocoder"),
                    info=i18n(
                        "Choose the vocoder for audio synthesis:\n- **HiFi-GAN**: Default option, compatible with all clients.\n- **MRF HiFi-GAN**: Higher fidelity, Applio-only.\n- **RefineGAN**: Superior audio quality, Applio-only."
                    ),
                    choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                    value="HiFi-GAN",
                    interactive=False,
                    visible=False,  # to be visible once pretraineds are ready
                )
        with gr.Accordion(
            i18n("Ekstra Ayarlar"),
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    cpu_cores = gr.Slider(
                        1,
                        min(cpu_count(), 32),  # max 32 parallel processes
                        min(cpu_count(), 32),
                        step=1,
                        label=i18n("CPU Çekirdekleri"),
                        info=i18n(
                            "Çıkarım sürecinde kullanılacak CPU çekirdeği sayısı. Varsayılan ayar, çoğu durumda önerilen CPU çekirdeklerinizdir."
                        ),
                        interactive=True,
                    )

                with gr.Column():
                    gpu = gr.Textbox(
                        label=i18n("GPU Numarası"),
                        info=i18n(
                            "Çıkarım işlemi için kullanmak istediğiniz GPU sayısını tire (-) ile ayırarak girin."
                        ),
                        placeholder=i18n("0 to ∞ separated by -"),
                        value=str(get_number_of_gpus()),
                        interactive=True,
                    )
                    gr.Textbox(
                        label=i18n("GPU Bilgisi"),
                        info=i18n("GPU bilgisi burada görüntülenecek."),
                        value=get_gpu_info(),
                        interactive=False,
                    )
    # Preprocess section
    with gr.Accordion(i18n("Ön İşleme")):
        dataset_path = gr.Dropdown(
            label=i18n("Veri Kümesi Yolu"),
            info=i18n("Veri kümesi klasörünün yolu."),
            # placeholder=i18n("Veri kümesi yolunu girin"),
            choices=get_datasets_list(),
            allow_custom_value=True,
            interactive=True,
        )
        dataset_creator = gr.Checkbox(
            label=i18n("Veri kümesi Oluşturucusu"),
            value=False,
            interactive=True,
            visible=True,
        )
        with gr.Column(visible=False) as dataset_creator_settings:
            with gr.Accordion(i18n("Veri kümesi Oluşturucusu")):
                dataset_name = gr.Textbox(
                    label=i18n("Veri Seti Adı"),
                    info=i18n("Yeni veri kümesinin adı."),
                    placeholder=i18n("Veri kümesi adını girin."),
                    interactive=True,
                )
                upload_audio_dataset = gr.File(
                    label=i18n("Ses Veri Setini Yükleyin."),
                    type="filepath",
                    interactive=True,
                )
        refresh = gr.Button(i18n("Yenile"))

        with gr.Accordion(i18n("Ekstra Ayarlar"), open=False):
            cut_preprocess = gr.Radio(
                label=i18n("Ses Kesme"),
                info=i18n(
                    "Ses dosyası dilimleme yöntemi: Dosyalar zaten önceden dilimlenmişse ‘Atla’yı seçin, dosyalardan aşırı sessizlik zaten kaldırılmışsa ‘Basit’i seçin veya otomatik sessizlik algılama ve buna göre dilimleme için ‘Otomatik’i seçin."
                ),
                choices=["Atla", "Basit", "Otomatik"],
                value="Automatic",
                interactive=True,
            )
            with gr.Row():
                chunk_len = gr.Slider(
                    0.5,
                    5.0,
                    3.0,
                    step=0.1,
                    label=i18n("Parça uzunluğu (saniye)"),
                    info=i18n("‘Basit’ yöntemi için ses diliminin uzunluğu."),
                    interactive=True,
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.4,
                    0.3,
                    step=0.1,
                    label=i18n("Çakışma uzunluğu (saniye)"),
                    info=i18n(
                        "‘Basit’ yöntemi için dilimler arasındaki çakışma uzunluğu."
                    ),
                    interactive=True,
                )

            with gr.Row():
                process_effects = gr.Checkbox(
                    label=i18n("İşlem efektleri"),
                    info=i18n(
                        "Veri kümeniz zaten işlenmişse bu seçeneğin devre dışı bırakılması önerilir."
                    ),
                    value=True,
                    interactive=True,
                    visible=True,
                )
                noise_reduction = gr.Checkbox(
                    label=i18n("Gürültü Azaltma"),
                    info=i18n(
                        "Veri kümeniz zaten işlenmişse bu seçeneğin devre dışı tutulması önerilir."
                    ),
                    value=False,
                    interactive=True,
                    visible=True,
                )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Gürültü Azaltma Gücü"),
                info=i18n(
                    "Verisetindeki seslerin temizlik düzeyini ayarlayın; ne kadar artırırsanız, o kadar fazla temizlenir, ancak sesin kalitesi düşer."
                ),
                visible=False,
                value=0.5,
                interactive=True,
            )
        preprocess_output_info = gr.Textbox(
            label=i18n("Çıktı Bilgisi"),
            info=i18n("Çıktı bilgisi burada gösterilecektir."),
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            preprocess_button = gr.Button(i18n("Preprocess Dataset"))
            preprocess_button.click(
                fn=run_preprocess_script,
                inputs=[
                    model_name,
                    dataset_path,
                    sampling_rate,
                    cpu_cores,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    clean_strength,
                    chunk_len,
                    overlap_len,
                ],
                outputs=[preprocess_output_info],
            )

    # Extract section
    with gr.Accordion(i18n("Çıkarım")):
        with gr.Row():
            f0_method = gr.Radio(
                label=i18n("Pitch çıkartma algoritması"),
                info=i18n(
                    "Ses dönüştürme için kullanılacak perde çıkarma algoritması. Varsayılan algoritma çoğu durumda önerilen rmvpe'dir."
                ),
                choices=["crepe", "crepe-tiny", "rmvpe"],
                value="rmvpe",
                interactive=True,
            )

            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Konuşmayı öğrenecek olan embedding model."),
                choices=[
                    "contentvec",
                    "chinese-hubert-base",
                    "japanese-hubert-base",
                    "korean-hubert-base",
                    "custom",
                ],
                value="contentvec",
                interactive=True,
            )
        include_mutes = gr.Slider(
            0,
            10,
            2,
            step=1,
            label=i18n("Sessiz eğitim dosyaları"),
            info=i18n(
                "Eğitim setine birkaç sessiz dosya eklemek, modelin çıkarılan ses dosyalarındaki saf sessizliği işlemesini sağlar. Veri setiniz temizse ve zaten saf sessizlik segmentleri içeriyorsa 0'ı seçin."
            ),
            value=True,
            interactive=True,
        )
        hop_length = gr.Slider(
            1,
            512,
            128,
            step=1,
            label=i18n("Hop Length"),
            info=i18n(
                "Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."
            ),
            visible=False,
            interactive=True,
        )
        with gr.Row(visible=False) as embedder_custom:
            with gr.Accordion("Custom Embedder", open=True):
                with gr.Row():
                    embedder_model_custom = gr.Dropdown(
                        label="Select Custom Embedder",
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    refresh_embedders_button = gr.Button("Refresh embedders")
                folder_name_input = gr.Textbox(label="Folder Name", interactive=True)
                with gr.Row():
                    bin_file_upload = gr.File(
                        label="Upload .bin", type="filepath", interactive=True
                    )
                    config_file_upload = gr.File(
                        label="Upload .json", type="filepath", interactive=True
                    )
                move_files_button = gr.Button("Move files to custom embedder folder")

        extract_output_info = gr.Textbox(
            label=i18n("Çıktı Bilgisi"),
            info=i18n("Çıktı bilgisi burada gösterilecektir."),
            value="",
            max_lines=8,
            interactive=False,
        )
        extract_button = gr.Button(i18n("Özellikleri Çıkar"))
        extract_button.click(
            fn=run_extract_script,
            inputs=[
                model_name,
                f0_method,
                hop_length,
                cpu_cores,
                gpu,
                sampling_rate,
                embedder_model,
                embedder_model_custom,
                include_mutes,
            ],
            outputs=[extract_output_info],
        )

    # Training section
    with gr.Accordion(i18n("Eğitim")):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                50,
                max_vram_gpu(0),
                step=1,
                label=i18n("Batch Boyutu"),
                info=i18n(
                    "Bu değeri GPU’nuzun kullanılabilir VRAM miktarıyla uyumlu hale getirmek önerilir. “4” daha yüksek doğruluk sunar ancak daha yavaş çalışır; “8” ise daha hızlı ve standart sonuçlar verir."
                ),
                interactive=True,
            )
            save_every_epoch = gr.Slider(
                1,
                100,
                10,
                step=1,
                label=i18n("Her Epochta Kaydet"),
                info=i18n("Modelin kaç epochta bir kaydedileceğini belirleyin."),
                interactive=True,
            )
            total_epoch = gr.Slider(
                1,
                10000,
                500,
                step=1,
                label=i18n("Toplam Epoch"),
                info=i18n(
                    "Model eğitim süreci için toplam epoch sayısını belirtir."
                ),
                interactive=True,
            )
        with gr.Accordion(i18n("Ekstra Ayarlar"), open=False):
            with gr.Row():
                with gr.Column():
                    save_only_latest = gr.Checkbox(
                        label=i18n("Yalnızca Sonuncuyu Kaydet"),
                        info=i18n(
                            "Bu ayarın etkinleştirilmesi, G ve D dosyalarının yalnızca en son sürümlerinin kaydedilmesini sağlar ve böylece depolama alanından tasarruf edilir."
                        ),
                        value=True,
                        interactive=True,
                    )
                    save_every_weights = gr.Checkbox(
                    label=i18n("Her Ağırlığı Kaydet"),
                    info=i18n(
                        "Bu ayar, her epoch sonunda modelin ağırlıklarının kaydedilmesini sağlar."
                    ),
                    value=True,
                    interactive=True,
                )
                pretrained = gr.Checkbox(
                    label=i18n("Önceden Eğitilmiş"),
                    info=i18n(
                        "Kendi modelinizi eğitirken önceden eğitilmiş modelleri kullanın. Bu yaklaşım, eğitim süresini kısaltır ve genel kaliteyi artırır."
                    ),
                    value=True,
                    interactive=True,
                )
            with gr.Column():
                cleanup = gr.Checkbox(
                    label=i18n("Yeni Eğitim"),
                    info=i18n(
                        "Bu ayarı yalnızca yeni bir model sıfırdan eğitiliyorsa veya eğitimi yeniden başlatıyorsanız etkinleştirin. Önceki tüm oluşturulmuş ağırlıkları ve tensorboard günlüklerini siler."
                    ),
                    value=False,
                    interactive=True,
                )
                cache_dataset_in_gpu = gr.Checkbox(
                    label=i18n("Veri Setini GPU'da Önbelleğe Al"),
                    info=i18n(
                        "Veri setini GPU belleğinde önbelleğe alarak eğitim sürecini hızlandırır."
                    ),
                    value=False,
                    interactive=True,
                )
                checkpointing = gr.Checkbox(
                    label=i18n("Kontrol Noktası"),
                    info=i18n(
                        "Bellek tasarruflu eğitimi etkinleştirir. Bu, VRAM kullanımını azaltır ancak eğitim hızını düşürür. Sınırlı belleğe sahip GPU'lar (örneğin, <6GB VRAM) veya normalden daha büyük bir toplu iş boyutuyla eğitim yaparken kullanışlıdır."
                    ),
                    value=False,
                    interactive=True,
                )
        with gr.Row():
            custom_pretrained = gr.Checkbox(
                label=i18n("Özel Önceden Eğitilmiş"),
                info=i18n(
                    "Özel önceden eğitilmiş modeller kullanmak, özellikle kullanım durumuna en uygun önceden eğitilmiş modeller seçildiğinde, üstün sonuçlar elde edilmesini sağlayabilir."
                ),
                value=False,
                interactive=True,
            )
            overtraining_detector = gr.Checkbox(
                label=i18n("Aşırı Eğitim Dedektörü"),
                info=i18n(
                    "Modelin eğitim verilerini aşırı öğrenmesini ve yeni verilere genelleme yeteneğini kaybetmesini önlemek için aşırı eğitimi tespit eder."
                ),
                value=False,
                interactive=True,
            )
            with gr.Row():
                with gr.Column(visible=False) as pretrained_custom_settings:
                    with gr.Accordion(i18n("Pretrained Custom Settings")):
                        upload_pretrained = gr.File(
                            label=i18n("Upload Pretrained Model"),
                            type="filepath",
                            interactive=True,
                        )
                        refresh_custom_pretaineds_button = gr.Button(
                            i18n("Refresh Custom Pretraineds")
                        )
                        g_pretrained_path = gr.Dropdown(
                            label=i18n("Custom Pretrained G"),
                            info=i18n(
                                "Select the custom pretrained model for the generator."
                            ),
                            choices=sorted(pretraineds_list_g),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        d_pretrained_path = gr.Dropdown(
                            label=i18n("Custom Pretrained D"),
                            info=i18n(
                                "Select the custom pretrained model for the discriminator."
                            ),
                            choices=sorted(pretraineds_list_d),
                            interactive=True,
                            allow_custom_value=True,
                        )

                with gr.Column(visible=False) as overtraining_settings:
                    with gr.Accordion(i18n("Overtraining Detector Settings")):
                        overtraining_threshold = gr.Slider(
                            1,
                            100,
                            50,
                            step=1,
                            label=i18n("Overtraining Threshold"),
                            info=i18n(
                                "Set the maximum number of epochs you want your model to stop training if no improvement is detected."
                            ),
                            interactive=True,
                        )
            index_algorithm = gr.Radio(
                label=i18n("Index Algorithm"),
                info=i18n(
                    "KMeans is a clustering algorithm that divides the dataset into K clusters. This setting is particularly useful for large datasets."
                ),
                choices=["Auto", "Faiss", "KMeans"],
                value="Auto",
                interactive=True,
            )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message
            return run_train_script(*args)

        terms_checkbox = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            info=i18n(
                "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your training."
            ),
            value=False,
            interactive=True,
        )
        train_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            train_button = gr.Button(i18n("Start Training"))
            train_button.click(
                fn=enforce_terms,
                inputs=[
                    terms_checkbox,
                    model_name,
                    save_every_epoch,
                    save_only_latest,
                    save_every_weights,
                    total_epoch,
                    sampling_rate,
                    batch_size,
                    gpu,
                    overtraining_detector,
                    overtraining_threshold,
                    pretrained,
                    cleanup,
                    index_algorithm,
                    cache_dataset_in_gpu,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                    vocoder,
                    checkpointing,
                ],
                outputs=[train_output_info],
            )

            stop_train_button = gr.Button(i18n("Stop Training"), visible=False)
            stop_train_button.click(
                fn=stop_train,
                inputs=[model_name],
                outputs=[],
            )

            index_button = gr.Button(i18n("Generate Index"))
            index_button.click(
                fn=run_index_script,
                inputs=[model_name, index_algorithm],
                outputs=[train_output_info],
            )

    # Export Model section
    with gr.Accordion(i18n("Export Model"), open=False):
        if not os.name == "nt":
            gr.Markdown(
                i18n(
                    "The button 'Upload' is only for google colab: Uploads the exported files to the ApplioExported folder in your Google Drive."
                )
            )
        with gr.Row():
            with gr.Column():
                pth_file_export = gr.File(
                    label=i18n("Exported Pth file"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                pth_dropdown_export = gr.Dropdown(
                    label=i18n("Pth file"),
                    info=i18n("Select the pth file to be exported"),
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Column():
                index_file_export = gr.File(
                    label=i18n("Exported Index File"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                index_dropdown_export = gr.Dropdown(
                    label=i18n("Index File"),
                    info=i18n("Select the index file to be exported"),
                    choices=get_index_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        with gr.Row():
            with gr.Column():
                refresh_export = gr.Button(i18n("Refresh"))
                if not os.name == "nt":
                    upload_exported = gr.Button(i18n("Upload"))
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return {"visible": checkbox, "__type__": "update"}

            def toggle_visible_hop_length(f0_method):
                if f0_method == "crepe" or f0_method == "crepe-tiny":
                    return {"visible": True, "__type__": "update"}
                return {"visible": False, "__type__": "update"}

            def toggle_pretrained(pretrained, custom_pretrained):
                if custom_pretrained == False:
                    return {"visible": pretrained, "__type__": "update"}, {
                        "visible": False,
                        "__type__": "update",
                    }
                else:
                    return {"visible": pretrained, "__type__": "update"}, {
                        "visible": pretrained,
                        "__type__": "update",
                    }

            def enable_stop_train_button():
                return {"visible": False, "__type__": "update"}, {
                    "visible": True,
                    "__type__": "update",
                }

            def disable_stop_train_button():
                return {"visible": True, "__type__": "update"}, {
                    "visible": False,
                    "__type__": "update",
                }

            def download_prerequisites():
                gr.Info(
                    "Checking for prerequisites with pitch guidance... Missing files will be downloaded. If you already have them, this step will be skipped."
                )
                run_prerequisites_script(
                    pretraineds_hifigan=True,
                    models=False,
                    exe=False,
                )
                gr.Info(
                    "Prerequisites check complete. Missing files were downloaded, and you may now start preprocessing."
                )

            def toggle_visible_embedder_custom(embedder_model):
                if embedder_model == "custom":
                    return {"visible": True, "__type__": "update"}
                return {"visible": False, "__type__": "update"}

            def toggle_architecture(architecture):
                if architecture == "Applio":
                    return {
                        "choices": ["32000", "40000", "48000"],
                        "__type__": "update",
                    }, {
                        "interactive": True,
                        "__type__": "update",
                    }
                else:
                    return {
                        "choices": ["32000", "40000", "48000"],
                        "__type__": "update",
                        "value": "40000",
                    }, {"interactive": False, "__type__": "update", "value": "HiFi-GAN"}

            def update_slider_visibility(noise_reduction):
                return gr.update(visible=noise_reduction)

            noise_reduction.change(
                fn=update_slider_visibility,
                inputs=noise_reduction,
                outputs=clean_strength,
            )
            architecture.change(
                fn=toggle_architecture,
                inputs=[architecture],
                outputs=[sampling_rate, vocoder],
            )
            refresh.click(
                fn=refresh_models_and_datasets,
                inputs=[],
                outputs=[model_name, dataset_path],
            )
            dataset_creator.change(
                fn=toggle_visible,
                inputs=[dataset_creator],
                outputs=[dataset_creator_settings],
            )
            upload_audio_dataset.upload(
                fn=save_drop_dataset_audio,
                inputs=[upload_audio_dataset, dataset_name],
                outputs=[upload_audio_dataset, dataset_path],
            )
            f0_method.change(
                fn=toggle_visible_hop_length,
                inputs=[f0_method],
                outputs=[hop_length],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            move_files_button.click(
                fn=create_folder_and_move_files,
                inputs=[folder_name_input, bin_file_upload, config_file_upload],
                outputs=[],
            )
            refresh_embedders_button.click(
                fn=refresh_embedders_folders, inputs=[], outputs=[embedder_model_custom]
            )
            pretrained.change(
                fn=toggle_pretrained,
                inputs=[pretrained, custom_pretrained],
                outputs=[custom_pretrained, pretrained_custom_settings],
            )
            custom_pretrained.change(
                fn=toggle_visible,
                inputs=[custom_pretrained],
                outputs=[pretrained_custom_settings],
            )
            refresh_custom_pretaineds_button.click(
                fn=refresh_custom_pretraineds,
                inputs=[],
                outputs=[g_pretrained_path, d_pretrained_path],
            )
            upload_pretrained.upload(
                fn=save_drop_model,
                inputs=[upload_pretrained],
                outputs=[upload_pretrained],
            )
            overtraining_detector.change(
                fn=toggle_visible,
                inputs=[overtraining_detector],
                outputs=[overtraining_settings],
            )
            train_button.click(
                fn=enable_stop_train_button,
                inputs=[],
                outputs=[train_button, stop_train_button],
            )
            train_output_info.change(
                fn=disable_stop_train_button,
                inputs=[],
                outputs=[train_button, stop_train_button],
            )
            pth_dropdown_export.change(
                fn=export_pth,
                inputs=[pth_dropdown_export],
                outputs=[pth_file_export],
            )
            index_dropdown_export.change(
                fn=export_index,
                inputs=[index_dropdown_export],
                outputs=[index_file_export],
            )
            refresh_export.click(
                fn=refresh_pth_and_index_list,
                inputs=[],
                outputs=[pth_dropdown_export, index_dropdown_export],
            )
