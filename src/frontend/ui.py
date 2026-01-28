# -----------LIBRARIES LOADING-------------

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
import os
import logging

# import datetime
import requests
import base64
import io
import numpy as np
import random

# Configurar logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------- PATH CONSTANTS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data"
IMAGES_DIR = BASE_DIR / "img"
VIDEO_PATH = BASE_DIR / "video" / "flask_demo.mp4"

# Configuración desde variables de entorno
# Busca la URL en la configuración de Streamlit Cloud
API_BASE_URL = "https://brain-tumor-detection-production-a58e.up.railway.app/"
# CSV files - rutas locales
ROUTE_LABEL_CSV_LOCAL = DATA_DIR / "route_label.csv"
SEGMENTATION_ROUTES_LABELS_CSV_LOCAL = DATA_DIR / "segmentation_routes_labels.csv"


def load_csv_local(csv_filename):
    """Carga un CSV desde archivo local"""
    local_path = DATA_DIR / csv_filename
    if local_path.exists():
        logger.info(f"Cargando {csv_filename} desde archivo local: {local_path}")
        return pd.read_csv(str(local_path))
    else:
        logger.error(f"CSV {csv_filename} no encontrado localmente en {local_path}")
        return pd.DataFrame()  # Retornar DataFrame vacío si no se encuentra


# Image files
KAGGLE_IMAGE = IMAGES_DIR / "kaggle.png"
TCIA_IMAGE = IMAGES_DIR / "TCIA.png"
GITHUB_IMAGE = IMAGES_DIR / "github.png"

# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- CUSTOM CSS STYLING ----------

st.markdown(
    """
    <style>
    /* Highlight analysis box */
    .highlight-box {
        border-left: 6px solid #0747d4;
        background-color: #F0F2F6;
        color: black;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14.5px;
    }

    </style>

    <style>
    /* Red highlight analysis box */
    .red-box {
        border-left: 6px solid #FF0000;
        background-color: #F0F2F6;
        color: black;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14.5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------DATAFRAME LOADING and ROUTES -------------

df = load_csv_local("route_label.csv")
df_tumors = load_csv_local("segmentation_routes_labels.csv")


def call_flask_model(
    api_url: str, pil_image: Image.Image, endpoint: str = "clasificacion"
):
    """
    Llama al modelo Flask con una imagen PIL.

    Args:
        api_url: URL base de la API
        pil_image: Imagen PIL
        endpoint: "clasificacion" o "segmentacion"

    Returns:
        dict: Respuesta JSON de la API

    Raises:
        requests.exceptions.HTTPError: Si la respuesta tiene un código de error HTTP
        requests.exceptions.ConnectionError: Si no se puede conectar al servidor
        requests.exceptions.Timeout: Si la solicitud excede el tiempo límite
        requests.exceptions.RequestException: Para otros errores de requests
    """
    try:
        pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        url = api_url.rstrip("/") + f"/{endpoint}/predict"
        logger.debug(f"Llamando a la API: {url}")

        files = {"image": ("image.png", img_bytes, "image/png")}

        resp = requests.post(url, files=files, timeout=60)

        # Verificar código de estado antes de procesar
        if resp.status_code >= 400:
            logger.error(
                f"Error HTTP {resp.status_code} desde {url}: {resp.text[:200]}"
            )

        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        # Re-lanzar HTTPError con información adicional
        logger.error(f"Error HTTP {e.response.status_code} llamando a {url}: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error llamando a la API {url}: {e}")
        raise


def decode_mask_from_b64(mask_b64: str) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_img)


def page_home():
    st.header("🏠 **Home**")
    st.markdown("")
    st.markdown(
        """
        Welcome to the Brain MRI Tumor Detection webpage!


        This project focuses on the development of a deep learning system for **brain tumor segmentation and detection in MRI scans**, aiming to support medical research and improve early identification of low-grade gliomas.

        <div class="highlight-box">

        The project combines:
        - **Medical and domain knowledge**, to formulate clinically relevant questions.
        - **AI engineering and AIOps**, to design, train and deploy robust models.
        - **Data engineering**, to process raw TIFF images into analysis-ready tensors.
        - **Frontend and UX design**, to create interfaces that fit real clinical workflows.
        Effective AI in healthcare always requires this kind of cross-disciplinary collaboration.

        </div>

        The website is organized into several sections to guide you through the project:
        - 🏠 **Home** – Overview of the project
        - 📚 **Introduction** – Context and motivation
        - 📂 **Data Sources** – Description of the datasets used
        - 🧬 **Deep Learning Model** – Architecture, training, and methodology
        - 📊 **Data Visualization** – Exploratory and technical visual analyses
        - 🔍 **Live Prediction** – Real-time model inference on user-uploaded MRI images
        - 🎥 **Visual Demo** – Practical demonstration of the segmentation results
        - 🤝 **Collaboration** – Ways to contribute to the project or cancer research
        - 👥 **About the Authors** – Information about the project contributors

        Thank you for visiting — your interest and participation help strengthen ongoing efforts in medical imaging and cancer research.

        """,
        unsafe_allow_html=True,
    )


def page_intro():
    st.header("📚 **Introduction**")
    st.markdown(
        """

        <h5 style="text-align: center;color: black;"> <b>What Is a Low-Grade Glioma?</b></h5>

        **Brain cancer**, and in particular **low-grade gliomas (LGG) requires early diagnosis and careful monitoring**.
        From a clinical perspective, low-grade gliomas often affect relatively young adults and may present with **seizures, headaches or subtle cognitive changes**.
        <br> </br>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h5 style="text-align: center;color: black;"> <b>Why Early Detection Is Important?</b></h5>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """

                **Early detection of brain tumors** plays a crucial role in **improving patient outcomes**. When identified at an early stage, tumors are often smaller, less aggressive, and more responsive to treatment, **allowing clinicians to intervene before neurological damage becomes extensive**.

                """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="red-box">

        - Around 80% of people living with a brain tumor require neurorehabilitation.
        - In 2022, 322,000 new cases of brain and central nervous system tumors were estimated globally.
        - Brain tumors account for approximately 2% of all cancers diagnosed in adults and 15% of those diagnosed in children.
        - About 80% of patients will present cognitive dysfunction, and 78% will present motor dysfunction.

        </div>
        <br></br>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h5 style="text-align: center;color: black;"> <b>Why MRI tumor segmentation is important in Low-Grade Glioma Patients?</b></h5>

        Even though they are classified as "low grade", **they can progress to high-grade gliomas**, so **longitudinal monitoring with MRI** and, when indicated, histopathological and molecular analysis are **key for prognosis and treatment planning**.

        **MRI-based diagnosis** is especially valuable, as it **provides detailed structural information without exposing patients to radiation**.

         <div class="highlight-box">

        For radiologists and data scientists, MRI is interesting because it combines:
        - **Anatomical detail** (T1- and T2-weighted sequences).
        - **Edema and tumor extent** visualization (FLAIR).
        - In some protocols, **functional information** such as diffusion and perfusion,
          which can correlate with cell density and vascularity.
        Integrating these heterogeneous sources of information is one of the main
        motivations for using deep learning in neuro-oncology.

         </div>
        <br></br>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """Advances in **automated image analysis and deep learning** now offer the possibility of **supporting radiologists with faster, more consistent tumor identification**. By accelerating the diagnostic process, **reducing human error, and enabling timely intervention**, early detection becomes a powerful tool in improving survival rates and enhancing quality of life for patients affected by brain tumors.")
    """,
        unsafe_allow_html=True,
    )


def page_sources():
    st.header("📂 **Data Sources**")
    st.markdown(
        """

    The **LGG MRI Segmentation** dataset comes from the TCGA-LGG collection hosted on [*The Cancer Imaging Archive (TCIA)*](https://www.cancerimagingarchive.net/collection/tcga-lgg/) and was curated and released on [Kaggle by Mateusz Buda](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data). It contains MRI scans of patients diagnosed with **low-grade gliomas**, along with expert-annotated **tumor segmentation masks**.
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4, col5 = st.columns(
        [2, 5, 2, 5, 2], gap="large", vertical_alignment="center"
    )
    with col2:
        with st.container(
            border=True,
        ):
            st.image(str(KAGGLE_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        Kaggle – [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

                        </center>
                        """,
                unsafe_allow_html=True,
            )
    with col4:
        with st.container(border=True):
            st.image(str(TCIA_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        TCIA – [TCGA-LGG Collection](https://www.cancerimagingarchive.net)

                        </center>
                        """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
    ##### **Data Key Characteristics**

    - **Patients:** ~110
    - **Total images:** ~3,900 MRI slices
    - **Modalities:** Multi-channel `.tiff` images (commonly including FLAIR and contrast variations)
    - **Annotations:** Single-channel masks marking the tumor region
    - **Structure:** Each patient folder includes MRI slices and corresponding segmentation masks

    ##### **Why It’s Useful for Brain Tumor Segmentation?**

    - Provides **reliable ground-truth labels** for supervised learning.
    - Includes **multiple slices per patient**, giving models diverse anatomical variation.

    """,
        unsafe_allow_html=True,
    )


def page_data():
    st.header("📊 Dataset Visualization")
    df_routes = load_csv_local("route_label.csv")
    if not df_routes.empty:
        df_routes = (
            df_routes.set_index(df_routes.columns[0])
            if len(df_routes.columns) > 0
            else df_routes
        )
    tab_plots, tab_table = st.tabs(["📈 Plots", "📄 Table"])

    # ===== PLOTS =====
    with tab_plots:
        st.subheader("Class distribution")

        # Count 0 and 1
        class_counts = df_routes["mask"].value_counts().reset_index()
        class_counts.columns = ["mask_value", "Number of images"]

        class_counts["Class"] = class_counts["mask_value"].map(
            {
                0: "0 – Negative (no tumor)",
                1: "1 – Positive (tumor present)",
            }
        )

        # Keep only the columns needed for the plot
        class_counts = class_counts[["Class", "Number of images"]]

        # Pie chart
        fig_pie = px.pie(
            class_counts,
            names="Class",
            values="Number of images",
            title="Tumor vs No Tumor",
        )
        col1, col2, col3 = st.columns(3)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)

        # Global prevalence (image-level)
        prevalence = df_routes["mask"].mean()
        st.markdown(
            f"""
                    <div style="text-align: center;">

                    *In this dataset ≈ {prevalence * 100:.2f}% of the images are labelled as positive (`mask = 1`).*

                     </div>

            """,
            unsafe_allow_html=True,
        )

        # ===== TABLE =====
        with tab_table:
            st.subheader("`route_label.csv`")
            st.markdown(
                "*Overview of routes MRI images and mask, with their corresponding label (0 – Negative (no tumor),1 – Positive (tumor present)*"
            )
            st.dataframe(df_routes[df_routes.columns])
    # =====================================================================
    #  🔬 Scientific medical + data science interpretation
    # =====================================================================
    prevalence_global = df_routes["mask"].mean()
    negative_pct = (1 - prevalence_global) * 100
    positive_pct = prevalence_global * 100

    show_analysis = st.expander("Show Authors' Analysis", expanded=True)
    with show_analysis:
        st.markdown(
            f"""

        <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
    <div class="highlight-box">
<p style="margin:0;">


<h5 style="text-align: center;color: black;"> <b> Cohort composition (image-level class distribution)</b></h5>

In this dataset:


- **≈ {negative_pct:.1f}%** of MRI slices are labelled as
  **0 – Negative (no tumor)**
- **≈ {positive_pct:.1f}%** of MRI slices are labelled as
  **1 – Positive (tumor present)**

This yields an **image-level tumor prevalence of approximately {positive_pct:.1f}%**.

From a methodological standpoint, this indicates a **moderately imbalanced dataset**,
with a dominant negative class and a substantial proportion of positive slices.
Therefore, **any classification model** must outperform a trivial baseline predicting
the majority class (≈ **{negative_pct:.1f}% accuracy**) to demonstrate meaningful discriminative value.


<h5 style="text-align: center;color: black;"> <b> Clinical and machine-learning implications </b></h5>

- The enrichment in tumor-positive slices (≈ {positive_pct:.1f}%) is higher than in routine clinical cohorts,
  which usually contain far fewer tumors. This is advantageous for model development, as it provides a
  sufficient number of positive examples to learn tumor-related patterns and to train segmentation models.

- Because of the moderate class imbalance, evaluation should not rely solely on accuracy. More informative metrics are:

  - **Sensitivity / recall** for positive cases (`mask = 1`)
  - **Specificity** for negative cases (`mask = 0`)
  - **AUC-ROC** and **AUC-PR**, which better capture performance under imbalance.

- If the model tends to under-detect tumors, one may consider:
  - **Class-weighted loss functions**
  - **Focal loss**
  - **Oversampling of positive slices** or undersampling of negatives.


<h5 style="text-align: center;color: black;"> <b>

Utility of the `mask` column
</b></h5>
Although voxel-wise segmentation masks are available via `mask_path`, the binary image-level label (`mask`) enables:


- Rapid assessment of **class distribution** (as visualised in the pie chart).
- Training of a **binary tumor vs. no-tumor classifier** as a screening or pre-filtering stage.
- Stratified analyses, for example comparing intensity distributions or radiomic features between positive and negative slices.

</p>
<div/>

From a clinical research perspective, the cohort can be succinctly described as:

> *"In this dataset, approximately {positive_pct:.1f}% of MRI slices contain visible tumor tissue according to expert segmentation. This prevalence establishes the baseline that any automated detection model must exceed in order to be clinically relevant."*


""",
            unsafe_allow_html=True,
        )


def page_cases():
    st.header("🧠 MRI Images Visualization")

    st.markdown(
        """
        Here we show slices of **brain magnetic resonance imaging (MRI)** with and without **segmented tumor**.
        In each example you will see:

        1. **Original MRI**
        2. **Binary tumor mask** (white = tumor, black = background)
        3. **MRI with the superimposed mask** (only in cases with a tumor)
        """
    )

    rows_dir = IMAGES_DIR

    # ------------------ CASOS CON TUMOR (row_*.png) ------------------
    tumor_rows = sorted(rows_dir.glob("tumor_*.png"))

    # ------------------ CASOS SIN TUMOR (example_no_tumor*.png) ------------------
    no_tumor_rows = sorted(rows_dir.glob("no_tumor*.png"))

    if not tumor_rows and not no_tumor_rows:
        st.error("Images Not found (0 and 1)")
        return

    # =========================
    # Contenedor central
    # =========================
    left_empty, center, right_empty = st.columns([1, 4, 1])
    with center:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tipo = st.radio(
                "Show a case example: ",
                ("🟢 Without tumor", "🔴 With tumor"),
                horizontal=True,
                index=0,
                width="stretch",
            )

        if tipo == "🔴 With tumor":
            active_rows = tumor_rows
            state_key = "random_row_idx_tumor"
            boton_texto = "🔀 Show another tumor MRI image"
            titulo_prefix = "Cancer Patient"
            subtitulo_suffix = "Tumor RMI image"
        else:
            active_rows = no_tumor_rows
            state_key = "random_row_idx_no_tumor"
            boton_texto = "🔀 Show another healthy patient MRI image"
            titulo_prefix = "Healthy Patient"
            subtitulo_suffix = "No tumor RMI image"

        if not active_rows:
            if tipo == "🔴 With tumor":
                st.warning("Tumor images not found")
            else:
                st.warning("No tumor images not fount")
            return

        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        bc1, bc2, bc3 = st.columns([1, 2, 1])
        with bc2:
            if st.button(boton_texto, use_container_width=True):
                st.session_state[state_key] = random.randrange(len(active_rows))

        st.markdown("<br>", unsafe_allow_html=True)

        current_idx = st.session_state[state_key]
        current_path = active_rows[current_idx]

        stem = current_path.stem
        num_part = "".join(ch for ch in stem if ch.isdigit())
        case_number = num_part if num_part else "–"

        st.markdown(
            f"<h3 style='text-align:center'>{titulo_prefix} {case_number}: {subtitulo_suffix}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # Show images
        # =========================

        if tipo == "🔴 Con tumor":
            # fila row_XX con 3 columnas en una misma imagen
            img_row = Image.open(current_path)
            st.image(img_row, use_container_width=True)
            st.markdown(
                """
                <br></br>

                #### Clinical / data analyst interpretation (with tumor)

                - **Region of interest:** a focal hyperintense lesion is visible within the brain
                  parenchyma. The binary mask highlights all pixels classified as tumor.
                - **Segmentation concept:** every white pixel in the mask corresponds to voxels
                  that the model (or the manual annotation) considers part of the tumor.
                - **Visual benefit:** the overlaid image makes it easier to appreciate tumor
                  borders, mass effect and relationship to surrounding tissue.
                - **From a data point of view:** this slice would be labelled as a **positive
                  sample**, and the mask provides dense supervision for training segmentation
                  models (Dice, IoU, pixel-wise accuracy, etc.).
                """,
                unsafe_allow_html=True,
            )

        else:
            img_mri = Image.open(current_path).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(img_mri, use_container_width=False)
            st.markdown(
                """
                <br></br>

                #### Clinical / data analyst interpretation (no visible tumor)

                - **Overall impression:** normal-appearing brain MRI for this slice, with
                  no focal mass, no clear edema pattern and preserved global symmetry.
                - **Segmentation point of view:** this is a **negative sample**; the
                  corresponding mask is empty, meaning no pixels are labelled as tumor.
                - **Why it matters for the model:** negative cases are crucial to reduce
                  false positives and to teach the network what healthy anatomy looks like.
                - **Expected behavior:** the model should assign low tumor probability to
                  all pixels in this image. Any high activation here would be a potential
                  false positive.
                """,
                unsafe_allow_html=True,
            )


def page_model():
    st.header("🧬 Deep learning model")
    st.markdown(
        '<h3 style="text-align: center;color: black;"> <b> General Pipeline </b></h3>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    with b:
        st.image(IMAGES_DIR / "general_pipeline.png", use_container_width=True)
    st.markdown(
        '<br></br> <h3 style="text-align: center;color: black;"> <b> ResNet-50 Classification Model Architecture </b></h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
For the classification stage of this project, we use **ResNet-50**, a deep convolutional neural network introduced in the paper [*“Deep Residual Learning for Image Recognition”* (He et al., 2015)](https://arxiv.org/pdf/1512.03385.pdf).

ResNet-50 belongs to the family of **Residual Networks (ResNets)**, whose key innovation is the use of **residual (skip) connections**. These connections allow the model to learn residual functions instead of full mappings, which solves two major problems of very deep networks: vanishing gradients and training degradation.

A typical residual block applies several convolutional layers to the input and then adds the original input back to the block output:

output = F(x) + x

This design helps preserve gradient flow and stabilizes training when the network is very deep.
- Architecturally, ResNet-50 consists of 50 layers arranged in several stages. It uses a “bottleneck” block structure: a 1×1 convolution to reduce dimensionality, a 3×3 convolution, and another 1×1 convolution to restore dimensionality — allowing efficient parameter usage while maintaining depth.
- Thanks to its design, ResNet-50 has proven extremely effective in large-scale image recognition tasks (e.g. ImageNet), and has become a standard backbone for many computer vision tasks, including medical image analysis.

In our context, classification (or feature extraction) over MRI images of low-grade gliomas, ResNet-50 gives us a robust, well-tested backbone: its residual learning capability helps with training stability, even with relatively deep layers; its feature representations are rich and suitable for transfer learning and fine-tuning on medical data; and its architecture is widely accepted in literature, which helps with reproducibility and comparability of our results.

<br></br>

 <div style="text-align: center;">

![](https://www.researchgate.net/profile/Master-Prince/publication/350421671/figure/fig1/AS:1005790324346881@1616810508674/An-illustration-of-ResNet-50-layers-architecture.png)

 </div>

 <br></br>
<br></br> <h3 style="text-align: center;color: black;"> <b> ResUNet Segmentation Model Architecture </b></h3>


Building upon the concepts previously introduced with **ResNet-50**, ResUNet represents a natural evolution of deep convolutional networks for image segmentation tasks. While ResNet-50 demonstrated how *residual connections* enable the training of very deep models by facilitating stable gradient flow, ResUNet integrates these same principles into the classic **U-Net encoder–decoder structure**, creating a model that is both deep and highly effective at capturing fine-grained spatial information. The ResUNet Architecture was first described in [Z. Zhang et al. 2017](https://arxiv.org/pdf/1711.10684.pdf)


##### **How ResUNet Extends the Ideas of ResNet-50?**

ResUNet incorporates *residual blocks* throughout its architecture, mirroring the philosophy behind ResNet-50:

- Residual connections allow the model to learn identity mappings more easily.
- They reduce vanishing gradients and support deeper, more expressive feature extractors.
- They promote efficient training, especially when datasets are limited.

By embedding these residual blocks inside the U-Net structure, ResUNet achieves a strong balance between **feature depth** and **spatial precision**, which is crucial for segmentation tasks.


##### **Core Components of the ResUNet Architecture**

1. **Encoder with Residual Blocks**
   The encoder operates similarly to ResNet-style feature extraction. Each stage includes residual convolutional blocks that progressively downsample the spatial resolution while increasing feature richness. These blocks stabilize training and enable the model to capture high-level semantics.

2. **Bottleneck Layer**
   At the deepest level, the network aggregates global context. Residual connections continue to support gradient flow even at this highly compressed representation.

3. **Decoder with Skip Connections**
   The decoder mirrors the encoder but performs upsampling to recover spatial structure. Standard U-Net skip connections bridge encoder and decoder levels, ensuring the model retains fine details lost during downsampling.

4. **Residual Refinement**
   Each decoder stage incorporates residual blocks, allowing the model to refine and correct features as they are upsampled. This combination of residual learning + multi-scale fusion is one of the key strengths of ResUNet.

5. **Final Segmentation Layer**
   A final convolutional layer maps the decoded features to pixel-wise class probabilities, producing a dense segmentation mask.

<br></br>

 <div style="text-align: center;">

![](https://idiotdeveloper.com/wp-content/uploads/2021/02/arch.png)

</div>

<br></br>

##### **Why ResUNet Is Effective for Segmentation?**

ResUNet is particularly advantageous because it unifies:

- **Deep semantic feature extraction** (thanks to residual blocks, much like in ResNet-50)
- **Precise spatial localization** (enabled by the U-Net skip connections)
- **Stable and efficient training**, even with limited data
- **Flexibility in depth and complexity**, allowing adaptations for various modalities (e.g., medical imaging, remote sensing)

This makes ResUNet a powerful architecture for tasks where accurate object boundaries and contextual understanding are both essential.



        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<br></br> <h3 style="text-align: center;color: black;"> <b> Data preprocessing and quality control </b></h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Before training any medical imaging model, a robust preprocessing pipeline is essential:
        - **Skull stripping** to remove non-brain tissue and reduce noise.
        - **Intensity normalization** per scan to mitigate scanner- or protocol-related variability.
        - **Spatial registration** to a common template when combining data from multiple patients.
        - **Resampling to isotropic voxels** so that physical distances are comparable.
        - **Data augmentation** (rotations, flips, elastic deformations, mild intensity shifts)
          to improve generalization and simulate real-world acquisition variability.
        A careful visual QC (quality control) step is usually performed with radiologists
        to exclude corrupted or mislabeled scans.
        """
    )

    st.markdown(
        '<br></br> <h4 style="text-align: center;color: black;"> <b> Training (summary) </b></h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        - **Data**: MRI dataset with tumor annotations.
        - **Labels**:
          - For classification: `0` = no tumor, `1` = tumor.
          - For segmentation: masks where each pixel indicates tumor/no tumor.
        - **Procedure**:
          - Split into *train / validation / test*.
          - Train for several epochs minimizing a loss function
            (*Binary Cross-Entropy (BCE) * for classification or
            *Dice-BCE loss* for segmentation).
        - **Metrics**:
          - Classification: accuracy, F1, precision and recall scores.
          - Segmentation: dice coefficient, intersection over union, accuracy.
        """
    )

    st.markdown(
        '<br></br> <h3 style="text-align: center;color: black;"> <b> Model Evaluation </b></h3>',
        unsafe_allow_html=True,
    )
    st.markdown("##### **ResNet-50 Classification Model**")
    A, B, C, D = st.columns([1, 4, 4, 1], gap="medium", vertical_alignment="center")
    with B:
        with st.container(border=True):
            st.image(
                IMAGES_DIR / "confusion_matrix_classification.png",
                use_container_width=True,
            )
    with C:
        with st.container(border=True):
            st.image(
                IMAGES_DIR / "ROC_curve_classification.png", use_container_width=True
            )
    a, b, c, d, e, f = st.columns(
        [1, 2, 2, 2, 2, 1], gap="large", vertical_alignment="center"
    )
    with b:
        with st.container(border=True):
            st.metric("Accuracy", "98.61%")
    with c:
        with st.container(border=True):
            st.metric("Precision", "97.84%")
    with d:
        with st.container(border=True):
            st.metric("Recall", "97.84%")
    with e:
        with st.container(border=True):
            st.metric("F1 score", "97.84%")

    st.markdown("##### **ResUNet Segmentation Model**")
    a, b, c, d, e, f = st.columns(
        [1, 2, 2, 2, 2, 1], gap="large", vertical_alignment="center"
    )
    with b:
        with st.container(border=True):
            st.metric("Accuracy", "99.25%")
    with c:
        with st.container(border=True):
            st.metric("Dice Coefficient", "84.58%")
    with d:
        with st.container(border=True):
            st.metric("Intersection over Union (IoU)", "73.38%")
    with e:
        with st.container(border=True):
            st.metric("Dice-BCE Loss", "18.29%")

    st.markdown("## Integration with Flask")
    st.info(
        """
        The model is deployed inside a **Flask API**:

        - The Flask app exposes an HTTP endpoint (for example, `/predict`).
        - Streamlit sends the MRI image to the endpoint in base64 format.
        - Flask runs the deep learning model and returns:
          - whether there is a tumor (`has_tumor`)
          - the probability (`probability`)
          - optionally, a mask (`mask_base64`).

        This separation allows us to:
        - Scale the model independently (GPU/CPU).
        - Use Streamlit only as a lightweight visual interface.
        """
    )

    st.markdown(
        """
        In a production setting, this architecture would be complemented with:
        - **Audit logs** to track who requested each prediction.
        - **Versioning** of models and training datasets to ensure reproducibility.
        - **Monitoring** of latency, error rates and data drift to detect when
          the model may need to be re-evaluated or retrained.
        - Integration with hospital systems (PACS/RIS) using standards such as DICOM and HL7/FHIR.
        """
    )

    st.markdown("## Limitations and responsible use")
    st.warning(
        """
        This application is a **proof of concept** (PoC):

        - It does not replace the judgment of a medical professional.
        - Predictions may contain errors.
        - Any real clinical use must undergo rigorous validation.
        """
    )

    st.info(
        """
        Even models that perform well in retrospective studies can fail once deployed
        if the patient population, scanners or imaging protocols change over time.
        Continuous surveillance, periodic re-validation and collaboration between
        data scientists, clinicians and MLOps engineers are essential for safe,
        responsible AI in healthcare.
        """
    )


def page_live_prediction():
    st.header("🔍 Live prediction with Flask model")

    st.markdown(
        """
        Upload an MRI image and the system will query the **deep learning model**
        deployed in Flask to predict whether there is a tumor or not.
        """
    )

    st.markdown(
        """
        In a realistic deployment, the input would often be an entire MRI study
        (many slices and sequences) rather than a single image. A more advanced system
        could:
        - Aggregate predictions across slices to provide a per-patient risk score.
        - Produce a 3D segmentation and estimate total tumor volume.
        - Track changes over time across multiple exams to quantify treatment response.
        Here we simplify this process to make the interaction easier to understand.
        """
    )

    
    def decode_base64_image(base64_str):
        """Decodifica una imagen en base64 a objeto PIL Image"""
        # Remover el prefijo data:image/png;base64, si existe
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        
        image_bytes = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_bytes))
    
    
    def classify_image(image_file=None, use_random=False):
        """
        Clasifica una imagen usando la API
        
        Args:
            image_file: Archivo de imagen subido (para opción 1)
            use_random: Si True, usa endpoint random (para opción 2)
        
        Returns:
            dict con la respuesta de la API o None si hay error
        """
        try:
            if use_random:
                # Opción 2: Imagen aleatoria
                response = requests.get(f"{API_BASE_URL}/clasificacion/predict/random")
            else:
                # Opción 1: Imagen subida
                if image_file is None:
                    return None
                
                # Resetear el puntero del archivo
                image_file.seek(0)
                files = {"image": image_file}
                response = requests.post(f"{API_BASE_URL}/clasificacion/predict", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error en clasificación: {response.status_code} - {response.text}")
                return None
        
        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar con la API. Verifica que esté ejecutándose.")
            return None
        except Exception as e:
            st.error(f"Error al clasificar imagen: {str(e)}")
            return None
    
    
    def segment_image(image_file=None, use_random=False):
        """
        Segmenta una imagen usando la API
        
        Args:
            image_file: Archivo de imagen subido (para opción 1)
            use_random: Si True, usa endpoint random (para opción 2)
        
        Returns:
            dict con la respuesta de la API o None si hay error
        """
        try:
            if use_random:
                # Opción 2: Imagen aleatoria
                response = requests.get(f"{API_BASE_URL}/segmentacion/predict/random")
            else:
                # Opción 1: Imagen subida
                if image_file is None:
                    return None
                
                # Resetear el puntero del archivo
                image_file.seek(0)
                files = {"image": image_file}
                response = requests.post(f"{API_BASE_URL}/segmentacion/predict", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error en segmentación: {response.status_code} - {response.text}")
                return None
        
        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar con la API. Verifica que esté ejecutándose.")
            return None
        except Exception as e:
            st.error(f"Error al segmentar imagen: {str(e)}")
            return None
    
    
    def show_classification_result(result):
        """Muestra el resultado de clasificación con formato"""
        if not result:
            return
        
        prediction_label = result.get("prediction_label", "")
        confidence = result.get("confidence", "0%")
        
        # Determinar si es positivo (tumor detectado)
        is_positive = "Detectado" in prediction_label or "(1)" in prediction_label
        
        # Mostrar resultado con color según predicción
        if is_positive:
            st.error(f"🔴 **Resultado:** {prediction_label}")
            st.metric("Nivel de Confianza", confidence)
            return True  # Retorna True si es positivo
        else:
            st.success(f"🟢 **Resultado:** {prediction_label}")
            st.metric("Nivel de Confianza", confidence)
            return False  # Retorna False si es negativo
    
    
    def overlay_mask_on_image(original_image, mask_image, alpha=0.5):
        """
        Superpone la máscara de segmentación sobre la imagen original
        
        Args:
            original_image: PIL Image original
            mask_image: PIL Image de la máscara
            alpha: Transparencia de la máscara (0-1)
        
        Returns:
            PIL Image con la máscara superpuesta
        """
        # Asegurar que ambas imágenes tienen el mismo tamaño
        if original_image.size != mask_image.size:
            mask_image = mask_image.resize(original_image.size, Image.Resampling.LANCZOS)
        
        # Convertir a RGB si es necesario
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")
        if mask_image.mode != "RGB":
            mask_image = mask_image.convert("RGB")
        
        # Convertir a arrays numpy
        orig_array = np.array(original_image).astype(np.float32)
        mask_array = np.array(mask_image).astype(np.float32)
        
        # Crear máscara coloreada (rojo para tumor)
        colored_mask = np.zeros_like(orig_array)
        colored_mask[:, :, 0] = mask_array[:, :, 0]  # Canal rojo
        
        # Superponer con transparencia
        overlay = (1 - alpha) * orig_array + alpha * colored_mask
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    
    def main():
        """Función principal de la página de predicción"""
        
        st.title("🧠 Detección de Tumores Cerebrales")
        st.markdown("### Sistema de Clasificación y Segmentación con IA")
        
        st.markdown("---")
        
        # Verificar conexión con API
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if health_response.status_code != 200:
                st.warning("⚠️ La API está respondiendo pero puede tener problemas. Verifica el estado.")
        except:
            st.error("❌ No se pudo conectar con la API. Asegúrate de que esté ejecutándose en " + API_BASE_URL)
            st.stop()
        
        # Selector de modo
        st.subheader("Selecciona el modo de entrada")
        
        mode = st.radio(
            "¿Cómo deseas proporcionar la imagen?",
            ["📤 Subir imagen MRI (.tif)", "🎲 Usar imagen aleatoria de la base de datos"],
            index=0
        )
        
        st.markdown("---")
        
        # Variables para almacenar datos
        uploaded_file = None
        use_random = False
        
        # Opción 1: Subir imagen
        if mode == "📤 Subir imagen MRI (.tif)":
            st.subheader("📤 Cargar Imagen MRI")
            uploaded_file = st.file_uploader(
                "Selecciona un archivo de imagen MRI",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                help="Formatos soportados: .tif, .tiff, .png, .jpg, .jpeg"
            )
            
            if uploaded_file is not None:
                # Mostrar vista previa
                st.success(f"✅ Archivo cargado: **{uploaded_file.name}**")
                
                try:
                    image = Image.open(uploaded_file)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(image, caption="Vista previa de la imagen", use_container_width=True)
                except Exception as e:
                    st.error(f"Error al cargar la imagen: {str(e)}")
                    uploaded_file = None
        
        # Opción 2: Imagen aleatoria
        else:
            st.subheader("🎲 Imagen Aleatoria")
            st.info("Se seleccionará una imagen aleatoria del dataset de la base de datos.")
            use_random = True
        
        st.markdown("---")
        
        # Botón de análisis
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analizar Imagen",
                type="primary",
                use_container_width=True,
                disabled=(not use_random and uploaded_file is None)
            )
        
        # Procesamiento cuando se presiona el botón
        if analyze_button:
            with st.spinner("🔄 Procesando imagen..."):
                
                # PASO 1: CLASIFICACIÓN
                st.markdown("---")
                st.subheader("📊 Paso 1: Clasificación")
                
                with st.spinner("Clasificando imagen..."):
                    classification_result = classify_image(
                        image_file=uploaded_file,
                        use_random=use_random
                    )
                
                if classification_result:
                    # Mostrar resultado de clasificación
                    is_positive = show_classification_result(classification_result)
                    
                    # Mostrar información adicional
                    if not use_random and "prediction_id" in classification_result:
                        with st.expander("ℹ️ Información adicional"):
                            st.text(f"ID de predicción: {classification_result['prediction_id']}")
                            if "filename" in classification_result:
                                st.text(f"Archivo: {classification_result['filename']}")
                    
                    # PASO 2: SEGMENTACIÓN (solo si es positivo)
                    if is_positive:
                        st.markdown("---")
                        st.subheader("🎯 Paso 2: Segmentación del Tumor")
                        st.info("⚠️ Se ha detectado un tumor. Procediendo con la segmentación...")
                        
                        with st.spinner("Generando máscara de segmentación..."):
                            segmentation_result = segment_image(
                                image_file=uploaded_file,
                                use_random=use_random
                            )
                        
                        if segmentation_result and segmentation_result.get("success"):
                            st.success("✅ Segmentación completada exitosamente")
                            
                            # Obtener la máscara
                            mask_base64 = segmentation_result.get("mask_base64", "")
                            
                            if mask_base64:
                                try:
                                    # Decodificar la máscara
                                    mask_image = decode_base64_image(mask_base64)
                                    
                                    # Mostrar resultados en columnas
                                    st.markdown("### 🖼️ Resultados de Segmentación")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Imagen Original**")
                                        if use_random:
                                            st.info("Imagen aleatoria del dataset")
                                        else:
                                            st.image(
                                                Image.open(uploaded_file),
                                                caption="Imagen MRI original",
                                                use_container_width=True
                                            )
                                    
                                    with col2:
                                        st.markdown("**Máscara de Segmentación**")
                                        st.image(
                                            mask_image,
                                            caption="Región del tumor detectada",
                                            use_container_width=True
                                        )
                                    
                                    # Opción de superposición (solo si no es random)
                                    if not use_random and uploaded_file:
                                        st.markdown("---")
                                        st.markdown("### 🔬 Visualización Superpuesta")
                                        
                                        try:
                                            original_image = Image.open(uploaded_file)
                                            overlay_image = overlay_mask_on_image(
                                                original_image,
                                                mask_image,
                                                alpha=0.4
                                            )
                                            
                                            col1, col2, col3 = st.columns([1, 2, 1])
                                            with col2:
                                                st.image(
                                                    overlay_image,
                                                    caption="Tumor superpuesto en la imagen original",
                                                    use_container_width=True
                                                )
                                        except Exception as e:
                                            st.warning(f"No se pudo generar la visualización superpuesta: {str(e)}")
                                    
                                    # Información adicional
                                    if "segmentation_id" in segmentation_result:
                                        with st.expander("ℹ️ Información de segmentación"):
                                            st.text(f"ID de segmentación: {segmentation_result['segmentation_id']}")
                                            if "filename" in segmentation_result:
                                                st.text(f"Archivo: {segmentation_result['filename']}")
                                    
                                except Exception as e:
                                    st.error(f"Error al procesar la máscara de segmentación: {str(e)}")
                        else:
                            st.error("❌ No se pudo realizar la segmentación")
                    
                    else:
                        # Si es negativo, no hay segmentación
                        st.markdown("---")
                        st.info("✅ No se requiere segmentación. No se detectó tumor en la imagen.")
                
                else:
                    st.error("❌ No se pudo realizar la clasificación")
        
        # Información adicional en la barra lateral
        with st.sidebar:
            st.markdown("### 📖 Información")
            st.markdown("""
            **Proceso de análisis:**
            
            1️⃣ **Clasificación**: El modelo ResNet determina si hay tumor presente
            
            2️⃣ **Segmentación**: Si se detecta tumor, el modelo ResUNet localiza su ubicación exacta
            
            ---
            
            **Clases de clasificación:**
            - 🟢 No detectado (0)
            - 🔴 Detectado (1)
            
            ---
            
            **API Status:**
            """)
            
            try:
                health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    if health_data.get("status") == "healthy":
                        st.success("✅ API Operativa")
                        st.json({
                            "Modelos cargados": health_data.get("models_loaded"),
                            "Almacenamiento": health_data.get("storage_configured")
                        })
                    else:
                        st.warning("⚠️ API con problemas")
                else:
                    st.error("❌ API no responde")
            except:
                st.error("❌ Sin conexión")
    main()

def page_media():
    st.header("🎥 Flask Backend Visual demo")

    st.subheader("Demo video of the app / model")
    with open(str(VIDEO_PATH), "rb") as video_file:
        video_bytes = video_file.read()
    st.video(video_bytes)


def page_collab():
    st.header("🤝 Collaboration")
    st.markdown("""

Collaboration is central to the success and scientific value of this brain tumor segmentation project. Our work builds directly on the collective efforts of the research community and the open-access initiatives that make high-quality medical imaging data available for machine learning research.

We acknowledge and thank the contributors of the **LGG MRI Segmentation** dataset, derived from the TCGA-LGG collection on *The Cancer Imaging Archive (TCIA)* and curated by Mateusz Buda. Their commitment to transparent data sharing enables researchers worldwide to develop, benchmark, and validate deep learning models for low-grade glioma segmentation. You can learn more about the dataset or contribute to their ongoing initiatives through the following links:
""")
    col1, col2, col3, col4, col5 = st.columns(
        [2, 5, 2, 5, 2], gap="large", vertical_alignment="center"
    )
    with col2:
        with st.container(
            border=True,
        ):
            st.image(str(KAGGLE_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        Kaggle – [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

                        </center>
                        """,
                unsafe_allow_html=True,
            )
    with col4:
        with st.container(border=True):
            st.image(str(TCIA_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        TCIA – [TCGA-LGG Collection](https://www.cancerimagingarchive.net)

                        </center>
                        """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """We also actively encourage collaboration within our own project. Our repository is publicly available, and we invite contributions related to model development, preprocessing pipelines, evaluation metrics, or exploratory radiogenomic analysis. Whether you are a researcher, clinician, or data scientist, your expertise can help improve the robustness and clinical relevance of our neural network models."""
    )

    col1, col2, col3, col4, col5 = st.columns(
        [2, 2, 5, 2, 2], gap="large", vertical_alignment="center"
    )
    with col3:
        with st.container(border=True):
            st.image(str(GITHUB_IMAGE))
            st.markdown(
                """
                        <center>

                        GitHub Repository [Brain Tumor Detection Project](https://github.com/FabsGMartin/brain-tumor-detection)

                        </center>
                        """,
                unsafe_allow_html=True,
            )
    st.markdown("""
We welcome pull requests, issue reporting, dataset discussions, and architectural improvements. In the spirit of open science, our goal is to create a collaborative space where insights and methods can be shared, replicated, and expanded. Through joint effort with both external data providers and the broader scientific community, we aim to produce reliable and reproducible tools that support research and clinical innovation in brain tumor analysis.
""")

    st.markdown(
        """
    ##### Support Cancer Research

    Beyond contributing to this project, you can also support the broader fight against cancer. Advancing treatments, improving diagnostics, and understanding tumor biology all depend on continued scientific and clinical research. Many organizations work tirelessly to fund studies, support patients, and accelerate the development of life-saving therapies.

    Here are several well-regarded associations you can collaborate with or donate to:

    - **American Cancer Society (ACS):** https://www.cancer.org
    - **Brain Tumor Foundation:** https://www.braintumorfoundation.org
    - **National Brain Tumor Society (NBTS):** https://braintumor.org
    - **Cancer Research UK:** https://www.cancerresearchuk.org
    - **European Organisation for Research and Treatment of Cancer (EORTC):** https://www.eortc.org

    Your support (whether through scientific collaboration, sharing expertise, or contributing to research foundations) helps move the field forward and brings us closer to better outcomes for patients around the world.
    """,
        unsafe_allow_html=True,
    )


def page_team():
    st.header("👥 Project team")

    st.markdown(
        """
        This work has been developed by a multidisciplinary team of data scientist with knowledge in  AIops.

        Below you can see our profiles and GitHub links.
        """
    )

    team = [
        {
            "name": "Luna Pérez T.",
            "github": "https://github.com/LunaPerezT",
            "linkedin": "https://www.linkedin.com/in/luna-p%C3%A9rez-troncoso-0ab21929b/",
        },
        {
            "name": "Raquel Hernández",
            "github": "https://github.com/RaquelH18",
            "linkedin": "https://www.linkedin.com/in/raquel-hern%C3%A1ndez-lozano/",
        },
        {
            "name": "Mary Marín",
            "github": "https://github.com/mmarin3011-cloud",
            "linkedin": "https://www.linkedin.com/in/mmarin30/",
        },
        {
            "name": "Fabián G. Martín",
            "github": "https://github.com/FabsGMartin",
            "linkedin": "",
        },
        {
            "name": "Miguel J. de la Torre",
            "github": "https://github.com/migueljdlt",
            "linkedin": "https://www.linkedin.com/in/miguel-jimenez-7403a2374/",
        },
        {
            "name": "Alejandro C.",
            "github": "https://github.com/alc98",
            "linkedin": "https://www.linkedin.com/in/alejandro-c-9b6525292/",
        },
    ]

    # Grid de 2 filas x 3 columnas, con GitHub justo debajo del nombre
    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start : row_start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<h5 style="text-align: center;color: black;"> <b>{member["name"]}</b></h5>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                                <center>

                                **GitHub:** [{member["github"]}]({member["github"]})

                                </center>
                                """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                                <center>

                                **LinkedIn:** [{member["linkedin"]}]({member["linkedin"]})

                                </center>
                                """,
                        unsafe_allow_html=True,
                    )

    st.info(
        """
        Although this is an academic project, any real-world deployment as a clinical
        tool would require collaboration with neuroradiologists, neurosurgeons,
        oncologists, medical physicists and hospital IT teams, as well as regulatory
        approval as a medical device.
        """
    )


# ---------- APP HEADER ----------


st.markdown(
    """<h1 style="text-align: center;color: black;"> <b> Brain MRI Tumor Detection </b></h1>
    <h5 style="text-align: center;color: gray"> <em> A Deep Learning based project to detect and segmetate brain tumors in MRI images </em> </h5>""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------- SIDEBAR NAVIGATION ----------

st.sidebar.header("Navigation Menu")
st.sidebar.caption("Choose a section to explore the project.")

menu = [
    "🏠 Home",
    "📚 Introduction",
    "📂 Data Sources",
    "📊 Dataset Visualization",
    "🧠 MRI Images Visualization",
    "🧬 Deep learning model",
    "🔍 Live prediction",
    "🎥 Flask Backend Visual demo",
    "🤝 Collaboration",
    "👥 About the Authors",
]

choice = st.sidebar.radio("Select a page:", menu)

# ---------- APP BODY ----------

if choice == "🏠 Home":
    page_home()
elif choice == "📚 Introduction":
    page_intro()
elif choice == "📂 Data Sources":
    page_sources()
elif choice == "📊 Dataset Visualization":
    page_data()
elif choice == "🧠 MRI Images Visualization":
    page_cases()
elif choice == "🧬 Deep learning model":
    page_model()
elif choice == "🔍 Live prediction":
    page_live_prediction()
elif choice == "🎥 Flask Backend Visual demo":
    page_media()
elif choice == "🤝 Collaboration":
    page_collab()
elif choice == "👥 About the Authors":
    page_team()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 1em;'>© 2025 Brain MRI Tumor Detection </p>",
    unsafe_allow_html=True,
)
