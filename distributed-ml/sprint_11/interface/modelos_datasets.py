import streamlit as st
import os, json
from paths import REGISTRY_JSON
def load_registry():
    if not os.path.exists(REGISTRY_JSON):
        return {"datasets": {}, "models": {}}
    with open(REGISTRY_JSON, "r") as f:
        return json.load(f)

def save_registry(data):
    with open(REGISTRY_JSON, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
# -----------------------------

registry = load_registry()
MODELS = registry["models"]
DATASETS = registry["datasets"]

st.markdown("""
    <style>

    button[data-baseweb="tab"] p {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ff4b4b !important;
    }
    </style>
""", unsafe_allow_html=True)
st.header("Modelos y Datasets")
st.caption("Gestiona el repertorio disponible para entrenar")
# CAMBIAR

dataset_tab, model_tab = st.tabs(["Datasets","Modelos"])

with dataset_tab:
    cols = st.columns(3)
    for i, (name, info) in enumerate(DATASETS.items()):
        with cols[i%3]:
            with st.container(border=True):
                head_col, badge_col = st.columns([5, 1])

                with head_col:
                    st.subheader(f"{name}")
                    subset_info = f" ({info['subset']})" if "subset" in info else ""
                    st.caption(f"{info['hf_name']}{subset_info} · {info['num_labels']} clases")
                with badge_col:
                    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
                    is_image = info["type"] == "image_classification"
                    st.badge("Imagen" if is_image else "Texto", color="blue" if is_image else "green")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if not is_image:
                    st.write(f"**Tokenizer:** `{info['tokenizer']}`")

                _, col_edit, col_del = st.columns([4, 4, 4])
                
                with col_del:
                    with st.popover("Eliminar", key=f"pop_{name}", use_container_width=True):
                        st.write(f"¿Borrar `{name}`?")
                        if st.button("Confirmar", key=f"del_{name}", type="primary", use_container_width=True):
                            del registry["datasets"][name]
                            save_registry(registry)
                            st.rerun()
            
    st.header("Añadir Dataset", divider=True)
    add_dt_cont = st.container(border=True)
    with add_dt_cont:
        select_col, _ = st.columns([2,6])
        with select_col:
            dt_type = st.selectbox("Tipo de dataset", options=["Clasificación de imágenes", "Clasificación de texto"])

        name_col, hf_col, class_col = st.columns([5,5,2])
        with name_col:
            dt_name = st.text_input("Nombre", key="dataset_name", placeholder="ej: CIFAR-10")
        with hf_col:
            dt_hf_name = st.text_input("Nombre del dataset en hugging face", placeholder="ej: cifar10")
        with class_col:        
            dt_clases = st.number_input("Número de clases", min_value=2, step=1)

        dt_subset = None
        df_label_column = "label"
        df_text_column = None
        df_tokenizer = None
        if dt_type == "Clasificación de imágenes":
            dt_type = "image_classification"
            col1, col2 = st.columns([1, 1])
            
            with col1:
                dt_subset = st.text_input("Subset (opcional)", placeholder="ej: default")
            with col2:
                df_label_column = st.text_input("Label Column", value="label")
        else:
            dt_type = "text_classification"
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                dt_subset = st.text_input("Subset (opcional)", placeholder="ej: default")
            with col2:
                df_text_column = st.text_input("Columna de texto", value="text")
            with col3:
                df_label_column = st.text_input("Label Column", value="label")
            with col4:
                df_tokenizer = st.text_input("Tokenizador", value="bert-base-uncased")
        
        if st.button("Añadir dataset"):
            if not dt_name or not dt_hf_name:
                st.error("Faltan valores: El nombre y el ID de Hugging Face son obligatorios.")
                st.stop()
            else:
                new_ds = {
                    "type": dt_type,
                    "num_labels": dt_clases,
                    "hf_name": dt_hf_name,
                    "label_column": df_label_column,
                }
                if dt_subset and dt_subset.strip():
                    new_ds["subset"] = dt_subset.strip()

                if dt_type == "text_classification":
                    if not df_text_column:
                        st.error("Error: Debes especificar la columna de texto.")
                        st.stop()
                    if not df_tokenizer:
                        st.error("Error: Debes especificar un tokenizer.")
                        st.stop()
                    new_ds["text_column"] = df_text_column
                    new_ds["tokenizer"] = df_tokenizer

            registry["datasets"][dt_name] = {k: v for k, v in new_ds.items() if v is not None}
            save_registry(registry)
            st.success(f"Dataset {dt_name} añadido")
            st.rerun()

with model_tab:
    cols = st.columns(3)
    for i, (name, info) in enumerate(MODELS.items()):
        with cols[i%3]:
            with st.container(border=True):
                head_col, badge_col = st.columns([5, 1])

                with head_col:
                    st.subheader(f"{name}")
                    is_vision = info["type"] == "image"
                    st.badge("Imagen" if is_vision else "Texto", color="blue" if is_vision else "green")

                st.write(f"{info['description']}")

                _, col_edit, col_del = st.columns([4, 4, 4])


                with col_del:
                    with st.popover("Eliminar", key=f"pop_{name}", use_container_width=True):
                        st.write(f"¿Borrar `{name}`?")
                        if st.button("Confirmar", key=f"del_{name}", type="primary", use_container_width=True):
                            del registry["models"][name]
                            save_registry(registry)
                            st.rerun()

    st.header("Añadir Modelo", divider=True)

    add_model_cont = st.container(border=True)
    with add_model_cont:
        select_col, _ = st.columns([2,6])
        with select_col:
            model_type = st.selectbox("Tipo de modelo", options=["Clasificación de imágenes", "Clasificación de texto"])
            if model_type == "Clasificación de imágenes":
                model_type="image"
            else:
                model_type="text"
        model_left, model_right = st.columns(2, gap="medium")
        
        with model_left:
            model_name = st.text_input("Nombre", placeholder="ej: ResNet18")
        with model_right:
            model_ident = st.text_input("Identificador", placeholder="ej: resnet18")

        model_desc = st.text_area("Descripción", placeholder="Descripción libre del modelo, sin uso real...")

        if st.button("Añadir modelo"):
            if not model_ident or not model_type:
                st.error("Faltan valores: El nombre y el ID son obligatorios.")
                st.stop()

            registry["models"][model_name] = {
                "name": model_ident,
                "type": model_type,
                "description": model_desc
            }
            save_registry(registry)
            st.rerun()