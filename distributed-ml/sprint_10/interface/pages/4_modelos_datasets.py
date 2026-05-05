import streamlit as st
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
MODELS = {
    "ResNet18": {
        "name": "resnet18",
        "type": "resnet18",
        "description": "Red neuronal de 18 capas de profundidad"
    },
    "BERT-Base": {
        "name": "bert-base-uncased",
        "type": "transformer",
        "description": "Modelo de lenguaje bidireccional para tareas de texto."
    },
    "MobileNet-V2": {
        "name": "mobilenet_v2",
        "type": "mobilenet",
        "description": "Modelo optimizado para dispositivos móviles y baja potencia."
    },
    "ViT-B16": {
        "name": "vit_base_patch16_224",
        "type": "vit",
        "description": "Vision Transformer que utiliza parches de imágenes como palabras."
    },
    "DistilRoBERTa": {
        "name": "distilroberta-base",
        "type": "transformer",
        "description": "Versión ligera y rápida del modelo RoBERTa para NLP."
    }
}
DATASETS = {
    "ag_news": {
        "type": "text_classification",
        "num_labels": 4,
        "hf_name": "ag_news",
        "text_column": "text",
        "label_column": "label",
        "tokenizer": "distilbert-base-uncased"
    },
    "imdb": {
        "type": "text_classification", 
        "num_labels": 2,
        "hf_name": "imdb",
        "text_column": "text",
        "label_column": "label",
        "tokenizer": "distilbert-base-uncased"
    },
    "sst2": {
        "type": "text_classification",
        "num_labels": 2,
        "hf_name": "glue",
        "subset": "sst2",
        "text_column": "sentence",
        "label_column": "label",
        "tokenizer": "bert-base-uncased"
    },
    "cifar10": {
        "type": "image_classification",
        "num_labels": 10,
        "hf_name": "cifar10",
        "label_column": "label",
    }
}

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
                
                with col_edit:
                    st.button("Editar", key=f"edit_{name}", use_container_width=True)

                with col_del:
                    with st.popover("Eliminar", key=f"pop_{name}", use_container_width=True):
                        st.write(f"¿Borrar `{name}`?")
                        if st.button("Confirmar", key=f"del_{name}", type="primary", use_container_width=True):
                            st.write("Eliminado")
            
    st.header("Añadir Dataset", divider=True)
    add_dt_cont = st.container(border=True)
    with add_dt_cont:
        select_col, blank_col = st.columns([2,6])
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
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                df_text_column = st.text_input("Columna de texto", value="text")
            with col2:
                df_label_column = st.text_input("Label Column", value="label")
            with col3:
                df_tokenizer = st.text_input("Tokenizador", value="bert-base-uncased")
        
        st.button("Añadir dataset")

with model_tab:
    cols = st.columns(3)
    for i, (name, info) in enumerate(MODELS.items()):
        with cols[i%3]:
            with st.container(border=True):
                head_col, badge_col = st.columns([5, 1])

                with head_col:
                    st.subheader(f"{name}")
                    st.caption(f"{info['type']}")

                st.write(f"{info['description']}")

                _, col_edit, col_del = st.columns([4, 4, 4])
                with col_edit:
                    st.button("Editar", key=f"edit_{name}", use_container_width=True)

                with col_del:
                    with st.popover("Eliminar", key=f"pop_{name}", use_container_width=True):
                        st.write(f"¿Borrar `{name}`?")
                        if st.button("Confirmar", key=f"del_{name}", type="primary", use_container_width=True):
                            st.write("Eliminado")
    st.header("Añadir Modelo", divider=True)

    add_model_cont = st.container(border=True)
    with add_model_cont:
        model_left, model_right = st.columns(2, gap="medium")
        
        with model_left:
            model_name = st.text_input("Nombre", placeholder="ej: ResNet18")
        with model_right:
            model_ident = st.text_input("Identificador", placeholder="ej: resnet18")

        model_desc = st.text_area("Descripción", placeholder="Descripción libre del modelo, sin uso real...")

        st.button("Añadir modelo")