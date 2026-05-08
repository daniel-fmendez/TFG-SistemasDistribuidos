import streamlit as st
import yaml

from paths import CONFIG_YAML
import yaml

def load_config():
    if CONFIG_YAML.exists():
        with open(CONFIG_YAML) as f:
            return yaml.safe_load(f) or {}
    return {}

def save_config(config):
    with open(CONFIG_YAML, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


actual_config = load_config()
model_cfg = actual_config.get("model", {})
dataset_cfg = actual_config.get("dataset", {})
training_cfg = actual_config.get("training", {})
workers_cfg = actual_config.get("workers", {})
heartbeat_cfg = actual_config.get("heartbeat", {})
# ==============================================
st.header("Configuración")
st.caption("Edita y guarda config.yaml")

# SI HAY CAMBIOS MOSTRAR CAJA
guardar = st.button("Guardar config.yaml", type="primary")
st.warning("Cambios sin guardar", icon="⚠️")

col1, col2 = st.columns(2)
with col1: 
    model_cont = st.container(border=True)
    with model_cont:
        st.subheader("Modelo", divider="gray")
        
        model_type = st.selectbox("Arquitectura", options=["resnet18","otros"])

        # Añadir para que eliga de entre los existentes
        model_name = st.text_input("Nombre (HuggingFace / torchvision)")

        num_labels = st.number_input("Número de clases (num_labels)", value=dataset_cfg.get("num_labels"), min_value=1, step=1)
    
    dataset_cont = st.container(border=True)

    with dataset_cont:
        st.subheader("Dataset", divider="gray")
        selected_dataset = dataset_cfg.get("name")
        dataset = st.selectbox("Dataset", options=["cifar10","imdb"])

        max_value = 50000 # ADAPTARLO A CADA SELECCION DE DATASET
        total_samples = st.slider("Total muestras", min_value=1, max_value=max_value, value=dataset_cfg.get("total_samples"))

    worker_cont = st.container(border=True)
    NODOS = [
        {"nombre": k, "ip": v["ip"], "tipo": v["tipo"]}
        for k, v in st.session_state.nodos.items()
    ]

    with worker_cont:
        st.subheader("Workers", divider="gray")
        workers = {}
        for nodo in NODOS:
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{nodo['nombre']}**  \n`{nodo['ip']}` · {nodo['tipo']}")
            with c2:
                st.caption("workers")
            with c3:
                workers[nodo["nombre"]] = st.number_input(
                    label=nodo["nombre"],   # requerido pero lo ocultamos
                    min_value=0,
                    max_value=8,
                    value=2 if nodo["tipo"] != "remoto" else 1,
                    step=1,
                    label_visibility="collapsed",
                    key=f"w_{nodo['nombre']}"
                )
        remotos = sum(v for k, v in workers.items()
                    if any(n["nombre"] == k and n["tipo"] == "remoto" for n in NODOS))

        shard = workers_cfg.get("remote_shard_ratio")
        master_ip = workers_cfg.get("master_ip")
        if remotos > 0:
            st.divider()
            st.caption("Parámetros para nodos remotos")
            shard = st.slider("Shard remoto", min_value=0.0, max_value=1.0, value=shard, step=0.05)
            master_ip = st.text_input("IP master", value=master_ip)

        total_workers = sum(workers.values())
        num_local_workers = workers["daniel-asus"]
        
        st.success(f"{total_workers} workers en total · {remotos} remotos")

with col2:
    trainig_cont = st.container(border=True)
    with trainig_cont:
        st.subheader("Entrenamiento", divider="gray")
        seed = st.number_input("Semilla", step=1, value=training_cfg.get("seed"))
        epochs = st.slider("Épocas", min_value=1, max_value=50, value=training_cfg.get("epochs"))
        posibles_potencias = [2 ** j for j in range(1, 8)]
        batch_size = st.select_slider("Bacth Size", options=posibles_potencias, value=training_cfg.get("batch_size"), key="batch_slider")
        opciones_micro = [v for v in posibles_potencias if v <= batch_size]
        micro_size = st.select_slider("Micro Batch Size", options=opciones_micro, value=training_cfg.get("micro_batch_size"), key="micro_slider")
        lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        learning_rate = st.select_slider(
            "Selecciona el Learning Rate",
            options=lr_options,
            value=1e-3,
            format_func=lambda x: f"{x:.0e}" if x % 1 == 0 or x < 0.001 else f"{x}",
            key="lr_slider_fancy"
        )
        max_step = 0# total muestras / (n de workers + batch size) dificili
        # sync_every = st.slider("Sincronizar cada:", min_value=1, max_value=max_step)
        sync_every = st.number_input("Sincronizar cada", value=training_cfg.get("sync_every"), min_value=1, step=1)
        sync_every_early = st.number_input("Sincronizar cada (temprano)", value=training_cfg.get("sync_every_early"), min_value=1, max_value=sync_every, step=1)
        st.divider()
        report_every_step = st.number_input("Reportar metricas cada", value=training_cfg.get("report_step"), min_value=10, max_value=100, step=1)
        st.divider()
        agg_col, comp_col = st.columns(2)
        with agg_col:
            agg_options = ["fed_avg", "fed_median", "fed_trimmed_mean"]
            agg_index = agg_options.index(training_cfg.get("aggregation_strategy"))
            aggregation_type = st.selectbox("Estrategia de agregación", options=["Fed. Average", "Fed. Median", "Fed. Trimmed Mean"], index=agg_index)
            aggregation_type = aggregation_type.lower().replace(".", "").replace(" ", "_")
        with comp_col:
            comp_options = ["none", "quantization", "top_k"]
            comp_index = comp_options.index(training_cfg.get("compression_strategy"))
            compresion_type = st.selectbox("Compresión", options=["None", "Quantization", "Top K"], index=comp_index)
            compresion_type = compresion_type.lower().replace(" ", "_")

        
        # Personalizar??
        quantization_bits = 16
        top_k_ratio = 0.1

    heartbeat_cont = st.container(border=True)

    with heartbeat_cont:
        st.subheader("Heartbeat", divider="gray")

        hb_inter = st.slider("Intervalo (s)",value=heartbeat_cfg.get("interval"), min_value=1, max_value=60)
        hb_multi = st.slider("Multiplicador (timeout = inter. x mult.)",value=heartbeat_cfg.get("multiplier"), min_value=1, max_value=10)

yaml_cont = st.container(border=True)
with yaml_cont:
    st.markdown("##### YAML resultante")
    
    config = {
        "model": {"type": model_type, "name": model_name, "num_labels": num_labels},
        "dataset": {"name": dataset, "total_samples": total_samples},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "micro_batch_size": micro_size,
            "sync_every": sync_every,
            "sync_every_early": sync_every_early,
            "aggregation_strategy": aggregation_type,
            "compression_strategy": compresion_type
        },
        "workers": {"nodes": workers, "remote_shard_ratio": shard, "master_ip": master_ip},
        "heartbeat": {"interval": hb_inter, "multiplier": hb_multi}
    }

    yaml_str = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
    formatted_yaml = ""
    lines = yaml_str.splitlines()
    for line in lines:
        if line and not line.startswith(" ") and ":" in line:
            if formatted_yaml:
                formatted_yaml += "\n"
        formatted_yaml += line + "\n"

    st.code(formatted_yaml, language="yaml")

    if guardar:
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        st.success("✅ Guardado en config.yaml")

