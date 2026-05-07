import streamlit as st
from styles import worker_row, log_line, pipeline_fases

st.header("Lanzar")
st.caption("Build, deploy y seguimiento del entrenamiento")
with st.container(border=True):
    c1, c2 = st.columns([8, 1])
    c1.caption("Configuración activa")
    c1.markdown("#### **ResNet18 · CIFAR-10 · 10 épocas · fed_avg · 5 workers**")
    c2.page_link("configuracion.py", label="Editar →")


with st.container(border=True):
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.button("▶ Build + Run", type="primary",    use_container_width=True)
    b2.button("Build", type="secondary",  use_container_width=True)
    b3.button("Deploy", type="secondary",  use_container_width=True)
    b4.button("■ Detener", type="secondary",  use_container_width=True)
    b5.button("Abrir Grafana ↗", type="secondary",  use_container_width=True)


FASES = ["dataset-init", "cleanup", "master", "workers", "ejecución", "fin"]
FASE_ACTUAL = 3
with st.container(border=True):
    h1, h2 = st.columns([8, 1])
    h1.markdown("#### Fases del despliegue")
    h2.caption("⏱ 00:03:42")

    pipeline_fases(FASES, FASE_ACTUAL)

    st.info("Esperando registro de workers… 3 / 5 listos", icon="⏳")


log_col, state_col = st.columns(2, gap="large")



with log_col:
    st.markdown("#### Log en vivo")
    with st.container(border=True, height=360):
        log_line("09:12:01", "dataset-init-local completado", "ok")
        log_line("09:12:03", "dataset-init-lan completado", "ok")
        log_line("09:12:11", "master job lanzado", "ok")
        log_line("09:12:14", "master service NodePort :30051", "ok")
        log_line("09:12:17", "worker-0 creado  (daniel-asus)", "ok")
        log_line("09:12:17", "worker-1 creado  (daniel-asus)", "ok")
        log_line("09:12:20", "worker-2 creado  (daniserver)", "ok")
        log_line("09:12:20", "worker-3 creado  (daniserver)", "ok")
        log_line("09:12:43", "worker-4 pendiente (danfer-vm1)", "warn")

MOCK_WORKERS = [
    ("worker-0", "daniel-asus", "local",  "registrado", "0/10", "—"),
    ("worker-1", "daniel-asus", "local",  "registrado", "0/10", "—"),
    ("worker-2", "daniserver",  "LAN",    "registrado", "0/10", "—"),
    ("worker-3", "daniserver",  "LAN",    "registrado", "0/10", "—"),
    ("worker-4", "danfer-vm1",  "remoto", "pendiente",  "—",    "—"),
]

with state_col:
    st.markdown("#### Estado de workers")
    with st.container(border=True, height=360):
        for wname, nodo, tipo, estado, epoch, loss in MOCK_WORKERS:
            worker_row(wname, nodo, tipo, estado, epoch, loss)

st.subheader("Resultado del último entrenamiento", divider=True)



a, b, c, d = st.columns(4)
a.metric("Accuracy final", "82.88 %", border=True)
b.metric("Loss final", "0.5071", border=True)
c.metric("Duración", "1h 24m", border=True)
d.metric("Épocas completadas", "10 / 10", border=True)

st.button("Ver en Grafana ↗",)
