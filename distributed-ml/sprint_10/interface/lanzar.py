import streamlit as st

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
    h2.caption("#### 00:03:42")

    # Barra de progreso
    progreso = FASE_ACTUAL / (len(FASES) - 1)
    st.progress(progreso)

    # Etiquetas de fases
    cols = st.columns(len(FASES))
    for i, (col, fase) in enumerate(zip(cols, FASES)):
        if i < FASE_ACTUAL:
            col.markdown(f"✅ **{str.capitalize(fase)}**")
        elif i == FASE_ACTUAL:
            col.markdown(f"⏳ **{str.capitalize(fase)}**")
        else:
            col.caption(str.capitalize(fase))

    st.info("Esperando registro de workers… 3 / 5 listos", icon="⏳")


log_col, state_col = st.columns(2, gap="large")

MOCK_LOGS = [
    ("09:12:01", "dataset-init-local completado"),
    ("09:12:03", "dataset-init-lan completado"),
    ("09:12:11", "master job lanzado"),
    ("09:12:14", "master service NodePort :30051"),
    ("09:12:17", "worker-0 creado  (daniel-asus)"),
    ("09:12:17", "worker-1 creado  (daniel-asus)"),
    ("09:12:20", "worker-2 creado  (daniserver)"),
    ("09:12:20", "worker-3 creado  (daniserver)"),
    ("09:12:43", "worker-4 pendiente (danfer-vm1)"),
]

with log_col:
    st.markdown("#### Log en vivo")
    with st.container(border=True, height=360):
        for ts, msg in MOCK_LOGS:
            st.markdown(f"`{ts}` {msg}")

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
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 2])
                c1.markdown(f"**{wname}**  \n`{nodo}` · {tipo}")
                badge_color = "green" if estado == "registrado" else "orange"
                c2.badge(estado, color=badge_color)
                c3.caption(f"epoch {epoch}  ·  loss {loss}")

st.subheader("Resultado del último entrenamiento", divider=True)



a, b, c, d = st.columns(4)
a.metric("Accuracy final", "82.88 %", border=True)
b.metric("Loss final", "0.5071", border=True)
c.metric("Duración", "1h 24m", border=True)
d.metric("Épocas completadas", "10 / 10", border=True)

st.button("Ver en Grafana ↗",)
