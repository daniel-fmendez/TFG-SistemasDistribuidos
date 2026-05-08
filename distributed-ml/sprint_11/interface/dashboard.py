import streamlit as st

st.header("Dashboard")
st.caption("Resumen general del sistema")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodos", "3 / 3", delta="todos online", delta_color="off", border=True)
c2.metric("Workers", "5", delta="2 local · 2 LAN · 1 remoto", delta_color="off", border=True)
c3.metric("Modelo activo", "ResNet18", delta="CIFAR-10 · 50 000 muestras", delta_color="off", border=True)
c4.metric("Último accuracy", "82.88 %", delta="loss 0.5071", delta_color="off", border=True)


left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### Cluster")
    with st.container(border=True):
        header = st.columns([3, 2, 1, 1, 1, 2])
        for col, label in zip(header, ["Nodo", "Estado", "CPU", "RAM", "Disco", "Latencia"]):
            col.caption(f"#### {label}")

        nodos = st.session_state.nodos
        for nombre, info in nodos.items():
            cols = st.columns([3, 2, 1, 1, 1, 2])
            estado = "🟢 online" if info["activo"] else "🟡 degradado"
            cols[0].markdown(f"**{nombre}**  \n`{info['ip']}`")
            cols[1].markdown(estado)
            cols[2].markdown(f"##### `{info['cpu']}%`")
            cols[3].markdown(f"##### `{info['ram']}G`")
            cols[4].markdown(f"##### `{info['disco']}G`")
            cols[5].markdown(f"##### `{info['latencia']} ms`")

        st.page_link("nodos.py", label="Ver detalle completo →")

    st.markdown("#### Configuración activa")
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            st.caption("modelo")
            st.markdown("##### **ResNet18**")
            st.caption("resnet18 · torchvision · 10 clases")
        with b:
            st.caption("dataset")
            st.markdown("##### **CIFAR-10**")
            st.caption("cifar10 · 50 000 muestras")

        st.divider()
        st.caption("entrenamiento")
        st.markdown("##### 10 épocas · batch 64 · lr `5e-4` · sync 5 · `fed_avg` · sin compresión · 5 workers")
        st.page_link("configuracion.py", label="Editar configuración →")


with right:
    st.markdown("#### Acción rápida")
    with st.container(border=True):
        st.caption("Configuración activa")
        with st.container(border=True):
            st.markdown("#### **ResNet18 · CIFAR-10 · 10 épocas · fed_avg**")
        b1, b2 = st.columns(2)
        b1.button("▶ Build + Run", type="primary", use_container_width=True)
        b2.page_link("lanzar.py", label="→ Ir a Lanzar", use_container_width=True)

    st.markdown("#### Catálogo")
    with st.container(border=True):
        st.caption("Datasets registrados")
        d1, d2, d3 = st.columns(3)
        for col, (nombre, tipo, clases) in zip([d1,d2,d3], [("CIFAR-10","imagen",10),("AG News","texto",4),("IMDB","texto",3)]):
            with col:
                with st.container(border=True):
                    st.markdown(f"##### {nombre}")
                    st.caption(f"{str.capitalize(tipo)} · {clases} clases")

        st.divider()
        st.caption("Modelos registrados")
        m1, m2, m3, m4 = st.columns(4)
        for col, nombre in zip([m1,m2,m3,m4], ["ResNet18","MobileNet","DistilBERT","BERT"]):
            with col:
                with st.container(border=True):
                    st.markdown(f"##### {nombre}")

        st.page_link("modelos_datasets.py", label="Añadir modelo o dataset →")
            
st.markdown("#### Último entrenamiento")
col_g, _ = st.columns([1, 4])
col_g.button("Abrir Grafana ↗", type="secondary")

a, b, c, d = st.columns(4)
a.metric("Accuracy final", "82.88 %", border=True)
b.metric("Loss final", "0.5071", border=True)
c.metric("Duración", "1h 24m",border=True)
d.metric("Épocas completadas", "10 / 10", border=True)

st.divider()
st.markdown("#### Alertas del sistema")
st.warning("**danfer-vm1** — recursos críticos: CPU 88% · RAM 3.9/4 GB · Disco 18/20 GB · heartbeat hace 48s", icon="⚠️")
st.info("Limpieza automática ejecutada en **danfer-vm1** — se eliminaron 17 GB de pesos huérfanos hace 2h", icon="ℹ️")
