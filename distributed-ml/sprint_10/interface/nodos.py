import streamlit as st

st.header("Modelos y Datasets")
st.caption("Gestiona conexiones y el estado del clsuter completo")

nodos = st.session_state.nodos

total    = len(nodos)
online   = sum(1 for n in nodos.values() if n["activo"])
degraded = total - online
lat_media = round(sum(n["latencia"] for n in nodos.values()) / total, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodos totales", total, border=True)
c2.metric("Online", online, border=True)
c3.metric("Degradado", degraded, border=True)
c4.metric("Latencia media gRPC", f"{lat_media} ms", border=True)


st.subheader("Nodos", divider=True)
st.divider()
h1, h2 = st.columns([6, 1])
h1.markdown("#### Nodos registrados")
h2.button("+ Añadir nodo", type="primary", use_container_width=True)


for nombre, info in nodos.items():
    with st.container(border=True):
        # Cabecera
        col_name, col_tags, col_hb, col_btn = st.columns([1, 4, 1, 2])
        estado_icon = "🟢" if info["activo"] else "🟡"
        col_name.markdown(f"##### {estado_icon} {nombre}")
        col_name.caption(f"`{info['ip']}` · k3s v1.28")
        with col_tags:
            st.badge(info["tipo"], color="blue" if info["tipo"]=="LAN" else ("green" if info["tipo"]=="local" else "orange"))
        col_hb.caption("Último heartbeat: hace 4s" if info["activo"] else "Último heartbeat: hace 48s · timeout en 72s")
        btn_label = "Comprobar SSH" if info["activo"] else "Reconectar"
        btn_type  = "secondary" if info["activo"] else "primary"
        col_btn.button(btn_label, key=f"ssh_{nombre}", type=btn_type, use_container_width=True)

        st.divider()

        # Recursos
        cpu_c, ram_c, disk_c, lat_c = st.columns(4)

        with cpu_c:
            st.markdown("#### CPU")
            st.markdown(f"### **{info['cpu']}%**")
            st.progress(info["cpu"] / 100)

        with ram_c:
            st.markdown("#### RAM")
            st.markdown(f"### **{info['ram']} / {info['ram_max']} GB**")
            st.progress(info["ram"] / info["ram_max"])

        with disk_c:
            st.markdown("#### Disco")
            st.markdown(f"### **{info['disco']} / {info['disco_max']} GB**")
            st.progress(info["disco"] / info["disco_max"])

        with lat_c:
            st.markdown("#### Latencia gRPC")
            st.markdown(f"### **{info['latencia']} ms**")

        st.divider()

        # Workers
        st.caption("Workers activos")
        wcols = st.columns(info["workers"] + 1)
        for i in range(info["workers"]):
            wcols[i].badge(f"worker-{i}", color="green" if info["activo"] else "orange")


st.subheader("Añadir nodo", divider=True)

with st.container(border=True):
    select_col, blank_col = st.columns([2,6])
    with select_col:
        node_type = st.selectbox("Tipo de Nodo", options=["Nodo LAN (mismo cluster)", "Cluster remoto (k3s independiente)"])

    name_col, ip_col = st.columns([5,5])
    with name_col:
        node_name = st.text_input("Nombre", key="dataset_name", placeholder="ej: CIFAR-10")
    with ip_col:
        ip = st.text_input("IP / hostname", placeholder="ej: 10.0.0.1")
    
    ssh_user_col, ssh_port_col, ssh_pass_col = st.columns(3)
    with ssh_user_col:
        ssh_user = st.text_input("Usuario SSH", placeholder="ej: daniel")
    with ssh_port_col:
        ssh_port = st.text_input("Puerto SSH", placeholder="ej: 22")
    with ssh_pass_col:
        ssh_port = st.text_input("Clave privada SSH", placeholder="~/.ssh/id_rsa", type="password")
    
    b1, b2, _ = st.columns([2, 2, 4])
    b1.button("Probar conexión",  type="primary",   use_container_width=True)
    b2.button("Registrar nodo",   type="secondary",  use_container_width=True)
