import streamlit as st
from styles import resource_bar, node_card_header, badge

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
        col_name, col_btn = st.columns([8,2])
        
        hb_txt = "Último heartbeat: hace 4s" if info["activo"] else "Último heartbeat: hace 48s · timeout en 72s"
        with col_name:
            node_card_header(nombre, info["ip"], info["tipo"], info["activo"], hb_txt, f"btn_{nombre}")
        btn_label = "Comprobar SSH" if info["activo"] else "Reconectar"
        btn_type  = "secondary" if info["activo"] else "primary"
        col_btn.button(btn_label, key=f"ssh_{nombre}", type=btn_type, use_container_width=True)

        st.divider()

        # Recursos
        c1, c2, c3, c4 = st.columns(4)
        with c1: resource_bar("CPU",   info["cpu"],   100,              "%")
        with c2: resource_bar("RAM",   info["ram"],   info["ram_max"],  " GB")
        with c3: resource_bar("Disco", info["disco"], info["disco_max"]," GB")
        with c4:
            lat_color = "#1D9E75" if info["latencia"] < 1 else "#EF9F27"
            st.markdown(f"""
            <div class="resource-bar-wrap">
              <div class="resource-label">Latencia gRPC</div>
              <div class="resource-value" style="color:{lat_color}">{info['latencia']} ms</div>
            </div>
            """, unsafe_allow_html=True)

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
