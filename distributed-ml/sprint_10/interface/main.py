import streamlit as st
from styles import inject_styles

st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        page_title="DistML Panel",
    )
inject_styles()
pages = [
    st.Page("dashboard.py", title="Dashboard"),
    st.Page("nodos.py", title="Nodos"),
    st.Page("lanzar.py", title="Lanzar"),
    st.Page("modelos_datasets.py", title="Modelos y Datasets"),
    st.Page("configuracion.py", title="Configuracion")
]
st.sidebar.caption("Nodos")
if "nodos" not in st.session_state:
    st.session_state.nodos = {
        "daniel-asus": {"activo": True,  "tipo": "local",  "ip": "192.168.1.129", "workers": 2, "cpu": 34, "ram": 5.2, "ram_max": 16, "disco": 48, "disco_max": 120, "latencia": 0.3},
        "daniserver":  {"activo": True,  "tipo": "LAN",    "ip": "192.168.1.10",  "workers": 2, "cpu": 61, "ram": 6.8, "ram_max": 8,  "disco": 22, "disco_max": 60,  "latencia": 2.5},
        "danfer-vm1":  {"activo": False, "tipo": "remoto", "ip": "***REMOVED***",       "workers": 1, "cpu": 88, "ram": 3.9, "ram_max": 4,  "disco": 18, "disco_max": 20,  "latencia": 2.7},
    }
with st.sidebar:
    st.markdown("**Cluster**")
    for nombre, info in st.session_state.nodos.items():
        color = "🟢" if info["activo"] else "🟡"
        st.markdown(f"{color} **{nombre}** · {info['tipo']}")
        
pg = st.navigation(pages=pages)
pg.run()