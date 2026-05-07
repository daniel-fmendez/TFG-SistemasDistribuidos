import streamlit as st

def inject_styles():
    st.markdown("""
    <style>
    /* ── Colores base del proyecto ─────────────────────────
       verde  : #1D9E75   (online, ok, local)
       azul   : #378ADD   (LAN, info)
       ámbar  : #EF9F27   (degradado, remoto, warning)
       rojo   : #E74C3C   (crítico)
    ────────────────────────────────────────────────────── */


    /* Quitar padding excesivo de métricas */
    div[data-testid="metric-container"] {
        padding: 0.6rem 0.8rem;
    }

    /* ── Cards de nodo ──────────────────────────────────── */
    .node-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid #1D9E75;
        background: white;
    }
    .node-card.degradado { border-left-color: #EF9F27; }
    .node-card.critico   { border-left-color: #E74C3C; }

    /* ── Badge custom ───────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 4px;
    }
    .badge-green  { background:#EAF3DE; color:#27500A; }
    .badge-blue   { background:#E6F1FB; color:#0C447C; }
    .badge-amber  { background:#FAEEDA; color:#633806; }
    .badge-red    { background:#FDECEA; color:#7B1B1B; }
    .badge-gray   { background:#EBEBEA; color:#444;    }

    /* ── Barra de recurso custom ────────────────────────── */
    .resource-bar-wrap { margin-bottom: 0.5rem; }
    .resource-label    { font-size: 11px; color: #888; margin-bottom: 2px; }
    .resource-value    { font-size: 15px; font-weight: 600; color: #1a1a1a; }
    .resource-bar-bg   { background: #0e1117; border-radius: 4px; height: 6px; }
    .resource-bar-fill { height: 6px; border-radius: 4px; }
    .bar-green  { background: #1D9E75; }
    .bar-amber  { background: #EF9F27; }
    .bar-red    { background: #E74C3C; }

    /* ── Worker card ────────────────────────────────────── */
    .worker-card {
        border: 1px solid #e0e0de;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .worker-card.pendiente { border-color: #F5CFA0; background: #FFFBF5; }
    .worker-card.registrado{ border-color: #9FE1CB; background: #F7FCF9; }

    /* ── Log terminal ───────────────────────────────────── */
    .log-line { font-family: monospace; font-size: 13px; margin: 2px 0; }
    .log-ts   { color: #888; }
    .log-ok   { color: #1D9E75; }
    .log-warn { color: #EF9F27; }
    .log-info { color: #888; }

    /* ── Pipeline fases ─────────────────────────────────── */
    .fase-done    { color: #1D9E75; font-weight: 600; font-size: 13px; }
    .fase-active  { color: #EF9F27; font-weight: 600; font-size: 13px; }
    .fase-pending { color: #aaa;    font-size: 13px; }

    /* ── Stat card grande ───────────────────────────────── */
    .stat-card {
        border: 1px solid #e0e0de;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background: white;
    }
    .stat-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .stat-value { font-size: 28px; font-weight: 700; color: #1a1a1a; }
    .stat-value.green { color: #1D9E75; }

    /* ── Yaml preview ───────────────────────────────────── */
    .yaml-section { color: #1D9E75; font-weight: 600; }
    .yaml-remote  { color: #EF9F27; }

    </style>
    """, unsafe_allow_html=True)

def badge(texto, tipo="gray"):
    """tipo: green | blue | amber | red | gray"""
    st.markdown(f'<span class="badge badge-{tipo}">{texto}</span>', unsafe_allow_html=True)

def resource_bar(label, value, max_value, unit=""):
    pct = value / max_value
    if pct > 0.85:  
        color = "#E74C3C"
    elif pct > 0.60: 
        color = "#EF9F27"
    else: 
        color = "#1D9E75"
    fill = int(pct * 100)
    html = f"""
    <div class="resource-bar-wrap">
      <div class="resource-label" style="font-size: 22px; color: #4A90E2; font-weight: bold;">
            {label}
        </div>
      <div class="resource-value" style="font-size: 20px; color: #e6e6e6;">
            {value}{unit} / {max_value}{unit}
        </div>
      <div class="resource-bar-bg">
        <div class="resource-bar-bg" style="background-color: #262730; border-radius: 15px; height: 10px; width: 100%; overflow: hidden;">
            <div class="resource-bar-fill" style="width: calc({fill}% + 1px); background-color: {color}; height: 10px;"></div>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def node_card_header(nombre, ip, tipo, activo, heartbeat_txt, btn_key):
    estado = "online" if activo else "degradado"
    dot_color = "#1D9E75" if activo else "#EF9F27"
    tipo_badge_class = {"local":"badge-green","LAN":"badge-blue","remoto":"badge-amber"}.get(tipo,"badge-gray")
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
      <span style="font-size:22px; font-weight:600; color:#ffffff;">
        <span style="color:{dot_color}">●</span> {nombre}
      </span>
      <span class="badge {tipo_badge_class}">{tipo}</span>
      <span class="badge {'badge-green' if activo else 'badge-amber'}">{estado}</span>
      <span style="font-size:11px; color:#888; margin-left:auto;">{heartbeat_txt}</span>
    </div>
    <div style="font-size:12px; color:#888; font-family:monospace; margin-bottom:8px;">{ip} · k3s v1.28</div>
    """, unsafe_allow_html=True)

def worker_row(wname, nodo, tipo, estado, epoch, loss):
    bg_color = "#1E2024" if estado == "registrado" else "#262730"
    border_color = "#1D9E75" if estado == "registrado" else "#EF9F27"
    badge_bg = "#1D9E75" if estado == "registrado" else "#EF9F27"

    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 5px solid {border_color};
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 15px;
    ">
      <!-- Información Principal -->
      <div style="flex-grow: 0; min-width: 150px;">
        <div style="font-weight: 600; color: white; font-size: 18px;">{wname}</div>
        <div style="font-size: 12px; color: #888;">{nodo} · {tipo}</div>
      </div>

      <!-- Badge de Estado -->
      <div style="
        background-color: {badge_bg};
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        text-transform: uppercase;
        font-weight: bold;
      ">
        {estado}
      </div>

      <!-- Datos de Entrenamiento (Alineados a la izquierda también) -->
      <div style="font-size: 14px; color: #aaa; line-height: 1.2; border-left: 1px solid #444; padding-left: 15px;">
        <b>Epoch:</b> {epoch}<br>
        <b>Loss:</b> {loss}
      </div>
    </div>
    """, unsafe_allow_html=True)

def log_line(ts, msg, tipo="info"):
    """tipo: ok | warn | info"""
    st.markdown(
        f'<div class="log-line"><span class="log-ts">{ts}</span> '
        f'<span class="log-{tipo}"> {msg}</span></div>',
        unsafe_allow_html=True
    )

def stat_card(label, value, color=""):
    color_class = f"green" if color == "green" else ""
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">{label}</div>
      <div class="stat-value {color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def pipeline_fases(fases, fase_actual):
    items_html = ""

    for i, fase in enumerate(fases):
        if i < fase_actual:
            dot_bg, dot_border, dot_inner, dot_text, label_color, shadow, font_weight = "#1D9E75", "#1D9E75", "✓", "white", "#1D9E75", "none", "600"
        elif i == fase_actual:
            dot_bg, dot_border, dot_inner, dot_text, label_color, shadow, font_weight = "#EF9F27", "#EF9F27", "▶", "white", "#EF9F27", "0 0 0 5px #FEF0D8", "600"
        else:
            dot_bg, dot_border, dot_inner, dot_text, label_color, shadow, font_weight = "#f0f0ee", "#ccc", str(i + 1), "#aaa", "#aaa", "none", "400"

        if i > 0:
            line_color = "#1D9E75" if i <= fase_actual else "#e0e0de"
            items_html += f'<div style="flex:1; height:3px; background:{line_color}; align-self:flex-start; margin-top:17px; min-width:10px;"></div>'

        items_html += f"""
        <div style="display:flex; flex-direction:column; align-items:center; gap:8px; flex-shrink:0;">
          <div style="width:34px; height:34px; border-radius:50%; background:{dot_bg}; border:2px solid {dot_border}; 
                      display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:700; 
                      color:{dot_text}; box-shadow:{shadow};">{dot_inner}</div>
          <div style="font-size:11px; color:{label_color}; font-weight:{font_weight}; white-space:nowrap;">{fase}</div>
        </div>
        """

    full_html = f"""
    <div style="display:flex; align-items:flex-start; justify-content:space-between; width:100%; padding:20px 0;">
        {items_html}
    </div>
    """.replace("\n", "")
    
    st.markdown(full_html, unsafe_allow_html=True)

def catalog_card(nombre, subtitulo, descripcion=None, tipo="imagen", extra_info=None):
    """
    tipo: "imagen" → borde verde #1D9E75
          "texto"  → borde azul  #378ADD
    """
    border_color = "#1D9E75" if tipo == "imagen" else "#378ADD"
    badge_bg     = "#EAF3DE" if tipo == "imagen" else "#E6F1FB"
    badge_color  = "#27500A" if tipo == "imagen" else "#0C447C"
    badge_label  = "Imagen"  if tipo == "imagen" else "Texto"

    extra_html = ""
    if descripcion:
        extra_html += f'<div style="font-size:12px; color:#888; margin-top:6px;">{descripcion}</div>'
    if extra_info:
        extra_html += f'<div style="font-size:11px; color:#aaa; margin-top:4px; font-family:monospace;">{extra_info}</div>'

    st.markdown(f"""
    <div style="
        border-left: 5px solid {border_color};
        border: 1px solid #e0e0de;
        border-left: 5px solid {border_color};
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        background: white;
    ">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
        <span style="font-weight:600; font-size:15px; color:#1a1a1a;">{nombre}</span>
        <span style="
            background:{badge_bg}; color:{badge_color};
            font-size:11px; font-weight:600;
            padding:2px 8px; border-radius:4px;
        ">{badge_label}</span>
      </div>
      <div style="font-size:12px; color:#666; font-family:monospace;">{subtitulo}</div>
      {extra_html}
    </div>
    """, unsafe_allow_html=True)
