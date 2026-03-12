"""
Analizador de Prensa con IA — v2
Temas (libre) + Atributos (libre o guiado) + Protagonismo + Vocería + Sentimiento 5 niveles + Score de impacto
"""

import streamlit as st
import pandas as pd
import anthropic
import json
import time
import re
from datetime import datetime

# ── CONFIG ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Analizador de Prensa IA", page_icon="📰", layout="wide")

ORANGE   = "#C87A2A"
BLUE     = "#2B3A4E"
GRAY     = "#7A7268"
GREEN    = "#2E6B4F"
RED      = "#8B1A1A"

ATRIBUTOS_DEFAULT = """Representativa del sector proveedor
Influyente en política pública
Innovadora y tecnológica
Comprometida con la diversidad
Articulada con el Estado
Formadora de capital humano
Con proyección internacional
Referente técnico
Con presencia territorial"""

SENT_SCORES = {
    "muy_positivo":  2,
    "positivo":      1,
    "neutro":        0,
    "negativo":     -1,
    "muy_negativo": -2,
}
PROT_MULT = {"real": 2, "secundario": 1, "referencial": 0.5}

SENT_COLORS = {
    "muy_positivo":  "#1a5c38",
    "positivo":      "#2E6B4F",
    "neutro":        "#6b6b6b",
    "negativo":      "#8B1A1A",
    "muy_negativo":  "#5a0a0a",
}
PROT_COLORS = {
    "real":        "#1a6b3a",
    "secundario":  "#7a5a0a",
    "referencial": "#4a4a4a",
    "error":       "#8B1A1A",
}


# ── PROMPT DINÁMICO ────────────────────────────────────────────────────────────

def build_system_prompt(modo_atributos: str, atributos_lista: list[str]) -> str:

    if modo_atributos == "guiado" and atributos_lista:
        bloque_atributos = f"""2. ATRIBUTOS — categorías valorativas que describen qué imagen proyecta la organización en la nota:
   - Si protagonismo es "real" o "secundario": elige los atributos que apliquen de esta lista. Si detectas uno no contemplado, agrégalo de todas formas con palabras breves y valorativas.
   - Si protagonismo es "referencial": devuelve lista vacía [].
   Lista de referencia: {", ".join(atributos_lista)}"""
    else:
        bloque_atributos = """2. ATRIBUTOS — categorías valorativas que describen qué imagen proyecta la organización en la nota:
   - Si protagonismo es "real" o "secundario": identifica libremente qué atributos de imagen proyecta la nota (ej: "innovadora", "influyente", "representativa", "comprometida con la seguridad"). Sé conciso y valorativo. Puede ser más de uno o ninguno.
   - Si protagonismo es "referencial": devuelve lista vacía []."""

    return f"""Eres un analista experto en reputación corporativa y relaciones con medios.

Para cada nota de prensa sigue este orden estricto:

1. PROTAGONISMO de la entidad analizada (determina esto primero, condiciona todo lo demás):
   - "real": aparece en el título O un vocero declara con verbo de habla (dijo, señaló, afirmó, indicó, explicó, sostuvo, destacó, precisó, advirtió, planteó, subrayó, llamó, instó, anunció)
   - "secundario": mencionada activamente sin declaración propia (organizó, firmó, participó, presentó)
   - "referencial": solo dato de contexto, listado, patrocinio o mención pasiva sin acción propia

{bloque_atributos}

3. TEMAS — descripción objetiva del tópico de la nota (independiente de la entidad):
   Identifica 1 a 3 temas que describan de qué trata la nota. Usa sustantivos concretos en minúsculas (ej: "política regulatoria", "seguridad laboral", "género e inclusión", "mercado del cobre", "formación técnica", "convenios gremiales").

4. VOCERÍA — quién habla en nombre de la entidad:
   - nombre: nombre completo del vocero, o null
   - cargo: cargo que se le atribuye, o null
   - tipo: "interna" (ejecutivo/directivo de la entidad) | "externa" (empresa socia u otro gremio) | "sin_voceria"

5. SENTIMIENTO del tono de la nota hacia la entidad o su industria:
   - "muy_positivo": nota claramente favorable con protagonismo destacado (logros, reconocimientos, liderazgo)
   - "positivo": tono favorable sin énfasis especial
   - "neutro": informativo sin valoración clara
   - "negativo": tono crítico o problemático
   - "muy_negativo": nota claramente desfavorable con protagonismo destacado (crisis, fracasos, críticas graves)

Responde SOLO con JSON válido, sin markdown, sin explicación adicional:
{{
  "protagonismo": "real|secundario|referencial",
  "protagonismo_razon": "explicación en máximo 20 palabras",
  "atributos": ["atributo1", "atributo2"],
  "temas": ["tema1", "tema2"],
  "voceria": {{
    "nombre": "Nombre Apellido o null",
    "cargo": "Cargo o null",
    "tipo": "interna|externa|sin_voceria"
  }},
  "sentimiento": "muy_positivo|positivo|neutro|negativo|muy_negativo",
  "sentimiento_razon": "explicación en máximo 15 palabras"
}}"""


def build_user_prompt(row: pd.Series, entidad: str) -> str:
    cuerpo = limpiar_texto(str(row.get('cuerpo', '') or ''), 800)
    titulo = limpiar_texto(str(row.get('titulo', '') or ''), 200)
    return f"""Analiza esta nota sobre "{entidad}".

Año: {row.get('anio','')} | Trimestre: {row.get('trimestre','')} | Tier: {row.get('tier','')} | Medio: {str(row.get('medio',''))[:50]}
TÍTULO: {titulo}
CUERPO: {cuerpo}"""


# ── FUNCIONES CORE ─────────────────────────────────────────────────────────────

def limpiar_texto(texto: str, max_chars: int = 800) -> str:
    texto = re.sub(r'<[^>]+>', ' ', str(texto or ''))
    texto = re.sub(r'&[a-z]+;', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto[:max_chars]


def calcular_impacto(protagonismo: str, sentimiento: str) -> float:
    s = SENT_SCORES.get(sentimiento, 0)
    m = PROT_MULT.get(protagonismo, 0.5)
    return round(s * m, 2)


def analizar_nota(client, row: pd.Series, entidad: str,
                  modo_atributos: str, atributos_lista: list[str],
                  modelo: str) -> dict:
    system = build_system_prompt(modo_atributos, atributos_lista)
    prompt = build_user_prompt(row, entidad)

    msg = client.messages.create(
        model=modelo,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()
    try:
        parsed = json.loads(raw.replace('```json', '').replace('```', '').strip())
    except Exception:
        m = re.search(r'\{[\s\S]+\}', raw)
        parsed = json.loads(m.group()) if m else None

    if not parsed:
        return {
            "protagonismo": "error", "protagonismo_razon": f"JSON inválido: {raw[:80]}",
            "atributos": [], "temas": [],
            "voceria": {"nombre": None, "cargo": None, "tipo": "sin_voceria"},
            "sentimiento": "neutro", "sentimiento_razon": "Error", "impacto_score": 0
        }

    parsed["impacto_score"] = calcular_impacto(
        parsed.get("protagonismo", "referencial"),
        parsed.get("sentimiento", "neutro")
    )
    return parsed


def enriquecer_df(df_orig: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    df = df_orig.copy()
    def get(i, *keys, default=''):
        r = resultados.get(i, {})
        for k in keys:
            r = r.get(k, {}) if isinstance(r, dict) else default
        return r if r is not None else default

    df['ai_protagonismo']       = df.index.map(lambda i: resultados.get(i, {}).get('protagonismo', ''))
    df['ai_protagonismo_razon'] = df.index.map(lambda i: resultados.get(i, {}).get('protagonismo_razon', ''))
    df['ai_temas']              = df.index.map(lambda i: ', '.join(resultados.get(i, {}).get('temas', [])))
    df['ai_atributos']          = df.index.map(lambda i: ', '.join(resultados.get(i, {}).get('atributos', [])))
    df['ai_vocero_nombre']      = df.index.map(lambda i: resultados.get(i, {}).get('voceria', {}).get('nombre') or '')
    df['ai_vocero_cargo']       = df.index.map(lambda i: resultados.get(i, {}).get('voceria', {}).get('cargo') or '')
    df['ai_vocero_tipo']        = df.index.map(lambda i: resultados.get(i, {}).get('voceria', {}).get('tipo', ''))
    df['ai_sentimiento']        = df.index.map(lambda i: resultados.get(i, {}).get('sentimiento', ''))
    df['ai_sentimiento_razon']  = df.index.map(lambda i: resultados.get(i, {}).get('sentimiento_razon', ''))
    df['ai_impacto_score']      = df.index.map(lambda i: resultados.get(i, {}).get('impacto_score', ''))
    return df


def calcular_stats(resultados: dict) -> dict:
    if not resultados:
        return {}
    vals = list(resultados.values())
    prot    = pd.Series([r.get('protagonismo', '') for r in vals])
    sent    = pd.Series([r.get('sentimiento', '') for r in vals])
    scores  = [r.get('impacto_score', 0) for r in vals if isinstance(r.get('impacto_score'), (int, float))]
    voceros = [r.get('voceria', {}).get('nombre') for r in vals if r.get('voceria', {}).get('nombre')]
    temas   = [t for r in vals for t in r.get('temas', [])]
    atribs  = [a for r in vals for a in r.get('atributos', [])]
    return {
        'total': len(vals),
        'protagonismo': prot.value_counts().to_dict(),
        'sentimiento':  sent.value_counts().to_dict(),
        'score_medio':  round(sum(scores) / len(scores), 2) if scores else 0,
        'score_max':    max(scores) if scores else 0,
        'score_min':    min(scores) if scores else 0,
        'top_voceros':  pd.Series(voceros).value_counts().head(5).to_dict(),
        'top_temas':    pd.Series(temas).value_counts().head(10).to_dict(),
        'top_atributos':pd.Series(atribs).value_counts().head(8).to_dict(),
    }


# ── UI HELPERS ─────────────────────────────────────────────────────────────────

def metric_card(label, value, color=ORANGE, sub=None):
    sub_html = f"<div style='font-size:10px;opacity:.8;margin-top:2px'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div style="background:{color};border-radius:6px;padding:12px 16px;text-align:center;color:white">
        <div style="font-size:22px;font-weight:800;line-height:1.1">{value}</div>
        <div style="font-size:10px;margin-top:4px;opacity:.9">{label}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def bar_mini(label, n, total, color=ORANGE):
    pct = round(n / total * 100) if total > 0 else 0
    st.markdown(f"""
    <div style="margin:4px 0">
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
            <span>{label}</span><span style="font-weight:700;color:{color}">{n} ({pct}%)</span>
        </div>
        <div style="background:#E8E0D4;border-radius:3px;height:6px">
            <div style="width:{pct}%;background:{color};border-radius:3px;height:6px"></div>
        </div>
    </div>""", unsafe_allow_html=True)


def score_badge(score):
    if score >= 3:    c, label = "#1a5c38", "↑↑ Muy alto"
    elif score >= 1:  c, label = GREEN,     "↑ Positivo"
    elif score == 0:  c, label = GRAY,      "→ Neutro"
    elif score >= -1: c, label = "#8B1A1A", "↓ Negativo"
    else:             c, label = "#5a0a0a", "↓↓ Muy bajo"
    return f'<span style="background:{c};color:white;border-radius:3px;padding:1px 7px;font-size:10px">{label} ({score})</span>'


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():

    st.markdown(f"""
    <div style="background:{BLUE};border-radius:8px;padding:16px 20px;margin-bottom:16px">
        <h2 style="color:white;margin:0;font-size:20px">📰 Analizador de Prensa con IA</h2>
        <p style="color:#9BB5C8;margin:4px 0 0;font-size:12px">
            Temas · Atributos · Protagonismo · Vocería · Sentimiento · Score de impacto · Powered by Claude
        </p>
    </div>""", unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        secret_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
        if secret_key:
            api_key = secret_key
            st.success("✓ API Key cargada desde Secrets")
        else:
            api_key = st.text_input("API Key de Anthropic", type="password")

        st.markdown("---")
        modelo = st.text_input(
            "Modelo de Claude",
            value="claude-sonnet-4-6",
            help="Actualiza aquí cuando Anthropic lance nuevas versiones."
        )

        st.markdown("---")
        entidad = st.text_input("Entidad a analizar", value="APRIMIN")

        st.markdown("---")
        st.markdown("**Modo de atributos**")
        modo_atributos = st.radio(
            "Detección de atributos",
            ["libre", "guiado"],
            format_func=lambda x: "🔍 Descubrimiento libre (Claude define)" if x == "libre" else "📋 Lista guiada (Claude elige + puede agregar)",
            label_visibility="collapsed"
        )

        atributos_lista = []
        if modo_atributos == "guiado":
            atributos_texto = st.text_area(
                "Atributos (uno por línea)",
                value=ATRIBUTOS_DEFAULT,
                height=220,
                help="Claude elegirá de esta lista y puede agregar nuevos si los detecta."
            )
            atributos_lista = [a.strip() for a in atributos_texto.strip().split('\n') if a.strip()]

        st.markdown("---")
        delay = st.slider("Pausa entre notas (seg)", 0.3, 2.0, 0.5, 0.1)

        col_titulo = st.text_input("Columna título", value="titulo")
        col_cuerpo = st.text_input("Columna cuerpo", value="cuerpo")
        col_dup    = st.text_input("Columna duplicados", value="es_duplicado")

        st.markdown("---")
        st.markdown(f"""<div style="font-size:10px;color:{GRAY}">
        <b>Score de impacto:</b><br>
        Muy positivo×Real = +4<br>
        Positivo×Real = +2<br>
        Neutro = 0<br>
        Negativo×Real = -2<br>
        Muy negativo×Real = -4
        </div>""", unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📂 Cargar y procesar", "📊 Resultados", "📖 Manual"])

    # ── TAB 1 ──────────────────────────────────────────────────────────────────
    with tab1:

        uploaded = st.file_uploader("Carga tu base (.xlsx o .csv)", type=["xlsx", "xls", "csv"])

        if uploaded:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            df_raw[col_titulo] = df_raw[col_titulo].fillna('') if col_titulo in df_raw.columns else ''
            df_raw[col_cuerpo] = df_raw[col_cuerpo].fillna('') if col_cuerpo in df_raw.columns else ''

            if col_dup in df_raw.columns:
                df_base = df_raw[df_raw[col_dup].astype(str).isin(['0', '0.0', 'False'])].copy()
            else:
                df_base = df_raw.copy()
                st.warning(f"⚠️ Columna '{col_dup}' no encontrada. Procesando todas las filas.")

            key = entidad.lower().strip()
            df_entidad = df_base[
                df_base[col_titulo].str.lower().str.contains(key, na=False) |
                df_base[col_cuerpo].str.lower().str.contains(key, na=False)
            ].copy().reset_index(drop=True)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Base total (sin duplicados)", f"{len(df_base):,}")
            with c2: st.metric(f"Notas con '{entidad}'", f"{len(df_entidad):,}")
            with c3:
                anios = sorted(df_base['anio'].unique()) if 'anio' in df_base.columns else ['—']
                st.metric("Período", f"{anios[0]}–{anios[-1]}" if len(anios) > 1 else str(anios[0]))

            st.dataframe(df_entidad[[col_titulo]].head(5), use_container_width=True, height=140)

            if 'df_entidad' not in st.session_state or st.session_state.get('_entidad') != entidad:
                st.session_state['df_entidad'] = df_entidad
                st.session_state['_entidad']   = entidad
                st.session_state['resultados'] = {}

            st.markdown("---")
            resultados = st.session_state.get('resultados', {})
            n_total = len(df_entidad)
            n_done  = len(resultados)

            if n_done > 0:
                st.progress(n_done / n_total, text=f"{n_done}/{n_total} notas procesadas ({round(n_done/n_total*100)}%)")
                c1, c2 = st.columns(2)
                with c1: st.success(f"✓ {n_done} notas ya procesadas")
                with c2:
                    if st.button("🗑️ Limpiar y reiniciar"):
                        st.session_state['resultados'] = {}
                        st.rerun()

            st.markdown("#### Procesamiento")

            if not api_key:
                st.warning("⚠️ Ingresa tu API Key en el panel lateral.")
            else:
                modo = st.radio(
                    "¿Qué procesar?",
                    ["Solo notas pendientes", "Muestra de prueba (10 notas)", "Todas las notas desde cero"],
                    horizontal=True
                )

                if st.button("▶ Iniciar análisis", type="primary", use_container_width=True):
                    client = anthropic.Anthropic(api_key=api_key)
                    df_proc = st.session_state['df_entidad']

                    if modo == "Muestra de prueba (10 notas)":
                        indices = list(df_proc.sample(min(10, len(df_proc)), random_state=42).index)
                    elif modo == "Todas las notas desde cero":
                        st.session_state['resultados'] = {}
                        indices = list(df_proc.index)
                    else:
                        indices = [i for i in df_proc.index if i not in st.session_state.get('resultados', {})]

                    st.info(f"Procesando {len(indices)} notas.")

                    prog_bar  = st.progress(0, text="Iniciando...")
                    nota_info = st.empty()

                    kpi_cols  = st.columns(5)
                    kpi_real  = kpi_cols[0].empty()
                    kpi_sec   = kpi_cols[1].empty()
                    kpi_ref   = kpi_cols[2].empty()
                    kpi_score = kpi_cols[3].empty()
                    kpi_err   = kpi_cols[4].empty()

                    st.markdown("**Feed en vivo**")
                    feed      = st.empty()
                    error_log = []
                    feed_rows = []

                    def render_feed(rows):
                        if not rows:
                            return
                        filas = ""
                        for r in reversed(rows[-30:]):
                            pc = PROT_COLORS.get(r['prot'], '#444')
                            sc = SENT_COLORS.get(r['sent'], '#444')
                            score_v = r.get('score', 0)
                            score_c = GREEN if score_v > 0 else (RED if score_v < 0 else GRAY)
                            filas += f"""
                            <tr style="border-bottom:1px solid #e0d8cc;font-size:11px">
                              <td style="padding:4px 6px;color:#888">{r['n']}</td>
                              <td style="padding:4px 6px;max-width:220px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{r['titulo']}</td>
                              <td style="padding:4px 6px;color:#888;white-space:nowrap">{r['anio']} {r['trim']}</td>
                              <td style="padding:4px 6px"><span style="background:{pc};color:white;border-radius:3px;padding:1px 6px;font-size:10px">{r['prot']}</span></td>
                              <td style="padding:4px 6px;font-size:10px;color:#555">{r['temas'][:50] or '—'}</td>
                              <td style="padding:4px 6px;font-size:10px;color:#555">{r['atribs'][:50] or '—'}</td>
                              <td style="padding:4px 6px;font-size:10px;color:#666">{r['vocero'][:22] or '—'}</td>
                              <td style="padding:4px 6px"><span style="background:{sc};color:white;border-radius:3px;padding:1px 6px;font-size:10px">{r['sent'].replace('_',' ')}</span></td>
                              <td style="padding:4px 6px;font-weight:700;color:{score_c};text-align:center">{score_v:+g}</td>
                            </tr>"""
                        feed.markdown(f"""
                        <div style="max-height:380px;overflow-y:auto;border:1px solid #ddd;border-radius:6px">
                        <table style="width:100%;border-collapse:collapse;font-family:monospace">
                          <thead><tr style="background:{BLUE};color:white;font-size:10px;position:sticky;top:0">
                            <th style="padding:5px 6px">#</th><th style="padding:5px 6px">Título</th>
                            <th style="padding:5px 6px">Período</th><th style="padding:5px 6px">Protagonismo</th>
                            <th style="padding:5px 6px">Temas</th><th style="padding:5px 6px">Atributos</th>
                            <th style="padding:5px 6px">Vocero</th><th style="padding:5px 6px">Sentimiento</th>
                            <th style="padding:5px 6px">Score</th>
                          </tr></thead>
                          <tbody>{filas}</tbody>
                        </table></div>""", unsafe_allow_html=True)

                    def render_kpis(rows):
                        t = len(rows) or 1
                        n_r = sum(1 for r in rows if r['prot'] == 'real')
                        n_s = sum(1 for r in rows if r['prot'] == 'secundario')
                        n_f = sum(1 for r in rows if r['prot'] == 'referencial')
                        n_e = sum(1 for r in rows if r['prot'] == 'error')
                        scores = [r['score'] for r in rows if isinstance(r.get('score'), (int, float))]
                        avg = round(sum(scores)/len(scores), 1) if scores else 0
                        kpi_real.metric("Real", n_r, f"{round(n_r/t*100)}%")
                        kpi_sec.metric("Secundario", n_s, f"{round(n_s/t*100)}%")
                        kpi_ref.metric("Referencial", n_f, f"{round(n_f/t*100)}%")
                        kpi_score.metric("Score medio", avg)
                        kpi_err.metric("Errores", n_e)

                    for j, idx in enumerate(indices):
                        row = df_proc.loc[idx]
                        nota_info.markdown(
                            f"⏳ **{j+1}/{len(indices)}** — "
                            f"`{str(row.get('medio',''))[:35]}` · "
                            f"{row.get('anio','')} {row.get('trimestre','')}"
                        )
                        try:
                            res = analizar_nota(client, row, entidad, modo_atributos, atributos_lista, modelo)
                            st.session_state['resultados'][idx] = res
                            feed_rows.append({
                                'n':      j + 1,
                                'titulo': str(row.get(col_titulo, ''))[:65],
                                'anio':   str(row.get('anio', '')),
                                'trim':   str(row.get('trimestre', '')),
                                'prot':   res.get('protagonismo', '?'),
                                'temas':  ', '.join(res.get('temas', [])),
                                'atribs': ', '.join(res.get('atributos', [])),
                                'vocero': res.get('voceria', {}).get('nombre') or '',
                                'sent':   res.get('sentimiento', '?'),
                                'score':  res.get('impacto_score', 0),
                            })
                        except Exception as e:
                            error_log.append(f"Nota {idx}: {str(e)[:200]}")
                            st.session_state['resultados'][idx] = {
                                "protagonismo": "error",
                                "protagonismo_razon": f"Error: {str(e)[:200]}",
                                "atributos": [], "temas": [],
                                "voceria": {"nombre": None, "cargo": None, "tipo": "sin_voceria"},
                                "sentimiento": "neutro", "sentimiento_razon": "Error",
                                "impacto_score": 0
                            }
                            feed_rows.append({
                                'n': j+1, 'titulo': str(row.get(col_titulo, ''))[:65],
                                'anio': str(row.get('anio','')), 'trim': str(row.get('trimestre','')),
                                'prot': 'error', 'temas': '', 'atribs': '', 'vocero': '',
                                'sent': 'error', 'score': 0
                            })

                        prog_bar.progress((j+1)/len(indices), text=f"{j+1}/{len(indices)} ({round((j+1)/len(indices)*100)}%)")
                        render_feed(feed_rows)
                        render_kpis(feed_rows)
                        time.sleep(delay)

                    nota_info.empty()
                    st.success(f"✓ {len(indices)} notas analizadas.")
                    if error_log:
                        with st.expander(f"⚠️ {len(error_log)} errores"):
                            st.write('\n'.join(error_log))
                    st.rerun()

            # Download
            if n_done > 0:
                st.markdown("---")
                df_enriq = enriquecer_df(st.session_state['df_entidad'], st.session_state['resultados'])
                df_export = df_enriq[df_enriq.index.isin(st.session_state['resultados'].keys())].copy()

                from io import BytesIO
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    df_export.to_excel(w, index=False, sheet_name='Notas enriquecidas')

                fname = f"analisis_{entidad.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                st.download_button(
                    label=f"⬇️ Descargar Excel enriquecido ({n_done} notas)",
                    data=buf.getvalue(), file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, type="primary"
                )

    # ── TAB 2: RESULTADOS ──────────────────────────────────────────────────────
    with tab2:
        resultados = st.session_state.get('resultados', {})
        df_e = st.session_state.get('df_entidad', pd.DataFrame())

        if not resultados:
            st.info("Procesa notas primero.")
        else:
            stats = calcular_stats(resultados)
            total = stats['total']

            st.markdown(f"### Resumen · {entidad} · {total:,} notas analizadas")

            # KPIs
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                n_real = stats['protagonismo'].get('real', 0)
                metric_card("Protagonismo real", f"{round(n_real/total*100)}%", BLUE, f"{n_real} notas")
            with c2:
                n_pos = stats['sentimiento'].get('muy_positivo', 0) + stats['sentimiento'].get('positivo', 0)
                metric_card("Sentimiento positivo", f"{round(n_pos/total*100)}%", GREEN, f"{n_pos} notas")
            with c3:
                n_neg = stats['sentimiento'].get('muy_negativo', 0) + stats['sentimiento'].get('negativo', 0)
                metric_card("Sentimiento negativo", f"{round(n_neg/total*100)}%", RED, f"{n_neg} notas")
            with c4:
                sc = stats['score_medio']
                color = GREEN if sc > 0 else (RED if sc < 0 else GRAY)
                metric_card("Score medio de impacto", f"{sc:+g}", color)
            with c5:
                n_voz = sum(1 for r in resultados.values() if r.get('voceria', {}).get('nombre'))
                metric_card("Con vocería", f"{round(n_voz/total*100)}%", ORANGE, f"{n_voz} notas")

            st.markdown("---")
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Protagonismo**")
                for k, n in sorted(stats['protagonismo'].items(), key=lambda x: -x[1]):
                    bar_mini(k.capitalize(), n, total, PROT_COLORS.get(k, GRAY))

                st.markdown("<br>**Sentimiento**", unsafe_allow_html=True)
                for k, n in sorted(stats['sentimiento'].items(), key=lambda x: -x[1]):
                    bar_mini(k.replace('_', ' ').capitalize(), n, total, SENT_COLORS.get(k, GRAY))

            with col_r:
                st.markdown("**Top temas detectados**")
                for k, n in list(stats.get('top_temas', {}).items())[:8]:
                    bar_mini(k, n, total, BLUE)

                st.markdown("<br>**Top atributos**", unsafe_allow_html=True)
                for k, n in list(stats.get('top_atributos', {}).items())[:6]:
                    bar_mini(k, n, total, ORANGE)

                st.markdown("<br>**Top voceros**", unsafe_allow_html=True)
                for k, n in stats.get('top_voceros', {}).items():
                    bar_mini(k, n, total, "#5B8AB0")

            st.markdown("---")
            st.markdown("**Vista de notas procesadas**")

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_prot = st.selectbox("Protagonismo", ["Todos", "real", "secundario", "referencial"])
            with fc2:
                f_sent = st.selectbox("Sentimiento", ["Todos", "muy_positivo", "positivo", "neutro", "negativo", "muy_negativo"])
            with fc3:
                f_score = st.selectbox("Score impacto", ["Todos", "Positivo (>0)", "Neutro (=0)", "Negativo (<0)"])

            df_enriq = enriquecer_df(df_e, resultados)
            df_view  = df_enriq[df_enriq.index.isin(resultados.keys())].copy()

            if f_prot != "Todos":
                df_view = df_view[df_view['ai_protagonismo'] == f_prot]
            if f_sent != "Todos":
                df_view = df_view[df_view['ai_sentimiento'] == f_sent]
            if f_score == "Positivo (>0)":
                df_view = df_view[df_view['ai_impacto_score'] > 0]
            elif f_score == "Neutro (=0)":
                df_view = df_view[df_view['ai_impacto_score'] == 0]
            elif f_score == "Negativo (<0)":
                df_view = df_view[df_view['ai_impacto_score'] < 0]

            cols_show = [c for c in ['titulo', 'anio', 'trimestre', 'tier',
                         'ai_protagonismo', 'ai_temas', 'ai_atributos',
                         'ai_vocero_nombre', 'ai_sentimiento', 'ai_impacto_score']
                         if c in df_view.columns]

            st.markdown(f"*{len(df_view):,} notas*")
            st.dataframe(df_view[cols_show], use_container_width=True, height=420)

    # ── TAB 3: MANUAL ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown(f"""
## Manual de uso — v2

### Qué analiza esta app por nota

| Campo | Descripción |
|---|---|
| `ai_protagonismo` | real / secundario / referencial |
| `ai_protagonismo_razon` | Justificación breve |
| `ai_temas` | Tópicos de la nota (descubiertos libremente) |
| `ai_atributos` | Imagen que proyecta la organización |
| `ai_vocero_nombre` | Nombre del vocero si declara |
| `ai_vocero_cargo` | Cargo atribuido |
| `ai_vocero_tipo` | interna / externa / sin_voceria |
| `ai_sentimiento` | muy_positivo / positivo / neutro / negativo / muy_negativo |
| `ai_sentimiento_razon` | Justificación breve |
| `ai_impacto_score` | Score numérico: sentimiento × multiplicador de protagonismo |

### Score de impacto reputacional

```
Score = valor_sentimiento × multiplicador_protagonismo

Valores sentimiento:  muy_positivo=+2, positivo=+1, neutro=0, negativo=-1, muy_negativo=-2
Multiplicadores:      real=×2, secundario=×1, referencial=×0.5

Rango posible: -4 (muy negativo + real) a +4 (muy positivo + real)
```

### Modos de atributos

**Descubrimiento libre:** Claude identifica libremente qué imagen proyecta la nota. Ideal para primera exploración o cuando no tienes hipótesis previas.

**Lista guiada:** Claude elige de tu lista y puede agregar si detecta algo no contemplado. Ideal para seguimiento sistemático entre períodos o clientes.

### Costos estimados

| Base | Tiempo | Costo aprox. |
|---|---|---|
| 500 notas | ~15 min | ~2 USD |
| 1.000 notas | ~30 min | ~4 USD |
| 3.434 notas | ~90 min | ~14 USD |

*(v2 usa más tokens por nota que v1 — temas + atributos + sentimiento 5 niveles)*
        """)


if __name__ == "__main__":
    main()
