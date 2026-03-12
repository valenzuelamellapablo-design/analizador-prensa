"""
Analizador de Prensa con IA
Enriquece bases de notas con: atributos, protagonismo, vocería y sentimiento.
Guarda progreso incremental para resistir interrupciones.
"""

import streamlit as st
import pandas as pd
import anthropic
import json
import time
import re
import os
from datetime import datetime
from pathlib import Path

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Analizador de Prensa IA",
    page_icon="📰",
    layout="wide",
)

# Colores corporativos
ORANGE  = "#C87A2A"
BLUE    = "#2B3A4E"
CREAM   = "#F7F3ED"
GRAY    = "#7A7268"

ATRIBUTOS_DEFAULT = [
    "Representación sector proveedor",
    "Convocatoria y arquitectura editorial",
    "Articulación gremial",
    "Diversidad de género",
    "Posicionamiento político",
    "Innovación y tecnología",
    "Formación y capital humano",
    "Articulación con el Estado",
    "Proyección internacional",
    "Vocería técnica",
    "Negociación bilateral",
    "Vanguardia tecnológica",
    "Responsabilidad territorial",
]

SYSTEM_PROMPT = """Eres un analista experto en reputación corporativa y relaciones con medios.

Para cada nota de prensa identifica:

1. ATRIBUTOS institucionales presentes (puede ser más de uno o ninguno). Elige solo de la lista provista.

2. PROTAGONISMO de la entidad analizada:
   - "real": aparece en el título O un vocero declara con verbo de habla (dijo, señaló, afirmó, indicó, explicó, sostuvo, destacó, precisó, advirtió, planteó, subrayó, llamó, instó, anunció)
   - "secundario": mencionada activamente sin declaración propia (organizó, firmó, participó, presentó)
   - "referencial": solo dato de contexto, listado, patrocinio o mención pasiva sin acción propia

3. VOCERÍA (quién habla en nombre de la entidad):
   - nombre: nombre completo del vocero, o null si no hay
   - cargo: cargo que se le atribuye, o null
   - tipo: "interna" (ejecutivo/directivo de la entidad) | "externa" (empresa socia u otro gremio) | "sin_voceria"

4. SENTIMIENTO del tono de la nota:
   - "positivo": logros, acuerdos, avances, reconocimientos
   - "negativo": críticas, problemas, conflictos, fracasos
   - "neutro": informativo sin valoración clara

Responde SOLO con JSON válido, sin markdown, sin explicación adicional:
{
  "atributos": ["atributo1", "atributo2"],
  "protagonismo": "real|secundario|referencial",
  "protagonismo_razon": "explicación en máximo 20 palabras",
  "voceria": {
    "nombre": "Nombre Apellido o null",
    "cargo": "Cargo o null",
    "tipo": "interna|externa|sin_voceria"
  },
  "sentimiento": "positivo|negativo|neutro",
  "sentimiento_razon": "explicación en máximo 15 palabras"
}"""


# ── FUNCIONES CORE ─────────────────────────────────────────────────────────────

def build_prompt(row: pd.Series, entidad: str, atributos: list[str]) -> str:
    cuerpo = str(row.get('cuerpo', '') or '')[:1000]
    titulo = str(row.get('titulo', '') or '')
    return f"""Analiza esta nota de prensa sobre "{entidad}".

LISTA DE ATRIBUTOS POSIBLES:
{chr(10).join(f'- {a}' for a in atributos)}

NOTA:
Año: {row.get('anio','')} | Trimestre: {row.get('trimestre','')} | Tier: {row.get('tier','')} | Medio: {str(row.get('medio',''))[:50]}
TÍTULO: {titulo}
CUERPO: {cuerpo}"""


def analizar_nota(client: anthropic.Anthropic, row: pd.Series, entidad: str, atributos: list[str]) -> dict:
    """Llama a Claude y retorna el JSON parseado."""
    prompt = build_prompt(row, entidad, atributos)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw.replace('```json', '').replace('```', '').strip())
    except Exception:
        m = re.search(r'\{[\s\S]+\}', raw)
        if m:
            return json.loads(m.group())
        return {
            "atributos": [],
            "protagonismo": "referencial",
            "protagonismo_razon": "Error parsing",
            "voceria": {"nombre": None, "cargo": None, "tipo": "sin_voceria"},
            "sentimiento": "neutro",
            "sentimiento_razon": "Error parsing"
        }


def enriquecer_df(df_orig: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    """Agrega columnas de análisis al dataframe original."""
    df = df_orig.copy()
    df['ai_atributos']           = df.index.map(lambda i: ', '.join(resultados.get(i, {}).get('atributos', [])))
    df['ai_protagonismo']        = df.index.map(lambda i: resultados.get(i, {}).get('protagonismo', ''))
    df['ai_protagonismo_razon']  = df.index.map(lambda i: resultados.get(i, {}).get('protagonismo_razon', ''))
    df['ai_vocero_nombre']       = df.index.map(lambda i: resultados.get(i, {}).get('voceria', {}).get('nombre') or '')
    df['ai_vocero_cargo']        = df.index.map(lambda i: resultados.get(i, {}).get('voceria', {}).get('cargo') or '')
    df['ai_vocero_tipo']         = df.index.map(lambda i: resultados.get(i, {}).get('voceria', {}).get('tipo', ''))
    df['ai_sentimiento']         = df.index.map(lambda i: resultados.get(i, {}).get('sentimiento', ''))
    df['ai_sentimiento_razon']   = df.index.map(lambda i: resultados.get(i, {}).get('sentimiento_razon', ''))
    return df


def calcular_estadisticas(resultados: dict, df_sub: pd.DataFrame) -> dict:
    """Calcula métricas del análisis completado."""
    total = len(resultados)
    if total == 0:
        return {}

    protagonismo = pd.Series([r.get('protagonismo', '') for r in resultados.values()])
    sentimiento  = pd.Series([r.get('sentimiento', '') for r in resultados.values()])
    voceros      = [r.get('voceria', {}).get('nombre') for r in resultados.values() if r.get('voceria', {}).get('nombre')]
    atributos    = []
    for r in resultados.values():
        atributos.extend(r.get('atributos', []))

    return {
        'total': total,
        'protagonismo': protagonismo.value_counts().to_dict(),
        'sentimiento':  sentimiento.value_counts().to_dict(),
        'top_voceros':  pd.Series(voceros).value_counts().head(5).to_dict(),
        'top_atributos': pd.Series(atributos).value_counts().head(8).to_dict(),
    }


# ── UI HELPERS ─────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, color: str = ORANGE):
    st.markdown(f"""
    <div style="background:{color};border-radius:6px;padding:12px 16px;text-align:center;color:white;margin:2px">
        <div style="font-size:22px;font-weight:800;line-height:1.1">{value}</div>
        <div style="font-size:10px;margin-top:4px;opacity:.9">{label}</div>
    </div>""", unsafe_allow_html=True)


def bar_mini(label: str, n: int, total: int, color: str = ORANGE):
    pct = round(n / total * 100) if total > 0 else 0
    st.markdown(f"""
    <div style="margin:3px 0">
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
            <span>{label}</span><span style="font-weight:700;color:{color}">{n} ({pct}%)</span>
        </div>
        <div style="background:#E8E0D4;border-radius:3px;height:6px">
            <div style="width:{pct}%;background:{color};border-radius:3px;height:6px"></div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── INTERFAZ PRINCIPAL ─────────────────────────────────────────────────────────

def main():

    # Header
    st.markdown(f"""
    <div style="background:{BLUE};border-radius:8px;padding:16px 20px;margin-bottom:16px">
        <h2 style="color:white;margin:0;font-size:20px">📰 Analizador de Prensa con IA</h2>
        <p style="color:#9BB5C8;margin:4px 0 0;font-size:12px">
            Enriquece bases de notas con atributos, protagonismo, vocería y sentimiento · Powered by Claude
        </p>
    </div>""", unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### ⚙️ Configuración")

        api_key = st.text_input(
            "API Key de Anthropic",
            type="password",
            help="Tu clave de la API de Anthropic. No se guarda."
        )

        st.markdown("---")
        entidad = st.text_input(
            "Entidad a analizar",
            value="APRIMIN",
            help="Nombre de la organización. Se usa para filtrar notas y contextualizar el análisis."
        )

        st.markdown("---")
        st.markdown("**Atributos a detectar**")
        atributos_texto = st.text_area(
            "Un atributo por línea",
            value="\n".join(ATRIBUTOS_DEFAULT),
            height=300,
            help="Puedes editar, agregar o eliminar atributos según el cliente."
        )
        atributos = [a.strip() for a in atributos_texto.strip().split('\n') if a.strip()]

        st.markdown("---")
        delay = st.slider(
            "Pausa entre notas (seg)",
            min_value=0.3, max_value=2.0, value=0.5, step=0.1,
            help="Evita sobrecargar la API. 0.5s recomendado."
        )

        col_titulo = st.text_input("Columna título", value="titulo")
        col_cuerpo = st.text_input("Columna cuerpo", value="cuerpo")
        col_dup    = st.text_input("Columna duplicados", value="es_duplicado")

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:10px;color:{GRAY}">
        <b>Costo estimado por nota:</b> ~0.003 USD<br>
        <b>1.000 notas:</b> ~3 USD · ~25 min<br>
        <b>3.400 notas:</b> ~10 USD · ~85 min
        </div>""", unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📂 Cargar y procesar", "📊 Resultados", "📖 Manual de uso"])

    # ─── TAB 1: CARGAR ────────────────────────────────────────────────────────
    with tab1:

        uploaded = st.file_uploader(
            "Carga tu base de notas (.xlsx o .csv)",
            type=["xlsx", "xls", "csv"]
        )

        if uploaded:
            # Leer archivo
            if uploaded.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            # Limpiar
            df_raw[col_titulo] = df_raw[col_titulo].fillna('') if col_titulo in df_raw.columns else ''
            df_raw[col_cuerpo] = df_raw[col_cuerpo].fillna('') if col_cuerpo in df_raw.columns else ''

            # Filtrar duplicados
            if col_dup in df_raw.columns:
                df_base = df_raw[df_raw[col_dup].astype(str).isin(['0', '0.0', 'False'])].copy()
            else:
                df_base = df_raw.copy()
                st.warning(f"⚠️ Columna '{col_dup}' no encontrada. Procesando todas las filas.")

            # Filtrar por entidad
            key = entidad.lower().strip()
            df_entidad = df_base[
                df_base[col_titulo].str.lower().str.contains(key, na=False) |
                df_base[col_cuerpo].str.lower().str.contains(key, na=False)
            ].copy()
            df_entidad = df_entidad.reset_index(drop=True)

            # Info
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total base (sin duplicados)", f"{len(df_base):,}")
            with col_b:
                st.metric(f"Notas con '{entidad}'", f"{len(df_entidad):,}")
            with col_c:
                anios = sorted(df_base['anio'].unique()) if 'anio' in df_base.columns else ['—']
                st.metric("Período", f"{anios[0]}–{anios[-1]}" if len(anios)>1 else str(anios[0]))

            st.dataframe(
                df_entidad[[col_titulo, col_cuerpo if col_cuerpo in df_entidad.columns else col_titulo]].head(5),
                use_container_width=True,
                height=160
            )

            # Guardar en session state
            if 'df_entidad' not in st.session_state or st.session_state.get('entidad_nombre') != entidad:
                st.session_state['df_entidad'] = df_entidad
                st.session_state['entidad_nombre'] = entidad
                st.session_state['resultados'] = {}

            st.markdown("---")

            # Estado del progreso
            resultados = st.session_state.get('resultados', {})
            n_total    = len(df_entidad)
            n_done     = len(resultados)
            n_pending  = n_total - n_done

            if n_done > 0:
                pct = round(n_done / n_total * 100)
                st.progress(pct / 100, text=f"Progreso: {n_done}/{n_total} notas procesadas ({pct}%)")

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✓ {n_done} notas ya procesadas — el progreso se mantiene aunque recargues")
                with col2:
                    if st.button("🗑️ Limpiar progreso y reiniciar"):
                        st.session_state['resultados'] = {}
                        st.rerun()

            # Botones de procesamiento
            st.markdown("#### Procesamiento")

            if not api_key:
                st.warning("⚠️ Ingresa tu API Key de Anthropic en el panel lateral para procesar.")
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

                    st.info(f"Procesando {len(indices)} notas. Puedes detener y continuar luego.")

                    prog_bar  = st.progress(0, text="Iniciando...")
                    nota_info = st.empty()
                    error_log = []

                    for j, idx in enumerate(indices):
                        row = df_proc.loc[idx]
                        nota_info.markdown(f"⏳ **Nota {j+1}/{len(indices)}** — {str(row.get('medio',''))[:40]} · {row.get('anio','')} {row.get('trimestre','')}")

                        try:
                            resultado = analizar_nota(client, row, entidad, atributos)
                            st.session_state['resultados'][idx] = resultado
                        except Exception as e:
                            error_log.append(f"Nota {idx}: {str(e)[:80]}")
                            st.session_state['resultados'][idx] = {
                                "atributos": [], "protagonismo": "referencial",
                                "protagonismo_razon": f"Error: {str(e)[:50]}",
                                "voceria": {"nombre": None, "cargo": None, "tipo": "sin_voceria"},
                                "sentimiento": "neutro", "sentimiento_razon": "Error"
                            }

                        prog_bar.progress((j + 1) / len(indices), text=f"{j+1}/{len(indices)} notas ({round((j+1)/len(indices)*100)}%)")
                        time.sleep(delay)

                    nota_info.empty()
                    st.success(f"✓ Proceso completado. {len(indices)} notas analizadas.")
                    if error_log:
                        with st.expander(f"⚠️ {len(error_log)} errores"):
                            st.write('\n'.join(error_log))
                    st.rerun()

            # Download siempre disponible si hay resultados
            if n_done > 0:
                st.markdown("---")
                st.markdown("#### Exportar resultados")

                df_enriq = enriquecer_df(st.session_state['df_entidad'], st.session_state['resultados'])

                # Filtrar solo las procesadas
                df_export = df_enriq[df_enriq.index.isin(st.session_state['resultados'].keys())].copy()

                @st.cache_data
                def to_excel_bytes(df):
                    from io import BytesIO
                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as w:
                        df.to_excel(w, index=False, sheet_name='Notas enriquecidas')
                    return buf.getvalue()

                fname = f"analisis_{entidad.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                st.download_button(
                    label=f"⬇️ Descargar Excel enriquecido ({n_done} notas)",
                    data=to_excel_bytes(df_export),
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )

    # ─── TAB 2: RESULTADOS ────────────────────────────────────────────────────
    with tab2:

        resultados = st.session_state.get('resultados', {})
        df_e       = st.session_state.get('df_entidad', pd.DataFrame())

        if not resultados:
            st.info("Aún no hay resultados. Procesa notas en la pestaña anterior.")
        else:
            stats = calcular_estadisticas(resultados, df_e)
            total = stats['total']

            st.markdown(f"### Resumen · {entidad} · {total:,} notas analizadas")

            # KPIs principales
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Notas analizadas", f"{total:,}", BLUE)
            with c2:
                n_real = stats['protagonismo'].get('real', 0)
                metric_card("Protagonismo real", f"{round(n_real/total*100)}%", ORANGE)
            with c3:
                n_pos = stats['sentimiento'].get('positivo', 0)
                metric_card("Sentimiento positivo", f"{round(n_pos/total*100)}%", "#2E6B4F")
            with c4:
                n_voz = sum(1 for r in resultados.values() if r.get('voceria',{}).get('nombre'))
                metric_card("Notas con vocería", f"{round(n_voz/total*100)}%", "#5B8AB0")

            st.markdown("---")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Protagonismo**")
                for k, n in sorted(stats['protagonismo'].items(), key=lambda x: -x[1]):
                    bar_mini(k.capitalize(), n, total, ORANGE)

                st.markdown("<br>**Sentimiento**", unsafe_allow_html=True)
                col_map = {'positivo': '#2E6B4F', 'negativo': '#B04030', 'neutro': GRAY}
                for k, n in sorted(stats['sentimiento'].items(), key=lambda x: -x[1]):
                    bar_mini(k.capitalize(), n, total, col_map.get(k, GRAY))

            with col_right:
                st.markdown("**Top atributos detectados**")
                top_a = stats.get('top_atributos', {})
                max_a = max(top_a.values()) if top_a else 1
                for k, n in sorted(top_a.items(), key=lambda x: -x[1]):
                    bar_mini(k, n, total, BLUE)

                st.markdown("<br>**Top voceros**", unsafe_allow_html=True)
                for k, n in stats.get('top_voceros', {}).items():
                    bar_mini(k, n, total, "#5B8AB0")

            st.markdown("---")

            # Tabla detallada
            st.markdown("**Vista de notas procesadas**")

            # Filtros
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_prot = st.selectbox("Protagonismo", ["Todos", "real", "secundario", "referencial"])
            with fc2:
                f_sent = st.selectbox("Sentimiento", ["Todos", "positivo", "negativo", "neutro"])
            with fc3:
                f_attr = st.selectbox("Atributo", ["Todos"] + atributos)

            df_enriq = enriquecer_df(df_e, resultados)
            df_view  = df_enriq[df_enriq.index.isin(resultados.keys())].copy()

            if f_prot != "Todos":
                df_view = df_view[df_view['ai_protagonismo'] == f_prot]
            if f_sent != "Todos":
                df_view = df_view[df_view['ai_sentimiento'] == f_sent]
            if f_attr != "Todos":
                df_view = df_view[df_view['ai_atributos'].str.contains(f_attr, na=False)]

            cols_show = [col_titulo, 'anio' if 'anio' in df_view.columns else col_titulo,
                         'trimestre' if 'trimestre' in df_view.columns else '',
                         'tier' if 'tier' in df_view.columns else '',
                         'ai_protagonismo', 'ai_atributos', 'ai_vocero_nombre', 'ai_sentimiento']
            cols_show = [c for c in cols_show if c and c in df_view.columns]

            st.markdown(f"*{len(df_view):,} notas con los filtros aplicados*")
            st.dataframe(df_view[cols_show], use_container_width=True, height=400)

    # ─── TAB 3: MANUAL ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
## Manual de uso

### Requisitos
- Python 3.10+
- `pip install streamlit anthropic pandas openpyxl`
- API Key de Anthropic (console.anthropic.com)

### Cómo usar

**1. Configuración (panel lateral)**
- Ingresa tu API Key
- Define la entidad a analizar (ej: APRIMIN, Sonami, Codelco)
- Edita la lista de atributos según el cliente
- Ajusta los nombres de columnas si tu Excel es diferente

**2. Cargar base**
- Sube tu Excel o CSV
- El sistema filtra automáticamente notas sin duplicados
- Luego filtra solo las notas que mencionan la entidad

**3. Procesar**
- Usa **"Muestra de prueba"** primero para validar que el análisis es correcto
- Luego procesa todas las notas
- Si se interrumpe, usa **"Solo notas pendientes"** para continuar
- El progreso se guarda en memoria durante la sesión

**4. Exportar**
- Descarga el Excel enriquecido con 8 columnas nuevas
- Columnas: `ai_atributos`, `ai_protagonismo`, `ai_protagonismo_razon`, `ai_vocero_nombre`, `ai_vocero_cargo`, `ai_vocero_tipo`, `ai_sentimiento`, `ai_sentimiento_razon`

### Costos estimados
| Base | Tiempo aprox. | Costo aprox. |
|------|--------------|-------------|
| 500 notas | 15 min | 1.5 USD |
| 1.000 notas | 30 min | 3 USD |
| 3.434 notas (APRIMIN) | 90 min | 10 USD |

### Para distintos clientes
Cambia solo:
1. El archivo Excel
2. El nombre de entidad
3. La lista de atributos

Todo lo demás es automático.

### Columnas esperadas en el Excel
| Columna | Descripción |
|---------|------------|
| `titulo` | Título de la nota |
| `cuerpo` | Texto completo |
| `anio` | Año de publicación |
| `trimestre` | T1/T2/T3/T4 |
| `tier` | Tier 1/2/3 |
| `medio` | Nombre del medio |
| `es_duplicado` | 0=original, 1=duplicado |

Los nombres de columna son configurables en el panel lateral.
        """)


if __name__ == "__main__":
    main()
