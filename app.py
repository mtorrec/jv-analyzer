import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import linregress
import seaborn as sns
from io import BytesIO
import os

# Import NREL's IV_Params
try:
    from iv_params import IV_Params
except ImportError:
    st.error("Please install NREL's iv_params: pip install git+https://github.com/NREL/iv_params.git")
    st.stop()

st.set_page_config(page_title="JV Curve Analysis", layout="wide", page_icon="⚡")

# ---------- Session state ----------
if 'imported_data' not in st.session_state:
    st.session_state.imported_data = []  # {filename, base, data}
if 'groups' not in st.session_state:
    st.session_state.groups = {}         # name -> {'files': [base], 'active_area': float}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'default_area' not in st.session_state:
    st.session_state.default_area = 0.1  # used to create first group + fallback

# ---------- Helpers ----------
def is_numeric_line(line: str) -> bool:
    try:
        parts = line.split()
        if not parts:
            return False
        _ = [float(x) for x in parts]
        return True
    except ValueError:
        return False

@st.cache_data(show_spinner=False)
def process_file_data(file_content: str, file_name: str):
    lines = file_content.split('\n')
    numeric = [ln for ln in lines if ln.strip() and is_numeric_line(ln)]
    if not numeric:
        return None
    data = [[float(x) for x in ln.split()] for ln in numeric]
    return np.array(data).T

def get_color_palette(n):
    colors = sns.color_palette("husl", n)
    return [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]

def split_forward_backward(v, i):
    k = np.argmin(v)
    return (v[:k+1], i[:k+1]), (v[k:], i[k:])

def _ensure_default_group():
    if not st.session_state.groups and st.session_state.imported_data:
        all_bases = [d['base'] for d in st.session_state.imported_data]
        st.session_state.groups['Group 1'] = {
            'files': all_bases,
            'active_area': float(st.session_state.default_area)
        }

def _build_area_lookup():
    lookup = {}
    for gname, meta in st.session_state.groups.items():
        area = float(meta.get('active_area', st.session_state.default_area))
        for b in meta['files']:
            lookup[b] = area
    return lookup

def _auto_calculate():
    if not st.session_state.imported_data:
        st.session_state.df = None
        return
    calculate_iv_params_per_file(_build_area_lookup())

def import_data(uploaded_files):
    """Import any new files in the uploader; skip duplicates."""
    if not uploaded_files:
        return 0
    existing = set(d['filename'] for d in st.session_state.imported_data)
    new_count = 0

    progress = st.progress(0.0)
    status = st.empty()
    total = len(uploaded_files)

    for idx, file in enumerate(uploaded_files):
        progress.progress((idx + 1) / total)
        status.text(f'Processing {idx + 1}/{total}: {file.name}')
        if file.name in existing:
            continue
        try:
            content = file.getvalue().decode('utf-8')
            arr = process_file_data(content, file.name)
            if arr is None:
                st.warning(f'⚠️ No valid numeric data in {file.name}')
                continue
            base = os.path.splitext(os.path.basename(file.name))[0]
            st.session_state.imported_data.append({'filename': file.name, 'base': base, 'data': arr})
            existing.add(file.name)
            new_count += 1
        except Exception as e:
            st.error(f'❌ Error importing {file.name}: {e}')

    progress.empty(); status.empty()

    if new_count:
        st.success(f'✅ Imported {new_count} new file(s). Total: {len(st.session_state.imported_data)}')
        _ensure_default_group()
        _auto_calculate()
    return new_count

def sync_with_uploader(uploaded_files):
    """If user clicks the small 'X' in file_uploader, remove the file from analysis as well."""
    present = set(f.name for f in uploaded_files) if uploaded_files else set()
    to_remove = [d for d in st.session_state.imported_data if d['filename'] not in present]
    if not to_remove:
        return
    for d in to_remove:
        base = d['base']
        st.session_state.imported_data = [x for x in st.session_state.imported_data if x['filename'] != d['filename']]
        for g in list(st.session_state.groups.keys()):
            st.session_state.groups[g]['files'] = [b for b in st.session_state.groups[g]['files'] if b != base]
    _auto_calculate()

def calculate_iv_params_per_file(area_lookup, visible_files=None):
    if not st.session_state.imported_data:
        st.warning('No data imported yet.')
        return None, None

    data = st.session_state.imported_data
    if visible_files is not None:
        keep = set(visible_files)
        data = [d for d in data if d['filename'] in keep]
    if not data:
        st.warning('No files selected for analysis.')
        return None, None

    fig = go.Figure()
    iv_list = []
    colors = get_color_palette(len(data))
    bar = st.progress(0.0)
    txt = st.empty()

    for idx, d in enumerate(data):
        bar.progress((idx + 1) / len(data))
        txt.text(f'Analyzing {idx + 1}/{len(data)}: {d["filename"]}')

        base, arr = d['base'], d['data']
        (bw_v, bw_i), (fw_v, fw_i) = split_forward_backward(arr[0], arr[1])

        area = float(area_lookup.get(base, st.session_state.default_area))
        if area <= 0:
            area = st.session_state.default_area

        color = colors[idx]
        j_bw, j_fw = bw_i / area, fw_i / area

        # curves
        fig.add_trace(go.Scatter(
            x=bw_v, y=j_bw, mode='lines', name=f'{base} (BW)',
            line=dict(color=color, width=2), legendgroup=base,
            hovertemplate='V: %{x:.3f} V<br>J: %{y:.3f} mA/cm²<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=fw_v, y=j_fw, mode='lines', name=f'{base} (FW)',
            line=dict(color=color, width=2, dash='dash'), legendgroup=base,
            hovertemplate='V: %{x:.3f} V<br>J: %{y:.3f} mA/cm²<extra></extra>'
        ))

        try:
            iv_bw = IV_Params(bw_v, bw_i * -1 / area).calc_iv_params()
            iv_bw['pce'] = iv_bw['isc'] * iv_bw['voc'] * iv_bw['ff'] / 100
            iv_fw = IV_Params(fw_v, fw_i * -1 / area).calc_iv_params()
            iv_fw['pce'] = iv_fw['isc'] * iv_fw['voc'] * iv_fw['ff'] / 100
            iv_fw['H'] = (iv_bw['pce'] - iv_fw['pce']) / iv_bw['pce'] if iv_bw['pce'] else 0

            # key points (label as Jsc/Voc/MPP)
            fig.add_trace(go.Scatter(
                x=[0, 0], y=[iv_bw['isc'] * -1, iv_fw['isc'] * -1],
                mode='markers', marker=dict(color=color, size=8),
                showlegend=False, legendgroup=base, hovertemplate='Jsc<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[iv_bw['voc'], iv_fw['voc']], y=[0, 0],
                mode='markers', marker=dict(color=color, size=8),
                showlegend=False, legendgroup=base, hovertemplate='Voc<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[iv_bw['vmp'], iv_fw['vmp']], y=[iv_bw['imp'] * -1, iv_fw['imp'] * -1],
                mode='markers', marker=dict(color=color, size=8),
                showlegend=False, legendgroup=base, hovertemplate='MPP<extra></extra>'
            ))
        except Exception as e:
            st.warning(f'⚠️ Error calculating parameters for {base}: {e}')
            iv_bw = {'isc': 0, 'voc': 0, 'imp': 0, 'vmp': 0, 'ff': 0, 'pce': 0}
            iv_fw = {'isc': 0, 'voc': 0, 'imp': 0, 'vmp': 0, 'ff': 0, 'pce': 0, 'H': 0}

        # Rs (BW)
        try:
            voc_bw = iv_bw['voc']; rng = 0.01
            mask = (bw_v > voc_bw - rng) & (bw_v < voc_bw + rng)
            if np.count_nonzero(mask) > 2:
                slope, *_ = linregress(bw_v[mask], (bw_i[mask] / area))
                iv_bw['rs'] = (1 / slope) * 1e3 if slope != 0 else np.inf
            else:
                iv_bw['rs'] = np.nan
        except Exception:
            iv_bw['rs'] = np.nan

        iv_list.append({
            'bw': iv_bw, 'fw': iv_fw,
            'filename': base,
            'group': next((g for g, meta in st.session_state.groups.items() if base in meta['files']), None),
            'active_area_used': area
        })

    bar.empty(); txt.empty()

    # Layout: dock legend to the right with its own scroll area
    fig.update_layout(
        xaxis_title='Voltage / V',
        yaxis_title='Current / mA cm⁻²',
        hovermode='closest',
        height=600,
        template='plotly_white',
        margin=dict(l=60, r=300, t=40, b=60),  # extra right margin for a tall legend
        legend=dict(
            orientation='v',
            yanchor='top', y=1.0,
            xanchor='left', x=1.02,
            itemclick='toggle', itemdoubleclick='toggleothers',
            itemsizing='constant',
            font=dict(size=10),
            # When the legend exceeds the plot height, Plotly provides a scrollbar automatically
        ),
        uirevision="keep"  # keep legend state while updating
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_xaxes(range=[-0.1, 1.25]); fig.update_yaxes(range=[-25, 5])

    # Build DataFrame, then rename isc -> jsc everywhere visible
    df = pd.json_normalize(iv_list).round(3)
    rename_map = {}
    for col in df.columns:
        if col.endswith('.isc') or col == 'isc':
            rename_map[col] = col.replace('isc', 'jsc')
    df = df.rename(columns=rename_map)
    st.session_state.df = df
    return fig, df

def create_statistics_plot(selected_params, num_columns):
    if not st.session_state.groups:
        st.warning('No groups defined.')
        return None
    if not selected_params:
        st.warning('No parameters selected.')
        return None
    df = st.session_state.df
    if df is None or df.empty:
        st.warning('No analysis results available.')
        return None

    # full-width layout across available columns
    num_params = len(selected_params)
    cols = max(1, min(num_columns, num_params))
    rows = num_params // cols + (num_params % cols > 0)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=selected_params,
        vertical_spacing=0.08, horizontal_spacing=0.06
    )

    for idx, param_name in enumerate(selected_params):
        r = idx // cols + 1
        c = idx % cols + 1

        # For each group, gather values and draw ONE box trace with overlaid points
        for gname, meta in st.session_state.groups.items():
            files = set(meta['files'])
            vals = []
            for _, row in df.iterrows():
                if row.get('filename') in files and param_name in df.columns:
                    try:
                        v = float(row[param_name])
                        if not np.isnan(v) and not np.isinf(v):
                            vals.append(v)
                    except Exception:
                        pass
            if not vals:
                continue

            # Box+whisker with scatter (no legend)
            fig.add_trace(
                go.Box(
                    y=vals,
                    x=[gname] * len(vals),     # category label shown on x-axis
                    name=gname,                # used for x tick text (legend hidden)
                    boxpoints='all',           # show points on top of box
                    jitter=0.35,
                    pointpos=0,                # center points
                    marker=dict(size=5, opacity=0.6),
                    showlegend=False
                ),
                row=r, col=c
            )

        fig.update_yaxes(title_text=param_name, row=r, col=c)
        fig.update_xaxes(categoryorder='array',
                         categoryarray=list(st.session_state.groups.keys()),
                         row=r, col=c)

    fig.update_layout(
        height=max(420, 360 * rows),
        template='plotly_white',
        showlegend=False,          # <-- hide legend entirely
        margin=dict(l=60, r=30, t=60, b=60)
    )
    return fig


# ---------- UI ----------
st.title('JV Curve Analysis')
st.markdown('---')

# Default active area (kept at top, used for first group + fallback)
top1, = st.columns([1])
with top1:
    st.session_state.default_area = st.number_input(
        'Default active area [cm²]',
        min_value=0.0, value=st.session_state.default_area,
        step=0.01, format="%.3f",
        help="Used to initialize the first auto-created group and as a fallback for files without a group."
    )

# Upload (auto-import). The small “X” now also removes from analysis
uploaded_files = st.file_uploader(
    "📁 Upload IV data files (auto-imports)",
    type=['txt'], accept_multiple_files=True, key='file_uploader'
)
if uploaded_files:
    import_data(uploaded_files)
sync_with_uploader(uploaded_files)

# File count
if st.session_state.imported_data:
    st.info(f'📊 {len(st.session_state.imported_data)} file(s) loaded')

# -------- Group Management (expander) --------
with st.expander("👥 Group Management (per-group active area)"):
    _ensure_default_group()

    g1, g2, g3 = st.columns([2, 1, 1])
    with g1:
        new_group = st.text_input('New group name', key='new_group_input', placeholder='e.g., Batch A')
    with g2:
        new_group_area = st.number_input('Active area [cm²] for new group',
                                         min_value=0.0, value=st.session_state.default_area,
                                         step=0.01, format="%.3f", key='new_group_area')
    with g3:
        if st.button('➕ Add Group', use_container_width=True):
            if not new_group:
                st.warning('Please enter a group name.')
            elif new_group in st.session_state.groups:
                st.warning('Group already exists.')
            else:
                st.session_state.groups[new_group] = {'files': [], 'active_area': float(new_group_area)}
                st.success(f'✅ Group "{new_group}" created (area={new_group_area:.3f} cm²)')

    if st.session_state.groups:
        left, right = st.columns([1.5, 2.5])
        with left:
            sel_group = st.selectbox('Select group', list(st.session_state.groups.keys()), key='group_selector')
            if sel_group:
                ga = st.number_input('Active area [cm²] for this group',
                                     min_value=0.0,
                                     value=float(st.session_state.groups[sel_group]['active_area']),
                                     step=0.01, format="%.3f", key='grp_area_editor')
                if st.button('💾 Save group area', use_container_width=True, key='save_group_area'):
                    st.session_state.groups[sel_group]['active_area'] = float(ga)
                    _auto_calculate()
                    st.success('Saved.')
        with right:
            if sel_group:
                all_bases = [d['base'] for d in st.session_state.imported_data]
                assigned = st.multiselect('Assign files to this group',
                                          all_bases,
                                          default=st.session_state.groups[sel_group]['files'],
                                          key='assign_files_box')
                if st.button('➕ Update assignments', use_container_width=True, key='update_assign'):
                    st.session_state.groups[sel_group]['files'] = list(dict.fromkeys(assigned))
                    _auto_calculate()
                    st.success('Assignments updated.')
                if st.button('🗑️ Delete group', use_container_width=True, key='delete_group'):
                    del st.session_state.groups[sel_group]
                    _auto_calculate()
                    st.rerun()

        with st.expander('📋 Groups summary'):
            for gname, meta in st.session_state.groups.items():
                st.markdown(f"- **{gname}** — area: `{meta['active_area']:.3f} cm²` — {len(meta['files'])} file(s)")
                if meta['files']:
                    st.caption(", ".join(meta['files']))

st.markdown('---')

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(['📊 JV Analysis', '📈 Statistics', '📥 Export'])

with tab1:
    if st.session_state.imported_data:
        all_files_with_ext = [d['filename'] for d in st.session_state.imported_data]
        st.markdown('**Select files to analyze (uses each file’s group active area):**')
        selected_for_analysis = st.multiselect('Files', all_files_with_ext, default=all_files_with_ext, label_visibility='collapsed')

        if st.button('Analyze', type='primary'):
            with st.spinner('Calculating IV parameters...'):
                fig, df = calculate_iv_params_per_file(_build_area_lookup(), selected_for_analysis)
                if fig is not None and df is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('Results Table')
                    st.dataframe(df, use_container_width=True, height=400)
        elif st.session_state.df is not None:
            st.info('Showing previous results. Click "Analyze" to recalculate.')
    else:
        st.info('Upload data files to begin analysis.')

with tab2:
    if st.session_state.df is not None and st.session_state.groups:
        # full-width placeholder for the chart (must be created BEFORE the button)
        stats_plot_area = st.empty()

        # Controls row
        c1, c2 = st.columns([3, 1])
        with c1:
            params = [c for c in st.session_state.df.columns
                      if c not in ('filename', 'group', 'active_area_used')]
            picked = st.multiselect(
                '**Select parameters to plot by group:**',
                params,
                default=params[:min(4, len(params))]
            )
        with c2:
            ncols = st.number_input('Columns', min_value=1, max_value=5, value=2)
            plot_stats = st.button('📊 Plot Statistics', type='primary', use_container_width=True)

        # Create or show plot in the FULL-WIDTH placeholder
        if plot_stats and picked:
            with st.spinner('Creating statistics plots...'):
                fig = create_statistics_plot(picked, ncols)
                if fig:
                    st.session_state.stats_fig = fig   # persist across reruns
                    stats_plot_area.plotly_chart(fig, use_container_width=True)
        else:
            # Show previous stats if available
            if 'stats_fig' in st.session_state and st.session_state.stats_fig:
                stats_plot_area.plotly_chart(st.session_state.stats_fig, use_container_width=True)
            else:
                st.info('Pick parameters and click "Plot Statistics".')

    elif not st.session_state.groups:
        st.info('👈 Add a group in “Group Management” to view statistics.')
    else:
        st.info('👈 Run JV Analysis first.')


with tab3:
    if st.session_state.df is not None:
        st.markdown('### 📊 Results Table')
        st.dataframe(st.session_state.df, use_container_width=True)
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.download_button('📥 Download CSV',
                               data=st.session_state.df.to_csv(index=False),
                               file_name='iv_params.csv', mime='text/csv',
                               use_container_width=True)
        with c2:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as w:
                st.session_state.df.to_excel(w, index=False, sheet_name='IV_Parameters')
            buf.seek(0)
            st.download_button('📥 Download Excel', data=buf, file_name='iv_params.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               use_container_width=True)
    else:
        st.info('👈 Run JV Analysis to generate exportable results.')
