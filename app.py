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
if 'plot_mode' not in st.session_state:
    st.session_state.plot_mode = 'JV'
if 'imported_data' not in st.session_state:
    st.session_state.imported_data = []  # {filename, base, data}
if 'groups' not in st.session_state:
    st.session_state.groups = {}         # name -> {'files': [base], 'active_area': float}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'default_area' not in st.session_state:
    st.session_state.default_area = 0.1  # used to create first group + fallback
if 'voltage_col' not in st.session_state:
    st.session_state.voltage_col = 1  # 1-indexed for user display
if 'current_col' not in st.session_state:
    st.session_state.current_col = 2  # 1-indexed for user display

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

def calculate_iv_params_per_file(area_lookup, visible_files=None, plot_mode='JV'):
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

    # Track PV range if needed
    pv_min, pv_max = None, None

    for idx, d in enumerate(data):
        bar.progress((idx + 1) / len(data))
        txt.text(f'Analyzing {idx + 1}/{len(data)}: {d["filename"]}')

        base, arr = d['base'], d['data']
        v_col = st.session_state.voltage_col - 1  # Convert to 0-indexed
        i_col = st.session_state.current_col - 1
        (bw_v, bw_i), (fw_v, fw_i) = split_forward_backward(arr[v_col], arr[i_col])

        area = float(area_lookup.get(base, st.session_state.default_area))
        if area <= 0:
            area = st.session_state.default_area

        color = colors[idx]
        # Current density [mA/cm²]
        j_bw, j_fw = bw_i / area, fw_i / area

        # Choose what to plot on Y
        if plot_mode == 'PV':
            # Power density [mW/cm²] = -J * V (delivered power positive in 4th quadrant)
            p_bw = -j_bw * bw_v
            p_fw = -j_fw * fw_v
            y_bw, y_fw = p_bw, p_fw
            y_title = 'Power density / mW cm⁻²'
            hover_tmpl = 'V: %{x:.3f} V<br>P: %{y:.3f} mW/cm²<extra></extra>'

            # Update PV global min/max for autoscaling
            cur_min = float(np.nanmin([np.nanmin(p_bw), np.nanmin(p_fw)])) if len(p_bw) and len(p_fw) else None
            cur_max = float(np.nanmax([np.nanmax(p_bw), np.nanmax(p_fw)])) if len(p_bw) and len(p_fw) else None
            if cur_min is not None and cur_max is not None:
                pv_min = cur_min if pv_min is None else min(pv_min, cur_min)
                pv_max = cur_max if pv_max is None else max(pv_max, cur_max)
        else:
            y_bw, y_fw = j_bw, j_fw
            y_title = 'Current / mA cm⁻²'
            hover_tmpl = 'V: %{x:.3f} V<br>J: %{y:.3f} mA/cm²<extra></extra>'

        # Curves
        fig.add_trace(go.Scatter(
            x=bw_v, y=y_bw, mode='lines', name=f'{base} (BW)',
            line=dict(color=color, width=2), legendgroup=base,
            hovertemplate=hover_tmpl
        ))
        fig.add_trace(go.Scatter(
            x=fw_v, y=y_fw, mode='lines', name=f'{base} (FW)',
            line=dict(color=color, width=2, dash='dash'), legendgroup=base,
            hovertemplate=hover_tmpl
        ))

        # IV parameters & markers
        try:
            # IV_Params expects current density [mA/cm²] as positive flowing out of device -> use (-i/area)
            iv_bw = IV_Params(bw_v, bw_i * -1 / area).calc_iv_params(mp_fit_order=9)
            iv_bw['pce'] = iv_bw['isc'] * iv_bw['voc'] * iv_bw['ff'] / 100
            iv_fw = IV_Params(fw_v, fw_i * -1 / area).calc_iv_params(mp_fit_order=9)
            iv_fw['pce'] = iv_fw['isc'] * iv_fw['voc'] * iv_fw['ff'] / 100
            iv_fw['H'] = (iv_bw['pce'] - iv_fw['pce']) / iv_bw['pce'] if iv_bw['pce'] else 0

            if plot_mode == 'JV':
                # Jsc markers (at V=0, J = -isc)
                fig.add_trace(go.Scatter(
                    x=[0, 0], y=[iv_bw['isc'] * -1, iv_fw['isc'] * -1],
                    mode='markers', marker=dict(color=color, size=8),
                    showlegend=False, legendgroup=base, hovertemplate='Jsc<extra></extra>'
                ))
                # Voc markers (at J=0)
                fig.add_trace(go.Scatter(
                    x=[iv_bw['voc'], iv_fw['voc']], y=[0, 0],
                    mode='markers', marker=dict(color=color, size=8),
                    showlegend=False, legendgroup=base, hovertemplate='Voc<extra></extra>'
                ))
                # MPP markers on J–V (Jmp = -imp)
                fig.add_trace(go.Scatter(
                    x=[iv_bw['vmp'], iv_fw['vmp']], y=[iv_bw['imp'] * -1, iv_fw['imp'] * -1],
                    mode='markers', marker=dict(color=color, size=8),
                    showlegend=False, legendgroup=base, hovertemplate='MPP (J)<extra></extra>'
                ))
            else:
                # MPP markers on P–V
                # imp from IV_Params is +J_out, so Pmp = Vmp * imp  (no minus)
                pmp_bw = iv_bw['vmp'] * iv_bw['imp']  # mW/cm²
                pmp_fw = iv_fw['vmp'] * iv_fw['imp']
                fig.add_trace(go.Scatter(
                    x=[iv_bw['vmp'], iv_fw['vmp']], y=[pmp_bw, pmp_fw],
                    mode='markers', marker=dict(color=color, size=8),
                    showlegend=False, legendgroup=base, hovertemplate='MPP (P)<extra></extra>'
                ))
        except Exception as e:
            st.warning(f'⚠️ Error calculating parameters for {base}: {e}')
            iv_bw = {'isc': 0, 'voc': 0, 'imp': 0, 'vmp': 0, 'ff': 0, 'pce': 0}
            iv_fw = {'isc': 0, 'voc': 0, 'imp': 0, 'vmp': 0, 'ff': 0, 'pce': 0, 'H': 0}

        # Series resistance estimates near Voc (both scans)
        try:
            voc_bw = iv_bw['voc']; rng = 0.01
            mask = (bw_v > voc_bw - rng) & (bw_v < voc_bw + rng)
            if np.count_nonzero(mask) > 2:
                slope, *_ = linregress(bw_v[mask], (bw_i[mask] / area))  # dJ/dV
                iv_bw['rs'] = (1 / slope) * 1e3 if slope != 0 else np.inf  # Ω·cm²
            else:
                iv_bw['rs'] = np.nan
        except Exception:
            iv_bw['rs'] = np.nan

        try:
            voc_fw = iv_fw['voc']; rng = 0.01
            mask = (fw_v > voc_fw - rng) & (fw_v < voc_fw + rng)
            if np.count_nonzero(mask) > 2:
                slope, *_ = linregress(fw_v[mask], (fw_i[mask] / area))  # dJ/dV
                iv_fw['rs'] = (1 / slope) * 1e3 if slope != 0 else np.inf  # Ω·cm²
            else:
                iv_fw['rs'] = np.nan
        except Exception:
            iv_fw['rs'] = np.nan

        iv_list.append({
            'bw': iv_bw, 'fw': iv_fw,
            'filename': base,
            'group': next((g for g, meta in st.session_state.groups.items() if base in meta['files']), None),
            'active_area': area
        })

    bar.empty(); txt.empty()

    # Layout: dock legend to the right with its own scroll area
    fig.update_layout(
        xaxis_title='Voltage / V',
        yaxis_title=y_title,
        hovermode='closest',
        height=600,
        template='plotly_white',
        margin=dict(l=60, r=300, t=40, b=60),
        legend=dict(
            orientation='v',
            yanchor='top', y=1.0,
            xanchor='left', x=1.02,
            itemclick='toggle', itemdoubleclick='toggleothers',
            itemsizing='constant',
            font=dict(size=10),
        ),
        uirevision="keep"
    )
    # Crosshairs
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Axis ranges
    fig.update_xaxes(range=[-0.1, 1.25])
    if plot_mode == 'JV':
        fig.update_yaxes(range=[-25, 5])
    else:
        # Autoscale around data, bias to show zero
        if pv_min is None or pv_max is None or not np.isfinite(pv_min) or not np.isfinite(pv_max):
            fig.update_yaxes(autorange=True)
        else:
            ymin = min(0.0, pv_min * 1.05)
            ymax = pv_max * 1.05 if pv_max > 0 else 1.0
            # Avoid degenerate range
            if abs(ymax - ymin) < 1e-6:
                ymax = ymin + 1.0
            fig.update_yaxes(range=[ymin, ymax])

    # Build DataFrame, then rename isc -> jsc and imp -> jmp everywhere visible
    df = pd.json_normalize(iv_list).round(3)
    rename_map = {}
    for col in df.columns:
        if col.endswith('.isc') or col == 'isc':
            rename_map[col] = col.replace('isc', 'jsc')
        if col.endswith('.imp') or col == 'imp':
            rename_map[col] = col.replace('imp', 'jmp')
        if col == 'fw.H':
            rename_map[col] = 'H'
    df = df.rename(columns=rename_map)
    st.session_state.df = df
    return fig, df


def create_statistics_plot(selected_params, num_columns, selected_groups=None):
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

    # Filter groups if specified
    groups_to_plot = st.session_state.groups
    if selected_groups:
        groups_to_plot = {k: v for k, v in st.session_state.groups.items() if k in selected_groups}
    
    if not groups_to_plot:
        st.warning('No groups selected.')
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

    # Create consistent color mapping for ALL groups (so colors stay consistent)
    group_names = list(st.session_state.groups.keys())
    group_colors = get_color_palette(len(group_names))
    color_map = dict(zip(group_names, group_colors))

    for idx, param_name in enumerate(selected_params):
        r = idx // cols + 1
        c = idx % cols + 1

        # For each group in groups_to_plot, gather values and draw ONE box trace with overlaid points
        for gname, meta in groups_to_plot.items():
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

            # Box+whisker with scatter using consistent group color
            fig.add_trace(
                go.Box(
                    y=vals,
                    x=[gname] * len(vals),
                    name=gname,
                    boxpoints='all',
                    jitter=0.35,
                    pointpos=0,
                    marker=dict(size=5, opacity=0.6, color=color_map[gname]),
                    line=dict(color=color_map[gname]),
                    fillcolor=color_map[gname].replace('rgb', 'rgba').replace(')', ',0.25)'),
                    showlegend=False
                ),
                row=r, col=c
            )

        fig.update_yaxes(title_text=param_name, row=r, col=c)
        fig.update_xaxes(categoryorder='array',
                         categoryarray=list(groups_to_plot.keys()),
                         row=r, col=c)

    fig.update_layout(
        height=max(420, 360 * rows),
        template='plotly_white',
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    return fig


# ---------- UI ----------
st.title("JV Curve Analysis")
st.caption(
    "Import IV `.txt` files containing two columns — voltage (V) and current (mA), for two scans — BW and FW, in that order. "
    "Assign groups and active areas, then calculate JV parameters under standard "
    "test conditions (100 mW/cm², 1 sun) using NREL iv_params https://github.com/NREL/iv_params."
)
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

# Upload (auto-import). The small "X" now also removes from analysis
upload_col, settings_col = st.columns([20, 1])
with upload_col:
    uploaded_files = st.file_uploader(
        "📁 Upload IV data files (auto-imports)",
        type=['txt'], accept_multiple_files=True, key='file_uploader'
    )
with settings_col:
    st.write("")  # Spacing to align with uploader
    with st.popover("⚙️", help="Column settings"):
        st.markdown("**Column Assignment**")
        st.caption("Specify which columns contain V and I data")
        st.session_state.voltage_col = st.number_input(
            "Voltage column",
            min_value=1, value=st.session_state.voltage_col,
            step=1, key='voltage_col_input',
            help="Column number for voltage (1 = first column)"
        )
        st.session_state.current_col = st.number_input(
            "Current column",
            min_value=1, value=st.session_state.current_col,
            step=1, key='current_col_input',
            help="Column number for current (2 = second column)"
        )
if uploaded_files:
    import_data(uploaded_files)
sync_with_uploader(uploaded_files)

# File count
if st.session_state.imported_data:
    st.info(f'📊 {len(st.session_state.imported_data)} file(s) loaded')

# -------- Group Management (expander) --------
with st.expander("Group management"):
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
                # Use a dynamic key that includes the group name to reset state when group changes
                assigned = st.multiselect('Assign files to this group',
                                          all_bases,
                                          default=st.session_state.groups[sel_group]['files'],
                                          key=f'assign_files_box_{sel_group}')
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
tab1, tab2, tab3 = st.tabs(['☀️ JV Analysis', '📊 Statistics', '📥 Export'])

with tab1:
    if st.session_state.imported_data:
        all_files_with_ext = [d['filename'] for d in st.session_state.imported_data]
        st.markdown("**Select files to analyze (uses the active area for each file's group):**")
        selected_for_analysis = st.multiselect('Files', all_files_with_ext, default=all_files_with_ext, label_visibility='collapsed')
        
        plot_choice = st.radio(
            "Plot type",
            ["J–V (current–voltage)", "P–V (power–voltage)"],
            horizontal=True,
            key='plot_type_radio'
        )
        new_plot_mode = 'JV' if plot_choice.startswith("J–V") else 'PV'
        
        # Detect mode change and auto-recalculate
        if new_plot_mode != st.session_state.plot_mode:
            st.session_state.plot_mode = new_plot_mode
            if selected_for_analysis:
                fig, df = calculate_iv_params_per_file(_build_area_lookup(), selected_for_analysis, plot_mode=new_plot_mode)
                if fig is not None:
                    st.session_state.jv_fig = fig
                    st.session_state.df = df
        
        # Update the Analyze button to use st.session_state.plot_mode:
        if st.button('Analyze', type='primary'):
            with st.spinner('Calculating IV parameters...'):
                fig, df = calculate_iv_params_per_file(_build_area_lookup(), selected_for_analysis, plot_mode=st.session_state.plot_mode)
                if fig is not None and df is not None:
                    # store separately per mode if you like
                    st.session_state.jv_fig = fig
                    st.session_state.df = df
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('Results Table')
                    st.dataframe(df, use_container_width=True, height=400)
        elif st.session_state.df is not None:
            st.info('Showing previous results. Click "Analyze" to recalculate.')
            if 'jv_fig' in st.session_state and st.session_state.jv_fig:
                st.plotly_chart(st.session_state.jv_fig, use_container_width=True)
            st.markdown('Results Table')
            st.dataframe(st.session_state.df, use_container_width=True, height=400)
    else:
        st.info('Upload data files to begin analysis.')

with tab2:
    if st.session_state.df is not None and st.session_state.groups:
        # Group filter
        all_groups = list(st.session_state.groups.keys())
        selected_groups = st.multiselect(
            '**Select groups to display:**',
            all_groups,
            default=all_groups,
            key='stats_group_filter'
        )
        # Controls row
        c1, c2 = st.columns([3, 1])
        with c1:
            params = [c for c in st.session_state.df.columns
                      if c not in ('filename', 'group', 'active_area')]
            picked = st.multiselect(
                '**Select parameters to plot by group:**',
                params,
                default=params[:min(4, len(params))]
            )
        with c2:
            ncols = st.number_input('Columns', min_value=1, max_value=5, value=2)
            plot_stats = st.button('📊 Plot Statistics', type='primary', use_container_width=True)

        # full-width placeholder for the chart (AFTER the controls)
        stats_plot_area = st.empty()

        # Create or show plot in the FULL-WIDTH placeholder
        if plot_stats and picked:
            with st.spinner('Creating statistics plots...'):
                fig = create_statistics_plot(picked, ncols, selected_groups)
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
        st.info('Add a group in "Group management" to view statistics.')
    else:
        st.info('Run JV Analysis first.')


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
            # Add statistics summary export
        if st.session_state.groups:
            st.markdown('---')
            st.markdown('### 📊 Statistics Summary')
            
            # Generate statistics table grouped by group
            stats_list = []
            for gname, meta in st.session_state.groups.items():
                files = set(meta['files'])
                group_data = st.session_state.df[st.session_state.df['filename'].isin(files)]
                
                if not group_data.empty:
                    # Calculate mean and std for numeric columns
                    numeric_cols = group_data.select_dtypes(include=[np.number]).columns
                    stats_row = {'Group': gname, 'N': len(group_data)}
                    for col in numeric_cols:
                        if col != 'active_area':
                            stats_row[f'{col}_mean'] = group_data[col].mean()
                            stats_row[f'{col}_std'] = group_data[col].std()
                    stats_list.append(stats_row)
            
            if stats_list:
                stats_df = pd.DataFrame(stats_list).round(3)
                st.dataframe(stats_df, use_container_width=True)
                
                c1, c2, _ = st.columns([1, 1, 2])
                with c1:
                    st.download_button('📥 Download Statistics CSV',
                                       data=stats_df.to_csv(index=False),
                                       file_name='iv_statistics.csv', mime='text/csv',
                                       use_container_width=True)
                with c2:
                    buf2 = BytesIO()
                    with pd.ExcelWriter(buf2, engine='openpyxl') as w:
                        stats_df.to_excel(w, index=False, sheet_name='Statistics')
                    buf2.seek(0)
                    st.download_button('📥 Download Statistics Excel', data=buf2,
                                       file_name='iv_statistics.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                       use_container_width=True)
    else:
        st.info('Run JV Analysis to generate exportable results.')

