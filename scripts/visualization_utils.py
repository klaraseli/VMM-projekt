"""
visualization_utils.py - Generování čistých vektorových SVG grafů bez externích závislostí.
Podporuje:
- Sloupcové grafy (Bar charts, horizontal benchmark charts)
- Vícepanelové porovnání ladění hyperparametrů (Two-panel tuning comparison)
- Bodové grafy (Scatter plots pro PCA projekce)
- Spojnicové grafy (Line plots pro scree plot a kumulativní rozptyl)
- Teplotní mapy a rozšířené matice záměn (Confusion Matrix s metrikami po řádcích a sloupcích)
Plně ošetřuje XML escapování (& -> &amp;, < -> &lt;, > -> &gt;) pro správné zobrazení ve všech prohlížečích.
"""

import math
import html
from typing import List, Dict, Tuple, Optional


def esc(text: any) -> str:
    """Ošetří speciální znaky pro korektní XML/SVG syntaxi."""
    return html.escape(str(text), quote=True)


def save_svg(content: str, filepath: str):
    """Uloží SVG kód do souboru."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def get_nice_ticks(min_val: float, max_val: float, n_ticks: int = 5) -> Tuple[float, float, List[float]]:
    """Vypočte zaokrouhlené hezké meze a značky osy (např. 0, 0.2, 0.4... nebo 0, 20, 40...)."""
    if min_val == max_val:
        max_val += 1.0
    span = max_val - min_val
    raw_step = span / n_ticks
    power = 10 ** math.floor(math.log10(raw_step))
    fraction = raw_step / power
    if fraction < 1.5:
        step = 1.0 * power
    elif fraction < 3.0:
        step = 2.0 * power
    elif fraction < 7.0:
        step = 5.0 * power
    else:
        step = 10.0 * power

    nice_min = math.floor(min_val / step) * step
    nice_max = math.ceil(max_val / step) * step

    ticks = []
    curr = nice_min
    while curr <= nice_max + step * 0.001:
        ticks.append(round(curr, 6))
        curr += step
    return nice_min, nice_max, ticks


def generate_horizontal_bar_chart(
    labels: List[str],
    values: List[float],
    stds: Optional[List[float]] = None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "",
    width: int = 860,
    bar_height: int = 24,
    bar_color: str = "#2563eb",
    baseline: Optional[float] = None,
    baseline_label: str = "",
    is_percentage: bool = False,
    highlight_max: bool = False,
    xlim: Optional[Tuple[float, float]] = None
) -> str:
    """
    Vytvoří vodorovný sloupcový graf (horizontal bar chart).
    - Umožňuje plné názvy modelů bez ořezávání (levý okraj 245 px).
    - Podporuje chybové úsečky (error bars +/- std).
    - Podporuje zobrazení referenční hodnoty (baseline: např. R2 = 0 nebo 50 % u klasifikace).
    - Zaokrouhlené značky na ose X (nice ticks) a volitelné procentuální formátování.
    """
    n = len(labels)
    if n == 0:
        return ""

    margin_top = 85 if subtitle else 65
    margin_bottom = 55
    margin_left = 250
    margin_right = 110

    plot_h = max(240, n * (bar_height + 12))
    height = margin_top + plot_h + margin_bottom
    plot_w = width - margin_left - margin_right

    # Určení mezí osy X
    all_low = [v - (s if stds else 0.0) for v, s in zip(values, stds or [0.0] * n)]
    all_high = [v + (s if stds else 0.0) for v, s in zip(values, stds or [0.0] * n)]
    if baseline is not None:
        all_low.append(baseline)
        all_high.append(baseline)

    data_min = min(0.0, min(all_low))
    data_max = max(all_high)

    if xlim is not None:
        x_min, x_max = xlim
        _, _, ticks = get_nice_ticks(x_min, x_max, n_ticks=5)
        ticks = [t for t in ticks if x_min <= t <= x_max + 1e-6]
    else:
        x_min, x_max, ticks = get_nice_ticks(data_min, data_max, n_ticks=5)

    def val_to_x(v):
        span = (x_max - x_min) if x_max != x_min else 1.0
        return margin_left + ((v - x_min) / span) * plot_w

    zero_x = val_to_x(0.0)

    # Grid & Ticks
    grid_lines = []
    tick_labels = []
    for t in ticks:
        tx = val_to_x(t)
        grid_lines.append(
            f'<line x1="{tx:.1f}" y1="{margin_top}" x2="{tx:.1f}" y2="{margin_top + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>'
        )
        t_str = f"{t:.1f} %" if is_percentage else (f"{t:.2f}" if abs(t) < 1 else f"{t:.1f}")
        tick_labels.append(
            f'<text x="{tx:.1f}" y="{margin_top + plot_h + 18}" text-anchor="middle" '
            f'font-family="system-ui, sans-serif" font-size="11" fill="#64748b">{t_str}</text>'
        )

    # Vykreslení sloupců
    bars = []
    labels_svg = []
    error_bars = []
    value_texts = []

    for i, lab in enumerate(labels):
        val = values[i]
        std = stds[i] if stds is not None else 0.0
        y = margin_top + i * (bar_height + 12) + 6

        x_val = val_to_x(val)
        if val >= 0:
            w = max(1.0, x_val - zero_x)
            x_start = zero_x
        else:
            w = max(1.0, zero_x - x_val)
            x_start = x_val

        color = bar_color
        bars.append(
            f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_height}" '
            f'rx="3" fill="{color}" opacity="0.88"/>'
        )

        # Chybové úsečky (+/- std)
        if stds is not None and std > 0:
            x_low = val_to_x(val - std)
            x_high = val_to_x(val + std)
            mid_y = y + bar_height / 2
            error_bars.append(
                f'<line x1="{x_low:.1f}" y1="{mid_y:.1f}" x2="{x_high:.1f}" y2="{mid_y:.1f}" '
                f'stroke="#1e293b" stroke-width="1.8"/>'
            )
            error_bars.append(
                f'<line x1="{x_low:.1f}" y1="{mid_y - 4:.1f}" x2="{x_low:.1f}" y2="{mid_y + 4:.1f}" '
                f'stroke="#1e293b" stroke-width="1.8"/>'
            )
            error_bars.append(
                f'<line x1="{x_high:.1f}" y1="{mid_y - 4:.1f}" x2="{x_high:.1f}" y2="{mid_y + 4:.1f}" '
                f'stroke="#1e293b" stroke-width="1.8"/>'
            )
            label_x = max(x_val, x_high) + 8
        else:
            label_x = x_val + 8

        # Textová hodnota
        if is_percentage:
            txt = f"{val:.1f} \u00b1 {std:.1f} %" if (stds is not None and std > 0) else f"{val:.1f} %"
        else:
            txt = f"{val:.3f} \u00b1 {std:.3f}" if (stds is not None and std > 0) else f"{val:.3f}"

        value_texts.append(
            f'<text x="{label_x:.1f}" y="{y + bar_height/2 + 4:.1f}" text-anchor="start" '
            f'font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#334155">{txt}</text>'
        )

        # Plný název modelu
        labels_svg.append(
            f'<text x="{margin_left - 12}" y="{y + bar_height/2 + 4:.1f}" text-anchor="end" '
            f'font-family="system-ui, sans-serif" font-size="11" fill="#1e293b">{esc(lab)}</text>'
        )

    # Baseline linka
    baseline_svg = []
    if baseline is not None:
        bx = val_to_x(baseline)
        baseline_svg.append(
            f'<line x1="{bx:.1f}" y1="{margin_top}" x2="{bx:.1f}" y2="{margin_top + plot_h}" '
            f'stroke="#ef4444" stroke-width="1.8" stroke-dasharray="4 3"/>'
        )
        b_label_text = f"Baseline ({baseline_label})" if baseline_label else f"Baseline ({baseline})"
        baseline_svg.append(
            f'<text x="{bx:.1f}" y="{margin_top - 8}" text-anchor="middle" '
            f'font-family="system-ui, sans-serif" font-size="10" font-weight="bold" fill="#b91c1c">{esc(b_label_text)}</text>'
        )

    # Subtitle
    sub_tag = (
        f'<text x="{width / 2}" y="48" text-anchor="middle" font-size="12" fill="#64748b">{esc(subtitle)}</text>'
        if subtitle else ""
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  {sub_tag}
  <text x="{margin_left + plot_w / 2}" y="{height - 14}" text-anchor="middle" font-size="12" fill="#64748b">{esc(xlabel)}</text>

  <!-- Grid -->
  {''.join(grid_lines)}

  <!-- Nulová osa -->
  <line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Sloupce -->
  {''.join(bars)}

  <!-- Chybové úsečky -->
  {''.join(error_bars)}

  <!-- Baseline -->
  {''.join(baseline_svg)}

  <!-- Hodnoty a popisky -->
  {''.join(value_texts)}
  {''.join(labels_svg)}
  {''.join(tick_labels)}
</svg>"""
    return svg


def generate_tuning_two_panel_chart(
    panel_A_models: List[str],
    panel_A_default: List[float],
    panel_A_tuned: List[float],
    panel_B_models: List[str],
    panel_B_default: List[float],
    panel_B_tuned: List[float],
    title: str,
    subtitle: str = "Nestranné srovnání: Výchozí (5-Fold CV) vs. Laděný model (Nested 5x5 CV). Scénáře mají samostatné měřítko.",
    metric_label: str = "Koeficient determinace (R²)",
    width: int = 940,
    height: int = 440,
    is_percentage: bool = False,
    panel_A_title: str = "Scénář A: Průběžná predikce (s G1 a G2)",
    panel_B_title: str = "Scénář B: Včasné varování (bez G1 a G2)",
    palette: Optional[List[str]] = None
) -> str:
    """
    Vytvoří dvoupanelový graf pro ladění hyperparametrů.
    Odděluje Scénář A a Scénář B do vlastních panelů s vlastním měřítkem osy Y,
    takže nedochází ke zkreslení mezi průběžnou predikcí (R2 ~ 0.84) a včasným varováním (R2 ~ 0.26).
    """
    colors = palette or ["#94a3b8", "#2563eb"]
    margin_top = 80
    margin_bottom = 60

    panel_w = 400
    pA_x = 65
    pB_x = 515
    plot_h = height - margin_top - margin_bottom

    def render_panel(px_start, models, def_vals, tuned_vals, p_title):
        n_m = len(models)
        all_v = def_vals + tuned_vals
        min_v = min(0.0 if not is_percentage else 40.0, min(all_v))
        max_v = max(all_v)
        nice_min, nice_max, ticks = get_nice_ticks(min_v, max_v * 1.08, n_ticks=4)

        def to_y(v):
            span = (nice_max - nice_min) if nice_max != nice_min else 1.0
            return margin_top + plot_h - ((v - nice_min) / span) * plot_h

        zero_y = to_y(0.0 if not is_percentage else nice_min)

        p_elements = []
        # Panel Title
        p_elements.append(
            f'<text x="{px_start + panel_w/2}" y="{margin_top - 18}" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="#1e293b">{esc(p_title)}</text>'
        )

        # Grid & Y-ticks
        for t in ticks:
            ty = to_y(t)
            p_elements.append(
                f'<line x1="{px_start}" y1="{ty:.1f}" x2="{px_start + panel_w}" y2="{ty:.1f}" '
                f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>'
            )
            t_str = f"{t:.0f} %" if is_percentage else f"{t:.2f}"
            p_elements.append(
                f'<text x="{px_start - 10}" y="{ty + 4:.1f}" text-anchor="end" '
                f'font-family="system-ui, sans-serif" font-size="10" fill="#64748b">{t_str}</text>'
            )

        # Zero line / base line
        p_elements.append(
            f'<line x1="{px_start}" y1="{zero_y:.1f}" x2="{px_start + panel_w}" y2="{zero_y:.1f}" '
            f'stroke="#94a3b8" stroke-width="1.5"/>'
        )

        # Grouped bars
        group_w = panel_w / n_m
        bar_w = min(42.0, (group_w * 0.70) / 2)

        for m_idx, m_name in enumerate(models):
            center_x = px_start + m_idx * group_w + group_w / 2
            v_def = def_vals[m_idx]
            v_tun = tuned_vals[m_idx]

            # 1. Bar Default
            bx_def = center_x - bar_w - 2
            by_def = to_y(v_def)
            bh_def = max(1.0, zero_y - by_def)
            p_elements.append(
                f'<rect x="{bx_def:.1f}" y="{by_def:.1f}" width="{bar_w:.1f}" height="{bh_def:.1f}" '
                f'rx="3" fill="{colors[0]}" opacity="0.9"/>'
            )
            txt_def = f"{v_def:.1f}%" if is_percentage else f"{v_def:.3f}"
            p_elements.append(
                f'<text x="{bx_def + bar_w/2:.1f}" y="{by_def - 5:.1f}" text-anchor="middle" '
                f'font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#475569">{txt_def}</text>'
            )

            # 2. Bar Tuned
            bx_tun = center_x + 2
            by_tun = to_y(v_tun)
            bh_tun = max(1.0, zero_y - by_tun)
            p_elements.append(
                f'<rect x="{bx_tun:.1f}" y="{by_tun:.1f}" width="{bar_w:.1f}" height="{bh_tun:.1f}" '
                f'rx="3" fill="{colors[1]}" opacity="0.95"/>'
            )
            txt_tun = f"{v_tun:.1f}%" if is_percentage else f"{v_tun:.3f}"
            p_elements.append(
                f'<text x="{bx_tun + bar_w/2:.1f}" y="{by_tun - 5:.1f}" text-anchor="middle" '
                f'font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#1e40af">{txt_tun}</text>'
            )

            # Model label below
            p_elements.append(
                f'<text x="{center_x:.1f}" y="{margin_top + plot_h + 18}" text-anchor="middle" '
                f'font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#1e293b">{esc(m_name)}</text>'
            )

            # Delta note
            delta = v_tun - v_def
            d_str = f"({delta:+.1f} %)" if is_percentage else f"({delta:+.3f})"
            d_col = "#15803d" if delta > 0.01 else ("#b91c1c" if delta < -0.01 else "#64748b")
            p_elements.append(
                f'<text x="{center_x:.1f}" y="{margin_top + plot_h + 32}" text-anchor="middle" '
                f'font-family="system-ui, sans-serif" font-size="10" fill="{d_col}">{d_str}</text>'
            )

        return "".join(p_elements)

    svg_panel_A = render_panel(pA_x, panel_A_models, panel_A_default, panel_A_tuned, panel_A_title)
    svg_panel_B = render_panel(pB_x, panel_B_models, panel_B_default, panel_B_tuned, panel_B_title)

    # Legenda nahoře vpravo
    legend_svg = f"""
    <g transform="translate({width - 250}, 24)">
      <rect x="0" y="0" width="12" height="12" rx="2" fill="{colors[0]}"/>
      <text x="18" y="10" font-size="11" fill="#334155">Default (5-Fold CV)</text>
      <rect x="130" y="0" width="12" height="12" rx="2" fill="{colors[1]}"/>
      <text x="148" y="10" font-size="11" fill="#1e40af" font-weight="bold">Tuned (Nested 5x5 CV)</text>
    </g>
    """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="40" y="32" font-size="16" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="40" y="52" font-size="12" fill="#64748b">{esc(subtitle)}</text>
  {legend_svg}

  <!-- Oddělovací vertikální čára mezi panely -->
  <line x1="{pB_x - 30}" y1="{margin_top - 25}" x2="{pB_x - 30}" y2="{margin_top + plot_h + 40}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 4"/>

  <!-- Panel A -->
  {svg_panel_A}

  <!-- Panel B -->
  {svg_panel_B}
</svg>"""
    return svg


def generate_confusion_matrix_svg(
    cm: List[List[int]],
    class_labels: List[str] = ["Neprospěl (<10)", "Prospěl (>=10)"],
    title: str = "Matice záměn: Random Forest (Balanced, Scénář B)",
    subtitle: str = "Out-of-Fold predikce včasného varování (5-Fold Stratified CV, bez G1 a G2)",
    width: int = 700,
    height: int = 510
) -> str:
    """
    Vytvoří rozšířenou a přehlednou matici záměn v SVG:
    - Bez překrývání nadpisů a s velkým horním okrajem (margin_top = 100).
    - V buňkách zobrazuje absolutní počet, řádkový podíl (Recall / Specificita) i celkový podíl.
    - Zobrazuje řádkový záchyt (Recall pro minority) a sloupcovou přesnost (Precision / NPV).
    - V souhrnné kartě dole uvádí Balanced Accuracy a varovné metriky.
    """
    margin_top = 105
    margin_left = 135
    cell_w = 175
    cell_h = 135

    total_samples = sum(sum(row) for row in cm)
    max_val = max(max(row) for row in cm) or 1

    cells = []
    row_totals = [sum(row) for row in cm]
    col_totals = [sum(cm[r][c] for r in range(2)) for c in range(2)]

    for r in range(2):
        for c in range(2):
            val = cm[r][c]
            pct_total = (val / total_samples) * 100 if total_samples > 0 else 0
            pct_row = (val / row_totals[r]) * 100 if row_totals[r] > 0 else 0
            intensity = val / max_val

            if r == c:
                bg = f"rgba(37, 99, 235, {max(0.12, intensity * 0.9):.2f})"
                text_col = "#ffffff" if intensity > 0.55 else "#0f172a"
                sub_col = "#e2e8f0" if intensity > 0.55 else "#475569"
            else:
                bg = f"rgba(239, 68, 68, {max(0.10, intensity * 0.85):.2f})"
                text_col = "#ffffff" if intensity > 0.60 else "#0f172a"
                sub_col = "#e2e8f0" if intensity > 0.60 else "#475569"

            x = margin_left + c * cell_w
            y = margin_top + r * cell_h

            # Popis buňky
            if r == 0 and c == 0:
                cell_title = "Správně zachyceno (TP)"
                row_note = f"Recall: {pct_row:.1f} %"
            elif r == 0 and c == 1:
                cell_title = "Nezachyceno (FN)"
                row_note = f"Chyba: {pct_row:.1f} %"
            elif r == 1 and c == 0:
                cell_title = "Falešný poplach (FP)"
                row_note = f"Chyba: {pct_row:.1f} %"
            else:
                cell_title = "Správně prospěl (TN)"
                row_note = f"Specificita: {pct_row:.1f} %"

            cells.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'fill="{bg}" stroke="#ffffff" stroke-width="2"/>'
            )
            cells.append(
                f'<text x="{x + cell_w/2:.1f}" y="{y + 26:.1f}" text-anchor="middle" '
                f'font-size="11" font-weight="bold" fill="{text_col}">{esc(cell_title)}</text>'
            )
            cells.append(
                f'<text x="{x + cell_w/2:.1f}" y="{y + 68:.1f}" text-anchor="middle" '
                f'font-size="28" font-weight="bold" fill="{text_col}">{val}</text>'
            )
            cells.append(
                f'<text x="{x + cell_w/2:.1f}" y="{y + 94:.1f}" text-anchor="middle" '
                f'font-size="12" font-weight="600" fill="{text_col}">{row_note}</text>'
            )
            cells.append(
                f'<text x="{x + cell_w/2:.1f}" y="{y + 114:.1f}" text-anchor="middle" '
                f'font-size="10" fill="{sub_col}">({pct_total:.1f} % z celku)</text>'
            )

    # Popisky sloupců a řádků
    labels_svg = []
    for i, lab in enumerate(class_labels):
        cx = margin_left + i * cell_w + cell_w / 2
        labels_svg.append(
            f'<text x="{cx:.1f}" y="{margin_top - 12}" text-anchor="middle" '
            f'font-size="12" font-weight="bold" fill="#334155">{esc(lab)}</text>'
        )
        cy = margin_top + i * cell_h + cell_h / 2
        labels_svg.append(
            f'<text x="{margin_left - 14}" y="{cy + 4:.1f}" text-anchor="end" '
            f'font-size="12" font-weight="bold" fill="#334155">{esc(lab)}</text>'
        )

    # Pravý sloupec (Souhrn řádků - Recall / Specificita)
    right_x = margin_left + 2 * cell_w + 18
    row_summary_svg = []
    row_summary_svg.append(
        f'<text x="{right_x}" y="{margin_top + 45}" font-size="11" font-weight="bold" fill="#1e293b">'
        f'Řádek Neprospěl ({row_totals[0]}):</text>'
    )
    rec_val = (cm[0][0] / row_totals[0]) * 100 if row_totals[0] > 0 else 0
    row_summary_svg.append(
        f'<text x="{right_x}" y="{margin_top + 65}" font-size="12" font-weight="bold" fill="#15803d">'
        f'\u2192 Záchyt: {rec_val:.1f} % ({cm[0][0]}/{row_totals[0]})</text>'
    )

    row_summary_svg.append(
        f'<text x="{right_x}" y="{margin_top + cell_h + 45}" font-size="11" font-weight="bold" fill="#1e293b">'
        f'Řádek Prospěl ({row_totals[1]}):</text>'
    )
    spec_val = (cm[1][1] / row_totals[1]) * 100 if row_totals[1] > 0 else 0
    row_summary_svg.append(
        f'<text x="{right_x}" y="{margin_top + cell_h + 65}" font-size="12" font-weight="bold" fill="#2563eb">'
        f'\u2192 Specificita: {spec_val:.1f} % ({cm[1][1]}/{row_totals[1]})</text>'
    )

    # Spodní řádek (Souhrn sloupců - Precision / NPV)
    col_summary_svg = []
    prec_val = (cm[0][0] / col_totals[0]) * 100 if col_totals[0] > 0 else 0
    npv_val = (cm[1][1] / col_totals[1]) * 100 if col_totals[1] > 0 else 0
    col_summary_svg.append(
        f'<text x="{margin_left + cell_w/2}" y="{margin_top + 2*cell_h + 20}" text-anchor="middle" '
        f'font-size="11" font-weight="600" fill="#334155">Přesnost varování: {prec_val:.1f} % ({cm[0][0]}/{col_totals[0]})</text>'
    )
    col_summary_svg.append(
        f'<text x="{margin_left + 1.5*cell_w}" y="{margin_top + 2*cell_h + 20}" text-anchor="middle" '
        f'font-size="11" font-weight="600" fill="#334155">NPV (bezpečnost): {npv_val:.1f} % ({cm[1][1]}/{col_totals[1]})</text>'
    )

    # Souhrnný spodní panel
    bal_acc = (rec_val + spec_val) / 2.0
    summary_card = f"""
    <g transform="translate(40, {height - 44})">
      <rect x="0" y="0" width="{width - 80}" height="32" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
      <text x="20" y="20" font-size="11" font-weight="bold" fill="#0f172a">Vyvážená přesnost (Balanced Accuracy): {bal_acc:.1f} %</text>
      <text x="{width - 100}" y="20" text-anchor="end" font-size="11" fill="#475569">Záchyt ohrožených studentů (Recall): <tspan font-weight="bold" fill="#15803d">{rec_val:.1f} %</tspan> | Přesnost varování: <tspan font-weight="bold" fill="#1e293b">{prec_val:.1f} %</tspan></text>
    </g>
    """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{width / 2}" y="52" text-anchor="middle" font-size="12" fill="#64748b">{esc(subtitle)}</text>

  <!-- Záhlaví os -->
  <text x="{margin_left + cell_w}" y="{margin_top - 36}" text-anchor="middle" font-size="12" font-weight="bold" fill="#475569">Predikovaná třída</text>
  <text x="24" y="{margin_top + cell_h}" text-anchor="middle" font-size="12" font-weight="bold" fill="#475569" transform="rotate(-90, 24, {margin_top + cell_h})">Skutečná třída</text>

  <!-- Buňky -->
  {''.join(cells)}

  <!-- Popisky tříd -->
  {''.join(labels_svg)}

  <!-- Řádkové a sloupcové statistiky -->
  {''.join(row_summary_svg)}
  {''.join(col_summary_svg)}

  <!-- Spodní souhrnná lišta -->
  {summary_card}
</svg>"""
    return svg


def generate_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    width: int = 800,
    height: int = 450,
    bar_color: str = "#2563eb",
    highlight_max: bool = False
) -> str:
    """Zpětně kompatibilní generátor vertikálního sloupcového grafu."""
    margin_top = 60
    margin_bottom = 80
    margin_left = 90
    margin_right = 40

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    n = len(values)
    if n == 0:
        return ""

    min_val = min(0.0, min(values))
    max_val = max(values)
    nice_min, nice_max, ticks = get_nice_ticks(min_val, max_val, n_ticks=5)

    def val_to_y(v):
        span = (nice_max - nice_min) if nice_max != nice_min else 1.0
        return margin_top + plot_h - ((v - nice_min) / span) * plot_h

    zero_y = val_to_y(0.0)

    grid_lines = []
    y_labels = []
    for tick_val in ticks:
        y_pos = val_to_y(tick_val)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>'
        )
        y_labels.append(
            f'<text x="{margin_left - 12}" y="{y_pos + 4:.1f}" text-anchor="end" '
            f'font-family="system-ui, sans-serif" font-size="11" fill="#64748b">{tick_val:.2f}</text>'
        )

    bar_width = min(50.0, (plot_w / n) * 0.7)
    gap = plot_w / n
    bars = []
    x_labels = []

    for i, (lab, val) in enumerate(zip(labels, values)):
        x_center = margin_left + i * gap + gap / 2
        x_bar = x_center - bar_width / 2
        y_bar = val_to_y(val)

        if val >= 0:
            h = zero_y - y_bar
            top = y_bar
        else:
            h = y_bar - zero_y
            top = zero_y

        bars.append(
            f'<rect x="{x_bar:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{max(1.0, h):.1f}" '
            f'rx="4" fill="{bar_color}" opacity="0.9"/>'
        )
        bars.append(
            f'<text x="{x_center:.1f}" y="{top - 6 if val >= 0 else top + h + 14:.1f}" text-anchor="middle" '
            f'font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#1e293b">{val:.3f}</text>'
        )

        short_lab = lab if len(lab) <= 24 else lab[:22] + "…"
        rotate = f'transform="rotate(-30, {x_center:.1f}, {margin_top + plot_h + 16:.1f})"' if len(labels) > 5 else ""
        x_labels.append(
            f'<text x="{x_center:.1f}" y="{margin_top + plot_h + 18:.1f}" '
            f'text-anchor="{"end" if len(labels) > 5 else "middle"}" font-family="system-ui, sans-serif" font-size="11" fill="#334155" {rotate}>{esc(short_lab)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{width / 2}" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="12" fill="#64748b">{esc(xlabel)}</text>
  <text x="24" y="{margin_top + plot_h / 2}" text-anchor="middle" font-size="12" fill="#64748b" transform="rotate(-90, 24, {margin_top + plot_h / 2})">{esc(ylabel)}</text>

  <!-- Grid -->
  {''.join(grid_lines)}

  <!-- Osa X -->
  <line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Sloupce -->
  {''.join(bars)}

  <!-- Popisky os -->
  {''.join(y_labels)}
  {''.join(x_labels)}
</svg>"""
    return svg


def generate_grouped_bar_chart(
    categories: List[str],
    series: Dict[str, List[float]],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    width: int = 860,
    height: int = 460,
    palette: Optional[List[str]] = None
) -> str:
    """Vytvoří vícenásobný (grouped) sloupcový graf."""
    margin_top = 60
    margin_bottom = 80
    margin_left = 80
    margin_right = 160

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    n_cats = len(categories)
    series_names = list(series.keys())
    n_series = len(series_names)
    if n_cats == 0 or n_series == 0:
        return ""

    default_palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
    colors = palette or default_palette[:n_series]

    all_vals = [v for s_vals in series.values() for v in s_vals]
    min_val = min(0.0, min(all_vals))
    max_val = max(all_vals)
    nice_min, nice_max, ticks = get_nice_ticks(min_val, max_val, n_ticks=5)

    def val_to_y(v):
        span = (nice_max - nice_min) if nice_max != nice_min else 1.0
        return margin_top + plot_h - ((v - nice_min) / span) * plot_h

    zero_y = val_to_y(0.0)

    grid_lines = []
    y_labels = []
    for tick_val in ticks:
        y_pos = val_to_y(tick_val)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>'
        )
        y_labels.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" '
            f'font-family="system-ui, sans-serif" font-size="11" fill="#64748b">{tick_val:.2f}</text>'
        )

    group_w = plot_w / n_cats
    bar_w = (group_w * 0.75) / n_series
    bars = []
    x_labels = []

    for c_idx, cat in enumerate(categories):
        group_center = margin_left + c_idx * group_w + group_w / 2
        start_x = group_center - (n_series * bar_w) / 2

        for s_idx, (s_name, s_vals) in enumerate(series.items()):
            val = s_vals[c_idx]
            bx = start_x + s_idx * bar_w
            by = val_to_y(val)
            col = colors[s_idx % len(colors)]

            if val >= 0:
                bh = max(1.0, zero_y - by)
                top = by
            else:
                bh = max(1.0, by - zero_y)
                top = zero_y

            bars.append(
                f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bar_w - 2:.1f}" height="{bh:.1f}" '
                f'rx="2" fill="{col}" opacity="0.88"/>'
            )

        short_cat = cat if len(cat) <= 20 else cat[:18] + "…"
        rotate = f'transform="rotate(-25, {group_center:.1f}, {margin_top + plot_h + 16:.1f})"' if len(categories) > 5 else ""
        x_labels.append(
            f'<text x="{group_center:.1f}" y="{margin_top + plot_h + 18:.1f}" '
            f'text-anchor="{"end" if len(categories) > 5 else "middle"}" font-family="system-ui, sans-serif" font-size="11" fill="#1e293b" {rotate}>{esc(short_cat)}</text>'
        )

    legend_items = []
    for s_idx, s_name in enumerate(series_names):
        ly = margin_top + 20 + s_idx * 24
        col = colors[s_idx % len(colors)]
        legend_items.append(f'<rect x="{width - margin_right + 20}" y="{ly - 8}" width="12" height="12" rx="2" fill="{col}"/>')
        legend_items.append(f'<text x="{width - margin_right + 38}" y="{ly + 2}" font-size="11" fill="#334155">{esc(s_name)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{margin_left + plot_w / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{margin_left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="12" fill="#64748b">{esc(xlabel)}</text>
  <text x="24" y="{margin_top + plot_h / 2}" text-anchor="middle" font-size="12" fill="#64748b" transform="rotate(-90, 24, {margin_top + plot_h / 2})">{esc(ylabel)}</text>

  <!-- Grid -->
  {''.join(grid_lines)}

  <!-- Zero Line -->
  <line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Sloupce -->
  {''.join(bars)}

  <!-- Popisky os -->
  {''.join(y_labels)}
  {''.join(x_labels)}

  <!-- Legenda -->
  {''.join(legend_items)}
</svg>"""
    return svg


def generate_scree_plot(
    explained_var: List[float],
    cumulative_var: List[float],
    title: str = "PCA Scree Plot (Vysvětlený rozptyl)",
    width: int = 750,
    height: int = 420
) -> str:
    """Vytvoří kombinovaný sloupcový a čárový graf pro PCA rozptyl."""
    margin_top = 50
    margin_bottom = 60
    margin_left = 70
    margin_right = 70

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n = len(explained_var)

    gap = plot_w / n
    bar_w = gap * 0.65

    bars = []
    line_points = []
    dots = []
    x_labels = []

    for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
        cx = margin_left + i * gap + gap / 2
        bx = cx - bar_w / 2

        max_ev = max(explained_var) * 1.2
        by = margin_top + plot_h - (ev / max_ev) * plot_h
        bh = (ev / max_ev) * plot_h

        bars.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="#3b82f6" opacity="0.8" rx="3"/>')
        bars.append(f'<text x="{cx:.1f}" y="{by - 5:.1f}" text-anchor="middle" font-size="10" fill="#1e40af">{ev*100:.1f}%</text>')

        cy = margin_top + plot_h - (cv * plot_h)
        line_points.append(f"{cx:.1f},{cy:.1f}")
        dots.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#ef4444" stroke="#ffffff" stroke-width="2"/>')
        dots.append(f'<text x="{cx:.1f}" y="{cy - 8:.1f}" text-anchor="middle" font-size="10" font-weight="600" fill="#b91c1c">{cv*100:.1f}%</text>')

        x_labels.append(f'<text x="{cx:.1f}" y="{margin_top + plot_h + 18:.1f}" text-anchor="middle" font-size="11" fill="#475569">{esc(f"PC{i+1}")}</text>')

    polyline = " ".join(line_points)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>

  <!-- Osy -->
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1"/>

  <!-- Sloupce (jednotlivé PC) -->
  {''.join(bars)}

  <!-- Čára kumulativního rozptylu -->
  <polyline fill="none" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="none" points="{polyline}"/>
  {''.join(dots)}

  <!-- Popisky os -->
  {''.join(x_labels)}

  <!-- Legenda -->
  <g transform="translate({width - margin_right - 220}, {margin_top - 20})">
    <rect x="0" y="0" width="12" height="12" fill="#3b82f6" rx="2"/>
    <text x="18" y="10" font-size="11" fill="#334155">Individuální rozptyl</text>
    <line x1="120" y1="6" x2="140" y2="6" stroke="#ef4444" stroke-width="2"/>
    <circle cx="130" cy="6" r="3" fill="#ef4444"/>
    <text x="146" y="10" font-size="11" fill="#334155">Kumulativní (%)</text>
  </g>
</svg>"""
    return svg


def generate_scatter_plot(
    x_vals: List[float],
    y_vals: List[float],
    categories: List[str],
    title: str,
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    width: int = 750,
    height: int = 500,
    color_palette: Optional[Dict[str, str]] = None
) -> str:
    """Vytvoří 2D bodový graf (např. projekce PCA obarvená podle třídy)."""
    margin_top = 50
    margin_bottom = 60
    margin_left = 70
    margin_right = 160

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    pad_x = (max_x - min_x) * 0.08 or 1.0
    pad_y = (max_y - min_y) * 0.08 or 1.0

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def to_x(x):
        return margin_left + ((x - min_x) / (max_x - min_x)) * plot_w

    def to_y(y):
        return margin_top + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h

    unique_cats = sorted(list(set(categories)))
    default_palette = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
    palette = color_palette or {c: default_palette[i % len(default_palette)] for i, c in enumerate(unique_cats)}

    points = []
    for x, y, cat in zip(x_vals, y_vals, categories):
        px = to_x(x)
        py = to_y(y)
        color = palette.get(cat, "#64748b")
        points.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}" opacity="0.75"/>')

    zero_x = to_x(0.0) if min_x <= 0 <= max_x else None
    zero_y = to_y(0.0) if min_y <= 0 <= max_y else None
    axes = []
    if zero_x:
        axes.append(f'<line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{margin_top + plot_h}" stroke="#cbd5e1" stroke-width="1"/>')
    if zero_y:
        axes.append(f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{margin_left + plot_w}" y2="{zero_y:.1f}" stroke="#cbd5e1" stroke-width="1"/>')

    legend_items = []
    for i, cat in enumerate(unique_cats):
        ly = margin_top + 20 + i * 24
        color = palette.get(cat, "#64748b")
        legend_items.append(f'<circle cx="{width - margin_right + 20}" cy="{ly}" r="5" fill="{color}"/>')
        legend_items.append(f'<text x="{width - margin_right + 32}" y="{ly + 4}" font-size="11" fill="#334155">{esc(cat)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{margin_left + plot_w / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{margin_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="#64748b">{esc(xlabel)}</text>
  <text x="22" y="{margin_top + plot_h / 2}" text-anchor="middle" font-size="12" fill="#64748b" transform="rotate(-90, 22, {margin_top + plot_h / 2})">{esc(ylabel)}</text>

  <!-- Plot border -->
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>

  {''.join(axes)}
  {''.join(points)}
  {''.join(legend_items)}
</svg>"""
    return svg


def generate_trajectory_chart(
    x_steps: List[str],
    series: Dict[str, List[float]],
    title: str,
    xlabel: str = "Ročník / Období studia",
    ylabel: str = "Průměrná známka (0 - 20)",
    width: int = 760,
    height: int = 440,
    palette: Optional[Dict[str, str]] = None
) -> str:
    """Vytvoří liniový graf trajektorie vývoje prospěchu v čase (G1 -> G2 -> G3)."""
    margin_top = 50
    margin_bottom = 60
    margin_left = 70
    margin_right = 170

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    n_steps = len(x_steps)
    if n_steps == 0 or len(series) == 0:
        return ""

    all_vals = [v for s_vals in series.values() for v in s_vals]
    min_val = min(all_vals)
    max_val = max(all_vals)

    pad = (max_val - min_val) * 0.15 or 1.0
    y_min = max(0.0, min_val - pad)
    y_max = min(20.0, max_val + pad)

    def to_x(i):
        if n_steps == 1:
            return margin_left + plot_w / 2
        return margin_left + (i / (n_steps - 1)) * plot_w

    def to_y(v):
        return margin_top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    default_palette = ["#10b981", "#2563eb", "#ef4444", "#8b5cf6"]
    colors = palette or {name: default_palette[i % len(default_palette)] for i, name in enumerate(series.keys())}

    grid_lines = []
    y_labels = []
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_val = y_min + i * (y_max - y_min) / n_ticks
        y_pos = to_y(tick_val)
        grid_lines.append(f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>')
        y_labels.append(f'<text x="{margin_left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{tick_val:.1f}</text>')

    x_labels = []
    for i, step in enumerate(x_steps):
        xp = to_x(i)
        x_labels.append(f'<text x="{xp:.1f}" y="{margin_top + plot_h + 22:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">{esc(step)}</text>')

    lines = []
    points = []
    legend_items = []

    for s_idx, (name, vals) in enumerate(series.items()):
        col = colors.get(name, "#2563eb")
        pts = [f"{to_x(i):.1f},{to_y(v):.1f}" for i, v in enumerate(vals)]
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="3" points="{" ".join(pts)}"/>')

        for i, v in enumerate(vals):
            px = to_x(i)
            py = to_y(v)
            points.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
            points.append(f'<text x="{px:.1f}" y="{py - 10:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="{col}">{v:.2f}</text>')

        ly = margin_top + 20 + s_idx * 26
        legend_items.append(f'<line x1="{width - margin_right + 15}" y1="{ly}" x2="{width - margin_right + 35}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        legend_items.append(f'<circle cx="{width - margin_right + 25}" cy="{ly}" r="4" fill="{col}"/>')
        legend_items.append(f'<text x="{width - margin_right + 42}" y="{ly + 4}" font-size="11" fill="#334155">{esc(name)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{margin_left + plot_w / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{margin_left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="12" fill="#64748b">{esc(xlabel)}</text>
  <text x="24" y="{margin_top + plot_h / 2}" text-anchor="middle" font-size="12" fill="#64748b" transform="rotate(-90, 24, {margin_top + plot_h / 2})">{esc(ylabel)}</text>

  <!-- Grid -->
  {''.join(grid_lines)}

  <!-- Lines -->
  {''.join(lines)}
  {''.join(points)}

  <!-- Popisky os -->
  {''.join(y_labels)}
  {''.join(x_labels)}

  <!-- Legenda -->
  {''.join(legend_items)}
</svg>"""
    return svg
