"""
visualization_utils.py - Generování čistých vektorových SVG grafů bez externích závislostí.
Podporuje:
- Sloupcové grafy (Bar charts, horizontal bar charts)
- Bodové grafy (Scatter plots pro PCA projekce)
- Spojnicové grafy (Line plots pro scree plot a kumulativní rozptyl)
- Teplotní mapy (Heatmaps pro matice záměn)
Plně ošetřuje XML escapování (& -> &amp;, < -> &lt;, > -> &gt;) pro správné zobrazení ve všech prohlížečích a prohlížečích obrázků.
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


def generate_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    width: int = 800,
    height: int = 450,
    bar_color: str = "#2563eb",
    highlight_max: bool = True
) -> str:
    """Vytvoří SVG sloupcový graf."""
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
    if max_val == min_val:
        max_val += 1.0

    # Zaokrouhlení horní meze na hezké číslo
    range_val = max_val - min_val
    y_max = max_val + 0.1 * range_val
    y_min = min_val if min_val < 0 else 0.0

    def val_to_y(v):
        return margin_top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    zero_y = val_to_y(0.0)

    # Grid & Y-ticks
    n_ticks = 5
    grid_lines = []
    y_labels = []
    for i in range(n_ticks + 1):
        tick_val = y_min + i * (y_max - y_min) / n_ticks
        y_pos = val_to_y(tick_val)
        grid_lines.append(f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>')
        y_labels.append(f'<text x="{margin_left - 12}" y="{y_pos + 4:.1f}" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#64748b">{tick_val:.2f}</text>')

    # Sloupce
    bar_width = min(50.0, (plot_w / n) * 0.7)
    gap = plot_w / n
    bars = []
    x_labels = []

    max_idx = values.index(max(values))

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

        color = "#10b981" if (highlight_max and i == max_idx and n > 1) else bar_color
        bars.append(f'<rect x="{x_bar:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{max(1.0, h):.1f}" rx="4" fill="{color}" opacity="0.9"/>')
        bars.append(f'<text x="{x_center:.1f}" y="{top - 6 if val >= 0 else top + h + 14:.1f}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#1e293b">{val:.3f}</text>')

        # Zkrácený label a bezpečné XML escapování
        short_lab = lab if len(lab) <= 14 else lab[:12] + "…"
        rotate = 'transform="rotate(-30, ' + f'{x_center:.1f}, {margin_top + plot_h + 16:.1f})"' if len(labels) > 6 else ''
        x_labels.append(f'<text x="{x_center:.1f}" y="{margin_top + plot_h + 18:.1f}" text-anchor="{"end" if len(labels) > 6 else "middle"}" font-family="system-ui, sans-serif" font-size="11" fill="#334155" {rotate}>{esc(short_lab)}</text>')

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


def generate_horizontal_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str = "",
    width: int = 750,
    bar_height: int = 24,
    bar_color: str = "#3b82f6"
) -> str:
    """Vytvoří horizontální sloupcový graf (ideální pro feature importance / correlation)."""
    n = len(labels)
    margin_top = 50
    margin_bottom = 50
    margin_left = 170
    margin_right = 70
    
    plot_h = max(200, n * (bar_height + 10))
    height = margin_top + plot_h + margin_bottom
    plot_w = width - margin_left - margin_right

    max_val = max(values) if values else 1.0
    min_val = min(0.0, min(values)) if values else 0.0
    span = (max_val - min_val) if max_val != min_val else 1.0

    def val_to_x(v):
        return margin_left + ((v - min_val) / span) * plot_w

    zero_x = val_to_x(0.0)

    bars = []
    labels_svg = []
    
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = margin_top + i * (bar_height + 10)
        x_val = val_to_x(val)
        
        if val >= 0:
            w = max(1.0, x_val - zero_x)
            x_start = zero_x
            text_x = x_start + w + 6
            text_anchor = "start"
        else:
            w = max(1.0, zero_x - x_val)
            x_start = x_val
            text_x = x_start - 6
            text_anchor = "end"

        color = "#10b981" if val > 0 else "#ef4444"
        bars.append(f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_height}" rx="3" fill="{color}" opacity="0.85"/>')
        bars.append(f'<text x="{text_x:.1f}" y="{y + bar_height/2 + 4:.1f}" text-anchor="{text_anchor}" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#334155">{val:.3f}</text>')
        
        short_lab = lab if len(lab) <= 24 else lab[:22] + "…"
        labels_svg.append(f'<text x="{margin_left - 10}" y="{y + bar_height/2 + 4:.1f}" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#1e293b">{esc(short_lab)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{margin_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" font-size="12" fill="#64748b">{esc(xlabel)}</text>
  
  <!-- Zero line -->
  <line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>
  
  {''.join(bars)}
  {''.join(labels_svg)}
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

    # Grid a osy
    zero_x = to_x(0.0) if min_x <= 0 <= max_x else None
    zero_y = to_y(0.0) if min_y <= 0 <= max_y else None
    axes = []
    if zero_x:
        axes.append(f'<line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{margin_top + plot_h}" stroke="#cbd5e1" stroke-width="1"/>')
    if zero_y:
        axes.append(f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{margin_left + plot_w}" y2="{zero_y:.1f}" stroke="#cbd5e1" stroke-width="1"/>')

    # Legenda
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


def generate_confusion_matrix_svg(
    cm: List[List[int]],
    class_labels: List[str],
    title: str = "Matice záměn (Confusion Matrix)",
    width: int = 480,
    height: int = 420
) -> str:
    """Vytvoří vizuální matici záměn v SVG."""
    margin_top = 70
    margin_bottom = 60
    margin_left = 110
    margin_right = 50

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n = len(class_labels)

    cell_w = plot_w / n
    cell_h = plot_h / n

    total_samples = sum(sum(row) for row in cm)
    max_val = max(max(row) for row in cm) or 1

    cells = []
    for r in range(n):
        for c in range(n):
            val = cm[r][c]
            pct = (val / total_samples) * 100 if total_samples > 0 else 0
            intensity = val / max_val
            
            # Barevné schéma: diagonála (zelená / modrá), mimo diagonálu (červená)
            if r == c:
                bg = f"rgba(37, 99, 235, {max(0.15, intensity):.2f})"
                text_col = "#ffffff" if intensity > 0.55 else "#1e293b"
            else:
                bg = f"rgba(239, 68, 68, {max(0.08, intensity * 0.8):.2f})"
                text_col = "#ffffff" if intensity > 0.6 else "#1e293b"

            x = margin_left + c * cell_w
            y = margin_top + r * cell_h

            cells.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{bg}" stroke="#ffffff" stroke-width="2"/>')
            cells.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 - 2:.1f}" text-anchor="middle" font-size="16" font-weight="bold" fill="{text_col}">{val}</text>')
            cells.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 16:.1f}" text-anchor="middle" font-size="10" fill="{text_col}">({pct:.1f}%)</text>')

    # Popisky os
    labels_svg = []
    for i, lab in enumerate(class_labels):
        # Nahoře: Predikce
        labels_svg.append(f'<text x="{margin_left + i*cell_w + cell_w/2:.1f}" y="{margin_top - 12}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">{esc(lab)}</text>')
        # Vlevo: Skutečnost
        labels_svg.append(f'<text x="{margin_left - 12}" y="{margin_top + i*cell_h + cell_h/2 + 4:.1f}" text-anchor="end" font-size="12" font-weight="600" fill="#334155">{esc(lab)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:system-ui, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>
  <text x="{margin_left + plot_w / 2}" y="{margin_top - 32}" text-anchor="middle" font-size="12" font-weight="bold" fill="#64748b">Predikovaná třída</text>
  <text x="24" y="{margin_top + plot_h / 2}" text-anchor="middle" font-size="12" font-weight="bold" fill="#64748b" transform="rotate(-90, 24, {margin_top + plot_h / 2})">Skutečná třída</text>

  {''.join(cells)}
  {''.join(labels_svg)}
</svg>"""
    return svg


def generate_grouped_bar_chart(
    categories: List[str],
    series: Dict[str, List[float]],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    width: int = 820,
    height: int = 460,
    palette: Optional[List[str]] = None
) -> str:
    """Vytvoří vícenásobný (grouped) sloupcový graf (např. porovnání vlivu na G1 vs G2 vs G3)."""
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
    span = (max_val - min_val) if max_val != min_val else 1.0
    y_max = max_val + 0.1 * span
    y_min = min_val if min_val < 0 else 0.0

    def val_to_y(v):
        return margin_top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    zero_y = val_to_y(0.0)

    # Grid
    grid_lines = []
    y_labels = []
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_val = y_min + i * (y_max - y_min) / n_ticks
        y_pos = val_to_y(tick_val)
        grid_lines.append(f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>')
        y_labels.append(f'<text x="{margin_left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#64748b">{tick_val:.2f}</text>')

    # Grouped bars
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

            bars.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bar_w - 2:.1f}" height="{bh:.1f}" rx="2" fill="{col}" opacity="0.85"/>')

        short_cat = cat if len(cat) <= 15 else cat[:13] + "…"
        rotate = 'transform="rotate(-25, ' + f'{group_center:.1f}, {margin_top + plot_h + 16:.1f})"' if len(categories) > 5 else ''
        x_labels.append(f'<text x="{group_center:.1f}" y="{margin_top + plot_h + 18:.1f}" text-anchor="{"end" if len(categories) > 5 else "middle"}" font-family="system-ui, sans-serif" font-size="11" fill="#1e293b" {rotate}>{esc(short_cat)}</text>')

    # Legenda
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
