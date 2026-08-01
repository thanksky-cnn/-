"""图 2.1：北极海域研究区地图 — 清洁底图版本.

高分辨率 Natural Earth 10m 底图，北极极射投影，北极点居中。
含经纬网格线、7个研究海域 + 东西伯利亚海标签。
全部矢量输出，所有海域名称均以可编辑 SVG <text> 元素呈现。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import sys, os, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nature_figure_config

fm.fontManager.addfont("C:/Windows/Fonts/arial.ttf")
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
from matplotlib.lines import Line2D
from svg.path import parse_path as svg_parse_path
from svg.path.path import Move, Line, CubicBezier, Close, QuadraticBezier, Arc

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Compass rose — shared with original version
# =============================================================================
def _svg_path_to_mpl_vertices(svg_path_d, n_bezier=24):
    """Convert SVG path d string → matplotlib Path, sampling Bezier curves.

    Returns a matplotlib.path.Path object.
    """
    path = svg_parse_path(svg_path_d)
    vertices, codes = [], []

    for seg in path:
        if isinstance(seg, Move):
            vertices.append((seg.start.real, seg.start.imag))
            codes.append(mpath.Path.MOVETO)
        elif isinstance(seg, Line):
            vertices.append((seg.end.real, seg.end.imag))
            codes.append(mpath.Path.LINETO)
        elif isinstance(seg, CubicBezier):
            for i in range(1, n_bezier + 1):
                t = i / n_bezier
                p = seg.point(t)
                vertices.append((p.real, p.imag))
                codes.append(mpath.Path.LINETO)
        elif isinstance(seg, QuadraticBezier):
            for i in range(1, n_bezier + 1):
                t = i / n_bezier
                p = seg.point(t)
                vertices.append((p.real, p.imag))
                codes.append(mpath.Path.LINETO)
        elif isinstance(seg, Arc):
            for i in range(1, n_bezier + 1):
                t = i / n_bezier
                p = seg.point(t)
                vertices.append((p.real, p.imag))
                codes.append(mpath.Path.LINETO)
        elif isinstance(seg, Close):
            # matplotlib CLOSEPOLY closes to MOVETO; skip explicit Close
            pass

    if len(vertices) == 0:
        return None
    return mpath.Path(np.array(vertices), codes)


def load_compass_from_svg(svg_path):
    """Parse all paths from the reference SVG compass file.

    The SVG uses transform="translate(0,1266) scale(0.1,-0.1)".
    Returns a list of matplotlib.path.Path, centred & normalised to [0,1]^2.
    """
    tree = ET.parse(svg_path)
    ns = 'http://www.w3.org/2000/svg'

    all_mpl_paths = []
    all_pts = []  # collect all vertices for bounding-box

    for g in tree.findall('.//{%s}g' % ns):
        transform_str = g.get('transform', '')
        fill_color = g.get('fill', '#018847')

        for path_elem in g.findall('{%s}path' % ns):
            d = path_elem.get('d', '')
            mpl_p = _svg_path_to_mpl_vertices(d)
            if mpl_p is None or len(mpl_p.vertices) == 0:
                continue

            verts = mpl_p.vertices
            # Apply SVG transform: scale(0.1, -0.1) then translate(0, 1266)
            xs = 0.1 * verts[:, 0]
            ys = 1266.0 - 0.1 * verts[:, 1]
            verts_t = np.column_stack([xs, ys])
            all_pts.append(verts_t)
            all_mpl_paths.append((verts_t, fill_color))

    if not all_pts:
        return None

    # Normalise: find bounding box and map to [0, 1] x [0, 1]
    all_concat = np.concatenate([p[0] for p in all_mpl_paths], axis=0)
    x_min, y_min = all_concat[:, 0].min(), all_concat[:, 1].min()
    x_max, y_max = all_concat[:, 0].max(), all_concat[:, 1].max()
    w, h = x_max - x_min, y_max - y_min

    out_paths = []
    for verts, fill_c in all_mpl_paths:
        vn = np.column_stack([
            (verts[:, 0] - x_min) / w,
            (verts[:, 1] - y_min) / h,
        ])
        out_paths.append((vn, fill_c))

    return out_paths


def draw_compass_rose(ax, x, y, radius, svg_ref_path, transform=None):
    """Draw compass rose from reference SVG, centred at (x,y) in Axes coordinates."""
    if transform is None:
        transform = ax.transAxes

    compass_data = load_compass_from_svg(svg_ref_path)
    if compass_data is None:
        print('  [WARN] Could not load compass SVG, using fallback compass')
        return _draw_fallback_compass(ax, x, y, radius, transform)

    for verts_norm, fill_color in compass_data:
        # Scale & translate to target position
        # verts_norm are in [0,1]^2 centred on the compass's own bounding box
        # We want the compass centred at (x, y) with given radius
        cx_norm = 0.5
        cy_norm = 0.5
        v_scaled = np.column_stack([
            x + (verts_norm[:, 0] - cx_norm) * 2.0 * radius,
            y + (verts_norm[:, 1] - cy_norm) * 2.0 * radius,
        ])
        patch = mpatches.Polygon(v_scaled, closed=True, transform=transform,
                                 facecolor=fill_color, edgecolor='none',
                                 zorder=42, linewidth=0)
        ax.add_patch(patch)


def _draw_fallback_compass(ax, x, y, radius, transform):
    """Fallback compass: simple N/S/E/W rose in case SVG fails to load."""
    circle = mpatches.Circle((x, y), radius, transform=transform,
                              facecolor='white', edgecolor='#444444',
                              linewidth=0.8, zorder=40)
    ax.add_patch(circle)
    for label, (dx, dy) in {'N': (0,1), 'S': (0,-1), 'E': (1,0), 'W': (-1,0)}.items():
        inner, outer = radius * 0.22, radius * 0.72
        lw = 2.0 if label == 'N' else 1.0
        ax.plot([x + dx*inner, x + dx*outer], [y + dy*inner, y + dy*outer],
                transform=transform, color='#018847',
                linewidth=lw, solid_capstyle='round', zorder=41)
        fw = 'bold' if label == 'N' else 'normal'
        fs = 10 if label == 'N' else 8
        ax.text(x + dx*radius*0.92, y + dy*radius*0.92, label,
                transform=transform, fontsize=fs, fontweight=fw,
                color='#018847', ha='center', va='center', zorder=43)
    hh = radius * 0.28; hw = radius * 0.15
    tip_y = y + radius * 0.78; base_y = tip_y - hh
    tri = mpatches.Polygon([[x, tip_y], [x-hw, base_y], [x+hw, base_y]],
                            closed=True, transform=transform,
                            facecolor='#018847', edgecolor='none', zorder=44)
    ax.add_patch(tri)
    ax.scatter([x], [y], transform=transform, color='#018847',
               s=10, zorder=45, linewidth=0)


# =============================================================================
# Scale bar — shared with original version
# =============================================================================
def draw_scale_bar(ax, lon_ref, lat_ref, length_km, label, transform=None,
                   color='#333333', linewidth=2.5, fontsize=7.5):
    """Draw an accurate scale bar at a given geographic position.

    Uses geodesic calculation to get the correct pixel length in the
    current map projection, then draws the bar in Axes coordinates.

    Parameters
    ----------
    ax : GeoAxes
    lon_ref, lat_ref : float
        Reference point (PlateCarree) for the left end of the scale bar.
    length_km : float
        Desired scale bar length in kilometres.
    label : str
        Label text (e.g. '500 km').
    transform : Transform, optional
        Transform for drawing (default: ax.transAxes).
    """
    geod = pyproj.Geod(ellps='WGS84')
    # Calculate end point at given distance along latitude parallel
    az_fwd = 90  # due east
    lon_end, lat_end, _ = geod.fwd(lon_ref, lat_ref, az_fwd, length_km * 1000)

    # Convert geographic -> map projection -> Axes coordinates
    map_proj = ax.projection
    x0, y0 = map_proj.transform_point(lon_ref, lat_ref, ccrs.PlateCarree())
    x1, y1 = map_proj.transform_point(lon_end, lat_end, ccrs.PlateCarree())

    # Convert data coords -> Axes fraction coords via the Axes transform chain
    ax_to_data = ax.transAxes.inverted()
    pts_data = np.array([[x0, y0], [x1, y1]])
    pts_axes = ax.transData.transform(pts_data)
    pts_ax_fraction = ax_to_data.transform(pts_axes)

    x0a, y0a = pts_ax_fraction[0]
    x1a, y1a = pts_ax_fraction[1]

    # Confirm the bar looks horizontal; use the y from the reference point
    bar_y = y0a
    # Adjust x range to keep it centred around the reference
    dx = x1a - x0a
    x_left = x0a

    # Draw
    if transform is None:
        transform = ax.transAxes

    # Bar
    ax.plot([x_left, x_left + dx], [bar_y, bar_y],
            transform=transform, color=color, linewidth=linewidth,
            solid_capstyle='butt', zorder=50)
    # End ticks
    tick_h = 0.006
    for bx in [x_left, x_left + dx]:
        ax.plot([bx, bx], [bar_y - tick_h, bar_y + tick_h],
                transform=transform, color=color, linewidth=linewidth * 0.6,
                solid_capstyle='round', zorder=50)
    # Label
    ax.text(x_left + dx / 2, bar_y - 0.012, label,
            transform=transform, fontsize=fontsize, color=color,
            ha='center', va='top', zorder=50,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.75, edgecolor='none'))


# =============================================================================
# Sector color scheme
# =============================================================================
SECTOR_COLORS = {
    'Pacific':  '#5B6EA5',   # blue-purple
    'Atlantic': '#C08090',   # pink-rose
    'Core':     '#888888',   # gray
}

# Thesis seas (7 regions) — bold, sector-colored labels
THESIS_SEAS = [
    # name, lon, lat, sector
    ('楚科奇海',   -170, 74.5, 'Pacific'),
    ('白令海',     -173, 61.5, 'Pacific'),
    ('巴伦支海',    42, 74,   'Atlantic'),
    ('喀拉海',      69, 76,   'Atlantic'),
    ('拉普捷夫海', 128, 75,   'Atlantic'),
    ('格陵兰海',   -18, 72,   'Atlantic'),
    ('中北冰洋',     5, 87,   'Core'),
]

# Context seas — 仅保留北极圈内紧邻研究区的海域
CONTEXT_SEAS = [
    ('东西伯利亚海', 155, 74),
]


# =============================================================================
# Main
# =============================================================================
def main():
    # ---- Create figure ----
    map_proj = ccrs.NorthPolarStereo(central_longitude=0)
    fig, ax = plt.subplots(figsize=(9.5, 9.5),
                           subplot_kw={'projection': map_proj})
    ax.set_extent([-180, 180, 58, 90], crs=ccrs.PlateCarree())
    ax.set_aspect('equal')  # 确保北极点位于图片正中心

    # ---- 10m resolution basemap (clean, no region fills) ----
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),
                   facecolor='#D4EAF5', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),
                   facecolor='#EAEAE2', edgecolor='#AAAAAA',
                   linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                   linewidth=0.6, edgecolor='#888888', zorder=4)

    # ---- Latitude parallels ----
    for lat_v in np.arange(55, 91, 5):
        t = np.linspace(-180, 180, 361)
        lw = 0.6 if lat_v % 10 == 0 else 0.25
        ax.plot(t, np.full_like(t, lat_v),
                color='#AAAAAA', linewidth=lw, linestyle='--', alpha=0.45,
                transform=ccrs.PlateCarree(), zorder=6)

    # ---- Longitude meridians ----
    for lon_v in range(0, 360, 30):
        l = np.linspace(58, 90, 81)
        lon_d = lon_v if lon_v <= 180 else lon_v - 360
        ax.plot(np.full_like(l, lon_d), l,
                color='#AAAAAA', linewidth=0.5, linestyle='--', alpha=0.45,
                transform=ccrs.PlateCarree(), zorder=6)

    # ---- Latitude labels (on ~5°E) ----
    for lat_v in [60, 70, 80]:
        ax.text(8, lat_v + 0.55, f'{lat_v}°N', transform=ccrs.PlateCarree(),
                fontsize=9, color='#777777', ha='left', va='bottom', zorder=10)

    # ---- Longitude labels (outer ring ~56°N) ----
    lon_markers = [
        (0,'0°','center','bottom'), (30,'30°E','left','bottom'),
        (60,'60°E','left','center'), (90,'90°E','left','center'),
        (120,'120°E','left','center'), (150,'150°E','left','top'),
        (180,'180°','center','top'),(-150,'150°W','right','top'),
        (-120,'120°W','right','center'),(-90,'90°W','right','center'),
        (-60,'60°W','right','center'),(-30,'30°W','right','bottom'),
    ]
    for lon_v, label, ha, va in lon_markers:
        ax.text(lon_v, 56, label, transform=ccrs.PlateCarree(),
                fontsize=9, color='#888888', ha=ha, va=va,
                fontstyle='italic', zorder=10)

    # ---- Thesis sea labels (bold, sector-colored) ----
    for name, lon, lat, sector in THESIS_SEAS:
        color = SECTOR_COLORS[sector]
        ax.text(lon, lat, name, transform=ccrs.PlateCarree(),
                fontsize=10, ha='center', va='center', fontweight='bold',
                color=color, zorder=14,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.85, edgecolor='none'))

    # ---- Context sea labels (lighter, gray) ----
    for name, lon, lat in CONTEXT_SEAS:
        ax.text(lon, lat, name, transform=ccrs.PlateCarree(),
                fontsize=10, ha='center', va='center', fontweight='normal',
                color='#999999', zorder=12,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.75, edgecolor='none'))

    # ---- Sector annotations (italic, lightweight, no bounding box) ----
    ax.text(-160, 55, '太平洋扇区', transform=ccrs.PlateCarree(),
            fontsize=10, ha='center', color='#5B6EA5', fontweight='bold',
            fontstyle='italic', zorder=14)
    ax.text(55, 55, '大西洋扇区', transform=ccrs.PlateCarree(),
            fontsize=10, ha='center', color='#C08090', fontweight='bold',
            fontstyle='italic', zorder=14)

    # ---- Legend (sector color key) ----
    legend_items = [
        Line2D([0], [0], color=SECTOR_COLORS['Pacific'],  lw=2.5,
               label='太平洋扇区'),
        Line2D([0], [0], color=SECTOR_COLORS['Atlantic'], lw=2.5,
               label='大西洋扇区'),
        Line2D([0], [0], color=SECTOR_COLORS['Core'],     lw=2.5,
               label='核心区'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=9, frameon=False)

    # ---- Data attribution (subtle, italics) ----
    ax.text(0.98, 0.015,
            '数据: Natural Earth 10m  •  投影: 北极球极立体投影  •  坐标: WGS 84',
            transform=ax.transAxes, fontsize=9, color='#AAAAAA',
            ha='right', va='bottom', fontstyle='italic', zorder=15)

    # ---- Save to all target paths ----
    plt.tight_layout(pad=0.8)

    # Star directory (primary target)
    star_dir = os.path.join(OUT_DIR, 'star')
    os.makedirs(star_dir, exist_ok=True)
    save_png_star = os.path.join(star_dir, 'fig_ch2_arctic_map.png')
    save_svg_star = os.path.join(star_dir, 'fig_ch2_arctic_map.svg')

    # Paper directory (secondary)
    save_png = os.path.join(OUT_DIR, 'fig_ch2_arctic_map.png')
    save_svg = os.path.join(OUT_DIR, 'fig_ch2_arctic_map.svg')

    for path in [save_png_star, save_svg_star, save_png, save_svg]:
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {path}')
    plt.close(fig)

    print(f'\n=== 图 2.1 北极海域研究区地图（清洁底图） ===')
    print(f'  Basemap: Natural Earth 10m (cartopy)')
    print(f'  Projection: North Polar Stereographic — 北极点居中')
    print(f'  Extent: 58°N–90°N  |  All-vector — 8 个可编辑文本标签')
    print(f'  Grid: 5°×30° lat/lon')
    print(f'  Thesis seas: 7 个 (bold + sector colour)')
    print(f'  Context seas: 1 个 (gray — 东西伯利亚海)')
    print('Done.')


if __name__ == "__main__":
    main()
