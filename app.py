import io
import re
import socket
import streamlit as st
import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import folium
import qrcode
from streamlit_folium import st_folium
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from branca.element import MacroElement
from jinja2 import Template
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
import time


class MapLegend(MacroElement):
    """A fixed-position, collapsible HTML legend rendered as a real Leaflet
    control -- click the title bar to hide the body and see the map underneath.

    A plain folium.Element added via map.get_root().html.add_child() is
    invisible under streamlit_folium: its serializer (_get_map_string /
    _get_siblings) only walks the map's own direct children, not elements
    nested under the root Figure's .html. Implementing the legend as an
    L.control (the same mechanism folium.LayerControl itself uses) makes it
    a direct child of the map, so it survives that serialization correctly.
    """

    def __init__(self, html, title="Legend"):
        super().__init__()
        self._name = "MapLegend"
        self.html = html
        self.title = title
        self._template = Template("""
        {% macro script(this, kwargs) %}
        var legend_{{ this.get_name() }} = L.control({position: 'bottomleft'});
        legend_{{ this.get_name() }}.onAdd = function (map) {
            var container = L.DomUtil.create('div', 'info legend');
            container.style.cssText = 'background:white;border:1px solid #ccc;border-radius:6px;' +
                'box-shadow:2px 2px 6px rgba(0,0,0,.2);max-width:320px;';
            var header = L.DomUtil.create('div', 'legend-header', container);
            header.style.cssText = 'padding:8px 12px;font-weight:700;font-size:12px;cursor:pointer;' +
                'user-select:none;font-family:Arial,sans-serif;';
            var arrowId = 'legend_arrow_{{ this.get_name() }}';
            header.innerHTML = '{{ this.title }} <span id="' + arrowId + '">&#9662;</span>';
            var body = L.DomUtil.create('div', 'legend-body', container);
            body.style.cssText = 'padding:0 14px 10px 14px;font-size:11px;line-height:1.9;' +
                'max-height:420px;overflow-y:auto;';
            body.innerHTML = `{{ this.html }}`;
            header.onclick = function () {
                var hidden = body.style.display === 'none';
                body.style.display = hidden ? 'block' : 'none';
                document.getElementById(arrowId).innerHTML = hidden ? '&#9662;' : '&#9656;';
            };
            L.DomEvent.disableClickPropagation(container);
            L.DomEvent.disableScrollPropagation(container);
            return container;
        };
        legend_{{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """)


class IconZoomScaler(MacroElement):
    """Shrinks context-layer icons (crossings/signals/hazards/landmarks -- the
    categories dense enough to overlap and hide the streets/route lines
    underneath, per direct user feedback) as the map zooms out, instead of
    rendering every icon at a fixed pixel size regardless of scale. Targets
    only ".ctx-icon-inner" wrapper elements (see _ctx_marker) -- the school
    marker and personal-feeder home marker are deliberately left full-size
    since they're few in number and act as wayfinding anchors, not clutter.

    Re-applied on 'overlayadd' too, not just 'zoomend': a hidden layer
    (show=False) has none of its marker DOM until its layer-control checkbox
    is switched on, so a freshly-revealed layer needs the current scale
    applied to it as well, not just future zoom changes."""

    def __init__(self, base_zoom=16, min_scale=0.35, step=0.15):
        super().__init__()
        self._name = "IconZoomScaler"
        self.base_zoom = base_zoom
        self.min_scale = min_scale
        self.step = step
        self._template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            function applyScale() {
                var z = map.getZoom();
                var scale = Math.min(1, Math.max({{ this.min_scale }},
                    1 - ({{ this.base_zoom }} - z) * {{ this.step }}));
                document.querySelectorAll('.ctx-icon-inner').forEach(function (el) {
                    el.style.transform = 'scale(' + scale + ')';
                });
            }
            map.on('zoomend', applyScale);
            map.on('overlayadd', applyScale);
            map.whenReady(applyScale);
            setTimeout(applyScale, 300);
        })();
        {% endmacro %}
        """)

st.set_page_config(
    page_title="Schulwegplan Berlin — Mitte",
    layout="wide",
)

INTERIM_DIR = Path(__file__).parent / "data" / "interim"

# ── Style constants ─────────────────────────────────────────────────────────
PB_STYLE = {
    "fillColor": "none", "color": "#444444",
    "weight": 1.5, "dashArray": "5, 5", "fillOpacity": 0,
}
ISO_STYLES = {
    # Purple family (ColorBrewer "Purples") -- deliberately not green, since Feeder Route
    # (TIER_STYLES) is now green and the two would otherwise blend together on the map.
    15: {"fillColor": "#dadaeb", "color": "#9e9ac8", "weight": 1.5, "fillOpacity": 0.20},
    10: {"fillColor": "#bcbddc", "color": "#807dba", "weight": 1.5, "fillOpacity": 0.25},
     5: {"fillColor": "#6a51a3", "color": "#6a51a3", "weight": 2.0, "fillOpacity": 0.35},
}
ISO_LABELS = {5: "5-min walk", 10: "10-min walk", 15: "15-min walk"}
# Same hue family as the reference pie chart ("color palette.png": light blue / orange /
# pale green for Primary / Secondary / Feeder), darkened+saturated so they read clearly
# against the basemap instead of the original red-orange-yellow gradient.
TIER_STYLES = {
    3: {"color": "#4aa329", "weight": 1.5, "opacity": 0.80},
    2: {"color": "#2480a8", "weight": 3.0, "opacity": 0.85},
    1: {"color": "#e88417", "weight": 5.0, "opacity": 0.95},
}
TIER_NAMES = {1: "Primary Spine", 2: "Secondary Spine", 3: "Feeder Route"}

# ── Icon colours shared with the (removed) static-export code's palette ─────
# Kept because the live map's own compact badge icons (SEVERITY_ICON/SIGN_ICON/
# LANDMARK_ICON below) reuse these exact hex values.
EXPORT_SEVERITY_COLORS = {"fatal": "#7b0000", "serious_injury": "#e67e22", "slight_injury": "#b7950b"}
EXPORT_SIGN_COLOR = "orange"
EXPORT_LANDMARK_COLORS = {
    "Bushaltestelle": "#2ecc71", "U-Bahn": "#2255a4", "S-Bahn": "#1b8a3c",
    "Apotheke": "#c0392b", "Krankenhaus": "#922b21", "Feuerwehr": "#a93226",
    "Spielplatz": "#2ecc71", "Kindergarten": "#7d6608", "Supermarkt": "#8e44ad",
    "Tankstelle": "#e74c3c", "Sportplatz": "#2c3e50",
}

# ── Personalised home-to-spine feeder (notebook 6.1) ─────────────────────────
# Same read-only contract as the notebook: the frozen jenks/spine file is only
# ever opened for reading here. Personal feeders live in their own file/layer.
CRS_METRIC = "EPSG:25833"
CRS_WGS84  = "EPSG:4326"
WALK_SPEED_M_MIN = 80                # m/min -- from 4.3_Walking_Time_Catchment_Modeling
ACCIDENT_BUFFER_M = 25               # m -- from 2.1_Master_Impedance_Equation's M_hazard buffer
SNAP_WARN_DIST_M = 200               # m -- from 4.1_Spatial_Catchment_Allocation's snap-quality warning
CATCHMENT_CELL_HALF_DIAG_M = 71.0    # 100 m grid cell half-diagonal
PERSONAL_FEEDER_STYLE = {"color": "#ffcc00", "weight": 1.5, "opacity": 1.0}
PERSONAL_FEEDERS_GPKG = INTERIM_DIR / "6.1_personal_feeders_utm33n.gpkg"
PERSONAL_FEEDER_SCHEMA = [
    "feeder_id", "raw_input", "school_idx", "school_name",
    "home_x", "home_y", "home_node", "snap_dist_m",
    "in_catchment", "assigned_school_name", "forced",
    "target_node", "tier_joined", "tier_joined_name",
    "length_m", "walk_time_min",
    "n_crossings", "n_traffic_signals", "n_hazard_points",
    "geometry",
]


class UnreachableHome(Exception):
    """Raised when a home cannot reach the school's spine through the network."""

# ── Context-layer symbols ────────────────────────────────────────────────────
# Icon design cross-checked against two real official Berlin Schulwegpläne:
# Hansa-Grundschule (CÖGA mbH / Landesverkehrswacht Berlin house style) and
# Bötzow-Grundschule (Steremat AFS GmbH, itself OSM-based like this model) --
# both downloaded from berlin.de/ba-mitte and boetzow-grundschule.de. Shared
# conventions from both: a mini red/yellow/green traffic-light glyph for
# Ampel, a red warning triangle for Gefahrenstelle, a green "H" badge for
# Bushaltestelle, and a green "S" badge for S-Bahn (the real German transit
# colours). The Bötzow plan additionally uses plain solid-colour dots for
# simple POI categories (playground, kindergarten, supermarket, fuel
# station) rather than pictograms -- adopted here for the same categories.
def _badge_html(letter, bg):
    return (
        f'<div style="background:{bg};color:#fff;border:1.5px solid #fff;border-radius:50%;'
        f'width:14px;height:14px;line-height:11px;text-align:center;font-weight:700;'
        f'font-family:Arial,sans-serif;font-size:8px;box-shadow:0 0 2px rgba(0,0,0,.6);">{letter}</div>'
    )


def _dot_html(color, size=10):
    return (
        f'<div style="background:{color};border:1.5px solid #fff;border-radius:50%;'
        f'width:{size}px;height:{size}px;box-shadow:0 0 2px rgba(0,0,0,.6);"></div>'
    )


def _emoji_badge_html(emoji, bg):
    return (
        f'<div style="background:{bg};border:1.5px solid #fff;border-radius:50%;'
        f'width:14px;height:14px;display:flex;align-items:center;justify-content:center;'
        f'font-size:8px;line-height:1;box-shadow:0 0 2px rgba(0,0,0,.6);">{emoji}</div>'
    )


def _severity_badge_html(color):
    return (
        f'<div style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'border:1.5px solid #fff;box-shadow:0 0 2px rgba(0,0,0,.6);display:flex;'
        f'align-items:center;justify-content:center;color:#fff;font-weight:700;'
        f'font-size:8px;line-height:1;">!</div>'
    )


def _sign_badge_html(color):
    return (
        f'<div style="width:10px;height:8px;background:{color};border:1.5px solid #fff;'
        f'border-radius:2px;box-shadow:0 0 2px rgba(0,0,0,.6);"></div>'
    )


def _crossing_stripe_html(accent):
    bars = ''.join(f'<div style="width:1.5px;height:7px;background:{accent};"></div>' for _ in range(3))
    return (
        f'<div style="background:#fff;border:1.5px solid {accent};border-radius:2px;'
        f'width:14px;height:11px;display:flex;align-items:center;justify-content:center;'
        f'gap:1.5px;box-shadow:0 0 2px rgba(0,0,0,.5);">{bars}</div>'
    )


TRAFFIC_LIGHT_HTML = (
    '<div style="background:#222;border-radius:3px;padding:1.5px 2px;display:inline-flex;'
    'flex-direction:column;gap:1px;box-shadow:0 0 2px rgba(0,0,0,.6);">'
    '<div style="width:6px;height:6px;border-radius:50%;background:#e74c3c;"></div>'
    '<div style="width:6px;height:6px;border-radius:50%;background:#f1c40f;"></div>'
    '<div style="width:6px;height:6px;border-radius:50%;background:#2ecc71;"></div>'
    '</div>'
)

CROSSING_ICON = {
    "traffic_signals": {"html": TRAFFIC_LIGHT_HTML},
    "zebra":            {"html": _crossing_stripe_html("#1a1a1a")},
    "refuge_island":    {"html": _crossing_stripe_html("#16a085")},
    "unmarked":         {"html": _crossing_stripe_html("#7f8c8d")},
}
# Hazard/sign/Feuerwehr/Kindergarten markers used to be classic AwesomeMarkers
# teardrop pins (~35x45px) -- by far the largest, most numerous icons on the
# map (every accident-hazard point gets one), so they were the single biggest
# contributor to the "icons cover the streets, the map isn't readable" problem.
# Switched to small HTML badges (same family as CROSSING_ICON/LANDMARK_ICON's
# other entries) so every context-layer marker is compact and, via _ctx_marker's
# wrapper div + IconZoomScaler below, can shrink further as you zoom out.
SEVERITY_ICON = {
    "fatal":          {"html": _severity_badge_html(EXPORT_SEVERITY_COLORS["fatal"])},
    "serious_injury": {"html": _severity_badge_html(EXPORT_SEVERITY_COLORS["serious_injury"])},
    "slight_injury":  {"html": _severity_badge_html(EXPORT_SEVERITY_COLORS["slight_injury"])},
}
LANDMARK_ICON = {
    "Bushaltestelle": {"html": _badge_html("H", "#2ecc71")},
    "U-Bahn":         {"html": _badge_html("U", "#2255a4")},
    "S-Bahn":         {"html": _badge_html("S", "#1b8a3c")},
    "Apotheke":       {"html": _badge_html("A", "#c0392b")},
    "Krankenhaus":    {"html": _emoji_badge_html("🏥", "#922b21")},
    "Feuerwehr":      {"html": _emoji_badge_html("🚒", EXPORT_LANDMARK_COLORS["Feuerwehr"])},
    "Spielplatz":     {"html": _dot_html("#2ecc71")},
    "Kindergarten":   {"html": _emoji_badge_html("🎓", EXPORT_LANDMARK_COLORS["Kindergarten"])},
    "Supermarkt":     {"html": _emoji_badge_html("🛒", "#8e44ad")},
    "Tankstelle":     {"html": _dot_html("#e74c3c")},
    "Sportplatz":     {"html": _emoji_badge_html("🏃", "#2c3e50")},
}
SIGNAL_ICON = {"html": TRAFFIC_LIGHT_HTML}
SIGN_ICON   = {"html": _sign_badge_html(EXPORT_SIGN_COLOR)}

# German StVO (Straßenverkehrs-Ordnung) traffic-sign catalogue -- decodes the raw
# OSM traffic_sign=DE:NNN[.N][-NN][,...] tag into a human-readable description
# instead of showing the bare catalogue number. Covers every base number that
# actually appears in this dataset (verified against official sign references);
# unrecognised numbers fall back to showing the raw code rather than guessing.
MAIN_SIGN_MEANINGS = {
    "120": "Verengte Fahrbahn (narrowed roadway)",
    "133": "Fußgänger (pedestrians warning)",
    "136": "Kinder (children warning)",
    "205": "Vorfahrt gewähren (yield)",
    "220": "Einbahnstraße (one-way street)",
    "222": "Vorgeschriebene Vorbeifahrt (pass on indicated side)",
    "224": "Haltestelle — Straßenbahn/Bus (tram/bus stop)",
    "229": "Taxenstand (taxi stand)",
    "237": "Radweg (mandatory cycle path)",
    "240": "Gemeinsamer Fuß- und Radweg (shared foot/cycle path)",
    "242": "Fußgängerzone (pedestrian zone)",
    "244": "Fahrradstraße (bicycle street)",
    "250": "Verbot für Fahrzeuge aller Art (no vehicles)",
    "262": "Verbot für Fahrzeuge über angegebene Gesamtmasse (weight limit)",
    "265": "Verbot für Fahrzeuge über angegebene Höhe (height limit)",
    "267": "Verbot der Einfahrt (no entry)",
    "274": "Zulässige Höchstgeschwindigkeit (speed limit)",
    "277": "Überholverbot für Kraftfahrzeuge über 3,5 t (overtaking ban, heavy vehicles)",
    "283": "Haltverbot (no stopping)",
    "314": "Parkplatz (parking)",
    "325": "Verkehrsberuhigter Bereich (traffic-calmed zone)",
    "357": "Sackgasse (dead end)",
    "437": "Straßennamensschild (street name sign)",
}
ZUSATZ_MEANINGS = {
    "1000": "Richtungspfeil (direction arrow)",
    "1008": "Vorfahrtregelung (right-of-way detail)",
    "1010": "zusätzliche Angabe mit Symbol (additional symbol detail)",
    "1020": "Ausnahme für bestimmte Personengruppen, z. B. Bewohner (exemption for a specific group, e.g. residents)",
    "1022": "Ausnahme für Radfahrer/Kleinkrafträder (exemption for cyclists/mopeds)",
    "1026": "Ausnahme für Taxi/Krankentransport (exemption for taxis/medical transport)",
    "1042": "zeitliche Beschränkung, z. B. werktags (time restriction, e.g. weekdays)",
    "1044": "Nur Bewohner mit Parkausweis (residents with parking permit only)",
    "1050": "Bevorzugung von Elektrofahrzeugen (priority for electric vehicles)",
    "1053": "mit Parkschein (paid parking ticket required)",
}


_HALTESTELLE_SIGN_RE = re.compile(r"^(?:DE|de):224(?:[.\-,;]|$)")


def _is_haltestelle_sign(raw_value):
    """Zeichen 224 (Haltestelle Straßenbahn/Bus) is redundant with the dedicated
    Bushaltestelle landmark layer -- verified 20/25 such sign points sit within
    15 m of an already-mapped highway=bus_stop node, i.e. the same physical
    stop tagged twice in OSM. Excluded here so it isn't shown as a duplicate."""
    return bool(_HALTESTELLE_SIGN_RE.match(str(raw_value).strip()))


def describe_traffic_sign(raw_value):
    """Translate an OSM traffic_sign tag value into a German StVO description,
    falling back to the raw code for catalogue numbers not in the lookup."""
    if raw_value is None or str(raw_value).strip().lower() in ("", "nan", "none"):
        return "Verkehrszeichen (unspecified)"
    raw = str(raw_value).strip()
    low = raw.lower()
    if low == "yes":
        return "Verkehrszeichen — nicht näher spezifiziert (sign present, type not recorded)"
    if low == "maxspeed":
        return "Geschwindigkeitsbegrenzung (speed limit, value not recorded)"
    if low == "verkehrsbeeinflussungsanlage":
        return "Verkehrsbeeinflussungsanlage (dynamic/variable message sign)"

    parts = []
    for part in re.split(r"[,;|]", raw):
        code = re.sub(r"^(DE|de):", "", part.strip()).strip()
        code = re.sub(r"\[[^\]]*\]", "", code).strip()
        if not code:
            continue
        m = re.match(r"^(\d+)", code)
        if not m:
            parts.append(code)
            continue
        base = m.group(1)
        if base.startswith("10") and len(base) == 4:
            desc = ZUSATZ_MEANINGS.get(base)
        else:
            desc = MAIN_SIGN_MEANINGS.get(base)
            if desc:
                if ".1" in code:
                    desc += " – Beginn (start)"
                elif ".2" in code:
                    desc += " – Ende (end)"
        parts.append(f"{code} – {desc}" if desc else code)

    return " · ".join(parts) if parts else raw

CONTEXT_LEGEND_HTML = f"""
  <div style="display:flex;align-items:center;gap:6px;">{TRAFFIC_LIGHT_HTML}<span>Ampel (traffic signal)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_crossing_stripe_html("#1a1a1a")}<span>Fußgänger Überweg — Zebrastreifen (zebra)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_crossing_stripe_html("#7f8c8d")}<span>Fußgänger Überweg — ungesichert (unmarked)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_crossing_stripe_html("#16a085")}<span>Fußgänger Überweg — Verkehrsinsel (refuge island)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_severity_badge_html(EXPORT_SEVERITY_COLORS["fatal"])}<span>Gefahrenstelle — tödlich (fatal)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_severity_badge_html(EXPORT_SEVERITY_COLORS["serious_injury"])}<span>Gefahrenstelle — schwer (serious)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_severity_badge_html(EXPORT_SEVERITY_COLORS["slight_injury"])}<span>Gefahrenstelle — leicht (slight)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_sign_badge_html(EXPORT_SIGN_COLOR)}<span>Verkehrszeichen (traffic sign, decoded on click)</span></div>
  <hr style="margin:4px 0">
  <div style="display:flex;align-items:center;gap:6px;">{_badge_html("H", "#2ecc71")}<span>Haltestelle Bus/Tram</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_badge_html("U", "#2255a4")}<span>U-Bahn</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_badge_html("S", "#1b8a3c")}<span>S-Bahn</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_badge_html("A", "#c0392b")}<span>Apotheke (pharmacy)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_emoji_badge_html("🏥", "#922b21")}<span>Krankenhaus (hospital)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_emoji_badge_html("🚒", EXPORT_LANDMARK_COLORS["Feuerwehr"])}<span>Feuerwehr (fire station)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_dot_html("#2ecc71")}<span>Spielplatz (playground)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_emoji_badge_html("🎓", EXPORT_LANDMARK_COLORS["Kindergarten"])}<span>Kindergarten</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_emoji_badge_html("🛒", "#8e44ad")}<span>Supermarkt</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_dot_html("#e74c3c")}<span>Tankstelle (fuel station)</span></div>
  <div style="display:flex;align-items:center;gap:6px;">{_emoji_badge_html("🏃", "#2c3e50")}<span>Sportplatz (sports pitch)</span></div>
"""


# ── Data loader — cached for the lifetime of the server process ─────────────
@st.cache_resource
def load_all_data():
    d = {}

    # 1. Mitte boundary
    pb_path = INTERIM_DIR / "mitte_boundary_utm33n.gpkg"
    if pb_path.exists():
        pb_gdf = gpd.read_file(pb_path).to_crs("EPSG:4326")
        c = pb_gdf.geometry.iloc[0].centroid
        d["pb_geojson"]  = pb_gdf.geometry.iloc[0].__geo_interface__
        d["map_center"]  = [c.y, c.x]
    else:
        d["pb_geojson"] = None
        d["map_center"] = [52.533, 13.366]

    # 2. Schools
    schools = gpd.read_file(INTERIM_DIR / "grundschulen_access_points_utm33n.gpkg")
    schools = schools.reset_index(drop=True)
    schools["school_idx"] = schools.index
    d["schools"]      = schools
    d["schools_4326"] = schools.to_crs("EPSG:4326")

    # 3. Isochrones
    iso_data = {}
    iso_path = INTERIM_DIR / "walktime_isochrones_utm33n.gpkg"
    if iso_path.exists():
        iso_gdf = gpd.read_file(iso_path).to_crs("EPSG:4326")
        for _, row in iso_gdf.iterrows():
            iso_data.setdefault(int(row["school_idx"]), {})[int(row["minutes"])] = row.geometry
    d["iso_data"] = iso_data

    # 4. Routes + route stats
    school_routes = {}
    school_stats  = {}
    routes_path = INTERIM_DIR / "school_specific_routes_jenks_utm33n.gpkg"
    if routes_path.exists():
        rt = gpd.read_file(routes_path)
        for _, srow in schools.iterrows():
            sidx     = int(srow["school_idx"])
            vol_col  = f"vol_school_{sidx}"
            tier_col = f"tier_jenks_school_{sidx}"
            if vol_col not in rt.columns:
                continue
            active = rt[rt[vol_col] > 0][[tier_col, "geometry"]].copy()
            active.rename(columns={tier_col: "tier"}, inplace=True)
            active = active[active["tier"].isin([1, 2, 3])]
            if len(active):
                school_routes[sidx] = active.to_crs("EPSG:4326")
            if tier_col in rt.columns:
                school_stats[sidx] = {
                    "primary_km"  : round(rt[rt[tier_col] == 1]["length_m"].sum() / 1000, 1),
                    "secondary_km": round(rt[rt[tier_col] == 2]["length_m"].sum() / 1000, 1),
                }
        del rt
    d["school_routes"] = school_routes
    d["school_stats"]  = school_stats

    # 5. Catchment + children stats
    school_catchments = {}
    ca_path = INTERIM_DIR / "catchment_allocation_utm33n.gpkg"
    if ca_path.exists():
        ca = gpd.read_file(ca_path)
        ca_4326 = ca[["assigned_school_idx", "children_6_15",
                       "network_dist_m", "geometry"]].to_crs("EPSG:4326")
        for sidx_val in ca_4326["assigned_school_idx"].dropna().unique():
            sidx     = int(sidx_val)
            sub_4326 = ca_4326[ca_4326["assigned_school_idx"] == sidx_val]
            school_catchments[sidx] = sub_4326[["children_6_15", "geometry"]].reset_index(drop=True)
            sub_utm   = ca[ca["assigned_school_idx"] == sidx_val]
            valid_utm = sub_utm[sub_utm["network_dist_m"] >= 0]
            school_stats.setdefault(sidx, {}).update({
                "children"    : int(sub_utm["children_6_15"].sum()),
                "cells"       : len(sub_utm),
                "mean_dist_m" : int(valid_utm["network_dist_m"].mean()) if len(valid_utm) else 0,
            })
        del ca, ca_4326
    d["school_catchments"] = school_catchments

    # 5b. Isochrone area stats (UTM source for accuracy)
    if iso_path.exists():
        iso_utm = gpd.read_file(iso_path)
        for _, row in iso_utm.iterrows():
            sidx = int(row["school_idx"])
            T    = int(row["minutes"])
            school_stats.setdefault(sidx, {})[f"iso_{T}_km2"] = round(row.geometry.area / 1e6, 2)
        del iso_utm

    # 6. Context location layers (from notebook 1.5) — 15 toggleable categories (4.5)
    ctx_path = INTERIM_DIR / "context_location_layers_utm33n.gpkg"
    if ctx_path.exists():
        d["ctx_crossings"] = gpd.read_file(ctx_path, layer="crossings").to_crs("EPSG:4326")
        d["ctx_signals"]   = gpd.read_file(ctx_path, layer="traffic_signals").to_crs("EPSG:4326")
        d["ctx_signs"]     = gpd.read_file(ctx_path, layer="traffic_signs").to_crs("EPSG:4326")
        d["ctx_hazards"]   = gpd.read_file(ctx_path, layer="hazard_points").to_crs("EPSG:4326")
        d["ctx_landmarks"] = gpd.read_file(ctx_path, layer="landmarks").to_crs("EPSG:4326")
    else:
        d["ctx_crossings"] = d["ctx_signals"] = d["ctx_signs"] = d["ctx_hazards"] = d["ctx_landmarks"] = None

    return d


# ── Personalised home-to-spine feeder: routing resources & logic (notebook 6.1) ──
@st.cache_resource
def load_routing_resources():
    """Everything needed to route a personal feeder, built once per server process.

    Same snapping method, same impedance-weighted graph, same frozen spine/catchment
    reads as notebooks 2.3 / 4.1 / 4.2 / 6.1 -- never recomputed, only read.
    """
    nodes_gdf = gpd.read_file(INTERIM_DIR / "osm_pedestrian_network_utm33n.gpkg", layer="nodes").set_index("osmid")
    edges_gdf = gpd.read_file(INTERIM_DIR / "routes_flow_utm33n.gpkg")
    jenks_gdf = gpd.read_file(INTERIM_DIR / "school_specific_routes_jenks_utm33n.gpkg")   # FROZEN -- read-only
    catchment_gdf = gpd.read_file(INTERIM_DIR / "catchment_allocation_utm33n.gpkg", layer="catchment_allocation_utm33n")
    ctx_path = INTERIM_DIR / "context_location_layers_utm33n.gpkg"
    ctx_crossings = gpd.read_file(ctx_path, layer="crossings")
    ctx_signals   = gpd.read_file(ctx_path, layer="traffic_signals")
    ctx_hazards   = gpd.read_file(ctx_path, layer="hazard_points")
    ctx_landmarks = gpd.read_file(ctx_path, layer="landmarks")   # native CRS_METRIC, for the route-plan export

    # Street names for the route-plan export's labels/hazard captions. routes_flow_utm33n.gpkg
    # (the impedance-weighted routing graph) has no "name" column, but its (u, v) node pairs are
    # 100%-verified to match osm_pedestrian_network_utm33n.gpkg's "edges" layer, which does --
    # so street names are looked up from that layer rather than invented or geometrically guessed.
    osm_edges_named = gpd.read_file(INTERIM_DIR / "osm_pedestrian_network_utm33n.gpkg", layer="edges")
    edge_name_lookup, node_street_lookup = {}, {}
    for row in osm_edges_named.itertuples(index=False):
        name = getattr(row, "name", None)
        if isinstance(name, str) and name.strip():
            u, v = int(row.u), int(row.v)
            edge_name_lookup.setdefault((u, v), name)
            edge_name_lookup.setdefault((v, u), name)
            node_street_lookup.setdefault(u, name)
            node_street_lookup.setdefault(v, name)
    del osm_edges_named

    G = nx.DiGraph()
    edge_geom_lookup, edge_length_lookup = {}, {}
    for row in edges_gdf.itertuples(index=False):
        u, v, w, length_m, geom = int(row.u), int(row.v), float(row.impedance), float(row.length_m), row.geometry
        G.add_edge(u, v, impedance=w, length_m=length_m)
        G.add_edge(v, u, impedance=w, length_m=length_m)
        edge_geom_lookup[(u, v)] = geom
        edge_geom_lookup[(v, u)] = geom
        edge_length_lookup[(u, v)] = length_m
        edge_length_lookup[(v, u)] = length_m

    node_ids = nodes_gdf.index.to_numpy(dtype=np.int64)
    node_xy  = np.column_stack([nodes_gdf.geometry.x.values, nodes_gdf.geometry.y.values])
    node_tree = cKDTree(node_xy)

    catchment_xy = np.column_stack([catchment_gdf.geometry.x.values, catchment_gdf.geometry.y.values])
    catchment_tree = cKDTree(catchment_xy)

    return dict(
        G=G, edge_geom_lookup=edge_geom_lookup, edge_length_lookup=edge_length_lookup,
        node_ids=node_ids, node_tree=node_tree,
        jenks_gdf=jenks_gdf, catchment_gdf=catchment_gdf, catchment_tree=catchment_tree,
        ctx_crossings=ctx_crossings, ctx_signals=ctx_signals, ctx_hazards=ctx_hazards,
        ctx_landmarks=ctx_landmarks, edge_name_lookup=edge_name_lookup, node_street_lookup=node_street_lookup,
    )


@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="schulwegplan-berlin-mitte-app", timeout=5)


def wgs84_to_metric(lat, lon):
    pt = gpd.GeoSeries([Point(lon, lat)], crs=CRS_WGS84).to_crs(CRS_METRIC).iloc[0]
    return pt.x, pt.y


def snap_to_node(r, x, y):
    dist, idx = r["node_tree"].query([x, y], k=1)
    return int(r["node_ids"][idx]), float(dist)


def validate_catchment(r, x, y, school_idx):
    dist, idx = r["catchment_tree"].query([x, y], k=1)
    cell = r["catchment_gdf"].iloc[idx]
    assigned_idx  = int(cell["assigned_school_idx"])
    assigned_name = str(cell["assigned_school"])
    if dist > CATCHMENT_CELL_HALF_DIAG_M:
        return {"in_catchment": None, "assigned_school_idx": None, "assigned_school_name": None, "cell_dist_m": float(dist)}
    return {
        "in_catchment": assigned_idx == school_idx,
        "assigned_school_idx": assigned_idx, "assigned_school_name": assigned_name,
        "cell_dist_m": float(dist),
    }


def spine_nodes_for_school(r, school_idx):
    tier_col = f"tier_jenks_school_{school_idx}"
    spine_edges = r["jenks_gdf"][r["jenks_gdf"][tier_col].isin([1, 2])]
    node_tier = {}
    for u, v, tier in zip(spine_edges["u"], spine_edges["v"], spine_edges[tier_col]):
        for n in (int(u), int(v)):
            if n not in node_tier or tier < node_tier[n]:   # 1 (Primary) beats 2 (Secondary)
                node_tier[n] = int(tier)
    return node_tier


def route_home_to_spine(r, home_node, spine_node_tier):
    lengths, paths = nx.single_source_dijkstra(r["G"], home_node, weight="impedance")
    reachable_spine = set(spine_node_tier) & set(lengths)
    if not reachable_spine:
        raise UnreachableHome(
            "This home cannot reach the school's Primary/Secondary spine through the "
            "network (barrier, gap, or home outside the mapped network)."
        )
    target_node = min(reachable_spine, key=lambda n: lengths[n])
    node_path = paths[target_node]
    total_length_m = sum(
        r["edge_length_lookup"].get((node_path[i], node_path[i + 1]), 0.0) for i in range(len(node_path) - 1)
    )
    return target_node, node_path, total_length_m


def path_geometry(r, node_path):
    segs = [r["edge_geom_lookup"][(node_path[i], node_path[i + 1])] for i in range(len(node_path) - 1)]
    coords = []
    for seg in segs:
        pts = list(seg.coords)
        if coords and coords[-1] == pts[0]:
            coords.extend(pts[1:])
        elif coords and coords[-1] == pts[-1]:
            coords.extend(list(reversed(pts))[1:])
        else:
            coords.extend(pts)
    return LineString(coords)


def _buffer_gdf(geom_metric, buffer_m):
    buf = gpd.GeoSeries([geom_metric], crs=CRS_METRIC).buffer(buffer_m).iloc[0]
    return gpd.GeoDataFrame(geometry=[buf], crs=CRS_METRIC)


def hazards_along_route(r, route_geom):
    buf_gdf = _buffer_gdf(route_geom, ACCIDENT_BUFFER_M)
    crossings_near = gpd.sjoin(r["ctx_crossings"], buf_gdf, predicate="within", how="inner")
    signals_near   = gpd.sjoin(r["ctx_signals"], buf_gdf, predicate="within", how="inner")
    hazards_near   = gpd.sjoin(r["ctx_hazards"], buf_gdf, predicate="within", how="inner")
    return {
        "n_crossings": len(crossings_near),
        "n_traffic_signals": len(signals_near),
        "n_hazard_points": len(hazards_near),
    }


def _load_or_init_personal_feeders():
    if PERSONAL_FEEDERS_GPKG.exists():
        return gpd.read_file(PERSONAL_FEEDERS_GPKG)
    return gpd.GeoDataFrame({c: [] for c in PERSONAL_FEEDER_SCHEMA}, geometry="geometry", crs=CRS_METRIC)


def add_personal_feeder(school_idx, school_name, home_x, home_y, raw_input, force=False):
    """Add one personalised home-to-spine feeder. Returns a status/report dict.

    status is one of: "ok", "flagged_catchment", "flagged_unreachable". On "ok" the
    feeder is appended to st.session_state["personal_feeders"] and saved to the
    notebook-6-owned 6.1_personal_feeders_utm33n.gpkg -- the frozen jenks/spine file
    is only ever read here, never written.
    """
    r = load_routing_resources()
    home_node, snap_dist = snap_to_node(r, home_x, home_y)

    cm = validate_catchment(r, home_x, home_y, school_idx)
    if cm["in_catchment"] is not True and not force:
        reason = ("outside the mapped catchment grid entirely" if cm["in_catchment"] is None
                   else f"assigned to {cm['assigned_school_name']!r}, not {school_name!r}")
        return {"status": "flagged_catchment", "message": f"Home is {reason}.", "detail": cm}

    spine_node_tier = spine_nodes_for_school(r, school_idx)
    try:
        target_node, node_path, total_length_m = route_home_to_spine(r, home_node, spine_node_tier)
    except UnreachableHome as e:
        return {"status": "flagged_unreachable", "message": str(e)}

    geom = path_geometry(r, node_path) if len(node_path) > 1 else Point(home_x, home_y)
    hazards = (hazards_along_route(r, geom) if len(node_path) > 1
               else {"n_crossings": 0, "n_traffic_signals": 0, "n_hazard_points": 0})
    tier_joined = spine_node_tier[target_node]
    walk_time_min = total_length_m / WALK_SPEED_M_MIN

    feeders = st.session_state["personal_feeders"]
    feeder_id = f"feeder_{len(feeders):04d}"
    row = {
        "feeder_id": feeder_id, "raw_input": raw_input,
        "school_idx": school_idx, "school_name": school_name,
        "home_x": home_x, "home_y": home_y, "home_node": home_node, "snap_dist_m": snap_dist,
        "in_catchment": bool(cm["in_catchment"]) if cm["in_catchment"] is not None else None,
        "assigned_school_name": cm["assigned_school_name"], "forced": force,
        "target_node": target_node, "tier_joined": tier_joined, "tier_joined_name": TIER_NAMES[tier_joined],
        "length_m": total_length_m, "walk_time_min": walk_time_min,
        "n_crossings": hazards["n_crossings"], "n_traffic_signals": hazards["n_traffic_signals"],
        "n_hazard_points": hazards["n_hazard_points"],
        "geometry": geom,
    }
    new_gdf = gpd.GeoDataFrame([row], geometry="geometry", crs=CRS_METRIC)
    st.session_state["personal_feeders"] = pd.concat([feeders, new_gdf], ignore_index=True)
    st.session_state["personal_feeders"].to_file(PERSONAL_FEEDERS_GPKG, driver="GPKG")
    return {"status": "ok", "feeder_id": feeder_id, "row": row}


# ── Map builders ─────────────────────────────────────────────────────────────
def _add_boundary(m, pb_geojson):
    if pb_geojson:
        folium.GeoJson(
            pb_geojson, name="Mitte",
            style_function=lambda x: PB_STYLE,
            tooltip="Mitte",
        ).add_to(m)


def build_default_map(d):
    m = folium.Map(location=d["map_center"], zoom_start=14, tiles="OpenStreetMap", prefer_canvas=True)
    _add_boundary(m, d["pb_geojson"])
    return m


def _ctx_marker(group, row, style, tooltip):
    # The badge HTML is wrapped in a ".ctx-icon-inner" div rather than passed
    # straight to DivIcon -- IconZoomScaler scales THAT inner element via CSS
    # transform, never the outer DivIcon container itself. Leaflet repositions
    # the outer container with its own translate3d transform on every pan/zoom,
    # which would silently clobber (or be clobbered by) a transform set there
    # directly; the inner wrapper is never touched by Leaflet, so a scale set
    # on it persists correctly.
    wrapped = (
        '<div class="ctx-icon-inner" style="width:100%;height:100%;display:flex;'
        f'align-items:center;justify-content:center;">{style["html"]}</div>'
    )
    icon = folium.DivIcon(html=wrapped, icon_size=(14, 14), icon_anchor=(7, 7))
    folium.Marker(location=[row.geometry.y, row.geometry.x], icon=icon, tooltip=tooltip).add_to(group)


def add_context_layers(m, sidx, d):
    """Add the 15 context-marking categories (notebook 4.5) as icon markers,
    clipped to the school's own 15-minute isochrone. Each category is hidden
    by default (show=False) and toggled on via the map's layer-control
    checkboxes, so the base Schulwegplan isn't obscured on first load. Plain
    FeatureGroups throughout (not MarkerCluster) -- every point keeps its own
    real icon at its real position at every zoom level, rather than
    aggregating into a numbered cluster bubble when zoomed out. At low zoom
    this means densely-packed points overlap visually (position stays exact,
    just not all individually legible); at high zoom every point separates
    out and is exact, which is what actually matters here."""
    if d["ctx_crossings"] is None:
        return
    iso15 = d["iso_data"].get(sidx, {}).get(15)
    if iso15 is None:
        return
    clip = lambda gdf: gdf[gdf.within(iso15)]

    fg = folium.FeatureGroup(name="Pedestrian crossings", show=False)
    for row in clip(d["ctx_crossings"]).itertuples(index=False):
        style = CROSSING_ICON.get(row.crossing_type, CROSSING_ICON["unmarked"])
        _ctx_marker(fg, row, style, f"Crossing ({row.crossing_type})")
    fg.add_to(m)

    fg = folium.FeatureGroup(name="Traffic signals", show=False)
    for row in clip(d["ctx_signals"]).itertuples(index=False):
        _ctx_marker(fg, row, SIGNAL_ICON, "Traffic signal")
    fg.add_to(m)

    fg = folium.FeatureGroup(name="Traffic signs", show=False)
    for row in clip(d["ctx_signs"]).itertuples(index=False):
        if _is_haltestelle_sign(row.sign_value):
            continue
        _ctx_marker(fg, row, SIGN_ICON, describe_traffic_sign(row.sign_value))
    fg.add_to(m)

    # A separate toggleable layer per severity (rather than one combined
    # "Accident hazards" layer) so e.g. slight-injury hazards can be hidden
    # while fatal/serious stay visible.
    hazards_clipped = clip(d["ctx_hazards"])
    for severity in ("fatal", "serious_injury", "slight_injury"):
        fg = folium.FeatureGroup(name=f"Accident hazards ({severity.replace('_', ' ')})", show=False)
        style = SEVERITY_ICON.get(severity, SEVERITY_ICON["slight_injury"])
        for row in hazards_clipped[hazards_clipped["severity"] == severity].itertuples(index=False):
            _ctx_marker(fg, row, style, f"Accident ({severity}, {row.year})")
        fg.add_to(m)

    landmarks_clipped = clip(d["ctx_landmarks"])
    for ltype, style in LANDMARK_ICON.items():
        sub = landmarks_clipped[landmarks_clipped["landmark_type"] == ltype]
        fg = folium.FeatureGroup(name=ltype, show=False)
        for row in sub.itertuples(index=False):
            label = row.name if isinstance(getattr(row, "name", None), str) and row.name else ltype
            _ctx_marker(fg, row, style, label)
        fg.add_to(m)


def build_school_map(sidx, d):
    school_row = d["schools_4326"][d["schools_4326"]["school_idx"] == sidx].iloc[0]
    lat, lon   = school_row.geometry.y, school_row.geometry.x

    m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap", prefer_canvas=True)
    _add_boundary(m, d["pb_geojson"])

    # Isochrones (15 → 5, back to front)
    for T in [15, 10, 5]:
        geom = d["iso_data"].get(sidx, {}).get(T)
        if geom:
            folium.GeoJson(
                geom.__geo_interface__, name=ISO_LABELS[T],
                style_function=lambda x, s=ISO_STYLES[T]: s,
                tooltip=ISO_LABELS[T],
            ).add_to(m)

    # Route spines (feeder → secondary → primary, back to front)
    routes = d["school_routes"].get(sidx)
    if routes is not None:
        for tier in [3, 2, 1]:
            sub = routes[routes["tier"] == tier]
            if len(sub):
                folium.GeoJson(
                    sub[["geometry"]].__geo_interface__, name=TIER_NAMES[tier],
                    style_function=lambda x, s=TIER_STYLES[tier]: s,
                    tooltip=TIER_NAMES[tier],
                ).add_to(m)

    # Census grid cells
    catchment = d["school_catchments"].get(sidx)
    if catchment is not None:
        fg = folium.FeatureGroup(name="Census cells (children 6–15)")
        for row in catchment.itertuples(index=False):
            n = int(row.children_6_15) if row.children_6_15 > 0 else 0
            if n == 0:
                continue
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=max(4, min(14, n * 0.4)),
                color="#1f4e79", fill=True, fill_color="#2166ac",
                fill_opacity=0.65, weight=1,
                tooltip=f"{n} children (6–15)",
            ).add_to(fg)
        fg.add_to(m)

    # School marker
    addr_popup = ""
    if "addr_street" in school_row.index and school_row["addr_street"]:
        addr_popup = f"<br><small>{school_row['addr_street']} {school_row.get('addr_housenumber', '')}</small>"
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(f"<b>{school_row['name']}</b>{addr_popup}", max_width=220),
        icon=folium.Icon(color="red", icon="school", prefix="fa"),
        tooltip=school_row["name"],
    ).add_to(m)

    # Personal feeders (notebook 6.1 layer) -- own colour (yellow), distinct from
    # any TIER_STYLES colour so it doesn't blend into a same-tier route it runs
    # alongside (a dark green here was previously indistinguishable from the
    # Feeder Route tier it's usually adjacent to). A white halo underneath keeps
    # it visible against any basemap tile colour, and the home pin sits at the
    # home's own entered coordinates (not derived from the route geometry, which
    # isn't guaranteed to start exactly at the home end of its first edge).
    feeders = st.session_state.get("personal_feeders")
    if feeders is not None and len(feeders):
        sub = feeders[feeders["school_idx"] == sidx]
        if len(sub):
            sub_4326 = sub.to_crs("EPSG:4326")
            fg = folium.FeatureGroup(name="Personal Feeder(s)")
            for row in sub_4326.itertuples(index=False):
                tooltip = f"{row.raw_input} — {row.length_m:.0f} m, {row.walk_time_min:.1f} min, joins {row.tier_joined_name}"
                geom = row.geometry
                if geom.geom_type == "LineString":
                    latlon = [[c[1], c[0]] for c in geom.coords]
                    folium.PolyLine(latlon, color="#ffffff", weight=PERSONAL_FEEDER_STYLE["weight"] + 2, opacity=0.9).add_to(fg)
                    folium.PolyLine(
                        latlon, color=PERSONAL_FEEDER_STYLE["color"], weight=PERSONAL_FEEDER_STYLE["weight"],
                        opacity=PERSONAL_FEEDER_STYLE["opacity"], tooltip=tooltip,
                    ).add_to(fg)
                home_pt = gpd.GeoSeries([Point(row.home_x, row.home_y)], crs=CRS_METRIC).to_crs("EPSG:4326").iloc[0]
                folium.Marker(
                    location=[home_pt.y, home_pt.x],
                    icon=folium.Icon(color="darkgreen", icon="home", prefix="fa"),
                    tooltip=tooltip,
                ).add_to(fg)
            fg.add_to(m)

    add_context_layers(m, sidx, d)
    MapLegend(CONTEXT_LEGEND_HTML, title="Die Zeichen bedeuten").add_to(m)
    IconZoomScaler().add_to(m)

    # collapsed=True: the layer-toggle panel starts as a small icon (click to
    # expand/collapse) instead of permanently taking up map space, matching
    # the legend's own click-to-collapse behaviour above.
    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ── Personalised feeder sidebar form (notebook 6.1) ─────────────────────────
def render_personal_feeder_form(selection, schools):
    srow = schools[schools["name"] == selection].iloc[0]
    sidx = int(srow["school_idx"])

    st.markdown("**🏠 Personalised Feeder**")
    st.caption("Route one home to this school's existing spine — never alters the spine itself.")
    mode = st.radio("Home input", ["Address", "Coordinates"], horizontal=True, label_visibility="collapsed", key="pf_mode")

    if mode == "Address":
        home_text = st.text_input("Home address", placeholder="e.g. Turmstraße 5, Berlin", key="pf_address")
    else:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=52.5250, format="%.6f", key="pf_lat")
        lon = c2.number_input("Longitude", value=13.3600, format="%.6f", key="pf_lon")

    force = st.checkbox("Force (ignore catchment mismatch)", value=False, key="pf_force")

    if st.button("Add feeder", use_container_width=True, key="pf_submit"):
        if mode == "Address":
            if not home_text:
                st.warning("Enter an address first.")
            else:
                # A pasted "lat, lon" pair (e.g. "52.523898, 13.401099") isn't a
                # geocodable place name -- detect it and use it directly instead
                # of sending it to Nominatim, where it would just fail.
                coord_match = re.match(
                    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", home_text
                )
                if coord_match:
                    lat_val, lon_val = float(coord_match.group(1)), float(coord_match.group(2))
                    if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                        hx, hy = wgs84_to_metric(lat_val, lon_val)
                        raw = f"({lat_val:.5f}, {lon_val:.5f})"
                        result = add_personal_feeder(sidx, selection, hx, hy, raw, force=force)
                        _show_feeder_result(result)
                        return
                geolocator = get_geolocator()
                # Only append ", Berlin, Germany" if the address doesn't already
                # name Berlin/Germany -- doing it unconditionally double-suffixes
                # full addresses (e.g. "...Str. 19A, 10115 Berlin, Germany") into a
                # malformed query Nominatim can't match.
                query = home_text if re.search(r"berlin|germany|deutschland", home_text, re.I) \
                    else f"{home_text}, Berlin, Germany"
                loc = None
                err = None
                # Nominatim's public server occasionally times out or rate-limits
                # under bursty use -- retry transient failures once after a short
                # pause before giving up; a genuinely bad address fails instantly
                # (no exception) and isn't retried.
                for attempt in range(2):
                    try:
                        loc = geolocator.geocode(query)
                        err = None
                        break
                    except GeocoderServiceError as e:
                        err = e
                        if attempt == 0:
                            time.sleep(1.5)
                    except Exception as e:
                        err = e
                        break
                if loc is None:
                    msg = f"Could not geocode {home_text!r} — try Coordinates instead."
                    if err is not None:
                        msg += f"  ({type(err).__name__}: {err})"
                    else:
                        msg += "  (Nominatim returned zero results — possibly blocking this server's IP)"
                    st.error(msg)
                else:
                    hx, hy = wgs84_to_metric(loc.latitude, loc.longitude)
                    result = add_personal_feeder(sidx, selection, hx, hy, home_text, force=force)
                    _show_feeder_result(result)
        else:
            hx, hy = wgs84_to_metric(lat, lon)
            raw = f"({lat:.5f}, {lon:.5f})"
            result = add_personal_feeder(sidx, selection, hx, hy, raw, force=force)
            _show_feeder_result(result)

    feeders = st.session_state.get("personal_feeders")
    n = len(feeders[feeders["school_idx"] == sidx]) if feeders is not None and len(feeders) else 0
    if n:
        st.caption(f"📍 {n} personal feeder(s) shown on the map for this school.")


def _show_feeder_result(result):
    if result["status"] == "ok":
        row = result["row"]
        st.success(
            f"{row['length_m']:.0f} m · {row['walk_time_min']:.1f} min walk · joins **{row['tier_joined_name']}**\n\n"
            f"{row['n_crossings']} crossing(s), {row['n_traffic_signals']} signal(s), "
            f"{row['n_hazard_points']} hazard point(s) along the way."
        )
    elif result["status"] == "flagged_catchment":
        st.warning(result["message"] + " Not routed — tick **Force** to route anyway.")
    elif result["status"] == "flagged_unreachable":
        st.error(result["message"])


def _share_url():
    """The URL to show/QR-code for sharing this app. Once deployed on Streamlit
    Community Cloud, set a PUBLIC_APP_URL secret to the app's real
    https://...streamlit.app address -- that's what gets shown, reachable from
    anywhere. Without that secret (i.e. running locally), falls back to this
    machine's LAN URL: opens a UDP "connection" to a public address purely so
    the OS reports which local interface/IP it would route through (nothing is
    actually sent -- UDP connect() doesn't transmit a packet by itself); a
    phone on the same Wi-Fi/network can use that, but nobody else can."""
    try:
        public_url = st.secrets.get("PUBLIC_APP_URL")
    except Exception:
        public_url = None
    if public_url:
        return public_url, True

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    port = st.get_option("server.port") or 8501
    return f"http://{ip}:{port}", False


# ── UI ───────────────────────────────────────────────────────────────────────
d = load_all_data()
schools = d["schools"]

if "personal_feeders" not in st.session_state:
    st.session_state["personal_feeders"] = _load_or_init_personal_feeders()

with st.sidebar:
    st.title("Schulwegplan Berlin")
    st.caption("Mitte · Grundschulen")
    with st.expander("📱 Open on phone"):
        share_url, is_public = _share_url()
        st.caption("Works from anywhere." if is_public else
                   "Phone must be on the same Wi-Fi/network as this computer.")
        st.code(share_url, language=None)
        qr_buf = io.BytesIO()
        qrcode.make(share_url).save(qr_buf, format="PNG")
        st.image(qr_buf.getvalue(), width=180)
    st.divider()
    options   = ["— Select School —"] + sorted(schools["name"].tolist())
    selection = st.selectbox("School", options, label_visibility="collapsed")
    st.divider()
    st.caption(
        "**Legend**\n\n"
        "🟠 Primary Spine\n\n"
        "🔵 Secondary Spine\n\n"
        "🟢 Feeder Route\n\n"
        "🟣 5 / 10 / 15-min isochrone\n\n"
        "🔷 Census cell (children 6–15)\n\n"
        "🟡 Personal Feeder (thin yellow line + 🏠 pin) — added by you\n\n"
        "➕ Traffic safety, hazards & landmarks — toggle in the map's layer panel (top right); legend at bottom left"
    )
    if selection != "— Select School —":
        st.divider()
        render_personal_feeder_form(selection, schools)

if selection == "— Select School —":
    st.subheader("Schulwegplan Berlin — Mitte")
    st.caption("Select a school from the sidebar to load its catchment and route plan.")
    st_folium(build_default_map(d), width=1300, height=620, returned_objects=[])

else:
    srow = schools[schools["name"] == selection].iloc[0]
    sidx = int(srow["school_idx"])
    s    = d["school_stats"].get(sidx, {})

    # Header
    addr = ""
    if srow.get("addr_street"):
        addr = f" · {srow['addr_street']} {srow.get('addr_housenumber', '')}"
    st.subheader(f"{selection}{addr}")

    # Metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Children (6–15)",  f"{s['children']:,}"    if s.get("children")     else "—")
    c2.metric("Mean walk",        f"{s['mean_dist_m']} m" if s.get("mean_dist_m")  else "—")
    c3.metric("5-min area",       f"{s['iso_5_km2']} km²" if s.get("iso_5_km2")    else "—")
    c4.metric("15-min area",      f"{s['iso_15_km2']} km²"if s.get("iso_15_km2")   else "—")
    c5.metric("Primary spine",    f"{s['primary_km']} km" if s.get("primary_km") is not None else "—")
    c6.metric("Secondary spine",  f"{s['secondary_km']} km" if s.get("secondary_km") is not None else "—")

    st.divider()
    st_folium(
        build_school_map(sidx, d), width=1300, height=620,
        key=f"school_map_{sidx}", returned_objects=[],
    )
