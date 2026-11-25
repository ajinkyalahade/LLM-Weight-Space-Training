"""Graphs as SVG files, written with plain Python (nothing to install).

Four forms, each returning the SVG as a string:
    columns(...)    grouped columns          one number per group and series (accuracy by source, original vs edited)
    segmented(...)  one bar per row, split   shares of a whole (what happened to the answers)
    scatter(...)    dots                     one dot per item (a neuron before and after an edit)
    lines(...)      lines                    a quantity along an axis (speed during a pass)

Colours are the first four categorical slots of the dataviz skill's reference palette, in its fixed order, checked
with its validator in light and dark mode. Text is never in a series colour. Every mark has a <title>, which a browser
shows on hover when the file is opened by itself.
"""
import math
from html import escape

W = 720
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
STYLE = """
svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
    --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--on4:#0b0b0b}
@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
    --grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--on4:#ffffff}}
text{font-family:%s;fill:var(--ink2);font-size:12px}
.title{fill:var(--ink);font-size:15px;font-weight:600}.sub{fill:var(--ink2)}.tick{fill:var(--muted);font-size:11px}
.val{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}.in{fill:#fff;font-size:11px;font-weight:600}
.in4{fill:var(--on4);font-size:11px;font-weight:600}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}.ref{stroke:var(--axis);stroke-width:1}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}
.l1{stroke:var(--s1)}.l2{stroke:var(--s2)}.l3{stroke:var(--s3)}.l4{stroke:var(--s4)}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}
""" % FONT


def width(s, size=12):
    """Rough width of a string in pixels (system sans)."""
    return 0.56 * size * len(str(s))


def nice(lo, hi, n=5):
    """Round tick values covering lo..hi."""
    span = (hi - lo) or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v < hi + step * 0.999:
        ticks.append(round(v, 10))
        v += step
    return ticks


def num(v):
    if v == 0:
        return "0"
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:g}"


def frame(h, title, sub, body, legend=()):
    """The SVG around a plot: surface, title, subtitle, and a legend row (name, class) when there are 2+ series."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" '
           f'aria-label="{escape(title)}"><style>{STYLE}</style>',
           f'<rect width="{W}" height="{h}" fill="var(--surface)"/>',
           f'<text class="title" x="20" y="28">{escape(title)}</text>',
           f'<text class="sub" x="20" y="47">{escape(sub)}</text>']
    x = 20
    for name, cls in legend:
        out.append(f'<rect class="{cls}" x="{x}" y="60" width="10" height="10" rx="2"/>'
                   f'<text x="{x + 15}" y="69">{escape(name)}</text>')
        x += 15 + width(name) + 18
    return "\n".join(out + [body, "</svg>"])


def _axes(x0, x1, y0, y1, yticks, ymap, ylabel, xlabel=None, yfmt=num):
    out = []
    for t in yticks:
        y = ymap(t)
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{yfmt(t)}</text>')
    out.append(f'<text class="tick" x="{x0 - 44}" y="{y0 - 10}">{escape(ylabel)}</text>')
    if xlabel:
        out.append(f'<text class="tick" x="{(x0 + x1) / 2}" y="{y1 + 36}" text-anchor="middle">{escape(xlabel)}</text>')
    return out


def columns(title, sub, groups, series, ylabel="%", ymax=100, fmt=lambda v: f"{v:.0f}%", h=380):
    """groups: names along the x axis. series: [(name, [value per group, None = no bar])]."""
    x0, x1, y0, y1 = 64, W - 20, 100, h - 44
    ymap = lambda v: y1 - (y1 - y0) * v / ymax  # noqa: E731
    out = _axes(x0, x1, y0, y1, nice(0, ymax)[:-1] if nice(0, ymax)[-1] > ymax else nice(0, ymax), ymap, ylabel)
    band = (x1 - x0) / len(groups)
    bw = min(24, (band * 0.7 - 2 * (len(series) - 1)) / len(series))
    for g, gname in enumerate(groups):
        cx = x0 + band * (g + 0.5)
        left = cx - (bw * len(series) + 2 * (len(series) - 1)) / 2
        for s, (sname, vals) in enumerate(series):
            v = vals[g]
            if v is None:
                continue
            x, y = left + s * (bw + 2), ymap(v)
            r = min(4, (y1 - y) / 2)
            out.append(f'<path class="s{s + 1}" d="M{x:.1f},{y1} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
                       f'H{x + bw - r:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{y1} Z">'
                       f'<title>{escape(gname)} — {escape(sname)}: {fmt(v)}</title></path>'
                       f'<text class="val" x="{x + bw / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{fmt(v)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{y1 + 18}" text-anchor="middle">{escape(gname)}</text>')
    out.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
    names = [(n, f"s{i + 1}") for i, (n, _) in enumerate(series)]
    return frame(h, title, sub, "\n".join(out), names if len(series) > 1 else ())


def segmented(title, sub, rows, parts, unit=""):
    """rows: [(row name, [amount per part])]; parts: names, in bar order. One 28 px bar per row, split by share."""
    x0, x1 = 20, W - 20
    out, y = [], 96
    for rname, vals in rows:
        total = sum(vals) or 1
        out.append(f'<text class="val" x="{x0}" y="{y}" style="font-size:12px">{escape(rname)}</text>')
        x = x0
        for p, (pname, v) in enumerate(zip(parts, vals)):
            if not v:
                continue
            wseg = (x1 - x0) * v / total
            label = f"{100 * v / total:.0f}%"
            out.append(f'<rect class="s{p + 1}" x="{x + 1:.1f}" y="{y + 8}" width="{max(wseg - 2, 1):.1f}" height="28" '
                       f'rx="4"><title>{escape(rname)} — {escape(pname)}: {v}{unit} ({label})</title></rect>')
            if wseg > width(label, 11) + 14:
                out.append(f'<text class="{"in4" if p == 3 else "in"}" x="{x + 8:.1f}" y="{y + 26}">{label}</text>')
            x += wseg
        y += 64
    return frame(y - 12, title, sub, "\n".join(out), [(n, f"s{i + 1}") for i, n in enumerate(parts)])


def _plane(series, xlabel, ylabel, h, xlim=None, ylim=None):
    xs = [p[0] for _, pts in series for p in pts]
    ys = [p[1] for _, pts in series for p in pts]
    xt = nice(*(xlim or (min(xs), max(xs))))
    yt = nice(*(ylim or (min(ys), max(ys))))
    x0, x1, y0, y1 = 72, W - 28, 100, h - 52
    xmap = lambda v: x0 + (x1 - x0) * (v - xt[0]) / (xt[-1] - xt[0])  # noqa: E731
    ymap = lambda v: y1 - (y1 - y0) * (v - yt[0]) / (yt[-1] - yt[0])  # noqa: E731
    out = _axes(x0, x1, y0, y1, yt, ymap, ylabel, xlabel)
    for t in xt:
        out.append(f'<text class="tick" x="{xmap(t):.1f}" y="{y1 + 16}" text-anchor="middle">{num(t)}</text>')
    out.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{ymap(yt[0]):.1f}" y2="{ymap(yt[0]):.1f}"/>')
    return out, xmap, ymap, xt, yt


def scatter(title, sub, series, xlabel, ylabel, ref=None, h=420):
    """series: [(name, [(x, y, hover text)])]. ref: (slope, label) draws the line y = slope * x through the plot."""
    out, xmap, ymap, xt, yt = _plane(series, xlabel, ylabel, h)
    for v, horizontal in ((0, True), (0, False)):                       # zero lines, when zero is inside the plot
        if horizontal and yt[0] < 0 < yt[-1]:
            out.append(f'<line class="axis" x1="{xmap(xt[0]):.1f}" x2="{xmap(xt[-1]):.1f}" y1="{ymap(0):.1f}" y2="{ymap(0):.1f}"/>')
        if not horizontal and xt[0] < 0 < xt[-1]:
            out.append(f'<line class="axis" x1="{xmap(0):.1f}" x2="{xmap(0):.1f}" y1="{ymap(yt[0]):.1f}" y2="{ymap(yt[-1]):.1f}"/>')
    if ref:
        slope, label = ref
        a = max(xt[0], min(yt[0] / slope, yt[-1] / slope))
        b = min(xt[-1], max(yt[0] / slope, yt[-1] / slope))
        corner = ymap(yt[-1]) + 16 if slope < 0 else ymap(yt[0]) - 10       # the corner the line does not pass through
        out.append(f'<line class="ref" x1="{xmap(a):.1f}" y1="{ymap(slope * a):.1f}" x2="{xmap(b):.1f}" y2="{ymap(slope * b):.1f}"/>'
                   f'<text x="{xmap(xt[-1]) - 6:.1f}" y="{corner:.1f}" text-anchor="end">Line: {escape(label)}</text>')
    for s, (name, pts) in enumerate(series):
        for x, y, tip in pts:
            out.append(f'<circle class="s{s + 1} dot" cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="5"><title>{escape(tip)}</title></circle>')
    names = [(n, f"s{i + 1}") for i, (n, _) in enumerate(series)]
    return frame(h, title, sub, "\n".join(out), names if len(series) > 1 else ())


def lines(title, sub, series, xlabel, ylabel, ylim=None, h=380):
    """series: [(name, [(x, y)])]. The last point of each line is marked and labelled with its value."""
    out, xmap, ymap, xt, yt = _plane(series, xlabel, ylabel, h, ylim=ylim)
    for s, (name, pts) in enumerate(series):
        d = " ".join(f"{'M' if i == 0 else 'L'}{xmap(x):.1f},{ymap(y):.1f}" for i, (x, y) in enumerate(pts))
        x, y = pts[-1]
        out.append(f'<path class="line l{s + 1}" d="{d}"><title>{escape(name)}</title></path>'
                   f'<circle class="s{s + 1} dot" cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="5"/>'
                   f'<text class="val" x="{xmap(x) - 8:.1f}" y="{ymap(y) - 9:.1f}" text-anchor="end">'
                   f'{num(y) if abs(y) >= 10 else f"{y:.2f}"}</text>')
    names = [(n, f"s{i + 1}") for i, (n, _) in enumerate(series)]
    return frame(h, title, sub, "\n".join(out), names if len(series) > 1 else ())
