from __future__ import annotations

import html
import json
from pathlib import Path


def write_outputs(atlas: dict, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "family_map.json"
    html_path = out_dir / "family_map.html"
    report_path = out_dir / "analysis_report.md"

    json_text = json.dumps(atlas, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    html_path.write_text(render_html(json_text), encoding="utf-8")
    report_path.write_text(render_report(atlas), encoding="utf-8")
    return html_path, json_path, report_path


def render_report(atlas: dict) -> str:
    lines = [
        "# Family Resemblance Analysis",
        "",
        "Локальный отчет по гипотезе: абстрактное слово не имеет одного центрального значения, если его употребления образуют несколько плотностей, переходные зоны и слабый медоид.",
        "",
        "| word | occurrences | clusters | noise | medoid dominance | semantic diameter | anti-essence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for word in atlas["words"]:
        metrics = atlas["metrics"][word]
        lines.append(
            f"| {word} | {metrics['occurrences']} | {metrics['cluster_count']} | "
            f"{metrics['noise_ratio']:.2f} | {metrics['medoid_dominance']:.2f} | "
            f"{metrics['semantic_diameter']:.2f} | {metrics['anti_essence_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Интерпретация: высокий anti-essence score не является философским доказательством в строгом смысле, но является проверяемым математическим свидетельством против модели одного центра.",
            "",
            "Метод: локальное самообучение PPMI+SVD с lexical-hash признаками контекста, UMAP-подобное спектральное вложение, density clustering в стиле HDBSCAN.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_html(json_text: str) -> str:
    escaped_json = html.escape(json_text, quote=False)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Family Resemblance Atlas</title>
  <style>
    :root {{
      --bg: #f8faf9;
      --ink: #202623;
      --muted: #69736f;
      --line: #d9dfdc;
      --panel: #ffffff;
      --accent: #2a9d8f;
      --accent2: #e76f51;
      --accent3: #457b9d;
      --accent4: #e9c46a;
      --shadow: 0 18px 45px rgba(32, 38, 35, 0.10);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(20px, 2.3vw, 30px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      align-items: end;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    label {{
      display: grid;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
    }}
    select, button {{
      height: 38px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      padding: 0 10px;
      font: inherit;
      max-width: 180px;
    }}
    button {{
      min-width: 38px;
      cursor: pointer;
      font-weight: 650;
    }}
    button:hover, select:hover {{ border-color: var(--accent); }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 0;
      min-height: 0;
    }}
    .map-area {{
      position: relative;
      min-height: 560px;
      border-right: 1px solid var(--line);
      background:
        linear-gradient(rgba(32, 38, 35, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(32, 38, 35, 0.035) 1px, transparent 1px);
      background-size: 36px 36px;
      overflow: hidden;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    aside {{
      min-height: 0;
      overflow: auto;
      padding: 18px;
      background: var(--panel);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 72px;
      background: #fff;
    }}
    .metric span {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .metric strong {{
      font-size: 23px;
      line-height: 1;
      letter-spacing: 0;
    }}
    .section-title {{
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 18px 0 8px;
    }}
    .cluster-list {{
      display: grid;
      gap: 8px;
    }}
    .cluster-item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 1px solid rgba(0,0,0,0.18);
    }}
    .selected {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 130px;
      background: #fff;
      box-shadow: var(--shadow);
    }}
    .selected .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .sentence {{
      font-size: 14px;
      line-height: 1.45;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      max-width: min(380px, calc(100% - 28px));
      padding: 9px 10px;
      background: rgba(255,255,255,0.96);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      font-size: 12px;
      line-height: 1.35;
      color: var(--ink);
      transform: translate(12px, 12px);
      display: none;
    }}
    .legend {{
      position: absolute;
      left: 16px;
      bottom: 16px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      max-width: calc(100% - 32px);
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.9);
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 860px) {{
      header {{ align-items: stretch; flex-direction: column; }}
      .controls {{ justify-content: flex-start; }}
      main {{ grid-template-columns: 1fr; }}
      .map-area {{ min-height: 64vh; border-right: 0; border-bottom: 1px solid var(--line); }}
      aside {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Family Resemblance Atlas</h1>
        <div class="subtitle">локальная карта контекстов без внешних API</div>
      </div>
      <div class="controls">
        <label>Слово<select id="wordSelect"></select></label>
        <label>Домен<select id="domainSelect"></select></label>
        <label>Кластер<select id="clusterSelect"></select></label>
        <button id="resetView" title="Сбросить выбор">↺</button>
      </div>
    </header>
    <main>
      <section class="map-area" id="mapArea">
        <canvas id="mapCanvas"></canvas>
        <div class="tooltip" id="tooltip"></div>
        <div class="legend" id="legend"></div>
      </section>
      <aside>
        <div class="metric-grid" id="metrics"></div>
        <div class="section-title">Кластеры</div>
        <div class="cluster-list" id="clusterList"></div>
        <div class="section-title">Контекст</div>
        <div class="selected" id="selectedPoint"></div>
      </aside>
    </main>
  </div>
  <script type="application/json" id="atlas-data">{escaped_json}</script>
  <script>
    const atlas = JSON.parse(document.getElementById('atlas-data').textContent);
    const colors = ['#2a9d8f', '#e76f51', '#457b9d', '#e9c46a', '#8ab17d', '#b56576', '#6d597a', '#4d908e'];
    const state = {{
      word: atlas.words[0],
      domain: 'all',
      cluster: 'all',
      selectedId: null,
      hoverId: null
    }};

    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const mapArea = document.getElementById('mapArea');
    const tooltip = document.getElementById('tooltip');

    function pointsForWord() {{
      return atlas.points.filter(point => point.word === state.word);
    }}

    function visiblePoints() {{
      return pointsForWord().filter(point => {{
        const domainOk = state.domain === 'all' || point.domain === state.domain;
        const clusterOk = state.cluster === 'all' || String(point.cluster) === state.cluster;
        return domainOk && clusterOk;
      }});
    }}

    function fitCanvas() {{
      const rect = mapArea.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * scale));
      canvas.height = Math.max(320, Math.floor(rect.height * scale));
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }}

    function project(point) {{
      const rect = canvas.getBoundingClientRect();
      const margin = Math.min(76, Math.max(38, rect.width * 0.08));
      const x = margin + ((point.x + 1) / 2) * (rect.width - margin * 2);
      const y = margin + ((1 - (point.y + 1) / 2)) * (rect.height - margin * 2);
      return {{x, y}};
    }}

    function draw() {{
      fitCanvas();
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      const all = pointsForWord();
      const visible = visiblePoints();
      const centroid = atlas.centroids[state.word];
      const c = project({{x: centroid[0], y: centroid[1]}});

      ctx.save();
      ctx.strokeStyle = 'rgba(32,38,35,0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(c.x - 10, c.y);
      ctx.lineTo(c.x + 10, c.y);
      ctx.moveTo(c.x, c.y - 10);
      ctx.lineTo(c.x, c.y + 10);
      ctx.stroke();
      ctx.restore();

      for (const point of all) {{
        const p = project(point);
        const isVisible = visible.includes(point);
        const selected = state.selectedId === point.id;
        const hovered = state.hoverId === point.id;
        const radius = selected ? 8 : hovered ? 7 : point.isMedoid ? 6 : 4.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isVisible ? colorFor(point.cluster) : 'rgba(105,115,111,0.16)';
        ctx.fill();
        ctx.lineWidth = selected || hovered ? 2.2 : point.isMedoid ? 1.8 : 0.9;
        ctx.strokeStyle = selected ? '#202623' : point.isMedoid ? '#202623' : 'rgba(255,255,255,0.95)';
        ctx.stroke();
      }}
    }}

    function colorFor(cluster) {{
      if (cluster < 0) return '#9aa3a0';
      return colors[cluster % colors.length];
    }}

    function nearestPoint(event) {{
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let best = null;
      let bestDistance = 14;
      for (const point of visiblePoints()) {{
        const p = project(point);
        const distance = Math.hypot(p.x - x, p.y - y);
        if (distance < bestDistance) {{
          best = point;
          bestDistance = distance;
        }}
      }}
      return best;
    }}

    function populateControls() {{
      fillSelect('wordSelect', atlas.words.map(word => [word, word]), state.word);
      updateDomainControl();
      updateClusterControl();
    }}

    function updateDomainControl() {{
      const domains = [...new Set(pointsForWord().map(point => point.domain))].sort();
      fillSelect('domainSelect', [['all', 'все'], ...domains.map(domain => [domain, domain])], state.domain);
    }}

    function updateClusterControl() {{
      const clusters = [...new Set(pointsForWord().map(point => point.cluster))].sort((a, b) => a - b);
      const options = [['all', 'все'], ...clusters.map(cluster => [String(cluster), cluster < 0 ? 'noise' : 'cluster ' + cluster])];
      fillSelect('clusterSelect', options, state.cluster);
    }}

    function fillSelect(id, options, selected) {{
      const select = document.getElementById(id);
      select.innerHTML = '';
      for (const [value, label] of options) {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        option.selected = value === selected;
        select.appendChild(option);
      }}
    }}

    function renderMetrics() {{
      const metrics = atlas.metrics[state.word];
      const entries = [
        ['anti-essence', metrics.anti_essence_score.toFixed(2)],
        ['clusters', metrics.cluster_count],
        ['medoid', metrics.medoid_dominance.toFixed(2)],
        ['diameter', metrics.semantic_diameter.toFixed(2)],
        ['noise', Math.round(metrics.noise_ratio * 100) + '%'],
        ['contexts', metrics.occurrences],
      ];
      document.getElementById('metrics').innerHTML = entries.map(([label, value]) =>
        `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`
      ).join('');
    }}

    function renderClusters() {{
      const counts = new Map();
      for (const point of pointsForWord()) {{
        counts.set(point.cluster, (counts.get(point.cluster) || 0) + 1);
      }}
      const items = [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([cluster, count]) => {{
        const label = cluster < 0 ? 'noise / bridges' : 'cluster ' + cluster;
        return `<div class="cluster-item">
          <span class="swatch" style="background:${{colorFor(cluster)}}"></span>
          <span>${{label}}</span>
          <strong>${{count}}</strong>
        </div>`;
      }});
      document.getElementById('clusterList').innerHTML = items.join('');
      renderLegend(counts);
    }}

    function renderLegend(counts) {{
      document.getElementById('legend').innerHTML = [...counts.keys()].sort((a, b) => a - b).map(cluster => {{
        const label = cluster < 0 ? 'noise' : 'c' + cluster;
        return `<span class="chip"><span class="swatch" style="background:${{colorFor(cluster)}}"></span>${{label}}</span>`;
      }}).join('');
    }}

    function renderSelected(point) {{
      const box = document.getElementById('selectedPoint');
      if (!point) {{
        const medoid = pointsForWord().find(item => item.isMedoid);
        point = medoid || pointsForWord()[0];
      }}
      if (!point) {{
        box.innerHTML = '';
        return;
      }}
      box.innerHTML = `<div class="meta">
          <span>${{point.domain}}</span>
          <span>${{point.cluster < 0 ? 'noise' : 'cluster ' + point.cluster}}</span>
          <span>${{point.isMedoid ? 'medoid' : 'context'}}</span>
        </div>
        <div class="sentence">${{escapeHtml(point.sentence)}}</div>`;
    }}

    function escapeHtml(value) {{
      return value.replace(/[&<>"']/g, char => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }}[char]));
    }}

    function refresh() {{
      updateDomainControl();
      updateClusterControl();
      renderMetrics();
      renderClusters();
      renderSelected(atlas.points.find(point => point.id === state.selectedId));
      draw();
    }}

    document.getElementById('wordSelect').addEventListener('change', event => {{
      state.word = event.target.value;
      state.domain = 'all';
      state.cluster = 'all';
      state.selectedId = null;
      refresh();
    }});
    document.getElementById('domainSelect').addEventListener('change', event => {{
      state.domain = event.target.value;
      state.selectedId = null;
      refresh();
    }});
    document.getElementById('clusterSelect').addEventListener('change', event => {{
      state.cluster = event.target.value;
      state.selectedId = null;
      refresh();
    }});
    document.getElementById('resetView').addEventListener('click', () => {{
      state.domain = 'all';
      state.cluster = 'all';
      state.selectedId = null;
      refresh();
    }});

    canvas.addEventListener('mousemove', event => {{
      const point = nearestPoint(event);
      state.hoverId = point ? point.id : null;
      if (point) {{
        tooltip.style.left = event.clientX - mapArea.getBoundingClientRect().left + 'px';
        tooltip.style.top = event.clientY - mapArea.getBoundingClientRect().top + 'px';
        tooltip.style.display = 'block';
        tooltip.innerHTML = `<strong>${{point.word}}</strong> · ${{point.domain}} · ${{point.cluster < 0 ? 'noise' : 'cluster ' + point.cluster}}<br>${{escapeHtml(point.sentence)}}`;
      }} else {{
        tooltip.style.display = 'none';
      }}
      draw();
    }});
    canvas.addEventListener('mouseleave', () => {{
      state.hoverId = null;
      tooltip.style.display = 'none';
      draw();
    }});
    canvas.addEventListener('click', event => {{
      const point = nearestPoint(event);
      if (point) {{
        state.selectedId = point.id;
        renderSelected(point);
        draw();
      }}
    }});
    window.addEventListener('resize', draw);

    populateControls();
    refresh();
  </script>
</body>
</html>
"""
