from __future__ import annotations

import argparse
from dataclasses import replace
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from family_resemblance.pipeline import AtlasConfig, CorpusModel
from family_resemblance.text import Document


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Family Resemblance Dashboard</title>
  <style>
    :root {
      --bg: #f7f9f8;
      --panel: #ffffff;
      --ink: #202623;
      --muted: #6c7672;
      --line: #dce2df;
      --accent: #2a9d8f;
      --accent2: #e76f51;
      --accent3: #457b9d;
      --accent4: #e9c46a;
      --danger: #b23a48;
      --shadow: 0 18px 45px rgba(32, 38, 35, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }
    header {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 18px;
      align-items: end;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
    }
    h1 {
      margin: 0;
      font-size: clamp(22px, 2.4vw, 31px);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 5px;
    }
    .query {
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(210px, 300px) 128px 118px 118px auto;
      gap: 10px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select, textarea, button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 0 10px;
      min-width: 0;
    }
    textarea {
      height: 78px;
      padding: 9px 10px;
      resize: vertical;
      line-height: 1.35;
    }
    input:focus, select:focus, textarea:focus {
      outline: 2px solid rgba(42, 157, 143, 0.18);
      border-color: var(--accent);
    }
    button {
      cursor: pointer;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
    }
    button.primary {
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
      min-width: 116px;
    }
    button.icon {
      min-width: 38px;
      padding: 0;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.62;
    }
    .toggle {
      height: 38px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      white-space: nowrap;
    }
    .toggle input {
      width: 16px;
      height: 16px;
      padding: 0;
      accent-color: var(--accent);
    }
    .paste-panel {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 180px;
      gap: 10px;
      padding: 12px 22px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 0;
    }
    .map-area {
      position: relative;
      min-height: 590px;
      border-right: 1px solid var(--line);
      background:
        linear-gradient(rgba(32, 38, 35, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(32, 38, 35, 0.035) 1px, transparent 1px);
      background-size: 36px 36px;
      overflow: hidden;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    aside {
      padding: 18px;
      overflow: auto;
      background: var(--panel);
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric, .selected, .notice, .cluster-item, .evidence-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .metric {
      padding: 10px;
      min-height: 72px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 8px;
    }
    .metric strong {
      font-size: 23px;
      line-height: 1;
      letter-spacing: 0;
    }
    .section-title {
      margin: 18px 0 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .cluster-list {
      display: grid;
      gap: 8px;
    }
    .cluster-item {
      padding: 9px 10px;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }
    .cluster-text {
      display: grid;
      gap: 2px;
      min-width: 0;
    }
    .cluster-name {
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .cluster-terms {
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .evidence-card {
      padding: 10px 12px;
      display: grid;
      gap: 7px;
      margin-bottom: 14px;
    }
    .evidence-title {
      font-weight: 750;
      font-size: 13px;
    }
    .evidence-row {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }
    .evidence-chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 7px;
      color: var(--muted);
      font-size: 11px;
      background: #fff;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 1px solid rgba(0,0,0,0.18);
    }
    .selected {
      min-height: 132px;
      padding: 12px;
      box-shadow: var(--shadow);
    }
    .selected .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .sentence {
      font-size: 14px;
      line-height: 1.45;
    }
    .notice {
      margin: 0 0 14px;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .notice.error {
      color: var(--danger);
      border-color: rgba(178, 58, 72, 0.35);
    }
    .tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      max-width: min(390px, calc(100% - 28px));
      padding: 9px 10px;
      background: rgba(255,255,255,0.96);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      font-size: 12px;
      line-height: 1.35;
      color: var(--ink);
      transform: translate(12px, 12px);
    }
    .legend {
      position: absolute;
      left: 16px;
      bottom: 16px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      max-width: calc(100% - 32px);
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 1360px) {
      header { grid-template-columns: 1fr; align-items: start; }
      .query {
        width: 100%;
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    }
    @media (max-width: 1100px) {
      .query, .paste-panel { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .map-area { min-height: 62vh; border-right: 0; border-bottom: 1px solid var(--line); }
      .query, .toolbar, .paste-panel { grid-template-columns: 1fr; }
      button.primary, button.icon { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Family Resemblance Dashboard</h1>
        <div class="subtitle">local research into word contexts</div>
      </div>
      <form class="query" id="queryForm">
        <label>Word
          <input id="wordInput" value="game" autocomplete="off">
        </label>
        <label>Corpus
          <input id="corpusInput" value="data/sample_corpus" autocomplete="off">
        </label>
        <label>Matching
          <select id="matchMode">
            <option value="exact">exact</option>
            <option value="lemma-lite" selected>lemma-lite</option>
            <option value="prefix">prefix</option>
          </select>
        </label>
        <label>Min. Cluster
          <input id="minClusterSize" type="number" min="2" max="50" value="3">
        </label>
        <label>Source
          <span class="toggle"><input id="autoContext" type="checkbox" checked> auto</span>
        </label>
        <button class="primary" id="analyzeButton" type="submit">Analyze</button>
      </form>
    </header>
    <section class="paste-panel">
      <label>Paste Text
        <textarea id="pastedText" placeholder="Paste an article, book excerpt, forum thread, legal fragment, or any other text here."></textarea>
      </label>
      <label>Text Domain
        <input id="pastedDomain" value="pasted" autocomplete="off">
      </label>
    </section>
    <main>
      <section class="map-area" id="mapArea">
        <canvas id="mapCanvas"></canvas>
        <div class="tooltip" id="tooltip"></div>
        <div class="legend" id="legend"></div>
      </section>
      <aside>
        <div class="toolbar">
          <label>Found Word<select id="wordSelect"></select></label>
          <label>Domain<select id="domainSelect"></select></label>
          <button class="icon" id="downloadJson" title="Download JSON" type="button">&#8681;</button>
        </div>
        <div id="notice" class="notice">Enter a word that appears in the selected local corpus.</div>
        <div id="evidenceList"></div>
        <div class="metric-grid" id="metrics"></div>
        <div class="section-title">Clusters</div>
        <div class="cluster-list" id="clusterList"></div>
        <div class="section-title">Context</div>
        <div class="selected" id="selectedPoint"></div>
      </aside>
    </main>
  </div>
  <script>
    const colors = ['#2a9d8f', '#e76f51', '#457b9d', '#e9c46a', '#8ab17d', '#b56576', '#6d597a', '#4d908e'];
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const mapArea = document.getElementById('mapArea');
    const tooltip = document.getElementById('tooltip');
    const notice = document.getElementById('notice');
    let atlas = null;
    let state = { word: null, domain: 'all', cluster: 'all', selectedId: null, hoverId: null };

    function setNotice(text, isError = false) {
      notice.textContent = text;
      notice.classList.toggle('error', isError);
    }

    function clearResult(message, isError = false) {
      atlas = null;
      state = { word: null, domain: 'all', cluster: 'all', selectedId: null, hoverId: null };
      fillSelect('wordSelect', [['', 'no result']], '');
      fillSelect('domainSelect', [['all', 'all']], 'all');
      document.getElementById('metrics').innerHTML = '';
      document.getElementById('clusterList').innerHTML = '';
      document.getElementById('evidenceList').innerHTML = '';
      document.getElementById('selectedPoint').innerHTML = '';
      document.getElementById('legend').innerHTML = '';
      tooltip.style.display = 'none';
      draw();
      setNotice(message, isError);
    }

    async function analyze() {
      const button = document.getElementById('analyzeButton');
      button.disabled = true;
      setNotice('Analysis is running...');
      try {
        const payload = {
          words: document.getElementById('wordInput').value,
          corpus: document.getElementById('corpusInput').value,
          matchMode: document.getElementById('matchMode').value,
          minClusterSize: Number(document.getElementById('minClusterSize').value || 3),
          autoContext: document.getElementById('autoContext').checked,
          pastedText: document.getElementById('pastedText').value,
          pastedDomain: document.getElementById('pastedDomain').value
        };
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          clearResult(data.error || 'Analysis error', true);
          return;
        }
        atlas = data;
        state = { word: atlas.words[0], domain: 'all', cluster: 'all', selectedId: null, hoverId: null };
        populateControls();
        refresh();
        const missing = atlas.metadata.missing || [];
        if (missing.length) {
          setNotice('No occurrences: ' + missing.map(item => item.word).join(', '), true);
        } else {
          const evidence = atlas.metadata.evidence[state.word] || {};
          setNotice('Contexts: ' + atlas.points.length + ' | evidence: ' + (evidence.label || atlas.metadata.sourceMode) + ' | mode: ' + atlas.metadata.matchMode);
        }
      } catch (error) {
        clearResult(error.message, true);
      } finally {
        button.disabled = false;
      }
    }

    function pointsForWord() {
      return atlas ? atlas.points.filter(point => point.word === state.word) : [];
    }

    function visiblePoints() {
      return pointsForWord().filter(point => {
        const domainOk = state.domain === 'all' || point.domain === state.domain;
        return domainOk;
      });
    }

    function fitCanvas() {
      const rect = mapArea.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * scale));
      canvas.height = Math.max(320, Math.floor(rect.height * scale));
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }

    function project(point) {
      const rect = canvas.getBoundingClientRect();
      const margin = Math.min(76, Math.max(38, rect.width * 0.08));
      const x = margin + ((point.x + 1) / 2) * (rect.width - margin * 2);
      const y = margin + ((1 - (point.y + 1) / 2)) * (rect.height - margin * 2);
      return { x, y };
    }

    function draw() {
      fitCanvas();
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (!atlas) return;

      const all = pointsForWord();
      const visible = new Set(visiblePoints().map(point => point.id));
      const centroid = atlas.centroids[state.word] || [0, 0];
      const c = project({ x: centroid[0], y: centroid[1] });
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

      for (const point of all) {
        const p = project(point);
        const isVisible = visible.has(point.id);
        const selected = state.selectedId === point.id;
        const hovered = state.hoverId === point.id;
        const radius = selected ? 8 : hovered ? 7 : point.isMedoid ? 6 : 4.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isVisible ? colorFor(point.cluster) : 'rgba(105,115,111,0.15)';
        ctx.fill();
        ctx.lineWidth = selected || hovered ? 2.2 : point.isMedoid ? 1.8 : 0.9;
        ctx.strokeStyle = selected ? '#202623' : point.isMedoid ? '#202623' : 'rgba(255,255,255,0.95)';
        ctx.stroke();
      }
    }

    function colorFor(cluster) {
      if (cluster < 0) return '#9aa3a0';
      return colors[cluster % colors.length];
    }

    function clusterInfo(cluster) {
      if (!atlas) return null;
      const clusters = atlas.clusters?.[state.word] || [];
      return clusters.find(item => item.cluster === cluster) || null;
    }

    function sourceLabel(point) {
      if (point.isSynthetic) return 'auto-context';
      if (point.isPasted) return 'pasted';
      return 'corpus';
    }

    function nearestPoint(event) {
      if (!atlas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let best = null;
      let bestDistance = 14;
      for (const point of visiblePoints()) {
        const p = project(point);
        const distance = Math.hypot(p.x - x, p.y - y);
        if (distance < bestDistance) {
          best = point;
          bestDistance = distance;
        }
      }
      return best;
    }

    function populateControls() {
      fillSelect('wordSelect', atlas.words.map(word => [word, word]), state.word);
      updateDomainControl();
    }

    function updateDomainControl() {
      const domains = [...new Set(pointsForWord().map(point => point.domain))].sort();
      fillSelect('domainSelect', [['all', 'all'], ...domains.map(domain => [domain, domain])], state.domain);
    }

    function fillSelect(id, options, selected) {
      const select = document.getElementById(id);
      select.innerHTML = '';
      for (const [value, label] of options) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        option.selected = value === selected;
        select.appendChild(option);
      }
      select.disabled = options.length === 1 && options[0][0] === '';
    }

    function renderMetrics() {
      if (!atlas) return;
      const metrics = atlas.metrics[state.word];
      const entries = [
        ['anti-essence', metrics.anti_essence_score.toFixed(2)],
        ['clusters', metrics.cluster_count],
        ['medoid', metrics.medoid_dominance.toFixed(2)],
        ['diameter', metrics.semantic_diameter.toFixed(2)],
        ['noise', Math.round(metrics.noise_ratio * 100) + '%'],
        ['contexts', metrics.occurrences]
      ];
      document.getElementById('metrics').innerHTML = entries.map(([label, value]) =>
        `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
      ).join('');
    }

    function renderEvidence() {
      if (!atlas) return;
      const evidence = atlas.metadata.evidence?.[state.word];
      if (!evidence) {
        document.getElementById('evidenceList').innerHTML = '';
        return;
      }
      document.getElementById('evidenceList').innerHTML = `<div class="evidence-card">
        <div class="evidence-title">${escapeHtml(evidence.label)}</div>
        <div class="evidence-row">
          <span class="evidence-chip">corpus ${evidence.corpus}</span>
          <span class="evidence-chip">pasted ${evidence.pasted}</span>
          <span class="evidence-chip">auto ${evidence.synthetic}</span>
        </div>
      </div>`;
    }

    function renderClusters() {
      const clusters = atlas.clusters?.[state.word] || [];
      document.getElementById('clusterList').innerHTML = clusters.sort((a, b) => a.cluster - b.cluster).map(info => {
        const cluster = info.cluster;
        const count = info.count;
        const label = info.cluster < 0 ? 'bridges / outliers' : info.name;
        const terms = (info.terms || []).join(', ');
        return `<div class="cluster-item">
          <span class="swatch" style="background:${colorFor(cluster)}"></span>
          <span class="cluster-text">
            <span class="cluster-name">${escapeHtml(label)}</span>
            <span class="cluster-terms">${escapeHtml(terms || 'no terms')}</span>
          </span>
          <strong>${count}</strong>
        </div>`;
      }).join('');
      document.getElementById('legend').innerHTML = clusters.map(info => {
        const cluster = info.cluster;
        const label = cluster < 0 ? 'bridges' : info.name;
        return `<span class="chip"><span class="swatch" style="background:${colorFor(cluster)}"></span>${escapeHtml(label)}</span>`;
      }).join('');
    }

    function renderSelected(point) {
      const box = document.getElementById('selectedPoint');
      if (!point) {
        point = pointsForWord().find(item => item.isMedoid) || pointsForWord()[0];
      }
      if (!point) {
        box.innerHTML = '';
        return;
      }
      box.innerHTML = `<div class="meta">
          <span>${point.domain}</span>
          <span>${escapeHtml(point.clusterName || (point.cluster < 0 ? 'noise' : 'cluster ' + point.cluster))}</span>
          <span>${point.matchedForm || point.word}</span>
          <span>${sourceLabel(point)}</span>
          <span>${point.isMedoid ? 'medoid' : 'context'}</span>
        </div>
        <div class="sentence">${escapeHtml(point.sentence)}</div>`;
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[char]));
    }

    function refresh() {
      if (!atlas) return;
      updateDomainControl();
      renderEvidence();
      renderMetrics();
      renderClusters();
      renderSelected(atlas.points.find(point => point.id === state.selectedId));
      draw();
    }

    document.getElementById('queryForm').addEventListener('submit', event => {
      event.preventDefault();
      analyze();
    });
    document.getElementById('wordSelect').addEventListener('change', event => {
      state.word = event.target.value;
      state.domain = 'all';
      state.selectedId = null;
      refresh();
    });
    document.getElementById('domainSelect').addEventListener('change', event => {
      state.domain = event.target.value;
      state.selectedId = null;
      refresh();
    });
    document.getElementById('downloadJson').addEventListener('click', () => {
      if (!atlas) return;
      const blob = new Blob([JSON.stringify(atlas, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'family_resemblance_atlas.json';
      a.click();
      URL.revokeObjectURL(url);
    });
    canvas.addEventListener('mousemove', event => {
      const point = nearestPoint(event);
      state.hoverId = point ? point.id : null;
      if (point) {
        tooltip.style.left = event.clientX - mapArea.getBoundingClientRect().left + 'px';
        tooltip.style.top = event.clientY - mapArea.getBoundingClientRect().top + 'px';
        tooltip.style.display = 'block';
        tooltip.innerHTML = `<strong>${point.word}</strong> | ${point.domain} | ${escapeHtml(point.clusterName || (point.cluster < 0 ? 'noise' : 'cluster ' + point.cluster))}<br>${escapeHtml(point.sentence)}`;
      } else {
        tooltip.style.display = 'none';
      }
      draw();
    });
    canvas.addEventListener('mouseleave', () => {
      state.hoverId = null;
      tooltip.style.display = 'none';
      draw();
    });
    canvas.addEventListener('click', event => {
      const point = nearestPoint(event);
      if (point) {
        state.selectedId = point.id;
        renderSelected(point);
        draw();
      }
    });
    window.addEventListener('resize', draw);
    analyze();
  </script>
</body>
</html>
"""


class ModelCache:
    def __init__(self, base_config: AtlasConfig) -> None:
        self.base_config = base_config
        self.key: tuple[str, int, int, int, int] | None = None
        self.model: CorpusModel | None = None

    def get(self, corpus: Path) -> CorpusModel:
        config = replace(self.base_config, corpus=corpus)
        key = (
            str(corpus.resolve()),
            config.embedding_dim,
            config.max_vocab,
            config.min_count,
            config.training_window,
        )
        if self.model is None or self.key != key:
            self.model = CorpusModel.train(config)
            self.key = key
        return self.model


def split_words(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;\n]+", value) if part.strip()]


def resolve_corpus(value: str) -> Path:
    path = Path(value.strip() or "data/sample_corpus").expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def pasted_documents(text: str, domain: str) -> list[Document]:
    text = text.strip()
    if not text:
        return []
    clean_domain = re.sub(r"[^0-9A-Za-z\u0401\u0451\u0410-\u044f_-]+", "_", domain.strip() or "pasted")
    clean_domain = clean_domain.strip("_") or "pasted"
    return [
        Document(
            domain=clean_domain,
            source=f"pasted://{clean_domain}",
            text=text,
        )
    ]


def make_handler(cache: ModelCache):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_text(200, DASHBOARD_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "defaultCorpus": str(cache.base_config.corpus),
                        "externalApi": False,
                    },
                )
                return
            self.send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/analyze":
                self.send_json(404, {"error": "Not found"})
                return

            try:
                payload = self.read_json()
                corpus = resolve_corpus(str(payload.get("corpus", "data/sample_corpus")))
                words = split_words(str(payload.get("words", "")))
                match_mode = str(payload.get("matchMode", "exact"))
                auto_context = bool(payload.get("autoContext", True))
                extra_documents = pasted_documents(
                    str(payload.get("pastedText", "")),
                    str(payload.get("pastedDomain", "pasted")),
                )
                min_cluster_size = int(payload.get("minClusterSize", cache.base_config.min_cluster_size))
                if match_mode not in {"exact", "prefix", "lemma-lite"}:
                    raise ValueError("Unsupported matching mode.")
                if min_cluster_size < 2:
                    raise ValueError("Minimum cluster size must be at least 2.")

                model = cache.get(corpus)
                atlas = model.analyze(
                    words,
                    match_mode=match_mode,
                    min_cluster_size=min_cluster_size,
                    allow_synthetic=auto_context,
                    extra_documents=extra_documents,
                )
                self.send_json(200, atlas)
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def send_text(self, status: int, text: str, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Family Resemblance dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "sample_corpus")
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--training-window", type=int, default=5)
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_handle = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle

    config = AtlasConfig(
        corpus=args.corpus.resolve(),
        embedding_dim=args.embedding_dim,
        max_vocab=args.max_vocab,
        min_count=args.min_count,
        training_window=args.training_window,
        context_window=args.context_window,
        neighbors=args.neighbors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    cache = ModelCache(config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cache))
    url = f"http://{args.host}:{args.port}/"
    print(f"Family Resemblance Dashboard: {url}", flush=True)
    try:
        server.serve_forever()
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
