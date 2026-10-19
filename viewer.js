// Body-shape change viewer: STL meshes per scan + per-corner signed change (SYNTHETIC data).
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

const params = new URLSearchParams(location.search);
const el = (id) => document.getElementById(id);
const GIRTHS = { waist: "--waist", hip: "--hip", thigh: "--thigh" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
if (params.get("ui") === "0") document.body.classList.add("capture");

const view = el("view");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(css("--bg"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => (scene.background = new THREE.Color(css("--bg"))));
const camera = new THREE.PerspectiveCamera(30, 1, 10, 20000);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666670, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(800, 1200, 2000);
scene.add(sun);

// data is z-up in mm; three.js is y-up
const body = new THREE.Group();
body.rotation.x = -Math.PI / 2;
scene.add(body);
const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0 });
const plain = new THREE.MeshStandardMaterial({ color: 0xc9ced8, roughness: 0.75 });
const ghostMat = new THREE.MeshBasicMaterial({ color: 0x7a8394, transparent: true, opacity: 0.18, depthWrite: false });
const mesh = new THREE.Mesh(undefined, material);
const ghost = new THREE.Mesh(undefined, ghostMat);
body.add(mesh, ghost);

let manifest, cache = {};
const loader = new STLLoader();

function diverging(t) {
  // t in [-1, 1]; blue (loss) -> near-white -> red (gain), RdBu-like stops
  const stops = [[-1, [33, 102, 172]], [-0.5, [103, 169, 207]], [0, [247, 247, 247]], [0.5, [239, 138, 98]], [1, [178, 24, 43]]];
  t = Math.max(-1, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i], u = (t - t0) / (t1 - t0);
      return c0.map((c, k) => (c + (c1[k] - c) * u) / 255);
    }
  }
  return stops[stops.length - 1][1].map((c) => c / 255);
}

function loadScan(sid, week) {
  // cache the promise, so a scan requested twice while loading is fetched once
  return (cache[`${sid}/${week}`] ??= fetchScan(sid, week));
}

async function fetchScan(sid, week) {
  const ww = String(week).padStart(2, "0");
  const [geom, buf] = await Promise.all([
    loader.loadAsync(`data/${sid}/week${ww}.stl`),
    fetch(`data/${sid}/week${ww}.disp.f32`).then((r) => r.arrayBuffer()),
  ]);
  const disp = new Float32Array(buf);
  const colours = new Float32Array(disp.length * 3);
  for (let i = 0; i < disp.length; i++) {
    const c = Number.isNaN(disp[i]) ? [0.62, 0.64, 0.68] : diverging(disp[i] / manifest.display_range_mm);
    colours.set(c, i * 3);
  }
  geom.deleteAttribute("normal");
  geom.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  const smooth = mergeVertices(geom, 1e-3);  // weld corners so normals are smooth
  smooth.computeVertexNormals();
  smooth.computeBoundingBox();
  return smooth;
}

function frame(geom) {
  const box = geom.boundingBox, c = new THREE.Vector3();
  box.getCenter(c);
  const h = box.max.z - box.min.z;
  const target = new THREE.Vector3(c.x, c.z, -c.y);  // after the -90 deg rotation about x
  camera.position.set(target.x + h * 0.55, target.y + h * 0.08, target.z + h * 2.6);
  controls.target.copy(target);
  controls.update();
}

function chart(s, k) {
  const svg = el("chart"), W = 300, H = 150, L = 34, R = 16, T = 8, B = 20;
  const weeks = manifest.weeks;
  const series = Object.keys(GIRTHS).map((n) => s.girths_mm[n].tape.map((v) => v - s.girths_mm[n].tape[0]));
  const lo = Math.min(...series.flat(), -10), hi = 2;
  const x = (w) => L + (w / weeks[weeks.length - 1]) * (W - L - R);
  const y = (v) => T + ((hi - v) / (hi - lo)) * (H - T - B);
  let out = `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="${css("--line")}"/>`;
  for (const v of [Math.ceil(lo / 20) * 20, Math.round(lo / 2 / 10) * 10]) out += `<text x="2" y="${y(v) + 3}">${v}</text>`;
  weeks.forEach((w) => (out += `<text x="${x(w) - 6}" y="${H - 5}">${w}w</text>`));
  out += `<line x1="${x(weeks[k])}" x2="${x(weeks[k])}" y1="${T}" y2="${H - B}" stroke="${css("--muted")}" stroke-dasharray="3 3"/>`;
  Object.values(GIRTHS).forEach((colVar, i) => {
    const pts = series[i].map((v, j) => `${x(weeks[j])},${y(v)}`).join(" ");
    out += `<polyline points="${pts}" fill="none" stroke="${css(colVar)}" stroke-width="2"/>`;
    out += `<circle cx="${x(weeks[k])}" cy="${y(series[i][k])}" r="3.5" fill="${css(colVar)}"/>`;
  });
  svg.innerHTML = out;
}

function panel(s, k) {
  el("mass").textContent = s.mass_kg[k].toFixed(1);
  el("dmass").textContent = `${s.mass_change_pct[k] > 0 ? "+" : ""}${s.mass_change_pct[k].toFixed(1)}%`;
  el("bmi").textContent = s.mesh_bmi[k].toFixed(1);
  el("height").textContent = s.height_m.toFixed(2);
  let rows = "<tr><th>girth</th><th>tape (mm)</th><th>Δ vs wk 0</th></tr>";
  for (const [n, colVar] of Object.entries(GIRTHS)) {
    const g = s.girths_mm[n].tape, d = g[k] - g[0];
    rows += `<tr><td><span style="color:${css(colVar)}">●</span> ${n}</td><td>${g[k].toFixed(0)}</td><td>${d > 0 ? "+" : ""}${d.toFixed(0)}</td></tr>`;
  }
  el("girths").innerHTML = rows;
  chart(s, k);
}

async function show(refit = false) {
  const s = manifest.subjects.find((x) => x.id === el("subject").value);
  const k = Number(el("week").value), week = manifest.weeks[k];
  const [geom, base] = await Promise.all([loadScan(s.id, week), loadScan(s.id, manifest.weeks[0])]);
  mesh.geometry = geom;
  mesh.material = el("colour").checked ? material : plain;
  ghost.geometry = base;
  ghost.visible = el("ghost").checked && k > 0;
  el("weekbadge").textContent = `week ${week}`;
  if (refit) frame(geom);
  panel(s, k);
  renderer.render(scene, camera);
  window.viewerReady = true;
}

function resize() {
  const { clientWidth: w, clientHeight: h } = view;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}

async function main() {
  manifest = await fetch("data/manifest.json").then((r) => r.json());
  el("lo").textContent = `−${manifest.display_range_mm} mm`;
  el("hi").textContent = `+${manifest.display_range_mm} mm`;
  el("ticks").innerHTML = manifest.weeks.map((w) => `<span>week ${w}</span>`).join("");
  for (const s of manifest.subjects) {
    el("subject").add(new Option(`${s.id} · ${s.sex} · ${s.height_m.toFixed(2)} m · BMI ${s.mesh_bmi[0]}`, s.id));
  }
  if (params.get("subject")) el("subject").value = params.get("subject");
  if (params.get("week")) el("week").value = Math.max(0, manifest.weeks.indexOf(Number(params.get("week"))));
  if (params.get("ghost") === "1") el("ghost").checked = true;
  el("subject").addEventListener("change", () => show(true));
  ["week", "colour", "ghost"].forEach((id) => el(id).addEventListener("input", () => show()));
  window.addEventListener("resize", resize);
  resize();
  await show(true);
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}

main();
