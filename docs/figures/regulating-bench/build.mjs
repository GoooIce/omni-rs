import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "/Users/goooice/.agents/skills/anatomy/kit/iso-kit.mjs";
import { kitScript } from "/Users/goooice/.agents/skills/anatomy/scripts/inline-kit.mjs";

// One Burn runtime on this Mac, six trains already cut. The dial reads the
// numbers the repo has actually measured. Sources are named in the caption.
// Metal ASR: README, 6.74 s, RTF 0.379, p95 82.4 ms.
// ANE long run: docs/ane-optimization-goal.md, 30 min, RTF 1.0174, p95 293.70 ms,
//   backlog 33.07 s. Acceptance was RTF ≤ 0.8 and p95 ≤ 160 ms. Not met.
// Web: README, 960 ms tier, RTF about 0.75 on a 20 s take. 160 ms still loses.
// Image: README, 8 steps, 1024² generation 50.8 s.
// Pages: README, 16 tokens/page, pool 16384, prefix cache 4096.
// Music: README 48 kHz; VALIDATION.md, one request at a time.
// Cutter: W4A16 is the mounted MiniCPM-V checkpoint. The drawer does not
// hold a banned implementation; ADR-0003 picks whichever is faster on the same job.

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";

const WIDTH = 700;
const HEIGHT = 430;

const PLATE = { x: -158, y: -98, w: 316, d: 186, r: 9 };
const PLATE_H = 12;
const FOOT_H = 5;

const HUB = [26, -2];
const BOSS_R = 18;
const BOSS_TOP = 11;
const HOLE_R = 13;
const STATION_R = 56;

const DIAL_R = 24;
const DIAL_Z = 72;
const DIAL_H = 11;
const DIAL_TOP = DIAL_Z + DIAL_H;
const STEM_END = 50;
const LIFT0 = 22;
const LIFT_K = 16;
const RTF_MAX = 1.4;
const ARC_FROM = -2.2;
const ARC_SPAN = 4.15;
const RING_R = DIAL_R - 2.4;

const SEAT_Z = 30;
const TRAVEL = 20;
const BACKLOG_FULL = 40;
const BOB_H = 8;

const DRAWER = { x: -146, y: 46, w: 68, d: 34, r: 2 };
const DRAWER_H = 8;
const DRAWER_PULL = 20;

const TRAINS = [
  { id: "metal", deg: 206, kind: "rtf", rtf: 0.379, p95: 82.4, name: "金属" },
  { id: "ane", deg: 326, kind: "rtf", rtf: 1.0174, p95: 293.7, backlog: 33.07, name: "神经引擎" },
  { id: "web", deg: 34, kind: "rtf", rtf: 0.75, chunk: 960, name: "浏览器" },
  { id: "image", deg: 106, kind: "press", steps: 8, seconds: 50.8, name: "图像" },
  { id: "pages", deg: 154, kind: "pages", page: 16, pool: 16384, cache: 4096, name: "译页" },
  { id: "music", deg: 228, kind: "single", rate: 48000, name: "乐曲" },
];

const rad = (deg) => (deg * Math.PI) / 180;
const place = (deg, radius) => [HUB[0] + Math.cos(rad(deg)) * radius, HUB[1] + Math.sin(rad(deg)) * radius];
const rtfAngle = (rtf) => ARC_FROM + (rtf / RTF_MAX) * ARC_SPAN;
const spoke = (angle) => [Math.sin(angle), -Math.cos(angle)];

const P = k.fitProjection(
  [
    ...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0),
    [HUB[0], HUB[1], DIAL_TOP + 6],
    [-118, -30, 44],
    [DRAWER.x + DRAWER.w, DRAWER.y + DRAWER.d + DRAWER_PULL, DRAWER_H + 12],
    [HUB[0], HUB[1] - 44, DIAL_Z + 16],
    [...place(326, STATION_R + 18), 56],
  ],
  WIDTH,
  HEIGHT,
  { pad: 28, azimuth: 44, elevation: 33 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const mark = (svg, id) => svg.replace("<g ", `<g data-watch="${id}" `);

function screws(points, z, size = 0.48) {
  return D(points.map(([x, y]) => at([x, y, z])), { size, tone: "mid" });
}

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 26))) {
    out.push(S(k.cylinder(x, y, 7.2, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.2, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 6), 0, P), { tone: "lo" }));
  const corners = k.corners(PLATE, 12);
  out.push(L(corners.map(([x, y]) => k.ring(x, y, 2.3, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(screws(corners, 0, 0.5));

  const strip = { x: -118, y: PLATE.y + 12, w: 196, d: 8, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.7, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(-96, 64, 4, 4, strip.y + 0.7, 0.7, [2.1, 4.2], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(screws([[strip.x + 3, strip.y + 4], [strip.x + strip.w - 3, strip.y + 4]], 0.7, 0.42));
  out.push(L(k.ring(HUB[0], HUB[1], 34, 0, P, 72), { tone: "lo" }));
  out.push(L(k.ring(HUB[0], HUB[1], 36.2, 0, P, 72), { tone: "faint" }));

  const label = { x: 78, y: 58, w: 42, d: 16, r: 1.4 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.5), 0.8, P), { tone: "faint" }));
  out.push(L(k.lineOnTop([label.x + 4, label.y + 5], [label.x + 30, label.y + 5], 0.85, P), { tone: "lo", free: true }));
  out.push(L(k.lineOnTop([label.x + 4, label.y + 9], [label.x + 24, label.y + 9], 0.85, P), { tone: "lo", free: true }));
  out.push(L(k.lineOnTop([label.x + 4, label.y + 13], [label.x + 18, label.y + 13], 0.85, P), { tone: "lo", free: true }));
  out.push(screws([[label.x + 3, label.y + 3], [label.x + label.w - 3, label.y + label.d - 3]], 0.8, 0.4));
  return out.join("");
}

function headstock() {
  const out = [];
  const plinth = { x: -148, y: -56, w: 78, d: 52, r: 3 };
  const top = 11;
  out.push(S(k.slabOf(plinth, 0, top, P, 6, 0.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(plinth, 3), top, P), { tone: "faint" }));
  out.push(screws([[plinth.x + 5, plinth.y + 5], [plinth.x + plinth.w - 5, plinth.y + 5], [plinth.x + 5, plinth.y + plinth.d - 5], [plinth.x + plinth.w - 5, plinth.y + plinth.d - 5]], top, 0.45));

  const bx = -122;
  const by = -32;
  out.push(S(k.cylinder(bx, by, 15, top, 3, P, 40, 0.4), { tone: "mid" }));
  out.push(S(k.cylinder(bx, by, 12.5, top + 3, 14, P, 40), { tone: "hi", crease: "lo" }));
  out.push(L([6, 12].map((z) => k.sideArc(bx, by, 12.5, top + 3 + z, P, 28)).join(""), { tone: "faint" }));
  out.push(L(k.knurl(bx, by, 12.5, top + 14, top + 17, 42, P, 0.35), { tone: "lo" }));
  out.push(S(k.cylinder(bx, by, 13.2, top + 17, 3.2, P, 40, 0.4), { tone: "mid" }));
  out.push(S(k.slabOf({ x: bx - 3.2, y: by - 3.2, w: 6.4, d: 6.4, r: 0.6 }, top + 20.2, 3.4, P, 2), { tone: "mid" }));
  out.push(L(k.lineOnTop([bx - 2.2, by], [bx + 2.2, by], top + 23.6, P), { tone: "hi" }));
  out.push(D([at([bx, by, top + 23.6])], { size: 0.45, tone: "hi" }));

  const pinions = [[-98, 4.4], [-86, 3.5], [-76, 2.8]];
  out.push(S(k.slabOf({ x: -106, y: by - 5.5, w: 38, d: 11, r: 2 }, top, 3.2, P, 4, 0.4), { tone: "mid" }));
  for (const [x, radius] of pinions) {
    out.push(S(k.cylinder(x, by, radius + 1.1, top + 3.2, 1.4, P, 24), { tone: "mid" }));
    out.push(S(k.cylinder(x, by, radius, top + 4.6, 5, P, 24), { tone: "hi", crease: "lo" }));
    const teeth = k.radialTicks(x, by, radius - 0.2, radius > 4 ? 10 : 8, 20, top + 9.6, [1.3, 1.3], P);
    out.push(L(k.ring(x, by, radius - 0.2, top + 9.6, P, 24), { tone: "lo" }));
    out.push(L(teeth.minor, { tone: "mid" }));
    out.push(D([at([x, by, top + 9.6])], { size: 0.4, tone: "mid" }));
  }

  const cup = [plinth.x + plinth.w - 8, plinth.y + 8];
  out.push(S(k.cylinder(cup[0], cup[1], 2.4, top, 3.2, P, 16), { tone: "mid" }));
  out.push(L(k.ring(cup[0], cup[1], 1.5, top + 3.2, P, 16), { tone: "faint" }));

  out.push(L(k.ring(bx, by, 12.5, top + 3, P, 36), { tone: "lo" }));
  const edge = [bx + 15 * Math.cos(0.9), by + 15 * Math.sin(0.9)];
  const pivot = [edge[0] + 6.5, edge[1] + 2.5];
  out.push(D([at([pivot[0], pivot[1], top])], { size: 0.55, tone: "mid" }));
  out.push(L(k.lineOnTop(pivot, edge, top, P), { tone: "lo" }));

  const tag = { x: bx - 10, y: by + 16, w: 22, d: 8, r: 1 };
  out.push(S(k.slabOf(tag, top, 0.7, P, 3), { tone: "lo", crease: "none" }));
  out.push(k.faceTextSvg(k.topMatrix([tag.x + 11, tag.y + 4], top + 0.75, P, "x"), "W4A16", { size: 2.3, tone: "mid", anchor: "middle" }));
  return out.join("");
}

function pad(x, y) {
  const plan = { x: x - 11, y: y - 11, w: 22, d: 22, r: 2.2 };
  return {
    top: 4.2,
    svg: [
      S(k.slabOf(plan, 0, 4.2, P, 4, 0.45), { tone: "mid" }),
      screws([[x - 6, y - 6], [x + 6, y + 6]], 4.2, 0.4),
    ].join(""),
  };
}

function gear(x, y, z, radius, teeth, tone) {
  const out = [];
  out.push(S(k.cylinder(x, y, radius, z, 3.4, P, 28), { tone, crease: "lo" }));
  out.push(L(k.ring(x, y, radius - 0.15, z + 3.4, P, 28), { tone: "lo" }));
  const ticks = k.radialTicks(x, y, radius - 0.15, teeth, 40, z + 3.4, [1.5, 1.5], P);
  out.push(L(ticks.minor, { tone: "mid" }));
  out.push(D([at([x, y, z + 3.4])], { size: 0.42, tone: "hi" }));
  return out.join("");
}

function stationMetal(x, y) {
  const base = pad(x, y);
  const out = [base.svg];
  out.push(S(k.cylinder(x, y, 6.2, base.top, 2.2, P, 28), { tone: "mid" }));
  out.push(gear(x, y, base.top + 2.2, 4.6, 12, "hi"));
  return out.join("");
}

function stationAne(x, y) {
  const base = pad(x, y);
  const out = [base.svg];
  const dir = Math.atan2(y - HUB[1], x - HUB[0]);
  const px = x + Math.cos(dir) * 14 + Math.cos(dir + Math.PI / 2) * 8;
  const py = y + Math.sin(dir) * 14 + Math.sin(dir + Math.PI / 2) * 8;
  out.push(S(k.cylinder(px, py, 2.4, base.top, 2.2, P, 16), { tone: "mid" }));
  out.push(S(k.cylinder(px, py, 1.6, base.top + 2.2, 44, P, 16), { tone: "mid", crease: "none" }));
  out.push(L([14, 28, 40].map((z) => k.sideArc(px, py, 1.6, base.top + 2.2 + z, P, 12)).join(""), { tone: "faint" }));
  const hx = px + 10;
  const hy = py;
  out.push(S(k.slabOf({ x: px - 1.2, y: py - 1.3, w: 13.6, d: 2.6, r: 1.1 }, base.top + 44, 2.4, P, 2), { tone: "mid" }));
  out.push(L(k.ring(hx, hy, 1.5, base.top + 46.4, P, 12), { tone: "lo" }));
  out.push(D([at([hx, hy, base.top + 46.4])], { size: 0.5, tone: "mid" }));
  const strip = { x: hx + 7.4, y: hy - 1.4, w: 1.6, d: 2.8, r: 0.3 };
  out.push(S(k.slabOf(strip, 8, 36, P, 2), { tone: "lo", crease: "none" }));
  const ticks = [];
  for (let seconds = 0; seconds <= 40; seconds += 10) {
    const z = SEAT_Z - (seconds / BACKLOG_FULL) * TRAVEL;
    const big = seconds % 20 === 0;
    ticks.push(k.segment([strip.x + strip.w, strip.y + 1.3, z], [strip.x + strip.w + (big ? 2.6 : 1.5), strip.y + 1.3, z], P));
  }
  out.push(L(ticks.join(""), { tone: "mid" }));
  return { svg: out.join(""), hook: [hx, hy, base.top + 46.4], post: [px, py] };
}

function stationWeb(x, y) {
  const base = pad(x, y);
  const out = [base.svg];
  const dir = Math.atan2(y - HUB[1], x - HUB[0]);
  const side = dir + Math.PI / 2;
  const cx = x - Math.cos(dir) * 1.5;
  const cy = y - Math.sin(dir) * 1.5;
  const fx = x + Math.cos(side) * 11;
  const fy = y + Math.sin(side) * 11;
  out.push(S(k.cylinder(fx, fy, 1.5, base.top, 5, P, 14), { tone: "mid" }));
  out.push(gear(cx, cy, base.top, 6.4, 8, "hi"));
  out.push(gear(fx, fy, base.top + 5, 3.6, 16, "mid"));
  return out.join("");
}

function stationImage(x, y) {
  const base = pad(x, y);
  const out = [base.svg];
  const plan = { x: x - 5.5, y: y - 5.5, w: 11, d: 11, r: 1 };
  for (let step = 0; step < 8; step++) {
    const z = base.top + step * 1.15;
    out.push(S(k.slabOf(plan, z, 1, P, 3, 0.15), { tone: "hi", crease: "lo" }));
  }
  const top = base.top + 8 * 1.15;
  out.push(S(k.cylinder(x - 8, y, 1.3, base.top, top - base.top + 6, P, 12), { tone: "mid" }));
  out.push(S(k.cylinder(x + 8, y, 1.3, base.top, top - base.top + 6, P, 12), { tone: "mid" }));
  out.push(S(k.slabOf({ x: x - 8, y: y - 1.4, w: 16, d: 2.8, r: 1 }, top + 4.6, 2.2, P, 2), { tone: "mid" }));
  out.push(mark(S(k.cylinder(x, y, 3.2, top, 4.4, P, 20), { tone: "hi", crease: "lo" }), "image"));
  out.push(L(k.ring(x, y, 2.2, top + 4.4, P, 16), { tone: "faint" }));
  return out.join("");
}

function stationPages(x, y) {
  const base = pad(x, y);
  const out = [base.svg];
  const pool = { x: x - 13, y: y - 5, w: 26, d: 10, r: 1 };
  out.push(S(k.slabOf(pool, base.top, 5.2, P, 4, 0.35), { tone: "hi", crease: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(pool, 1.2), base.top + 5.2, P), { tone: "faint" }));
  const cache = { x: pool.x, y: pool.y + 1.2, w: 6.5, d: 7.6, r: 0.8 };
  out.push(mark(S(k.slabOf(cache, base.top + 5.2, 2.4, P, 3, 0.2), { tone: "hi", crease: "lo" }), "pages"));
  out.push(S(k.cylinder(cache.x + 3.2, cache.y + 3.8, 1.1, base.top + 7.6, 3.2, P, 12), { tone: "mid" }));
  out.push(L(k.knurl(cache.x + 3.2, cache.y + 3.8, 1.1, base.top + 9.2, base.top + 10.8, 12, P, 0.2), { tone: "lo" }));
  out.push(screws([[pool.x + 2, pool.y + 2], [pool.x + pool.w - 2, pool.y + pool.d - 2]], base.top + 5.2, 0.38));
  return out.join("");
}

function stationMusic(x, y) {
  const base = pad(x, y);
  const out = [base.svg];
  const dir = Math.atan2(y - HUB[1], x - HUB[0]);
  const side = dir + Math.PI / 2;
  const wx = x + Math.cos(side) * 12;
  const wy = y + Math.sin(side) * 12;
  out.push(S(k.cylinder(x, y, 5.4, base.top, 3.6, P, 24), { tone: "mid" }));
  out.push(mark(S(k.cylinder(x, y, 4.2, base.top + 3.6, 1.3, P, 24), { tone: "hi", crease: "lo" }), "music"));
  out.push(L(k.ring(x, y, 2.4, base.top + 4.9, P, 16), { tone: "faint" }));
  out.push(S(k.slabOf({ x: wx - 6, y: wy - 1.3, w: 12, d: 2.6, r: 0.8 }, base.top, 2.4, P, 2), { tone: "mid" }));
  out.push(S(k.cylinder(wx, wy, 3.4, base.top + 2.4, 1.1, P, 20), { tone: "mid", crease: "lo" }));
  out.push(D([at([wx, wy, base.top + 3.5])], { size: 0.4, tone: "lo" }));
  return out.join("");
}

function boss() {
  const out = [];
  out.push(S(k.cylinder(HUB[0], HUB[1], BOSS_R, 0, BOSS_TOP, P, 48, 0.6), { tone: "mid" }));
  out.push(L(k.ring(HUB[0], HUB[1], 8, BOSS_TOP, P, 32), { tone: "lo" }));
  const holes = TRAINS.map((train) => place(train.deg, HOLE_R));
  out.push(L(holes.map(([x, y]) => k.ring(x, y, 1.7, BOSS_TOP, P, 14)).join(""), { tone: "lo" }));
  out.push(screws([0, 90, 180, 270].map((deg) => place(deg, BOSS_R - 4)), BOSS_TOP, 0.42));
  return out.join("");
}

function dial() {
  const out = [];
  const [x, y] = HUB;
  const post = [x, y - 42];
  out.push(S(k.cylinder(post[0], post[1], 5.2, 0, 4, P, 24), { tone: "mid" }));
  out.push(S(k.cylinder(post[0], post[1], 3.2, 4, DIAL_Z - 10, P, 24), { tone: "mid", crease: "none" }));
  out.push(L([18, 36, 54].map((z) => k.sideArc(post[0], post[1], 3.2, z, P, 16)).join(""), { tone: "faint" }));
  out.push(S(k.slabOf({ x: post[0] - 6.5, y: post[1] - 6.5, w: 13, d: 13, r: 2 }, DIAL_Z - 8, 12, P, 4, 0.5), { tone: "mid" }));
  out.push(S(k.cylinder(post[0], post[1], 3.2, DIAL_Z + 4, 10, P, 20), { tone: "mid" }));
  out.push(S(k.cylinder(post[0], post[1], 4.2, DIAL_Z + 14, 2.2, P, 20), { tone: "mid" }));
  out.push(S(k.slabOf({ x: x - 3.1, y: post[1], w: 6.2, d: y - post[1], r: 2 }, DIAL_Z - 3, 5, P, 4, 0.45), { tone: "mid" }));
  out.push(screws([[x - 1.4, post[1] + 6], [x + 1.4, y - 8]], DIAL_Z + 2, 0.4));
  out.push(S(k.cylinder(x, y, 2.2, STEM_END, DIAL_Z - STEM_END, P, 16), { tone: "mid", crease: "none" }));
  out.push(L(k.sideArc(x, y, 2.2, STEM_END + 8, P, 12), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 64, 0.7), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 3.2, DIAL_TOP, 72, P, 0.28), { tone: "lo" }));
  out.push(L(k.sideArc(x, y, DIAL_R, DIAL_TOP - 3.2, P, 40), { tone: "faint" }));
  out.push(L(k.ring(x, y, RING_R, DIAL_TOP, P, 64), { tone: "lo" }));
  const ticks = k.radialTicks(x, y, RING_R, 14, 5, DIAL_TOP, [1.35, 2.7], P, ARC_FROM, ARC_SPAN);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  for (const rtf of [0.8, 1]) {
    const angle = rtfAngle(rtf);
    const [sx, sy] = spoke(angle);
    out.push(L(k.lineOnTop([x + sx * (RING_R - 4.6), y + sy * (RING_R - 4.6)], [x + sx * RING_R, y + sy * RING_R], DIAL_TOP, P), { tone: "hi" }));
    out.push(D([at([x + sx * (RING_R - 4.6), y + sy * (RING_R - 4.6), DIAL_TOP])], { size: 0.55, tone: "hi" }));
  }
  for (const rtf of [-0.06, RTF_MAX + 0.06]) {
    const angle = rtfAngle(rtf);
    const [sx, sy] = spoke(angle);
    out.push(D([at([x + sx * RING_R, y + sy * RING_R, DIAL_TOP])], { size: 0.85, tone: "hi" }));
  }
  out.push(L(k.ring(x, y, 1.7, DIAL_TOP, P, 16), { tone: "mid" }));
  out.push(S(k.slabOf({ x: x + DIAL_R - 1, y: y - 2.2, w: 4.2, d: 4.4, r: 1 }, DIAL_Z + 2, 4.2, P, 3), { tone: "mid" }));
  out.push(S(k.cylinder(x + DIAL_R + 3.4, y, 1.8, DIAL_Z + 1.4, 5, P, 14), { tone: "mid" }));
  out.push(L(k.knurl(x + DIAL_R + 3.4, y, 1.8, DIAL_Z + 4.2, DIAL_Z + 6.4, 12, P, 0.2), { tone: "lo" }));
  return out.join("");
}

function crown() {
  const atFront = place(78, 24);
  const [x, y] = atFront;
  const out = [];
  out.push(S(k.cylinder(x, y, 3.2, 0, 3, P, 20), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, 5.4, 3, 4.2, P, 28), { tone: "mid" }));
  out.push(L(k.knurl(x, y, 5.4, 3.4, 6.6, 28, P, 0.3), { tone: "lo" }));
  out.push(L(k.ring(x, y, 5.2, 7.2, P, 28), { tone: "lo" }));
  out.push(D([at([x, y, 7.2])], { size: 0.48, tone: "mid" }));
  return { svg: out.join(""), at: [x, y], top: 7.2, radius: 5.2 };
}

function drawer() {
  const wall = 3;
  const out = [];
  out.push(S(k.slabOf({ x: DRAWER.x, y: DRAWER.y, w: wall, d: DRAWER.d, r: 0.6 }, 0, DRAWER_H, P, 3, 0.3), { tone: "mid" }));
  out.push(S(k.slabOf({ x: DRAWER.x + DRAWER.w - wall, y: DRAWER.y, w: wall, d: DRAWER.d, r: 0.6 }, 0, DRAWER_H, P, 3, 0.3), { tone: "mid" }));
  out.push(S(k.slabOf({ x: DRAWER.x, y: DRAWER.y, w: DRAWER.w, d: wall, r: 0.6 }, 0, DRAWER_H, P, 3, 0.3), { tone: "mid" }));
  return out.join("");
}

function drawerBox() {
  const box = { x: DRAWER.x + 4.2, y: DRAWER.y + 4.2, w: DRAWER.w - 8.4, d: DRAWER.d - 6, r: 1.2 };
  const out = [];
  out.push(S(k.slabOf(box, 0, 5, P, 4, 0.4), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(box, 1.8), 5, P), { tone: "faint" }));
  const cutter = [box.x + 16, box.y + 14];
  out.push(L(k.ring(cutter[0], cutter[1], 4.4, 5, P, 20), { tone: "lo" }));
  out.push(mark(S(k.cylinder(cutter[0], cutter[1], 3.2, 5, 12, P, 20), { tone: "hi", crease: "lo" }), "drawer"));
  out.push(L(k.knurl(cutter[0], cutter[1], 3.2, 5, 17, 18, P, 0), { tone: "lo" }));
  out.push(D([at([cutter[0], cutter[1], 17])], { size: 0.42, tone: "mid" }));
  out.push(L(k.lineOnTop([box.x + 26, box.y + 8], [box.x + 48, box.y + 8], 5.15, P), { tone: "lo", free: true }));
  out.push(L(k.lineOnTop([box.x + 26, box.y + 12], [box.x + 42, box.y + 12], 5.15, P), { tone: "lo", free: true }));
  out.push(L(k.lineOnTop([box.x + 26, box.y + 16], [box.x + 36, box.y + 16], 5.15, P), { tone: "lo", free: true }));
  const latch = [box.x + box.w - 8, box.y + box.d - 4];
  out.push(S(k.cylinder(latch[0], latch[1], 2.4, 5, 3.6, P, 16), { tone: "mid" }));
  out.push(L(k.knurl(latch[0], latch[1], 2.4, 6.2, 8.6, 14, P, 0.2), { tone: "lo" }));
  return out.join("");
}

const stations = {
  metal: stationMetal,
  web: stationWeb,
  image: stationImage,
  pages: stationPages,
  music: stationMusic,
};

const ane = stationAne(...place(TRAINS.find((train) => train.id === "ane").deg, STATION_R));
const CROWN = crown();

const layers = [];
for (const train of TRAINS) {
  if (train.id === "ane") continue;
  const [x, y] = place(train.deg, STATION_R);
  layers.push({ depth: k.depthOf([x, y, 6], P), svg: stations[train.id](x, y) });
}
layers.push({ depth: k.depthOf([...place(TRAINS[1].deg, STATION_R), 6], P), svg: ane.svg });
layers.push({ depth: k.depthOf([HUB[0], HUB[1], 20], P), svg: boss() });
layers.push({ depth: k.depthOf([-136, -40, 20], P), svg: headstock() });
layers.push({ depth: k.depthOf([HUB[0], HUB[1], DIAL_Z], P), svg: dial() });
layers.push({ depth: k.depthOf([...CROWN.at, 4], P), svg: CROWN.svg });
layers.push({ depth: k.depthOf([DRAWER.x + 40, DRAWER.y + 20, 4], P), svg: drawer() });
layers.sort((a, b) => a.depth - b.depth);

const anvil = "";

const needleAt = (angle) => {
  const [sx, sy] = spoke(angle);
  return k.lineOnTop(
    [HUB[0] - sx * 3.2, HUB[1] - sy * 3.2],
    [HUB[0] + sx * RING_R, HUB[1] + sy * RING_R],
    DIAL_TOP + 0.15,
    P,
  );
};

const shaftLine = (angle, radius, z) => {
  const [x, y] = CROWN.at;
  return k.lineOnTop([x, y], [x + Math.cos(angle) * radius, y + Math.sin(angle) * radius], z, P);
};

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "A regulating bench for one clock. A mainspring barrel drives a six-station index. The dial reads real-time factor for the speech trains, and parks off the scale for the press, the page stack and the single disc. A weight beside the neural-engine station drops by the measured backlog. A cutter marked W4A16 is mounted. The drawer holds spare cutters; the faster one on the same job replaces it.",
  body: [
    plate(),
    ...layers.map((layer) => layer.svg),
    `<g data-part="anvil">${anvil}</g>`,
    `<g data-part="tip"></g>`,
    `<g data-part="bob">${S(k.cylinder(ane.hook[0], ane.hook[1], 4.6, SEAT_Z, BOB_H, P, 22), { tone: "hi", crease: "lo" })}${D([at([ane.hook[0], ane.hook[1], SEAT_Z + BOB_H])], { size: 0.6, tone: "hi" })}</g>`,
    `<path data-part="cord" class="iso-line" data-tone="lo" d="" />`,
    `<g data-part="pawl">${S(k.cylinder(HUB[0] + HOLE_R, HUB[1], 1.45, BOSS_TOP, 2.6, P, 14), { tone: "hi", crease: "lo" })}${D([at([HUB[0] + HOLE_R, HUB[1], BOSS_TOP + 2.6])], { size: 0.5, tone: "hi" })}</g>`,
    `<g data-part="drawer" transform="${k.translateAlong(P, [0, 0, 0])}">${drawerBox()}</g>`,
    `<path data-part="needle" class="iso-line" data-tone="lit" data-free="start" d="${needleAt(rtfAngle(TRAINS[0].rtf))}" style="stroke-width:1.15"/>`,
    `<path data-part="crown-hand" class="iso-line" data-tone="hi" d="${shaftLine(rad(TRAINS[0].deg), CROWN.radius - 0.2, CROWN.top)}" />`,
    `<g data-part="index-dots"></g>`,
    `<path data-part="index-hand" class="iso-line" data-tone="hi" d="" />`,
  ],
});

const live = `
const P = ${JSON.stringify(P)};
const HUB = ${JSON.stringify(HUB)};
const TRAINS = ${JSON.stringify(TRAINS.map((train) => ({ ...train, angle: rad(train.deg) })))};
const DIAL_R = ${DIAL_R}, DIAL_TOP = ${DIAL_TOP}, STEM_END = ${STEM_END};
const LIFT0 = ${LIFT0}, LIFT_K = ${LIFT_K}, RTF_MAX = ${RTF_MAX}, ARC_FROM = ${ARC_FROM}, ARC_SPAN = ${ARC_SPAN};
const SEAT_Z = ${SEAT_Z}, TRAVEL = ${TRAVEL}, BACKLOG_FULL = ${BACKLOG_FULL}, BOB_H = ${BOB_H}, RING_R = ${RING_R};
const HOOK = ${JSON.stringify(ane.hook)};
const HOLE_R = ${HOLE_R}, BOSS_TOP = ${BOSS_TOP};
const CROWN = ${JSON.stringify(CROWN.at)};
const CROWN_TOP = ${CROWN.top}, CROWN_R = ${CROWN.radius};
const DRAWER_PULL = ${DRAWER_PULL};
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const anvil = svg.querySelector('[data-part="anvil"]');
const tip = svg.querySelector('[data-part="tip"]');
const bob = svg.querySelector('[data-part="bob"]');
const cord = svg.querySelector('[data-part="cord"]');
const pawl = svg.querySelector('[data-part="pawl"]');
const drawer = svg.querySelector('[data-part="drawer"]');
const needle = svg.querySelector('[data-part="needle"]');
const crownHand = svg.querySelector('[data-part="crown-hand"]');
const indexHand = svg.querySelector('[data-part="index-hand"]');
const indexDots = svg.querySelector('[data-part="index-dots"]');
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const spoke = (angle) => [Math.sin(angle), -Math.cos(angle)];
const rtfAngle = (rtf) => ARC_FROM + (rtf / RTF_MAX) * ARC_SPAN;
const rtfOf = (angle) => ((angle - ARC_FROM) / ARC_SPAN) * RTF_MAX;
const onArc = (angle) => { const r = rtfOf(angle); return r >= -0.012 && r <= RTF_MAX + 0.012; };
const needleAt = ${needleAt.toString().replace("k.lineOnTop", "lineOnTop")};
const follow = (value, target, dt, tau, calm) => {
  if (calm || Math.abs(target - value) < 0.0008) return target;
  return value + (target - value) * (1 - Math.exp(-dt / tau));
};
const approach = (value, target, dt, tau, calm) => {
  let delta = target - value;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  if (calm || Math.abs(delta) < 0.004) return target;
  return value + delta * (1 - Math.exp(-dt / tau));
};
const sentence = (train, backlog) => {
  if (train.id === "metal") return "金属 · RTF " + train.rtf.toFixed(3) + " · p95 " + train.p95 + " ms";
  if (train.id === "ane") return "神经引擎 · RTF " + train.rtf.toFixed(3) + " · 积压 " + backlog.toFixed(0) + " s";
  if (train.id === "web") return "浏览器 · " + train.chunk + " ms · RTF " + train.rtf.toFixed(2);
  if (train.id === "image") return "图像 · " + train.steps + " 步 · " + train.seconds + " s";
  if (train.id === "pages") return "译页 · " + train.page + " 一页 · 池 " + train.pool;
  return "乐曲 · " + (train.rate / 1000) + " kHz · 一次一首";
};
const describe = (train, open) => {
  if (open) return "抽屉拉开。没有封存的方案。同一件活上谁更快，谁就装到床头。";
  if (train.id === "metal") return "金属语音，RTF 0.379，在 0.8 的验收线以内。";
  if (train.id === "ane") return "神经引擎三十分钟长流，RTF 1.017，越过实时，积压 33 秒。";
  if (train.id === "web") return "浏览器粗齿 960 毫秒，大约 RTF 0.75。";
  if (train.id === "image") return "图像压床，八步，1024 见方生成 50.8 秒。";
  if (train.id === "pages") return "翻译页，每页 16 个 token，池 16384，前缀缓存 4096。";
  return "乐曲一次一首，48 kHz，没有连续合批。";
};
let index = 0, shaft = TRAINS[0].angle, hand = rtfAngle(TRAINS[0].rtf);
let drop = 0, drawerOpen = 0, drawerTarget = 0;
let frame = 0, last = 0, clock = 0, touring = !still.matches, visible = false, idle = 0;
let said = "";
const tourAt = (t) => Math.floor(t / 2.8) % TRAINS.length;
const nearerPin = (rtf) => (rtf > 0.7 ? rtfAngle(RTF_MAX + 0.06) : rtfAngle(-0.06));
function targets() {
  const train = TRAINS[index];
  return {
    train,
    shaft: train.angle,
    hand: train.kind === "rtf" ? rtfAngle(train.rtf) : nearerPin(rtfOf(hand)),
    drop: train.id === "ane" ? train.backlog / BACKLOG_FULL * TRAVEL : 0,
  };
}
function paint() {
  const train = TRAINS[index];
  const reading = rtfOf(hand);
  const lift = LIFT0 + Math.min(RTF_MAX + 0.06, Math.max(-0.06, reading)) * LIFT_K;
  const onScale = onArc(hand);
  anvil.innerHTML = solidSvg(cylinder(HUB[0], HUB[1], 5, BOSS_TOP, Math.max(1.4, lift - BOSS_TOP), P, 24, 0.35), { tone: "hi", crease: "lo" });
  tip.innerHTML = solidSvg(cylinder(HUB[0], HUB[1], 2.2, lift, Math.max(1.2, STEM_END - lift), P, 16), { tone: "hi", crease: "none", lit: onScale && drawerTarget === 0 });
  bob.setAttribute("transform", translateAlong(P, [0, 0, -drop]));
  cord.setAttribute("d", segment([HOOK[0], HOOK[1], HOOK[2]], [HOOK[0], HOOK[1], SEAT_Z + BOB_H - drop], P));
  const px = HUB[0] + Math.cos(shaft) * HOLE_R;
  const py = HUB[1] + Math.sin(shaft) * HOLE_R;
  pawl.setAttribute("transform", translateAlong(P, [px - (HUB[0] + HOLE_R), py - HUB[1], 0]));
  drawer.setAttribute("transform", translateAlong(P, [0, drawerOpen * DRAWER_PULL, 0]));
  needle.setAttribute("d", needleAt(hand));
  needle.setAttribute("data-tone", onScale && drawerTarget === 0 ? "lit" : "hi");
  crownHand.setAttribute("d", lineOnTop(CROWN, [CROWN[0] + Math.cos(shaft) * CROWN_R, CROWN[1] + Math.sin(shaft) * CROWN_R], CROWN_TOP, P));
  const handNear = 6.8;
  const handFar = HOLE_R - 1.75;
  indexHand.setAttribute("d", lineOnTop(
    [HUB[0] + Math.cos(shaft) * handNear, HUB[1] + Math.sin(shaft) * handNear],
    [HUB[0] + Math.cos(shaft) * handFar, HUB[1] + Math.sin(shaft) * handFar],
    BOSS_TOP + 0.2,
    P,
  ));
  indexDots.innerHTML = dotsSvg([
    iso([HUB[0] + Math.cos(shaft) * handNear, HUB[1] + Math.sin(shaft) * handNear, BOSS_TOP + 0.2], P),
    iso([HUB[0] + Math.cos(shaft) * handFar, HUB[1] + Math.sin(shaft) * handFar, BOSS_TOP + 0.2], P),
  ], { size: 0.55, tone: "hi" });
  const want = targets();
  let shaftGap = shaft - want.shaft;
  while (shaftGap > Math.PI) shaftGap -= Math.PI * 2;
  while (shaftGap < -Math.PI) shaftGap += Math.PI * 2;
  const pointed = Math.abs(shaftGap) < 0.02 && Math.abs(hand - want.hand) < 0.02;
  const drawerThere = Math.abs(drawerOpen - drawerTarget) < 0.02;
  const backlog = (drop / TRAVEL) * BACKLOG_FULL;
  let text;
  if (!drawerThere) text = drawerTarget > 0 ? "拉开抽屉" : "推回抽屉";
  else if (drawerTarget > 0 && pointed) text = "抽屉 · 谁快用谁";
  else if (pointed && train.id === "ane") text = "神经引擎 · RTF " + train.rtf.toFixed(3) + " · 积压 " + backlog.toFixed(0) + " s";
  else if (pointed) text = sentence(train, 0);
  else if (onScale) text = "指针 · RTF " + Math.max(0, reading).toFixed(3);
  else text = "离开刻度";
  if (readout.textContent !== text) readout.textContent = text;
  const next = describe(train, drawerTarget > 0);
  if (next !== said) { said = next; stage.setAttribute("aria-valuetext", next); }
  svg.querySelectorAll("[data-watch]").forEach((node) => {
    const on = drawerThere && pointed && drawerTarget === 0 && node.getAttribute("data-watch") === train.id;
    const show = drawerThere && drawerTarget > 0 && node.getAttribute("data-watch") === "drawer";
    if (on || show) node.setAttribute("data-lit", "");
    else node.removeAttribute("data-lit");
  });
  stage.setAttribute("aria-valuenow", String(index + 1));
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  if (touring && drawerTarget === 0) { clock += dt; index = tourAt(clock); }
  const want = targets();
  shaft = approach(shaft, want.shaft, dt, 0.34, calm);
  hand = follow(hand, want.hand, dt, 0.28, calm);
  drop = follow(drop, want.drop, dt, 0.45, calm);
  drawerOpen = follow(drawerOpen, drawerTarget, dt, 0.32, calm);
  paint();
  const settled = !touring && shaft === want.shaft && hand === want.hand && drop === want.drop && drawerOpen === drawerTarget;
  frame = visible && !settled ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const steer = (next) => {
  touring = false;
  index = (next + TRAINS.length) % TRAINS.length;
  clearTimeout(idle);
  idle = setTimeout(() => { touring = !still.matches && drawerTarget === 0; clock = index * 2.8; run(); }, 3600);
  run();
};
new IntersectionObserver((entries) => {
  visible = entries[entries.length - 1].isIntersecting;
  if (visible) run();
}, { rootMargin: "120px 0px" }).observe(stage);
stage.addEventListener("pointermove", (event) => {
  const box = svg.getBoundingClientRect();
  const share = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
  steer(Math.round(share * (TRAINS.length - 1)));
});
stage.addEventListener("pointerleave", () => steer(index));
stage.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight" || event.key === "ArrowUp") { event.preventDefault(); steer(index + 1); }
  else if (event.key === "ArrowLeft" || event.key === "ArrowDown") { event.preventDefault(); steer(index - 1); }
  else if (event.key === "Home") { event.preventDefault(); steer(0); }
  else if (event.key === "End") { event.preventDefault(); steer(TRAINS.length - 1); }
  else if (event.key === "d" || event.key === "D") {
    event.preventDefault();
    touring = false;
    drawerTarget = drawerTarget > 0 ? 0 : 1;
    clearTimeout(idle);
    idle = setTimeout(() => { touring = !still.matches && drawerTarget === 0; clock = index * 2.8; run(); }, 3600);
    run();
  } else if (event.key === "Escape" && drawerTarget > 0) {
    event.preventDefault();
    drawerTarget = 0;
    run();
  } else if (/^[1-6]$/.test(event.key)) {
    event.preventDefault();
    steer(Number(event.key) - 1);
  }
});
paint();
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "校表台",
  hint: "方向键换轮系 · D 拉开抽屉",
  readout: "金属 · RTF 0.379 · p95 82.4 ms",
  keys: [
    { mark: "lit", label: "正在看的读数：刻度上的针，或这一工位本身" },
    { mark: "raised", label: "已经装在同一根发条上的零件" },
    { mark: "line", label: "两根长刻度：验收 0.8，实时 1.0" },
  ],
  caption:
    "发条筒是这台 Mac 上的同一份 Burn。金属语音在刻度内：RTF 0.379，p95 82.4 毫秒。神经引擎的 30 分钟长流在刻度外：RTF 1.017，p95 294 毫秒，结束积压 33 秒，砝码按这个秒数落下；要回到 RTF 0.8 还得少掉约 21% 的计算。浏览器 160 毫秒的细齿跟不上，960 毫秒的粗齿在 20 秒样本上大约 RTF 0.75。图像是八步压床，1024 见方的生成要 50.8 秒。译页已经能共享，一页 16 个 token，池 16384，缓存是池的四分之一。乐曲一次一首。床头的刀刻着 W4A16。抽屉里是备用的刀，同一件活上谁更快谁换上去。尺上每一大格是 160 毫秒，十大格 1.6 秒。这些是已经量过的差距，不是一份新的排期。",
  body: `<style>.iso-stage{touch-action:pan-y}.iso-stage:focus{outline:1px solid var(--anatomy-muted);outline-offset:6px}</style><div class="iso-stage" tabindex="0" role="slider" aria-label="校表台。方向键在六条轮系之间切换，D 拉开抽屉。" aria-valuemin="1" aria-valuemax="6" aria-valuenow="1" aria-valuetext="金属语音，RTF 0.379，在 0.8 的验收线以内。">${svg}</div>`,
});

const file = theme === "light" ? "regulating-bench-light.html" : "regulating-bench.html";
writeFileSync(join(HERE, file), k.pageHtml({ title: "校表台", theme, body, script: kitScript() + live, width: 760 }));
console.log("wrote", file);
