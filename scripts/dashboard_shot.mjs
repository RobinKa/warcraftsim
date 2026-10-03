// A screenshot of the dashboard (headless Chrome from Playwright's cache, driven over the DevTools
// protocol; no packages needed), optionally with the mouse over an element:
//   node scripts/dashboard_shot.mjs "http://localhost:8765/#run=fgself-10&tab=matchups" out.png \
//        [--wait "#matchups table"] [--hover "#c-win" --index 3] [--click "sel1;sel2"] [--width 1500 --height 1000]
// --hover: a chart's canvas id hovers its restart badges (.mark), any other selector the element.
import { spawn } from "node:child_process";
import { readdirSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [url, out] = args;
const width = +opt("--width", 1500), height = +opt("--height", 1000);
const cache = process.env.HOME + "/.cache/ms-playwright/";
const shell = readdirSync(cache).filter(d => d.startsWith("chromium_headless_shell-")).sort().pop();
const bin = `${cache}${shell}/chrome-headless-shell-linux64/chrome-headless-shell`;
const port = 9333 + Math.floor(Math.random() * 500);
const proc = spawn(bin, [`--remote-debugging-port=${port}`, `--window-size=${width},${height}`, "--no-sandbox", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws, id = 0;
const pending = new Map();
for (let i = 0; i < 50 && !ws; i++) {
  try {
    const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(x => x.type === "page");
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch (e) {}
  await sleep(200);
}
if (ws.readyState !== WebSocket.OPEN) await new Promise(r => ws.onopen = r);
ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async e => (await send("Runtime.evaluate", { expression: e, returnByValue: true })).result?.result?.value;
await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
const wait = opt("--wait", opt("--hover") ? null : "body");
for (let i = 0; i < 120; i++) {  // a big run's first load reads its files (seconds)
  await sleep(1000);
  if (await ev(`!!document.querySelector(${JSON.stringify(wait || ".mark")})`)) break;
}
await sleep(1200);
// --click "sel1;sel2": click these elements in turn (e.g. a panel's buttons), 600 ms apart
for (const sel of (opt("--click") || "").split(";").filter(Boolean)) {
  const ok = await ev(`(() => { const el = document.querySelector(${JSON.stringify(sel)}); if (!el) return false;
    el.scrollIntoView({ block: "center" }); el.click(); return true; })()`);
  console.log("clicked:", sel, ok);
  await sleep(600);
}
const hover = opt("--hover");
if (hover) {
  const pos = await ev(`(() => { let el = document.querySelector(${JSON.stringify(hover)}); if (!el) return null;
    el.scrollIntoView({ block: "center" });
    if (el.tagName === "CANVAS") { const ms = el.parentElement.querySelectorAll(".mark"); el = ms[Math.min(${+opt("--index", 0)}, ms.length - 1)]; }
    if (!el) return null; const r = el.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
  if (pos) { await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: pos.x, y: pos.y }); await sleep(400); }
  console.log("hovered:", JSON.stringify(pos));
}
const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.result.data, "base64"));
proc.kill();
process.exit(0);
