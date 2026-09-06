import { buildTreatmentResponseEvidence, type TreatmentResponseEvidence } from "@phenometrix/evidence-core";
import type { TreatmentResponseRunV1, TreatmentResponseScopeV1, TreatmentResponseReviewV1, TreatmentResponseRowV1 } from "@phenometrix/contracts";

export interface TreatmentResponsePanelOptions {
  scope: TreatmentResponseScopeV1;
  /** Host supplies its existing authenticated chart context; never put credentials in URLs. */
  load: (signal: AbortSignal) => Promise<{
    run: TreatmentResponseRunV1 | null; reviews?: TreatmentResponseReviewV1[]; supersededByRunId?: string | null;
  }>;
  /** Silent integration telemetry, never a patient-facing error or retry prompt. */
  onUnavailable?: (reason: "no-evidence" | "unavailable") => void;
}

const css = `
  :host { display:block; color:#203443; font:14px/1.5 system-ui,sans-serif; }
  * { box-sizing:border-box; }
  article { background:white; border:1px solid #dce4e9; border-radius:12px; overflow:hidden; }
  header { padding:22px 26px 18px; border-bottom:1px solid #e5ebef; }
  .eyebrow { display:flex; flex-wrap:wrap; gap:10px; color:#536b7c; font-size:11px; letter-spacing:.09em; text-transform:uppercase; }
  .badge { color:#385e79; background:#edf3f7; padding:1px 7px; border-radius:4px; letter-spacing:.03em; }
  h2 { margin:9px 0 5px; font-size:22px; font-weight:600; letter-spacing:-.025em; }
  p { margin:6px 0; } .muted { color:#617583; font-size:12px; }
  .tools { padding:14px 26px; display:flex; justify-content:space-between; align-items:center; gap:10px; background:#fbfcfd; }
  button { border:1px solid #c6d4dd; background:white; color:#315671; padding:7px 12px; border-radius:5px; font:inherit; cursor:pointer; }
  button:focus-visible, summary:focus-visible { outline:3px solid #6e9dbd; outline-offset:2px; }
  .metric { padding:18px 26px; border-top:1px solid #edf0f3; }
  h3 { font-size:15px; font-weight:600; margin:0; }
  .metric-heading { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
  .value { font-size:18px; font-variant-numeric:tabular-nums; }
  .chart { width:100%; height:150px; display:block; margin-top:5px; }
  .baseline { stroke:#91a2ae; stroke-dasharray:4 4; }
  .axis { stroke:#dce5eb; } .point { fill:#34688c; stroke:white; stroke-width:2; }
  .time-range { stroke:#34688c; stroke-width:2; } .treatment { stroke:#968778; stroke-dasharray:3 3; }
  svg text { font:11px system-ui,sans-serif; fill:#6b7d89; }
  details { margin-top:10px; } summary { color:#41647e; cursor:pointer; font-size:12px; }
  table { border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }
  td,th { text-align:left; padding:6px 8px; border-bottom:1px solid #e8edf0; overflow-wrap:anywhere; }
  th { color:#597180; font-weight:500; } .scroll { overflow:auto; }
  .coverage { display:flex; flex-wrap:wrap; gap:7px; margin-top:8px; font-size:11px; color:#596f7f; }
  .coverage span { background:#f1f5f7; padding:3px 7px; border-radius:3px; }
  footer { padding:18px 26px; background:#f8fafb; border-top:1px solid #e5ebef; }
  footer p { font-size:12px; color:#647885; } code { font-size:11px; overflow-wrap:anywhere; }
  @media(max-width:600px) { header,.tools,.metric,footer { padding-left:16px; padding-right:16px; } h2 { font-size:19px; } .metric-heading { align-items:flex-start; } }
`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string, className?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string): SVGElementTagNameMap[K] {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
const number = (value: number) => new Intl.NumberFormat("en", { maximumSignificantDigits: 4 }).format(value);
const date = (value: string) => new Date(value).toLocaleDateString("en", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const canAlign = (run: TreatmentResponseRunV1) => run.status !== "anchor-unavailable" && run.anchor.reasonCodes.length === 0 &&
  run.anchor.effectiveTime?.earliest != null && run.rows.every(row => row.points.every(point => point.daysSinceTreatment !== null));

function plot(row: TreatmentResponseRowV1, mode: "calendar" | "cycle", run: TreatmentResponseRunV1): SVGSVGElement {
  const chart = svg("svg", { class: "chart", viewBox: "0 0 820 150", role: "img", "aria-label": `${row.label}: observed points only; intervening periods are unobserved.` });
  const points = row.points;
  const xs = points.flatMap(p => mode === "cycle" && p.daysSinceTreatment ? [p.daysSinceTreatment.minimum, p.daysSinceTreatment.maximum] : [Date.parse(p.startedAt)]);
  const treatmentX = !canAlign(run) ? null : mode === "cycle" ? 0 : Date.parse(run.anchor.effectiveTime!.earliest!);
  if (treatmentX !== null) xs.push(treatmentX);
  let lowX = Math.min(...xs), highX = Math.max(...xs);
  if (lowX === highX) { lowX -= mode === "cycle" ? 1 : 86400000; highX += mode === "cycle" ? 1 : 86400000; }
  const ys = points.map(p => p.value); if (row.baseline.value !== null) ys.push(row.baseline.value);
  const minY = Math.min(...ys), maxY = Math.max(...ys), pad = (maxY - minY || Math.abs(minY) || 1) * .2;
  const x = (v: number) => 62 + (v - lowX) / (highX - lowX) * 720;
  const y = (v: number) => 111 - (v - minY + pad) / (maxY - minY + 2 * pad) * 88;
  chart.append(svg("line", { x1: 62, x2: 782, y1: 122, y2: 122, class: "axis" }));
  if (treatmentX !== null) chart.append(svg("line", { x1: x(treatmentX), x2: x(treatmentX), y1: 12, y2: 122, class: "treatment" }), svg("text", { x: x(treatmentX) + 5, y: 13 }, "Treatment"));
  if (row.baseline.value !== null) chart.append(svg("line", { x1: 62, x2: 782, y1: y(row.baseline.value), y2: y(row.baseline.value), class: "baseline" }));
  chart.append(svg("text", { x: 2, y: y(maxY) + 3 }, number(maxY)), svg("text", { x: 2, y: y(minY) + 3 }, number(minY)));
  for (const point of points) {
    const a = mode === "cycle" && point.daysSinceTreatment ? point.daysSinceTreatment.minimum : Date.parse(point.startedAt);
    const b = mode === "cycle" && point.daysSinceTreatment ? point.daysSinceTreatment.maximum : a;
    if (a !== b) chart.append(svg("line", { x1: x(a), x2: x(b), y1: y(point.value), y2: y(point.value), class: "time-range" }));
    const dot = svg("circle", { cx: x((a + b) / 2), cy: y(point.value), r: 5, class: "point", "data-revision": point.revisionId });
    dot.append(svg("title", {}, `${date(point.startedAt)}: ${number(point.value)} ${row.unit}; source ${point.revisionId}`)); chart.append(dot);
  }
  chart.append(svg("text", { x: 62, y: 143 }, mode === "cycle" ? `Day ${number(lowX)}` : date(new Date(lowX).toISOString())), svg("text", { x: 782, y: 143, "text-anchor": "end" }, mode === "cycle" ? `Day ${number(highX)}` : date(new Date(highX).toISOString())));
  return chart;
}

function render(root: ShadowRoot, evidence: TreatmentResponseEvidence, mode: "calendar" | "cycle", changeMode: () => void): void {
  const style = el("style", css), article = el("article");
  article.setAttribute("aria-label", "PhenoMetrix treatment response evidence");
  const header = el("header"), eyebrow = el("div", undefined, "eyebrow");
  eyebrow.append(el("span", "PhenoMetrix · Personal trajectory"), el("span", "Research measurements", "badge"), el("span", evidence.status, "badge"));
  if (evidence.run.contextFlags.some(flag => flag.kind === "source-correction")) eyebrow.append(el("span", "Corrected source history", "badge"));
  header.append(eyebrow, el("h2", evidence.headline));
  header.append(el("p", `${evidence.encounterCount} observed encounter${evidence.encounterCount === 1 ? "" : "s"} · Updated ${date(evidence.run.generatedAt)}`, "muted"));
  const tools = el("div", undefined, "tools");
  tools.append(el("span", "Observed points · Dashed horizontal line = pre-cycle reference", "muted"));
  const toggle = el("button", mode === "calendar" ? "Days since treatment" : "Calendar dates");
  toggle.type = "button"; toggle.addEventListener("click", changeMode);
  if (canAlign(evidence.run)) tools.append(toggle);
  article.append(header, tools);
  for (const row of evidence.run.rows) {
    const section = el("section", undefined, "metric"), heading = el("div", undefined, "metric-heading");
    heading.append(el("h3", row.label));
    const latest = [...row.points].sort((a, b) => Date.parse(a.endedAt) - Date.parse(b.endedAt)).at(-1);
    heading.append(el("span", latest ? `${number(latest.value)} ${row.unit}` : "Not measurable", "value")); section.append(heading);
    if (latest) {
      section.append(el("p", latest.delta === null ? "No eligible pre-cycle reference; descriptive value only." : `Latest difference from reference: ${latest.delta > 0 ? "+" : ""}${number(latest.delta)} ${row.unit}. Direction does not establish clinical benefit.`, "muted"));
      section.append(plot(row, mode, evidence.run));
    }
    section.append(el("p", row.baseline.value === null ? "Reference unavailable" : `${row.baseline.encounterCount === 1 ? "Single reference encounter" : `${row.baseline.encounterCount} reference encounters`} · ${number(row.baseline.value)} ${row.unit} · repeatability unknown`, "muted"));
    const coverage = el("div", undefined, "coverage");
    for (const phase of row.phaseCoverage) coverage.append(el("span", `${phase.phaseId.replace(/^days-(\d+)-(?:to-)?(\d+)$/i, "Days $1–$2")}: ${phase.count} encounter${phase.count === 1 ? "" : "s"} · ${phase.status}`));
    section.append(coverage);
    const details = el("details"); details.append(el("summary", "Inspect observations and exclusions"));
    const scroll = el("div", undefined, "scroll"), table = el("table"), thead = el("thead"), tr = el("tr");
    for (const title of ["Observed", "Value", "Source / exclusion"]) tr.append(el("th", title)); thead.append(tr); table.append(thead);
    const tbody = el("tbody");
    for (const point of row.points) {
      const line = el("tr"); line.append(el("td", date(point.startedAt)), el("td", `${number(point.value)} ${row.unit}`), el("td", `${point.revisionId} · ${point.context} · ${number(point.usableDurationMs / 1000)} usable seconds · technical quality ${number(point.technicalQualityScore)} · ${point.captureAdapterId} ${point.captureAdapterVersion} · ${point.sourceKind} · processor ${point.processorRef} · windows ${point.sourceWindowIds.join(", ")}`)); tbody.append(line);
    }
    for (const excluded of row.exclusions) { const line = el("tr"); line.append(el("td", "Excluded"), el("td", "—"), el("td", `${excluded.revisionId}: ${excluded.reasonCodes.join(", ")}`)); tbody.append(line); }
    table.append(tbody); scroll.append(table); details.append(scroll); section.append(details); article.append(section);
  }
  const footer = el("footer"); for (const limitation of evidence.limitations) footer.append(el("p", limitation));
  const trace = el("details"); trace.append(el("summary", "Analysis provenance"), el("p", `Run ${evidence.run.runId}`), el("code", evidence.run.contentSha256), el("p", `Snapshot ${evidence.run.snapshotId}`), el("p", `Protocol ${evidence.run.protocolRef.id} · ${evidence.run.protocolRef.version}`), el("p", `Specification ${evidence.run.specificationRef.id} · ${evidence.run.specificationRef.version}`));
  if (evidence.latestReview) trace.append(el("p", `Review: ${evidence.latestReview.disposition} · ${date(evidence.latestReview.recordedAt)}`));
  if (evidence.supersededByRunId) trace.append(el("p", `Replaced by ${evidence.supersededByRunId}`));
  footer.append(trace); article.append(footer); root.replaceChildren(style, article);
}

/** Mount inside the existing chart. No capture controls, login, popups, or mandatory review. */
export function mountTreatmentResponsePanel(container: HTMLElement, options: TreatmentResponsePanelOptions): { refresh(): Promise<void>; dispose(): void } {
  const host = el("div"), root = host.attachShadow({ mode: "open" }); container.append(host);
  let generation = 0, disposed = false, controller: AbortController | null = null, mode: "calendar" | "cycle" = "calendar";
  let evidence: TreatmentResponseEvidence | null = null;
  const draw = () => { if (evidence) render(root, evidence, mode, () => { mode = mode === "calendar" ? "cycle" : "calendar"; draw(); }); };
  const refresh = async () => {
    if (disposed) return;
    controller?.abort(); controller = new AbortController(); const current = ++generation;
    try {
      const result = await options.load(controller.signal);
      if (disposed || current !== generation) return;
      const next = result.run ? await buildTreatmentResponseEvidence({ run: result.run, reviews: result.reviews, expectedScope: options.scope, supersededByRunId: result.supersededByRunId }) : null;
      if (disposed || current !== generation) return;
      evidence = next;
      if (!next || !canAlign(next.run)) mode = "calendar";
      if (!next?.hasObservations) { root.replaceChildren(); options.onUnavailable?.("no-evidence"); return; }
      draw();
    } catch {
      if (disposed || current !== generation) return;
      evidence = null; root.replaceChildren(); options.onUnavailable?.("unavailable");
    }
  };
  return { refresh, dispose: () => { disposed = true; generation++; controller?.abort(); evidence = null; host.remove(); } };
}
