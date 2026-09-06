import type {
  ConditionEvidenceCardV1,
  EvidenceRef,
  ExcludedPreviousVisitMetricSourceV1,
  MeasuredPreviousVisitMetricSourceV1
} from "@phenometrix/contracts";

export interface ConditionEvidenceViewElements {
  card: HTMLElement;
  summary: HTMLElement;
  rows: HTMLElement;
  reviewState: HTMLElement;
  acceptButton: HTMLButtonElement;
  dismissButton: HTMLButtonElement;
}

function formatNumber(value: number, unit: string): string {
  if (unit === "Hz") return value.toFixed(2);
  if (unit.includes("second") || unit.includes("minute")) {
    return value.toFixed(2);
  }
  return value.toFixed(3);
}

type DisplaySource =
  | MeasuredPreviousVisitMetricSourceV1
  | ExcludedPreviousVisitMetricSourceV1;

function sourceOutcome(source: DisplaySource) {
  return "outcome" in source ? source.outcome : source;
}

function sourceQuality(source: DisplaySource): number | null {
  return "outcome" in source
    ? source.outcome.technicalQualityScore
    : source.technicalQualityScore;
}

function traceRef(ref: EvidenceRef): string {
  if (ref.kind === "event") return ref.eventId;
  if (ref.kind === "window") return ref.windowId;
  if (ref.kind === "measurement") return ref.measurementId;
  return ref.aggregateId;
}

function sourceValue(source: DisplaySource | null): {
  value: string;
  detail: string;
  missing: boolean;
} {
  if (!source) {
    return { value: "Not comparable", detail: "No compatible source", missing: true };
  }
  if (source.status === "withheld") {
    return {
      value: "Not measurable",
      detail: source.withheldReasonCode,
      missing: true
    };
  }
  if (!("outcome" in source)) {
    return {
      value: "Not comparable",
      detail: source.unit,
      missing: true
    };
  }
  return {
    value: formatNumber(source.outcome.value, source.outcome.unit),
    detail: source.outcome.unit,
    missing: false
  };
}

function valueCell(label: string, value: string, detail: string, missing: boolean): HTMLElement {
  const cell = document.createElement("div");
  cell.className = `condition-row-value${missing ? " is-missing" : ""}`;
  const caption = document.createElement("span");
  caption.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const unit = document.createElement("span");
  unit.textContent = detail;
  cell.append(caption, strong, unit);
  return cell;
}

function appendTrace(trace: HTMLElement, label: string, value: string): void {
  const item = document.createElement("div");
  const name = document.createElement("span");
  const code = document.createElement("code");
  name.textContent = label;
  code.textContent = value;
  item.append(name, code);
  trace.append(item);
}

export function renderConditionEvidenceCard(
  elements: ConditionEvidenceViewElements,
  card: ConditionEvidenceCardV1
): void {
  elements.card.hidden = false;
  elements.summary.textContent =
    `${card.qualitySummary.measuredComparisonCount} measured comparison${
      card.qualitySummary.measuredComparisonCount === 1 ? "" : "s"
    } · ${card.qualitySummary.withheldComparisonCount} withheld · ` +
    `${card.qualitySummary.incompatibleComparisonCount} incompatible. ` +
    "Raw differences have unknown repeatability and unknown minimum detectable change.";
  elements.rows.replaceChildren();

  for (const row of card.rows) {
    const article = document.createElement("article");
    article.className = "condition-row";
    article.dataset.displayRole = row.metric.displayRole;

    const assertedSide = card.comparison.demoContext.assertedAffectedSide.side;
    const assertedSideMetric =
      row.metric.laterality === `explicit-subject-${assertedSide}`;
    article.dataset.assertedAffectedSide = assertedSideMetric ? "true" : "false";

    const label = document.createElement("div");
    label.className = "condition-row-label";
    const strong = document.createElement("strong");
    strong.textContent = row.metric.displayLabel;
    const role = document.createElement("span");
    role.textContent = row.metric.displayRole === "experimental"
      ? "Experimental engineering metric"
      : "Engineering metric";
    label.append(strong, role);
    if (row.metric.laterality === "subject-left-minus-subject-right") {
      const signConvention = document.createElement("span");
      signConvention.className = "sign-convention";
      signConvention.textContent = "Sign: subject-left minus subject-right";
      label.append(signConvention);
    }
    if (assertedSideMetric) {
      const marker = document.createElement("span");
      marker.className = "affected-side-marker";
      marker.textContent = "Participant-asserted affected side";
      label.append(marker);
    }

    const reference = sourceValue(row.comparison.reference);
    const current = sourceValue(row.comparison.current);
    const delta = row.comparison.status === "measured"
      ? {
          value: `${row.comparison.delta >= 0 ? "+" : ""}${formatNumber(
            row.comparison.delta,
            row.comparison.nativeUnit
          )}`,
          detail: `${row.comparison.nativeUnit} · current minus reference`,
          missing: false
        }
      : {
          value: "Not available",
          detail: row.comparison.compatibilityReasonCodes.join(", "),
          missing: true
        };

    const reason = document.createElement("p");
    reason.className = "condition-row-detail";
    reason.textContent = row.comparison.status === "measured"
      ? "Numeric difference only; no direction of health change is assigned."
      : `No difference calculated: ${row.comparison.compatibilityReasonCodes.join(", ")}.`;

    const details = document.createElement("details");
    const detailsSummary = document.createElement("summary");
    detailsSummary.textContent = "Evidence and compatibility details";
    const trace = document.createElement("div");
    trace.className = "trace-grid";
    appendTrace(trace, "Reference observation", card.comparison.referenceObservation.observationId);
    appendTrace(trace, "Reference session", card.comparison.referenceObservation.sessionId);
    appendTrace(trace, "Current observation", card.comparison.currentObservation.observationId);
    appendTrace(trace, "Current session", card.comparison.currentObservation.sessionId);
    appendTrace(trace, "Compatibility", row.comparison.compatibilityReasonCodes.length === 0
      ? "included"
      : row.comparison.compatibilityReasonCodes.join(", "));
    if (row.comparison.reference) {
      const source = row.comparison.reference;
      const outcome = sourceOutcome(source);
      const quality = sourceQuality(source);
      appendTrace(trace, "Reference outcome", outcome.outcomeId);
      appendTrace(trace, "Reference aggregate", outcome.aggregateId);
      appendTrace(trace, "Reference processor", row.comparison.reference.processor.processorRef);
      appendTrace(
        trace,
        "Reference technical quality",
        quality === null ? "Not available" : quality.toFixed(3)
      );
      appendTrace(
        trace,
        "Reference eligible duration",
        `${(outcome.evidence.eligibleDurationMs / 1_000).toFixed(1)} s`
      );
      appendTrace(trace, "Reference usable bins", String(outcome.evidence.binCount));
      appendTrace(trace, "Reference expression events", String(outcome.evidence.eventCount));
      appendTrace(trace, "Reference windows", String(outcome.evidence.windowCount));
      appendTrace(
        trace,
        "Reference evidence",
        outcome.evidence.refs.map(traceRef).join(", ")
      );
    }
    if (row.comparison.current) {
      const source = row.comparison.current;
      const outcome = sourceOutcome(source);
      const quality = sourceQuality(source);
      appendTrace(trace, "Current outcome", outcome.outcomeId);
      appendTrace(trace, "Current aggregate", outcome.aggregateId);
      appendTrace(trace, "Current processor", row.comparison.current.processor.processorRef);
      appendTrace(
        trace,
        "Current technical quality",
        quality === null ? "Not available" : quality.toFixed(3)
      );
      appendTrace(
        trace,
        "Current eligible duration",
        `${(outcome.evidence.eligibleDurationMs / 1_000).toFixed(1)} s`
      );
      appendTrace(trace, "Current usable bins", String(outcome.evidence.binCount));
      appendTrace(trace, "Current expression events", String(outcome.evidence.eventCount));
      appendTrace(trace, "Current windows", String(outcome.evidence.windowCount));
      appendTrace(
        trace,
        "Current evidence",
        outcome.evidence.refs.map(traceRef).join(", ")
      );
    }
    details.append(detailsSummary, trace);

    article.append(
      label,
      valueCell("Reference", reference.value, reference.detail, reference.missing),
      valueCell("Current", current.value, current.detail, current.missing),
      valueCell("Raw difference", delta.value, delta.detail, delta.missing),
      reason,
      details
    );
    elements.rows.append(article);
  }

  elements.reviewState.textContent = card.review.status === "pending"
    ? "Not reviewed"
    : card.review.status === "accepted"
      ? "Accepted for this page session"
      : "Dismissed for this page session";
  elements.acceptButton.disabled = card.review.status !== "pending";
  elements.dismissButton.disabled = card.review.status !== "pending";
}

export function clearConditionEvidenceCard(
  elements: ConditionEvidenceViewElements
): void {
  elements.card.hidden = true;
  elements.summary.textContent = "";
  elements.rows.replaceChildren();
  elements.reviewState.textContent = "Not reviewed";
  elements.acceptButton.disabled = false;
  elements.dismissButton.disabled = false;
}
