export function mean(values: number[]): number {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

export function stdDev(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(mean(values.map((v) => (v - m) ** 2)));
}

export function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  // Halving before addition keeps the midpoint finite when both central
  // operands are large finite numbers with the same sign.
  return sorted.length % 2 === 0
    ? sorted[mid - 1] / 2 + sorted[mid] / 2
    : sorted[mid];
}

export function medianAbsoluteDeviation(values: number[]): number {
  const center = median(values);
  return median(values.map((v) => Math.abs(v - center)));
}

export function percentile(values: number[], probability: number): number {
  if (values.length === 0) {
    throw new Error("A percentile requires at least one value.");
  }
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error("Percentile probability must be between 0 and 1.");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}
