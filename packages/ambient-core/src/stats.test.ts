import { describe, expect, it } from "vitest";
import {
  mean,
  stdDev,
  median,
  medianAbsoluteDeviation,
  percentile
} from "./stats.js";

describe("stats utilities", () => {
  it("computes the mean", () => {
    expect(mean([2, 4, 6])).toBe(4);
  });

  it("computes population standard deviation, 0 for fewer than 2 values", () => {
    expect(stdDev([120, 130, 110, 140])).toBeCloseTo(11.1803, 3);
    expect(stdDev([5])).toBe(0);
  });

  it("computes the median for odd and even counts", () => {
    expect(median([2, 6, 4])).toBe(4);
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });

  it("computes the median absolute deviation", () => {
    expect(medianAbsoluteDeviation([2, 4, 6])).toBe(2);
  });

  it("keeps the midpoint of large finite values finite", () => {
    expect(median([Number.MAX_VALUE, Number.MAX_VALUE])).toBe(
      Number.MAX_VALUE
    );
  });

  it("interpolates percentiles deterministically", () => {
    expect(percentile([0, 10, 20], 0.9)).toBe(18);
    expect(() => percentile([], 0.5)).toThrow(/at least one/);
  });
});
