/** Bound a dependency even when an adapter ignores its AbortSignal. */
export function withinDeadline<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const aborted = () => reject(signal.reason);
    signal.addEventListener("abort", aborted, { once: true });
    promise.then(value => {
      signal.removeEventListener("abort", aborted);
      if (signal.aborted) reject(signal.reason); else resolve(value);
    }, error => { signal.removeEventListener("abort", aborted); reject(error); });
    if (signal.aborted) { signal.removeEventListener("abort", aborted); aborted(); }
  });
}
