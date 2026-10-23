/** Bounded local NDJSON transport for the pure acoustic kernel.
 * Run with the installed Node/tsx runtime; not part of production capture.
 * One request is processed at a time, with no queued waveform or output state.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { inspectPeriodicityEvidence, PERIODICITY_EVIDENCE_SCHEMA } from './periodicity_evidence';

const MAX_SAMPLES = 48000 * 120;
const MAX_FRAMES = 12000;
const MAX_JSON_BYTES = 16 * 1024 * 1024;
const MAX_RSS_BYTES = 2 * 1024 ** 3;
const MAX_REQUEST_MS = 120000;
const REQUEST_KEYS = new Set(['id', 'pcm_path', 'sample_rate_hz', 'ends',
  'candidate_hz', 'candidate_mask', 'timeout_ms']);

function requireInput(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message);
}

function inspect(request: Record<string, unknown>) {
  const started = performance.now();
  requireInput(request && typeof request === 'object' && !Array.isArray(request), 'Object request required');
  requireInput(Object.keys(request).every(key => REQUEST_KEYS.has(key)), 'Unknown request field');
  requireInput(Number.isSafeInteger(request.id) && Number(request.id) > 0, 'Positive integer request id required');
  requireInput(request.sample_rate_hz === 48000, '48000 Hz required');
  const timeout = request.timeout_ms ?? MAX_REQUEST_MS;
  requireInput(typeof timeout === 'number' && Number.isFinite(timeout)
    && timeout > 0 && timeout <= MAX_REQUEST_MS, 'Request deadline must be positive and at most 120000 ms');
  const deadline = started + timeout;
  let peakRss = 0;
  function guard() {
    requireInput(performance.now() <= deadline, 'Periodicity request deadline exceeded');
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
    requireInput(peakRss <= MAX_RSS_BYTES, 'Periodicity worker RSS exceeds 2 GiB');
  }
  guard();
  requireInput(typeof request.pcm_path === 'string' && path.isAbsolute(request.pcm_path)
    && path.extname(request.pcm_path) === '.f32le', 'Absolute .f32le PCM file required');
  const stat = fs.statSync(request.pcm_path);
  requireInput(stat.isFile() && stat.size >= 1920 * 4 && stat.size % 4 === 0
    && stat.size <= MAX_SAMPLES * 4, 'PCM size must contain complete support and be at most 120 seconds');
  const ends = request.ends;
  const hz = request.candidate_hz;
  const masks = request.candidate_mask;
  requireInput(Array.isArray(ends) && ends.length <= MAX_FRAMES, 'At most 12000 endpoint frames required');
  requireInput(Array.isArray(hz) && Array.isArray(masks)
    && hz.length === ends.length && masks.length === ends.length, 'Candidate frame inventory must match endpoints');
  for (let frame = 0; frame < ends.length; frame += 1) {
    requireInput(Number.isInteger(ends[frame]) && ends[frame] >= 1920 && ends[frame] <= stat.size / 4
      && (frame === 0 || ends[frame] > ends[frame - 1]), 'Strictly increasing complete-support endpoints required');
    requireInput(Array.isArray(hz[frame]) && Array.isArray(masks[frame])
      && hz[frame].length === 7 && masks[frame].length === 7, 'Seven grouped candidate slots per frame required');
    for (let c = 0; c < 7; c += 1) {
      const mask = masks[frame][c];
      requireInput(mask === true || mask === false || mask === 0 || mask === 1, 'Boolean or zero/one mask required');
      requireInput(mask ? typeof hz[frame][c] === 'number' && Number.isFinite(hz[frame][c])
        && hz[frame][c] >= 50 && hz[frame][c] <= 700
        : hz[frame][c] === null || (typeof hz[frame][c] === 'number' && Number.isFinite(hz[frame][c])),
      'Finite in-range active grouped frequencies or masked null required');
    }
  }
  guard();
  const bytes = fs.readFileSync(request.pcm_path);
  requireInput(bytes.length === stat.size, 'PCM size changed during read');
  const pcm = new Float32Array(bytes.length / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < pcm.length; i += 1) {
    pcm[i] = view.getFloat32(i * 4, true);
    requireInput(Number.isFinite(pcm[i]), 'Finite PCM required');
  }
  const pcmSha256 = crypto.createHash('sha256').update(bytes).digest('hex');
  guard();
  const channels: number[][] = [], correlations: number[][] = [], valid: number[][] = [];
  const periods: number[][] = [], lengths: number[][] = [];
  let kernelElapsed = 0;
  for (let frame = 0; frame < ends.length; frame += 1) {
    guard();
    const frequencies = Float64Array.from(hz[frame], value => value === null ? NaN : value);
    const beforeKernel = performance.now();
    const result = inspectPeriodicityEvidence(pcm.subarray(ends[frame] - 1920, ends[frame]),
      48000, frequencies, masks[frame]);
    kernelElapsed += performance.now() - beforeKernel;
    channels.push(Array.from(result.channels)); correlations.push(Array.from(result.correlations));
    valid.push(Array.from(result.valid)); periods.push(Array.from(result.periodSamples));
    lengths.push(Array.from(result.anchorLength));
  }
  guard();
  const response = { id: request.id, ok: true, schema: PERIODICITY_EVIDENCE_SCHEMA,
    pcm_sha256: pcmSha256, pcm_sample_count: pcm.length, ends, channels, correlations, valid,
    period_samples: periods, common_length: lengths, rss_bytes: peakRss,
    elapsed_ms: performance.now() - started, kernel_elapsed_ms: kernelElapsed };
  const output = JSON.stringify(response) + '\n';
  requireInput(Buffer.byteLength(output) <= MAX_JSON_BYTES, 'Periodicity response exceeds 16 MiB');
  guard();
  return { output, deadline };
}

async function write(output: string, deadline: number) {
  const remaining = deadline - performance.now();
  requireInput(remaining > 0, 'Periodicity response deadline exceeded');
  // Sync kernel/I/O checks bound processing; this timer also bounds a blocked
  // stdout consumer while waiting for the write callback below.
  const timer = setTimeout(() => process.exit(1), remaining);
  try {
    await new Promise<void>((resolve, reject) => {
      process.stdout.write(output, error => error ? reject(error) : resolve());
    });
  } finally {
    clearTimeout(timer);
  }
}

async function main() {
  process.stdin.setEncoding('utf8');
  let pending = '';
  for await (const chunk of process.stdin) {
    pending += chunk;
    requireInput(Buffer.byteLength(pending) <= MAX_JSON_BYTES, 'Periodicity request exceeds 16 MiB');
    let newline: number;
    while ((newline = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, newline);
      pending = pending.slice(newline + 1);
      let request: Record<string, unknown> | undefined;
      try {
        request = JSON.parse(line);
        const result = inspect(request!);
        await write(result.output, result.deadline);
      } catch (error) {
        await write(JSON.stringify({ id: request?.id ?? null, ok: false,
          error: String(error).slice(0, 1000) }) + '\n', performance.now() + 1000);
        process.exitCode = 1;
        return;
      }
    }
  }
  requireInput(pending.length === 0, 'Incomplete NDJSON request');
}

main().catch(error => {
  process.stderr.write(String(error).slice(0, 1000) + '\n');
  process.exitCode = 1;
});
