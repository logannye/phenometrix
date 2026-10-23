/** Consumed-input curve diagnosis only. Run from apps/capture-web with node --import tsx. */
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
import {full40Peaks} from './coverage_peaks.js';
const repo=path.resolve(__dirname,'../..');
const args=process.argv.slice(2);
if(args.length!==2||args[0]!=='--output')throw Error('Supply --output inside ignored research runs.');
const output=path.resolve(args[1]);
const runs=path.join(repo,'research/voice-prosody/runs')+path.sep;
if(!output.startsWith(runs)||fs.existsSync(output)||!fs.existsSync(path.dirname(output)))throw Error('Fresh ignored output path required.');
const hash=(b:Buffer)=>crypto.createHash('sha256').update(b).digest('hex');
const inputs:Record<string,string>={};
function read(relative:string,expected?:string){const b=fs.readFileSync(path.join(repo,relative));const h=hash(b);if(expected&&h!==expected)throw Error('Input changed');inputs[relative]=h;return b;}
const base='research/voice-prosody/runs/harmonic-augmentation-v1/challenge/';
const manifest=JSON.parse(read(base+'manifest.json','a5bbe584f61515eaa142fda7ee5a2a397bc4bb2e72a4d73f7fb8c4684ec2a8f0').toString());
const names=['augmentation-balance-f4-p0','augmentation-balance-f4-p1','augmentation-balance-f5-p0','augmentation-balance-f5-p1'];
const rows=[];
for(const id of names){
 const r=manifest.records.find((x:{id:string})=>x.id===id);if(!r)throw Error('Fixed consumed input missing');
 const bytes=read(base+r.pcm_path,r.pcm_sha256);read(base+r.feature_path,r.feature_sha256);
 const pcm=Float32Array.from({length:bytes.length/4},(_,i)=>bytes.readFloatLE(i*4));
 const acquired=[];
 for(let end=2208;end<=pcm.length;end+=480){const target=end-960;if(target<3072||target+1536>pcm.length||end-1920<39840)continue;
  const peaks=full40Peaks(pcm.subarray(end-1920,end));
  acquired.push({end,peaks:[...peaks].sort((a,b)=>b.confidence-a.confidence||a.lagSamples-b.lagSamples)});
 }
 // Known consumed analytic truth enters diagnosis only after all peak curves exist.
 const hz=id.includes('-f4-')?517.3:641.9;
 const measured=acquired.map(({end,peaks})=>{
  const closest=peaks.map((p,i)=>({...p,rank:i+1,cents:1200*Math.log2(p.hz/hz)})).sort((a,b)=>Math.abs(a.cents)-Math.abs(b.cents))[0];
  const earliest=[...peaks].sort((a,b)=>a.lagSamples-b.lagSamples)[0];
  return {end,closest,old:peaks[0],earliest};
 });
 rows.push({id,known_frequency_hz:hz,fully_post_common_frames:measured.length,
  valid_near_truth_peak_count:measured.filter(x=>Math.abs(x.closest.cents)<=50).length,
  nearest_peak_score_rank_range:[Math.min(...measured.map(x=>x.closest.rank)),Math.max(...measured.map(x=>x.closest.rank))],
  earliest_admitted_peak_near_truth_count:measured.filter(x=>Math.abs(1200*Math.log2(x.earliest.hz/hz))<=50).length,
  median_index_window:measured[Math.floor(measured.length/2)]});
}
for(const rel of ['research/voice-prosody/coverage_peaks.ts','apps/capture-web/src/voice-dsp.ts',
 'apps/capture-web/src/dsp-primitives.ts','apps/capture-web/src/voice-pitch-features-v2.ts',
 path.relative(repo,__filename)])read(rel);
for(const [rel,expected]of Object.entries(inputs))if(hash(fs.readFileSync(path.join(repo,rel)))!==expected)throw Error('Input changed during diagnosis');
const result={schema:'phenometrix-candidate-coverage-diagnosis-v1',status:'complete',created_at:new Date().toISOString(),
 scope:'Already-consumed four analytic failures only; no fit, new waveform, new challenge, natural reference, selector inference or threshold search',
 mechanism:'True-frequency peaks are admitted and accurately interpolated, then discarded by integer-height ranking. Longer period multiples align more closely to integer8k lags.',
 policy_proposal:'shortest-admitted-full40-v1; replace only full40 selection, same .35 admission/interpolation/bounds/source-score semantics; no candidate expansion',
 limitations:['Short-lag selection may admit overtone/noise candidates and remove useful long-lag candidates elsewhere; no global coverage or model-benefit claim.',
 'This is descriptive evidence from consumed cases, not a new validation set.'],records:rows,bound_sha256:inputs};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,sha256:hash(fs.readFileSync(output)),records:rows.length,frames:rows.reduce((s,r)=>s+r.fully_post_common_frames,0)}));
