import { createServer,type IncomingMessage,type ServerResponse } from "node:http";
import { ZodError } from "zod";
import type { Authenticate } from "./auth.js";
import { ServiceError,invariant } from "./errors.js";
import type { EncounterService } from "./service.js";

async function readJson(request:IncomingMessage):Promise<unknown> {
  invariant(request.headers["content-type"]?.split(";")[0]==="application/json",415,"json-required","Only JSON derived records are accepted; raw media uploads are disabled.");
  let size=0;
  const chunks:Buffer[]=[];
  for await(const chunk of request){
    const buffer=Buffer.isBuffer(chunk)?chunk:Buffer.from(chunk);
    size+=buffer.length;
    invariant(size<=2_000_000,413,"body-too-large","JSON request exceeds the size limit.");
    chunks.push(buffer);
  }
  try{return JSON.parse(Buffer.concat(chunks).toString("utf8"));}
  catch{throw new ServiceError(400,"invalid-json","Request body must be valid JSON.");}
}
function respond(response:ServerResponse,status:number,payload:unknown){
  response.writeHead(status,{"Content-Type":"application/json","Cache-Control":"no-store","X-Content-Type-Options":"nosniff"});
  response.end(JSON.stringify(payload));
}
export function createEncounterHttpServer(options:{
  service:EncounterService;authenticate:Authenticate;allowedOrigins:readonly string[];
  /** Exposed only by the explicit loopback synthetic launcher. */
  developmentSession?:()=>Promise<unknown>
}){
  const server=createServer(async(request,response)=>{
    try{
      const origin=request.headers.origin;
      if(origin){
        invariant(options.allowedOrigins.includes(origin),403,"origin-not-allowed","This browser origin is not configured.");
        response.setHeader("Access-Control-Allow-Origin",origin);response.setHeader("Vary","Origin");
      }
      if(request.method==="OPTIONS"){
        response.setHeader("Access-Control-Allow-Methods","GET,POST,OPTIONS");
        response.setHeader("Access-Control-Allow-Headers","Authorization,Content-Type,Idempotency-Key");
        response.writeHead(204);response.end();return;
      }
      const url=new URL(request.url??"/","http://localhost");
      invariant(!url.search,400,"query-not-supported","Do not put credentials or clinical data in URL queries.");
      if(url.pathname==="/healthz" && request.method==="GET"){respond(response,200,{status:"ok",rawMediaUpload:"disabled"});return;}
      if(url.pathname==="/dev/session" && request.method==="POST" && options.developmentSession){
        const address=request.socket.remoteAddress;
        invariant(address==="127.0.0.1" || address==="::1" || address==="::ffff:127.0.0.1",403,"loopback-only","Synthetic development sessions are loopback only.");
        respond(response,200,await options.developmentSession());return;
      }
      const principal=await options.authenticate(request.headers.authorization);
      const captureContext=/^\/v1\/episodes\/([A-Za-z0-9._:-]+)\/capture-context\/([A-Za-z0-9._:-]+)$/.exec(url.pathname);
      if(captureContext && request.method==="GET"){
        respond(response,200,await options.service.captureContext(principal,captureContext[1]!,captureContext[2]!));return;
      }
      if(url.pathname==="/v1/episodes" && request.method==="POST"){
        respond(response,201,await options.service.createEpisode(principal,await readJson(request)));return;
      }
      const match=/^\/v1\/episodes\/([A-Za-z0-9._:-]+)(?:\/(consents|bindings|sessions|clinical-events|annotations|fhir|evidence|history|reviews))?$/.exec(url.pathname);
      invariant(match,404,"route-not-found","Route not found.");
      const episodeId=match[1]!,route=match[2];
      if(request.method==="GET"){
        if(!route){respond(response,200,await options.service.getEpisode(principal,episodeId));return;}
        if(route==="evidence" || route==="history"){respond(response,200,await options.service.evidence(principal,episodeId,route==="history"));return;}
      }
      if(request.method==="POST"){
        const key=request.headers["idempotency-key"];
        invariant(typeof key==="string" && /^[A-Za-z0-9._:-]{1,160}$/.test(key),400,"idempotency-key-required","A bounded Idempotency-Key header is required.");
        const body=await readJson(request);
        if(route==="reviews"){respond(response,201,await options.service.review(principal,episodeId,body,key));return;}
        if(route==="fhir"){respond(response,202,await options.service.importFhir(principal,episodeId,body,key));return;}
        if(route==="consents" || route==="bindings" || route==="sessions" || route==="clinical-events" || route==="annotations"){
          respond(response,202,await options.service.write(principal,episodeId,route,body,key));return;
        }
      }
      throw new ServiceError(404,"route-not-found","Route not found.");
    }catch(error){
      if(error instanceof ServiceError)respond(response,error.status,{error:{code:error.code,message:error.message}});
      else if(error instanceof ZodError)respond(response,422,{error:{code:"invalid-contract",message:"Input does not satisfy the versioned data contract.",paths:error.issues.map(issue=>issue.path.join("."))}});
      else respond(response,500,{error:{code:"internal-error",message:"The operation could not be completed."}});
    }
  });
  server.requestTimeout=30_000;server.headersTimeout=10_000;server.keepAliveTimeout=5_000;
  return server;
}
