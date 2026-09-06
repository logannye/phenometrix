import { afterEach,describe,it,expect,vi } from "vitest";
import { generateKeyPair,exportJWK,SignJWT } from "jose";
import { randomBytes } from "node:crypto";
import type { Server } from "node:http";
import { createAuthenticator,signDevelopmentSession } from "./auth.js";
import { createEncounterHttpServer } from "./http.js";
import { loadServiceConfig } from "./config.js";
import { EncounterService } from "./service.js";
import { MemoryRepository } from "./repository.memory.js";
import { SYNTHETIC_PRINCIPAL } from "./synthetic.js";
const servers:Server[]=[];
afterEach(async()=>{vi.unstubAllGlobals();await Promise.all(servers.splice(0).map(server=>new Promise<void>(resolve=>server.close(()=>resolve()))));});
async function start(){
  const secret=randomBytes(32),issuer="synthetic-issuer",audience="service";
  const server=createEncounterHttpServer({service:new EncounterService(new MemoryRepository(),{mode:"synthetic"}),authenticate:createAuthenticator({mode:"synthetic",issuer,audience,developmentSecret:secret}),allowedOrigins:["http://localhost:5173"]});
  servers.push(server);await new Promise<void>(resolve=>server.listen(0,"127.0.0.1",resolve));
  const address=server.address();if(!address||typeof address==="string")throw new Error("missing port");
  return {base:`http://127.0.0.1:${address.port}`,token:await signDevelopmentSession(SYNTHETIC_PRINCIPAL,secret,issuer,audience)};
}
describe("HTTP authentication and boundary",()=>{
  it("requires a valid signature and rejects identity headers and token queries",async()=>{
    const {base,token}=await start();
    expect((await fetch(`${base}/v1/episodes/x`,{headers:{"x-user-id":"synthetic-clinician"}})).status).toBe(401);
    expect((await fetch(`${base}/v1/episodes/x`,{headers:{Authorization:`Bearer ${token.slice(0,-10)}0123456789`}})).status).toBe(401);
    expect((await fetch(`${base}/v1/episodes/x`,{headers:{Authorization:`Bearer ${token}`}})).status).toBe(404);
    expect((await fetch(`${base}/v1/episodes/x?token=${token}`)).status).toBe(400);
  });
  it("rejects unconfigured browser origins and binary ingestion",async()=>{
    const {base,token}=await start();
    expect((await fetch(`${base}/healthz`,{headers:{Origin:"https://untrusted.example"}})).status).toBe(403);
    const response=await fetch(`${base}/v1/episodes/x/sessions`,{method:"POST",headers:{Authorization:`Bearer ${token}`,"Idempotency-Key":"one","Content-Type":"video/webm"},body:"media"});
    expect(response.status).toBe(415);
  });
  it("fails closed if live deployment and trust configuration are incomplete",()=>{
    expect(()=>loadServiceConfig({})).toThrow("PHENOMETRIX_MODE");
    expect(()=>loadServiceConfig({PHENOMETRIX_MODE:"live",DATABASE_URL:"postgresql://local/db"})).toThrow("enablement");
    expect(()=>loadServiceConfig({PHENOMETRIX_MODE:"synthetic",DATABASE_URL:"postgresql://remote.example/real"})).toThrow("loopback");
    expect(()=>createAuthenticator({mode:"live",issuer:"https://id.example",audience:"api",jwksUrl:"http://id.example/keys"})).toThrow("HTTPS");
  });
  it("verifies a live asymmetric session against configured JWKS and exact audience",async()=>{
    const {publicKey,privateKey}=await generateKeyPair("RS256");
    const jwk=await exportJWK(publicKey);
    vi.stubGlobal("fetch",vi.fn(async(url:URL)=>{
      expect(String(url)).toBe("https://issuer.example/keys");
      return new Response(JSON.stringify({keys:[{...jwk,kid:"trusted-key",alg:"RS256",use:"sig"}]}),{status:200,headers:{"Content-Type":"application/json"}});
    }));
    const authenticate=createAuthenticator({mode:"live",issuer:"https://issuer.example",audience:"clinical-api",jwksUrl:"https://issuer.example/keys"});
    const claims={...SYNTHETIC_PRINCIPAL,dataClass:"consented-research"};
    const sign=(audience:string)=>new SignJWT(claims).setProtectedHeader({alg:"RS256",kid:"trusted-key"}).setIssuer("https://issuer.example").setAudience(audience).setIssuedAt().setExpirationTime("1m").sign(privateKey);
    expect((await authenticate(`Bearer ${await sign("clinical-api")}`)).dataClass).toBe("consented-research");
    await expect(authenticate(`Bearer ${await sign("other-api")}`)).rejects.toMatchObject({code:"invalid-session"});
  });
});
