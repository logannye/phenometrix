import { randomBytes } from "node:crypto";
import { invariant } from "./errors.js";

export function loadServiceConfig(env:NodeJS.ProcessEnv=process.env){
  const mode=env.PHENOMETRIX_MODE;
  invariant(mode==="synthetic" || mode==="live",500,"mode-required","Set PHENOMETRIX_MODE explicitly to synthetic or live.");
  const databaseUrl=env.DATABASE_URL;
  invariant(databaseUrl,500,"database-required","DATABASE_URL is required; persistence never silently falls back to memory.");
  const database=new URL(databaseUrl);
  invariant(["postgres:","postgresql:"].includes(database.protocol),500,"database-config","A PostgreSQL URL is required.");
  const port=Number(env.PORT??4318);
  invariant(Number.isInteger(port)&&port>0&&port<65536,500,"port-config","PORT must be a valid TCP port.");
  const host=env.HOST??"127.0.0.1";
  const allowedOrigins=(env.ALLOWED_ORIGINS??"http://127.0.0.1:5173,http://localhost:5173").split(",").map(value=>value.trim());
  invariant(allowedOrigins.every(value=>{try{const url=new URL(value);return url.origin===value&&["http:","https:"].includes(url.protocol);}catch{return false;}}),500,"origin-config","Browser origins must be exact HTTP(S) origins.");
  if(mode==="synthetic"){
    invariant(["127.0.0.1","::1","localhost"].includes(host)&&["127.0.0.1","[::1]","localhost"].includes(database.hostname),500,"synthetic-loopback-only","Synthetic development must bind loopback and use a local synthetic PostgreSQL database.");
    invariant(/(?:test|synthetic|dev)/i.test(database.pathname),500,"synthetic-database-name","Synthetic mode requires a database explicitly named test, synthetic or dev.");
    return {mode:"synthetic" as const,host,port,databaseUrl,allowedOrigins,issuer:"phenometrix-synthetic",audience:"phenometrix-encounter-service",developmentSecret:randomBytes(32),jwksUrl:undefined};
  }
  invariant(env.PHENOMETRIX_LIVE_ENABLED==="1",500,"live-not-enabled","Live mode requires explicit deployment enablement.");
  invariant(env.OIDC_ISSUER&&env.OIDC_AUDIENCE&&env.OIDC_JWKS_URL,500,"oidc-required","Live mode requires issuer, audience and trusted HTTPS JWKS configuration.");
  invariant(new URL(env.OIDC_ISSUER).protocol==="https:"&&new URL(env.OIDC_JWKS_URL).protocol==="https:",500,"oidc-https-required","Live identity trust must use HTTPS.");
  invariant(database.searchParams.get("sslmode")==="verify-full",500,"database-tls-required","Live PostgreSQL requires sslmode=verify-full and trusted CA configuration.");
  invariant(allowedOrigins.every(value=>new URL(value).protocol==="https:"),500,"live-origin-https","Live browser origins must use HTTPS.");
  return {mode:"live" as const,host,port,databaseUrl,allowedOrigins,issuer:env.OIDC_ISSUER,audience:env.OIDC_AUDIENCE,jwksUrl:env.OIDC_JWKS_URL,developmentSecret:undefined};
}
