import { Pool } from "pg";
import { createAuthenticator,signDevelopmentSession } from "./auth.js";
import { loadServiceConfig } from "./config.js";
import { createEncounterHttpServer } from "./http.js";
import { PostgresRepository } from "./repository.postgres.js";
import { EncounterService } from "./service.js";
import { migrate } from "./migration.js";
import { SYNTHETIC_EPISODE_ID,SYNTHETIC_PRINCIPAL,seedSyntheticEpisode } from "./synthetic.js";
import { runOneJob } from "./worker.js";
import { loadFhirSources } from "./fhir-config.js";
import { createHfsTreatmentResponseProtocol } from "@phenometrix/condition-profiles";

const config=loadServiceConfig();
const pool=new Pool({connectionString:config.databaseUrl,max:10});
await migrate(pool);
const repository=new PostgresRepository(pool);
const service=new EncounterService(repository,{mode:config.mode,fhirSources:await loadFhirSources(process.env.FHIR_SOURCE_CONFIG_PATH),allowedProtocolDigests:[(await createHfsTreatmentResponseProtocol()).contentSha256]});
if(config.mode==="synthetic")await seedSyntheticEpisode(service);
const server=createEncounterHttpServer({service,authenticate:createAuthenticator(config),allowedOrigins:config.allowedOrigins,
  ...(config.mode==="synthetic"?{developmentSession:async()=>({dataClass:"synthetic",episodeId:SYNTHETIC_EPISODE_ID,token:await signDevelopmentSession(SYNTHETIC_PRINCIPAL,config.developmentSecret!,config.issuer,config.audience)})}:{})});
let working=false;
const timer=setInterval(async()=>{
  if(working)return;working=true;
  try{await runOneJob(repository);}catch{process.stderr.write("Encounter worker temporarily unavailable.\n");}finally{working=false;}
},1000);
server.listen(config.port,config.host,()=>process.stdout.write(`Encounter service ${config.mode} at http://${config.host}:${config.port}; raw media upload disabled.\n`));
async function stop(){clearInterval(timer);await new Promise<void>(resolve=>server.close(()=>resolve()));while(working)await new Promise(resolve=>setTimeout(resolve,10));await pool.end();}
process.once("SIGINT",()=>{void stop();});process.once("SIGTERM",()=>{void stop();});
