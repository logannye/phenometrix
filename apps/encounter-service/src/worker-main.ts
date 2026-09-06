import { Pool } from "pg";
import { loadServiceConfig } from "./config.js";
import { PostgresRepository } from "./repository.postgres.js";
import { runOneJob } from "./worker.js";

const config=loadServiceConfig();
const pool=new Pool({connectionString:config.databaseUrl,max:3});
const repository=new PostgresRepository(pool);
let stopping=false;
process.once("SIGINT",()=>{stopping=true;});process.once("SIGTERM",()=>{stopping=true;});
while(!stopping){
  try{
    const result=await runOneJob(repository);
    if(result==="idle" || result==="failed")await new Promise(resolve=>setTimeout(resolve,1000));
  }catch{process.stderr.write("Encounter worker temporarily unavailable.\n");await new Promise(resolve=>setTimeout(resolve,1000));}
}
await pool.end();
