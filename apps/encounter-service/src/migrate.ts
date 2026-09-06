import { Pool } from "pg";
import { loadServiceConfig } from "./config.js";
import { migrate } from "./migration.js";
const config=loadServiceConfig();
const pool=new Pool({connectionString:config.databaseUrl});
try{await migrate(pool);}finally{await pool.end();}
