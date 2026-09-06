import { readFile } from "node:fs/promises";
import type { Pool } from "pg";

export async function migrate(pool: Pool): Promise<void> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query("SELECT pg_advisory_xact_lock(74021610)");
    for(const name of ["001_initial.sql","002_source_provenance.sql","003_access_audit.sql"]){
      const sql = await readFile(new URL(`../migrations/${name}`, import.meta.url), "utf8");
      await client.query(sql);
    }
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally { client.release(); }
}
