CREATE TABLE IF NOT EXISTS encounter_access_audit (
  id uuid PRIMARY KEY,tenant_id text NOT NULL,episode_id text NOT NULL REFERENCES encounter_episodes(id),
  actor_id text NOT NULL,recorded_at timestamptz NOT NULL,
  action text NOT NULL CHECK(action IN ('capture-context-read','evidence-read','evidence-history-read')),
  result text NOT NULL CHECK(result IN ('allowed','denied')),input_revision integer NOT NULL,artifact_id text
);
CREATE INDEX IF NOT EXISTS encounter_access_audit_episode ON encounter_access_audit(episode_id,recorded_at);
DROP TRIGGER IF EXISTS encounter_access_immutable ON encounter_access_audit;
CREATE TRIGGER encounter_access_immutable BEFORE UPDATE OR DELETE ON encounter_access_audit FOR EACH ROW EXECUTE FUNCTION encounter_reject_mutation();
CREATE UNIQUE INDEX IF NOT EXISTS encounter_unique_source_revision ON encounter_inputs(episode_id,kind,(payload->>'revisionId')) WHERE kind IN ('session','clinical-event') AND payload ? 'revisionId';
INSERT INTO encounter_schema_migrations(version) VALUES(3) ON CONFLICT DO NOTHING;
