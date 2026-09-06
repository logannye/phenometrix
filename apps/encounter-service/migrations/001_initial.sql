CREATE TABLE IF NOT EXISTS encounter_schema_migrations (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS encounter_episodes (
  id text PRIMARY KEY, tenant_id text NOT NULL, subject_ref text NOT NULL,
  data_class text NOT NULL CHECK (data_class IN ('synthetic','consented-research')),
  created_at timestamptz NOT NULL, revision integer NOT NULL DEFAULT 0 CHECK (revision >= 0), metadata jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS encounter_episodes_tenant ON encounter_episodes(tenant_id, subject_ref);
CREATE TABLE IF NOT EXISTS encounter_inputs (
  id uuid PRIMARY KEY, episode_id text NOT NULL REFERENCES encounter_episodes(id),
  revision integer NOT NULL, kind text NOT NULL CHECK(kind IN ('session','clinical-event','consent','binding','annotation')),
  logical_id text NOT NULL, replaces_record_id uuid REFERENCES encounter_inputs(id),
  idempotency_key text NOT NULL, content_hash text NOT NULL, payload jsonb NOT NULL,
  actor_id text NOT NULL, recorded_at timestamptz NOT NULL,
  UNIQUE(episode_id, revision), UNIQUE(episode_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS encounter_jobs (
  id uuid PRIMARY KEY, tenant_id text NOT NULL, episode_id text NOT NULL REFERENCES encounter_episodes(id),
  input_revision integer NOT NULL,
  status text NOT NULL CHECK(status IN ('pending','running','completed','superseded','failed')),
  attempts integer NOT NULL DEFAULT 0, lease_token uuid, lease_expires_at timestamptz, error_code text,
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(episode_id,input_revision)
);
CREATE INDEX IF NOT EXISTS encounter_jobs_claim ON encounter_jobs(status,created_at);
CREATE TABLE IF NOT EXISTS encounter_snapshots (
  id uuid PRIMARY KEY, episode_id text NOT NULL REFERENCES encounter_episodes(id), input_revision integer NOT NULL,
  algorithm_version text NOT NULL, input_hash text NOT NULL, analysis jsonb NOT NULL, created_at timestamptz NOT NULL,
  UNIQUE(episode_id,input_revision,algorithm_version)
);
CREATE TABLE IF NOT EXISTS encounter_reviews (
  id uuid PRIMARY KEY, episode_id text NOT NULL REFERENCES encounter_episodes(id), snapshot_id uuid NOT NULL REFERENCES encounter_snapshots(id),
  input_revision integer NOT NULL, payload jsonb NOT NULL,
  recorded_at timestamptz NOT NULL, idempotency_key text NOT NULL,
  UNIQUE(episode_id,idempotency_key)
);
CREATE OR REPLACE FUNCTION encounter_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Encounter source, evidence, and review history is append-only'; END;
$$;
DROP TRIGGER IF EXISTS encounter_inputs_immutable ON encounter_inputs;
CREATE TRIGGER encounter_inputs_immutable BEFORE UPDATE OR DELETE ON encounter_inputs FOR EACH ROW EXECUTE FUNCTION encounter_reject_mutation();
DROP TRIGGER IF EXISTS encounter_snapshots_immutable ON encounter_snapshots;
CREATE TRIGGER encounter_snapshots_immutable BEFORE UPDATE OR DELETE ON encounter_snapshots FOR EACH ROW EXECUTE FUNCTION encounter_reject_mutation();
DROP TRIGGER IF EXISTS encounter_reviews_immutable ON encounter_reviews;
CREATE TRIGGER encounter_reviews_immutable BEFORE UPDATE OR DELETE ON encounter_reviews FOR EACH ROW EXECUTE FUNCTION encounter_reject_mutation();
INSERT INTO encounter_schema_migrations(version) VALUES(1) ON CONFLICT DO NOTHING;
