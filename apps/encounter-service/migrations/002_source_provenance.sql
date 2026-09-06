ALTER TABLE encounter_inputs ADD COLUMN IF NOT EXISTS source_provenance jsonb;
CREATE OR REPLACE FUNCTION encounter_protect_episode_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF ROW(NEW.id,NEW.tenant_id,NEW.subject_ref,NEW.data_class,NEW.created_at,NEW.metadata)
     IS DISTINCT FROM ROW(OLD.id,OLD.tenant_id,OLD.subject_ref,OLD.data_class,OLD.created_at,OLD.metadata) THEN
    RAISE EXCEPTION 'Episode identity, protocol and analysis specification are immutable';
  END IF;
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS encounter_episode_identity ON encounter_episodes;
CREATE TRIGGER encounter_episode_identity BEFORE UPDATE ON encounter_episodes FOR EACH ROW EXECUTE FUNCTION encounter_protect_episode_identity();
INSERT INTO encounter_schema_migrations(version) VALUES(2) ON CONFLICT DO NOTHING;
