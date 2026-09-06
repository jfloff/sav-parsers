from sav_parsers.schema import load_schema


def test_load_schema_is_cwd_independent(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  entities = load_schema("fpb_modelo_1")

  assert "licenca_fpb" in entities
  assert "licenca_fpb_region" in entities
  assert "carimbo_clube_presente" in entities
