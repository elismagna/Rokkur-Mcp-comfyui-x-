from pathlib import Path

from rokkur_studio.config import env_overrides, load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_defaults_and_profiles_load():
    s = load_settings(ROOT / "config", environ={})
    assert s.studio.autonomy_level == 2
    assert s.gpu.vram_gb == 8 and s.gpu.max_heavy_jobs == 1
    assert {"PREVIEW", "RTX3070_QUALITY", "HYBRID_MAX", "FUTURE_24GB"} <= set(s.profiles)
    assert s.profile("RTX3070_QUALITY").resource_class == "GPU_HEAVY"
    assert s.rights.block_unknown is True
    assert s.youtube.auto_publish is False


def test_env_overrides_are_typed_and_nested():
    s = load_settings(ROOT / "config", environ={
        "STUDIO_GPU__VRAM_GB": "24", "STUDIO_YOUTUBE__ENABLED": "true",
        "DATABASE_URL": "postgresql+psycopg://x@db/y",
        "STUDIO_PROFILES__PREVIEW__FPS": "8"})
    assert s.gpu.vram_gb == 24
    assert s.youtube.enabled is True
    assert s.database.url.endswith("@db/y")
    assert s.profiles["PREVIEW"].fps == 8 and s.profiles["PREVIEW"].max_frames == 48


def test_env_override_parser_ignores_unrelated():
    assert env_overrides({"PATH": "/bin", "STUDIO_X": "1"}) == {}
