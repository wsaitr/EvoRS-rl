from rs_hmcomm.data.obs_loader import OBSConfig
from rs_hmcomm.data.cache import DataCache, CacheEntry
import tempfile
from pathlib import Path


def test_obs_config_from_env(monkeypatch):
    monkeypatch.setenv("OBS_ENDPOINT", "obs.test.com")
    monkeypatch.setenv("OBS_BUCKET", "test-bucket")
    monkeypatch.setenv("OBS_ACCESS_KEY_ID", "test-ak")
    monkeypatch.setenv("OBS_SECRET_ACCESS_KEY", "test-sk")
    config = OBSConfig.from_env()
    assert config.endpoint == "obs.test.com"
    assert config.bucket == "test-bucket"
    assert config.is_configured


def test_obs_config_not_configured():
    config = OBSConfig()
    assert not config.is_configured


def test_data_cache_put_get():
    with tempfile.TemporaryDirectory() as td:
        cache = DataCache(td)
        # Create a temp file
        test_file = Path(td) / "test.txt"
        test_file.write_text("hello")
        cache.put("test/key", test_file)
        assert cache.has("test/key")
        assert cache.get("test/key") == test_file
        assert cache.stats["total_entries"] == 1


def test_data_cache_remove():
    with tempfile.TemporaryDirectory() as td:
        cache = DataCache(td)
        test_file = Path(td) / "test.txt"
        test_file.write_text("hello")
        cache.put("key1", test_file)
        assert cache.has("key1")
        cache.remove("key1")
        assert not cache.has("key1")


def test_data_cache_manifest_persistence():
    with tempfile.TemporaryDirectory() as td:
        cache1 = DataCache(td)
        test_file = Path(td) / "test.txt"
        test_file.write_text("data")
        cache1.put("key1", test_file)

        # Create new cache instance - should load manifest
        cache2 = DataCache(td)
        assert cache2.has("key1")


def test_data_cache_clear():
    with tempfile.TemporaryDirectory() as td:
        cache = DataCache(td)
        for i in range(3):
            f = Path(td) / f"file{i}.txt"
            f.write_text(f"data{i}")
            cache.put(f"key{i}", f)
        assert cache.stats["total_entries"] == 3
        cache.clear()
        assert cache.stats["total_entries"] == 0
