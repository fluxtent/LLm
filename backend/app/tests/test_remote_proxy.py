import unittest
from types import SimpleNamespace

from backend.app.main import _is_recursive_remote_proxy, _should_proxy_to_remote_backend
from backend.app.settings import Settings


class RemoteBackendProxyTests(unittest.TestCase):
    def test_global_api_paths_are_proxied_when_remote_backend_is_configured(self) -> None:
        settings = Settings(remote_backend_url="https://api.medbrief.example")

        for path in ("/health", "/api/config", "/api/keys", "/runtime-config.json", "/v1/chat/completions"):
            with self.subTest(path=path):
                request = SimpleNamespace(url=SimpleNamespace(path=path))
                self.assertTrue(_should_proxy_to_remote_backend(request, settings))

    def test_static_frontend_assets_are_not_proxied(self) -> None:
        settings = Settings(remote_backend_url="https://api.medbrief.example")
        request = SimpleNamespace(url=SimpleNamespace(path="/app.js"))

        self.assertFalse(_should_proxy_to_remote_backend(request, settings))

    def test_recursive_remote_backend_is_detected(self) -> None:
        settings = Settings(remote_backend_url="https://medbriefai.vercel.app")
        request = SimpleNamespace(url=SimpleNamespace(hostname="medbriefai.vercel.app"))

        self.assertTrue(_is_recursive_remote_proxy(request, settings))


if __name__ == "__main__":
    unittest.main()
