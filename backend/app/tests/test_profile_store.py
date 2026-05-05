import tempfile
import unittest
from pathlib import Path

from backend.app.profile_store import MemoryStore
from backend.app.schemas import UserProfile


class MemoryStoreTests(unittest.TestCase):
    def test_profile_and_summary_persist_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "store.json"
            first = MemoryStore(storage_path=store_path)

            first.upsert_profile(UserProfile(user_id="user-1", recurring_topics=["burnout"]))
            first.set_session_summary("user-1", "session-1", "The user is tracking burnout patterns.")

            second = MemoryStore(storage_path=store_path)

            profile = second.get_profile("user-1")
            self.assertIsNotNone(profile)
            self.assertEqual(profile.recurring_topics, ["burnout"])
            self.assertEqual(second.latest_summary("user-1"), "The user is tracking burnout patterns.")

    def test_api_keys_are_hashed_validated_and_revoked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(storage_path=Path(temp_dir) / "store.json")

            raw_key, record = store.create_api_key("test key")

            self.assertTrue(raw_key.startswith("mbk_"))
            self.assertEqual(record["label"], "test key")
            self.assertNotIn(raw_key, store.storage_path.read_text(encoding="utf-8"))

            authenticated = store.authenticate_api_key(raw_key)
            self.assertIsNotNone(authenticated)
            self.assertEqual(authenticated["id"], record["id"])

            store.revoke_api_key(record["id"])

            self.assertIsNone(store.authenticate_api_key(raw_key))


if __name__ == "__main__":
    unittest.main()
