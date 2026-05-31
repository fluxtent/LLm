import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_storage_write_failure_keeps_chat_state_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(storage_path=Path(temp_dir) / "store.json")
            with patch.object(Path, "write_text", side_effect=OSError("read-only")):
                stored = store.upsert_profile(UserProfile(user_id="user-1", recurring_topics=["school"]))

            self.assertEqual(stored.recurring_topics, ["school"])
            self.assertIsNone(store.storage_path)
            self.assertEqual(store.get_profile("user-1").recurring_topics, ["school"])

    def test_learning_events_persist_and_export_trainable_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "store.json"
            first = MemoryStore(storage_path=store_path)

            first.add_learning_event(
                prompt="What helps with burnout?",
                response="Burnout often needs rest, boundaries, and practical load reduction.",
                mode="psych",
                user_id="user-1",
                conversation_id="conversation-1",
                request_id="request-1",
                model="medbrief-local",
                safety_flag="allowed",
                trainable=True,
            )
            first.add_learning_event(
                prompt="I want to die",
                response="Please call or text 988 right now if you are in the US.",
                mode="crisis",
                user_id="user-1",
                conversation_id="conversation-1",
                request_id="request-2",
                model="medbrief-local",
                safety_flag="crisis_intercept",
                trainable=False,
            )

            second = MemoryStore(storage_path=store_path)
            trainable = second.export_learning_events(trainable_only=True)
            all_events = second.export_learning_events(trainable_only=False)

            self.assertEqual(len(trainable), 1)
            self.assertEqual(trainable[0]["prompt"], "What helps with burnout?")
            self.assertEqual(len(all_events), 2)

    def test_delete_user_removes_learning_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(storage_path=Path(temp_dir) / "store.json")
            store.add_learning_event(
                prompt="Remember this",
                response="Stored locally.",
                mode="general",
                user_id="user-1",
                conversation_id=None,
                request_id="request-1",
                model="medbrief-local",
                safety_flag="allowed",
                trainable=True,
            )

            store.delete_user("user-1")

            self.assertEqual(store.export_learning_events(trainable_only=False), [])


if __name__ == "__main__":
    unittest.main()
