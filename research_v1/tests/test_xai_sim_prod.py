"""Stage 10C tests: chunk naming, jointly-eligible baseline-vs-condition counting, and resume safety of the chunk
publisher (a .tmp dir or a dir with a missing/altered file is never mistaken for a completed chunk; quarantine moves
incomplete work aside; a committed chunk verifies)."""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
from rv1 import xai_prod as XP, xai_sim as XS, xai_sim_prod as XSP  # noqa: E402


class TestHelpers(unittest.TestCase):
    def test_chunk_name(self):
        self.assertEqual(XSP.chunk_dir_name(7), "chunk_007")
        self.assertEqual(XSP.chunk_dir_name(87), "chunk_087")

    def test_jointly_eligible(self):
        st = {XS.ref_key("a", "none", 0.0): {"ig2048": "primary"}, XS.ref_key("a", "blur", 1.0): {"ig2048": "primary"},
              XS.ref_key("b", "none", 0.0): {"ig2048": "ineligible"}, XS.ref_key("b", "blur", 1.0): {"ig2048": "primary"},
              XS.ref_key("c", "none", 0.0): {"ig2048": "primary"}, XS.ref_key("c", "blur", 1.0): {"ig2048": "endpoint_failure"}}
        self.assertEqual(XSP.jointly_eligible(st, ["a", "b", "c", "d"], "blur|1.000", "ig2048"), ["a"])


class TestResumeSafety(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _publish(self, name, sha="S"):
        tmp = os.path.join(self.d, f"{name}.tmp-abcd1234")
        os.makedirs(tmp)
        with open(os.path.join(tmp, "records.json"), "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        XP.publish_dir(tmp, os.path.join(self.d, name), {"spec_sha256": sha})
        return os.path.join(self.d, name)

    def test_committed_verifies_and_tmp_not(self):
        final = self._publish("chunk_000")
        self.assertTrue(XP.verify_dir(final, "S"))
        tmp = os.path.join(self.d, "chunk_001.tmp-deadbeef")
        os.makedirs(tmp)
        self.assertFalse(XP.verify_dir(tmp, "S"))                         # interrupted work is not a chunk
        self.assertFalse(XP.verify_dir(os.path.join(self.d, "chunk_001"), "S"))
        q = XP.quarantine_tmp(self.d)
        self.assertEqual(q, ["chunk_001.quarantined-deadbeef"])
        self.assertTrue(XP.verify_dir(final, "S"))

    def test_altered_or_foreign_chunk_refused(self):
        final = self._publish("chunk_002")
        with self.assertRaises(RuntimeError):
            XP.verify_dir(final, "OTHER_SPEC")
        with open(os.path.join(final, "records.json"), "a", encoding="utf-8") as f:
            f.write(" ")
        with self.assertRaises(RuntimeError):
            XP.verify_dir(final, "S")

    def test_dir_without_manifest_not_complete(self):
        os.makedirs(os.path.join(self.d, "chunk_003"))
        self.assertFalse(XP.verify_dir(os.path.join(self.d, "chunk_003"), "S"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
