import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("nanoattention_setup", Path(__file__).with_name("setup.py"))
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def fixture(self, directory):
        archive = directory / "source.tar.gz"
        files = {"build.zig": b"pub fn build() void {\n    b.installArtifact(nano);\n}\n",
                 "src/main.zig": b"const checkpoint = 42;\n", "LICENSE": b"MIT license fixture\n"}
        with tarfile.open(archive, "w:gz") as bundle:
            for name, data in files.items():
                info = tarfile.TarInfo(f"nanoattention-{setup.SOURCE_COMMIT}/{name}")
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        return archive, setup.sha256(archive), files

    def test_prepare_preserves_inference_and_license_and_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, digest, files = self.fixture(root)
            build = root / "build"
            first = setup.prepare(archive, build, digest, archive.stat().st_size)
            second = setup.prepare(archive, build, digest, archive.stat().st_size)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "prepared")
            self.assertIsNone(first["binary_sha256"])
            self.assertEqual((build / "src/main.zig").read_bytes(), files["src/main.zig"])
            self.assertEqual((build / "LICENSE").read_bytes(), files["LICENSE"])
            self.assertEqual(first["upstream_build_sha256"], hashlib.sha256(files["build.zig"]).hexdigest())
            self.assertIn(b'b.step("campaign-nano"', (build / "build.zig").read_bytes())

    def test_modified_existing_source_is_preserved_and_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, digest, _ = self.fixture(root)
            build = root / "build"
            setup.prepare(archive, build, digest, archive.stat().st_size)
            changed = build / "src/main.zig"
            changed.write_text("local investigation\n")
            with self.assertRaisesRegex(ValueError, "existing comparator source differs"):
                setup.prepare(archive, build, digest, archive.stat().st_size)
            self.assertEqual(changed.read_text(), "local investigation\n")

    def test_corrupt_archive_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, digest, _ = self.fixture(root)
            size = archive.stat().st_size
            archive.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "pinned size/SHA-256"):
                setup.prepare(archive, root / "build", digest, size)
            self.assertFalse((root / "build").exists())


if __name__ == "__main__":
    unittest.main()
