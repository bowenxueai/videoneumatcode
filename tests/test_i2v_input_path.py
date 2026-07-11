import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class I2VInputPathTest(unittest.TestCase):
    def setUp(self):
        (REPO / "outputs").mkdir(exist_ok=True)
        self.tempdir = tempfile.TemporaryDirectory(dir=REPO / "outputs")
        self.root = Path(self.tempdir.name)
        self.cwd = self.root / "caller"
        self.cwd.mkdir()
        self.image = self.cwd / "input.ppm"
        self.image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="ascii")
        self.repo_runtime = self.root / "runtime"
        self.repo_runtime.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def base_env(self):
        env = os.environ.copy()
        env.update(
            {
                "NEUGEN_REPO": str(self.repo_runtime),
                "WEIGHTS_DIR": str(self.repo_runtime / "weights"),
                "ACCOUNT": "test-account",
                "PARTITION": "test-partition",
                "GPUS_PER_JOB": "1",
            }
        )
        return env

    def test_slurm_entrypoint_exports_absolute_image_path(self):
        capture = self.root / "sbatch-inputs.txt"
        fake_bin = self.root / "bin"
        write_executable(
            fake_bin / "sbatch",
            "#!/bin/bash\nprintf '%s\\n' \"${INPUT_IMAGE:-}\" >> \"$TEST_CAPTURE\"\nprintf '123\\n'\n",
        )
        env = self.base_env()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["TEST_CAPTURE"] = str(capture)

        subprocess.run(
            [
                str(REPO / "release/oneclick_material.sh"),
                "--i2v",
                self.image.name,
                "test prompt",
            ],
            cwd=self.cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        exported = [line for line in capture.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(exported)
        self.assertEqual(set(exported), {str(self.image.resolve())})

    def test_local_entrypoint_passes_absolute_image_path(self):
        capture = self.root / "python-args.txt"
        fake_env = self.root / "fake-env"
        write_executable(
            fake_env / "bin/python",
            "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$TEST_CAPTURE\"\nexit 23\n",
        )
        env = self.base_env()
        env.update(
            {
                "GEN_ENV": str(fake_env),
                "NEUGEN_ENV": str(fake_env),
                "TEST_CAPTURE": str(capture),
            }
        )

        completed = subprocess.run(
            [
                str(REPO / "release/run_local.sh"),
                "--i2v",
                self.image.name,
                "test prompt",
            ],
            cwd=self.cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 23)
        args = capture.read_text(encoding="utf-8").splitlines()
        image_index = args.index("--image") + 1
        self.assertEqual(args[image_index], str(self.image.resolve()))


if __name__ == "__main__":
    unittest.main()
