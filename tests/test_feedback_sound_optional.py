import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


class FeedbackSoundImportTests(unittest.TestCase):
    def test_imports_without_simpleaudio(self):
        real_import = builtins.__import__

        def import_without_simpleaudio(name, *args, **kwargs):
            if name == "simpleaudio":
                raise ModuleNotFoundError("test blocks optional simpleaudio")
            return real_import(name, *args, **kwargs)

        source = Path(__file__).parents[1] / "nocode_robot_programming/feedback_sound.py"
        spec = importlib.util.spec_from_file_location(
            "feedback_sound_without_audio", source
        )
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=import_without_simpleaudio):
            spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
