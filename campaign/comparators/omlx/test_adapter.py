import json
import io
import hashlib
import contextlib
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter import mtp_cycles, parse_result, read_prompt, redact_other_processes, validate_model_path
from prompt_ids import prompt_ids
import setup
from setup import MODEL_FILES, MODEL_REVISION, REVISION, SHA256, model_entries, source_entry


class AdapterTests(unittest.TestCase):
    def test_prompt_identity_and_context(self):
        prompt = Path(__file__).resolve().parents[2] / 'prompts/inkling/code-1024.json'
        self.assertEqual(read_prompt(prompt, 512), json.loads(prompt.read_text()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'prompt.json'
            tokens = [2, 123, 456]
            path.write_text(json.dumps(tokens))
            self.assertEqual(read_prompt(path, 512), tokens)
            with self.assertRaises(ValueError):
                read_prompt(path, 32768)

    def test_result_validation_and_rates(self):
        result = {'type': 'omlx_comparator', 'prompt_tokens': 1024, 'generated_ids': [12, 13, 14],
                  'ttft_seconds': 2, 'decode_seconds': 1, 'mlx_peak_bytes': 1024}
        parsed = parse_result('log\n' + json.dumps(result), 1024, 3)
        self.assertEqual(parsed['generated_ids'], [12, 13, 14])
        self.assertEqual(parsed['prefill_tokens_per_second'], 512)
        self.assertEqual(parsed['decode_tokens_per_second'], 2)
        result['generated_ids'] = [12]
        with self.assertRaises(ValueError):
            parse_result(json.dumps(result), 1024, 3)
        self.assertEqual(mtp_cycles('INFO MTP[7] finish=length tokens=512 cycles=123 tok/cycle=4'), 123)
        with self.assertRaises(ValueError):
            mtp_cycles('no speculation')
        self.assertEqual(redact_other_processes('contended\n  GPU time users: name pid=1\nfailed'),
                         'contended\n[unrelated process details removed]\nfailed')

    def test_prompt_template_and_generation_tag(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            tokenizer = Tokenizer(models.WordLevel({'hello': 0, 'world': 1, '[UNK]': 2}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(model / 'tokenizer.json'))
            (model / 'chat_template.jinja').write_text(
                "{% for message in messages %}{{ message.content }}{% endfor %} "
                "{% generation %}world{% endgeneration %}")
            self.assertEqual(prompt_ids(model, 'hello'), [0, 1])

    def test_wrong_source_revision_rejected(self):
        validate_model_path(Path.home() / 'models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd')
        with self.assertRaises(ValueError):
            validate_model_path(Path.home() / 'models/Inkling-2-Mini-oQ4e-mtp@wrong')
        entry = {'model': 'omlx', 'version': REVISION, 'sha256': SHA256}
        self.assertEqual(source_entry([entry]), entry)
        with self.assertRaises(ValueError):
            source_entry([entry | {'version': 'wrong'}])
        models = [{'model': 'inkling-2-mini-oq4e-mtp', 'version': MODEL_REVISION,
                   'name': f'inkling-2-mini-oq4e-mtp/{name}'} for name in MODEL_FILES]
        self.assertEqual(model_entries(models), models)
        models[0]['version'] = 'wrong'
        with self.assertRaises(ValueError):
            model_entries(models)

    def test_changed_extraction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (root / 'tmp').mkdir()
            archive = source / 'omlx.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                for name, contents in [('LICENSE', b'license'), ('code.py', b'code')]:
                    info = tarfile.TarInfo(f'omlx-{REVISION}/{name}')
                    info.size = len(contents)
                    bundle.addfile(info, io.BytesIO(contents))
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            source_row = {'model': 'omlx', 'version': REVISION, 'sha256': sha,
                          'name': 'jundot/omlx.tar.gz', 'size': archive.stat().st_size}
            models = [{'model': 'inkling-2-mini-oq4e-mtp', 'version': MODEL_REVISION,
                       'name': f'inkling-2-mini-oq4e-mtp/{name}'} for name in MODEL_FILES]
            (root / 'sources.lock.json').write_text(json.dumps({'inputs': [source_row, *models]}))
            extracted = source / f'omlx-{REVISION}'
            with patch.multiple(setup, CAMPAIGN=root, WORK=root, SOURCE=source,
                                EXTRACTION=extracted, SHA256=sha), contextlib.redirect_stdout(io.StringIO()):
                setup.prepare()
                (extracted / 'extra').write_text('changed')
                with self.assertRaises(ValueError):
                    setup.prepare()
