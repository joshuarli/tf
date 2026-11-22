import argparse
import hashlib
import json
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REVISION = 'e72868dac12df4be4ced2dc8355f031686283974'
SHA256 = '30472a3375c2623c16dbf50945425391f209eb4ddd58f3aab2933091048b9967'
MODEL_REVISION = '3eea042751cd296b7266628a8342c1749242e1fe'
MODEL_FILES = frozenset({
    '.gitattributes', 'README.md', 'chat_template.jinja', 'config.json', 'generation_config.json',
    'model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors',
    'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors',
    'model.safetensors.index.json', 'mtp.safetensors', 'processor_config.json',
    'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
})
CAMPAIGN = Path(__file__).resolve().parents[2]
COMMON = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=CAMPAIGN, text=True).strip()).parent
WORK = COMMON / '.work'
SOURCE = WORK / 'src' / f'omlx@{REVISION[:12]}'
EXTRACTION = SOURCE / f'omlx-{REVISION}'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_entry(entries):
    matches = [item for item in entries if item.get('model') == 'omlx']
    if len(matches) != 1 or matches[0]['version'] != REVISION or matches[0]['sha256'] != SHA256:
        raise ValueError('oMLX source revision or hash does not match the pinned release')
    return matches[0]


def model_entries(entries):
    matches = [item for item in entries if item.get('model') == 'inkling-2-mini-oq4e-mtp']
    names = [item['name'].split('/', 1)[1] for item in matches]
    if (len(matches) != len(MODEL_FILES) or set(names) != MODEL_FILES
            or any(item['version'] != MODEL_REVISION for item in matches)):
        raise ValueError('oMLX model lock does not match the pinned revision')
    return matches


def validate_lock():
    entries = json.loads((CAMPAIGN / 'sources.lock.json').read_text())['inputs']
    entry = source_entry(entries)
    model_entries(entries)
    return entry


def prepare():
    entry = validate_lock()
    archive = SOURCE / entry['name'].split('/', 1)[1]
    if archive.stat().st_size != entry['size'] or digest(archive) != SHA256:
        raise ValueError('oMLX archive failed size or SHA-256 verification')
    with tempfile.TemporaryDirectory(dir=WORK / 'tmp', prefix='omlx-extract-') as temp:
        with tarfile.open(archive, 'r:gz') as bundle:
            bundle.extractall(temp, filter='data')
        extracted = Path(temp) / f'omlx-{REVISION}'
        paths = sorted(path.relative_to(extracted) for path in extracted.rglob('*'))
        if any((extracted / path).is_symlink() for path in paths):
            raise ValueError('oMLX source archive contains a symlink')
        files = [path for path in paths if (extracted / path).is_file()]
        if not files or not (extracted / 'LICENSE').is_file():
            raise ValueError('oMLX archive is missing source or license')
        if EXTRACTION.exists():
            existing = sorted(path.relative_to(EXTRACTION) for path in EXTRACTION.rglob('*'))
            if EXTRACTION.is_symlink() or existing != paths:
                raise ValueError('existing oMLX extraction has changed file set')
            for path in paths:
                old, new = EXTRACTION / path, extracted / path
                if old.is_symlink() or old.is_dir() != new.is_dir():
                    raise ValueError(f'existing oMLX extraction differs: {path}')
                if stat.S_IMODE(old.stat().st_mode) != stat.S_IMODE(new.stat().st_mode):
                    raise ValueError(f'existing oMLX extraction mode differs: {path}')
                if new.is_file() and (not old.is_file() or digest(old) != digest(new)):
                    raise ValueError(f'existing oMLX extraction differs: {path}')
        else:
            extracted.rename(EXTRACTION)
    print(json.dumps({'source_revision': REVISION, 'archive_sha256': SHA256, 'source_files': len(files)}))


def verify_install():
    import omlx
    installed = Path(omlx.__file__).parent
    native = [path for path in (installed / 'custom_kernels').rglob('*')
              if path.suffix in ('.so', '.metallib')]
    if native:
        raise ValueError('unexpected native custom kernel in oMLX installation')
    files = sorted(path for path in (EXTRACTION / 'omlx').rglob('*')
                   if path.is_file() and path.suffix in ('.py', '.json'))
    if not files:
        raise ValueError('verified oMLX extraction is missing')
    for source in files:
        target = installed / source.relative_to(EXTRACTION / 'omlx')
        if not target.is_file() or digest(source) != digest(target):
            raise ValueError(f'installed oMLX differs from archive: {source.relative_to(EXTRACTION)}')
    print(json.dumps({'installed_source_files_verified': len(files), 'installed': str(installed)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-install', action='store_true')
    parser.add_argument('--validate-lock', action='store_true')
    args = parser.parse_args()
    if args.validate_lock:
        validate_lock()
    elif args.verify_install:
        verify_install()
    else:
        prepare()


if __name__ == '__main__':
    main()
