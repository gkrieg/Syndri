"""Download or verify the exact file-level release manifest."""
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from . import ROOT
from .parameters import settings, resolve, configuration_path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def show_sources():
    """Public study landing pages are distinct from the processed asset archive."""
    sources = json.loads(configuration_path('dataset_sources').read_text())
    for name, entry in sources['datasets'].items():
        print(f'{name}: {entry["name"]}')
        for kind in ['spatial', 'reference']:
            source = entry[kind]
            print(f'  {kind}: {source["url"]}')
            if source.get('note'):
                print(f'    {source["note"]}')


def download(verify=None, base_url=None, allow_download=None, quiet=False):
    """Keep included files, copy configured local files, then fetch missing assets.

    Imported/downloaded files always require a matching manifest checksum.
    All missing sources are checked before any copying or network operations.
    """
    cfg = settings()['assets']
    verify = cfg.get('verify', True) if verify is None else verify
    allow_download = cfg.get('auto_download', True) if allow_download is None else allow_download
    base_url = base_url or cfg.get('base_url')
    manifest = json.loads(resolve(cfg['manifest']).read_text())
    pending, missing = [], []
    for item in manifest['files']:
        path = resolve(item['path'])
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Asset destination must be inside the repository: {item["path"]}')
        if path.is_file():
            if verify and not item.get('mutable', False) and sha256(path) != item['sha256']:
                raise ValueError(f'Checksum mismatch: {item["path"]}')
            if not quiet:
                print(f'Present: {item["path"]}', flush=True)
            continue
        local = cfg.get('local_files', {}).get(item['path'])
        if local:
            local = resolve(local)
        elif cfg.get('local_directory'):
            local = resolve(cfg['local_directory']) / item['path']
        if local is not None and local.is_file():
            pending.append((item, path, local, None))
            continue
        url = cfg.get('urls', {}).get(item['path']) or (
            f'{base_url.rstrip("/")}/{item["path"]}' if base_url else item.get('url', ''))
        if not allow_download or not url.startswith('https://') or 'FILL' in url:
            missing.append(item['path'])
        else:
            pending.append((item, path, None, url))
    if missing:
        raise FileNotFoundError(
            'Missing assets:\n  ' + '\n  '.join(missing) +
            '\nSet assets.local_directory, assets.local_files, assets.base_url, or assets.urls '
            'in your --params file. Processed-data/checkpoint release URLs are still FILL IN; '
            'public study landing pages cannot supply these exact artifacts.')
    for item, path, local, url in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.part', delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            print(f'{"Copying" if local else "Downloading"}: {item["path"]}', flush=True)
            source = local.open('rb') if local else urllib.request.urlopen(url, timeout=cfg.get('timeout_seconds', 120))
            with source as stream, partial.open('wb') as output:
                shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
            if sha256(partial) != item['sha256']:
                raise ValueError(f'Asset checksum mismatch: {item["path"]}')
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if not quiet:
        print(f'Ready: {len(manifest["files"])} required files.', flush=True)

