import json,sys,subprocess,shutil,tempfile,os
from pathlib import Path
manifest,index,stage=sys.argv[1:]
r=json.loads(Path(manifest).read_text())[int(index)]
scripts=Path(r['scripts'])
if stage=='train':
    subprocess.run(['/home/jwinchel/.venv/resolvi/bin/python','-u',str(scripts/'run_resolvi.py'),
        r['input'],'--output-root',r['output_root'],'--model-dir',r['model_dir'],
        '--epochs',str(r['epochs']),'--samples',str(r['samples'])],check=True)
elif stage=='metrics':
    for family,config in r['configs'].items():
        subprocess.run(['/home/jwinchel/miniconda3/envs/heimdall/bin/python','-u',
            str(scripts/'replay_resolvi_metrics.py'),config,r['corrected'],
            str(Path(r['metrics_root'])/family/Path(r['input']).stem)],check=True)
    # Publish only after every requested metric family has passed validation.
    # Stage copies beside destinations so replacement is atomic per file.
    replacements=[(Path(r['corrected']),Path(r['publish_corrected']))]
    for family in r['configs']:
        source=Path(r['metrics_root'])/family/Path(r['input']).stem
        destination=Path(r['publish_metrics_root'])/family/Path(r['input']).stem
        replacements.extend((p,destination/p.name) for p in source.iterdir() if p.is_file())
    staged=[]
    try:
        for source,destination in replacements:
            destination.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(prefix='.resolvi-publish-',dir=destination.parent)
            os.close(fd)
            staged.append((Path(tmp),destination))
            shutil.copy2(source,tmp)
        for tmp,destination in staged:
            os.replace(tmp,destination)
        report=Path(r['model_dir']).parent.parent/'published'/ (Path(r['input']).stem+'.json')
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps({'published':[str(d) for _,d in replacements],
                                     'model_dir':r['model_dir']},indent=2))
        print('Published successful corrected matrix and metrics:',r['publish_corrected'],flush=True)
    finally:
        for tmp,_ in staged:
            if tmp.exists():tmp.unlink()
else:raise ValueError(stage)
