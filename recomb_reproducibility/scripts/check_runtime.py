#!/usr/bin/env python
"""Exercise simulation, correction, and exactly one optimizer step on a small input."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['ACCELERATE_USE_CPU'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['WANDB_MODE'] = 'disabled'
os.environ['MPLCONFIGDIR'] = str(ROOT/'outputs/.checks/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'outputs/.checks/numba')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy import sparse
from omegaconf import OmegaConf
import Heimdall
from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.trainer import HeimdallTrainer
from accelerate.utils import set_seed
from syndri_repro.io import read, write, data_dir
from syndri_repro.model import runtime_config
from syndri_repro.correction import correct
from syndri_repro.simulation import sample, mmd2


def main():
    assert Path(Heimdall.__file__).resolve().is_relative_to(ROOT)
    for module in [sys.modules['Heimdall.cell_representations'],sys.modules['Heimdall.models'],sys.modules['Heimdall.trainer']]:
        assert Path(module.__file__).resolve().is_relative_to(ROOT), module.__file__
    counts=sparse.csr_matrix([[10.,4.],[5.,3.]])
    probabilities=sparse.csr_matrix([[.9,.1],[.8,.1]])
    cleaned,diagnostics=correct(counts,probabilities,['A','A'],target_fraction=.5)
    np.testing.assert_array_equal(cleaned.toarray(),[[0,4],[0,3]])
    assert counts.sum()==22 and np.isclose(diagnostics.achieved_removed_fraction.iloc[0],15/22)
    assert mmd2(np.eye(3),np.eye(3),1)==0
    reference=read(data_dir('NC')/'training.h5ad')
    rng=np.random.default_rng(1);n=24;g=40
    values=rng.poisson(60,size=(n,g)).astype(np.float32)
    values[rng.random((n,g))<.65]=0
    obs=pd.DataFrame(index=[f'check_{i}' for i in range(n)])
    obs['cell_type']=np.array(['A','B','C'])[np.arange(n)%3]
    obs['cell_type_harmonized']=obs.cell_type
    obs['split']=np.where(np.arange(n)%5==4,'val','train')
    obs['CenterX_global_px']=rng.normal(size=n);obs['CenterY_global_px']=rng.normal(size=n)
    obs['spatial_slice']='check';obs['batch']='check'
    obj=ad.AnnData(sparse.csr_matrix(values),obs=obs,var=reference.var.iloc[:g].copy())
    obj.uns['cell_type_dict']={'A':-1,'B':-2,'C':-3}
    simulated=sample(obj,seed=1)
    added=simulated.layers['synthetic_admixture_counts']
    assert added.nnz>0 and added.multiply(obj.X>0).nnz==0
    residual=simulated.X-obj.X-added
    assert residual.nnz==0
    folder=ROOT/'outputs/.checks/training';folder.mkdir(parents=True,exist_ok=True)
    input_path=folder/'toy.h5ad';write(obj,input_path)
    cfg=runtime_config('NC',folder, input_path)
    cfg.tasks.args.epochs=1;cfg.tasks.args.batchsize=8
    cfg.fc.args.max_input_length=80;cfg.tasks.args.context_length=74
    cfg.trainer.accumulate_grad_batches=1
    cfg.trainer.num_workers=0
    cfg.trainer.fastdev=True
    cfg.dataset.preprocess_args.admixture_args.tissue_col='batch'
    cfg.dataset.preprocess_args.admixture_args.min_source_transcript_count=0
    cfg.dataset.preprocess_args.admixture_args.tau_range=[.1,.2]
    set_seed(cfg.seed);torch.set_num_threads(2)
    data=CellRepresentation(cfg)
    model=HeimdallModel(data=data,model_config=cfg.model,task_config=cfg.tasks.args)
    trainer=HeimdallTrainer(cfg=cfg,model=model,data=data,run_wandb=False)
    before={name: parameter.detach().clone() for name,parameter in trainer.model.named_parameters()}
    losses=[]
    hook=trainer.loss_fn.register_forward_hook(lambda module,inputs,output: losses.append(float(output.detach())))
    trainer._set_dataset_epoch(0)
    trainer.train_epoch(0)  # fastdev stops after the first batch.
    hook.remove()
    assert trainer.step==1, f'Expected one optimizer step, got {trainer.step}'
    assert len(losses)==1 and np.isfinite(losses[0]), f'Invalid training loss: {losses}'
    assert any(not torch.equal(before[name],parameter.detach()) for name,parameter in trainer.model.named_parameters()), 'Optimizer did not update model weights'
    assert all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters()), 'Nonfinite model weights'
    trainer.initialize_checkpointing(folder)
    trainer.save_checkpoint(0)
    checkpoints=list(folder.glob('model-*.pt'))
    assert checkpoints, 'Training did not save a checkpoint'
    assert (folder/'config.txt').exists()
    saved={name:parameter.detach().clone() for name,parameter in trainer.model.named_parameters()}
    trainer.load_checkpoint(specific_milestone=0)
    assert all(torch.equal(saved[name],parameter.detach()) for name,parameter in trainer.model.named_parameters()), 'Checkpoint reload changed model weights'
    report={'status':'passed','training_steps':trainer.step,'training_loss':losses[0],
            'checks':['isolated runtime imports','count correction','synthetic truth','MMD identity',
                      'forward pass','backpropagation','optimizer update','checkpoint save and reload'],
            'full_training_run':False,'full_inference_run':False}
    from syndri_repro.io import save_json
    save_json(report,folder/'check_result.json')
    print(f'PASS: one training step (loss={losses[0]:.6f}), finite updated weights, checkpoint save/reload, and supporting checks')


if __name__=='__main__':main()

