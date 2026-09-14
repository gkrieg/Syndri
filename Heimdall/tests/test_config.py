from hydra import compose, initialize
from omegaconf import OmegaConf


def test_config_with_overrides():
    with initialize(version_base=None, config_path="../config", job_name="test"):
        cfg = compose(config_name="config", overrides=["fg=identity", "fe=binning_scgpt"])
        resolved_config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    # assert resolved_config.fe.args.embedding_parameters.args.out_features == resolved_config.fe.args.d_embedding


def test_admixture_external_eval_config_defaults():
    with initialize(version_base=None, config_path="../config", job_name="test"):
        cfg = compose(
            config_name="config",
            overrides=["tasks=admixture_correction", "fg=identity", "fe=binning_scgpt"],
        )

    external_eval = cfg.tasks.args.external_eval
    assert external_eval.enabled is False
    assert external_eval.data_path is None
    assert external_eval.split == "full"
    assert external_eval.prefix == "external_eval"
    assert external_eval.every_n_epochs == 1
    assert external_eval.fixed_epoch is None
    assert external_eval.require_matching_var_names is True
    assert external_eval.run_model_metrics is True
    assert external_eval.metrics_eval.enabled is False
    assert external_eval.metrics_eval.script_path is None
    assert external_eval.metrics_eval.probability_layer == "_training_syndri_probs"
    assert external_eval.metrics_eval.write_outputs is True
