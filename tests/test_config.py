"""Tests for the config system.

Configs are the experimental apparatus: an ablation is a config diff, so a
config that silently accepts a typo can invalidate a whole experiment.
"""

from __future__ import annotations

import pytest
import yaml

from minime.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    parse_cli_overrides,
)


def write(tmp_path, name: str, payload: dict):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


# ------------------------------------------------------------- inheritance


def test_extends_merges_parent_and_child(tmp_path):
    write(tmp_path, "base.yaml", {
        "name": "base",
        "model": {"d_model": 128, "n_heads": 4, "n_layers": 2},
        "training": {"max_steps": 100, "learning_rate": 1e-3},
    })
    child = write(tmp_path, "child.yaml", {
        "extends": "base.yaml",
        "name": "child",
        "model": {"n_layers": 6},
    })

    cfg = load_config(child)
    assert cfg.name == "child"
    assert cfg.model.n_layers == 6        # overridden
    assert cfg.model.d_model == 128       # inherited
    assert cfg.training.learning_rate == 1e-3


def test_extends_chains_through_multiple_levels(tmp_path):
    write(tmp_path, "a.yaml", {"model": {"d_model": 64, "n_heads": 4}})
    write(tmp_path, "b.yaml", {"extends": "a.yaml", "model": {"n_layers": 3}})
    c = write(tmp_path, "c.yaml", {"extends": "b.yaml", "name": "c"})

    cfg = load_config(c)
    assert (cfg.model.d_model, cfg.model.n_layers) == (64, 3)


def test_circular_extends_is_detected(tmp_path):
    write(tmp_path, "a.yaml", {"extends": "b.yaml"})
    write(tmp_path, "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# -------------------------------------------------------------- strictness


def test_unknown_key_is_rejected(tmp_path):
    path = write(tmp_path, "typo.yaml", {"model": {"n_layer": 6}})   # missing 's'
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = write(tmp_path, "typo.yaml", {"trainng": {"max_steps": 1}})
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_config(path)


# -------------------------------------------------------------- validation


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(d_model=100, n_heads=7)


def test_rope_requires_even_head_dimension():
    with pytest.raises(ValueError, match="even head dimension"):
        ModelConfig(d_model=30, n_heads=10, pos="rope")   # head_dim 3


def test_vocab_must_fit_uint16():
    with pytest.raises(ValueError, match="uint16"):
        ModelConfig(vocab_size=70000)


@pytest.mark.parametrize("field,value", [("norm", "batchnorm"), ("pos", "sinusoidal"), ("ffn", "relu")])
def test_unknown_architecture_switch_is_rejected(field, value):
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig(**{field: value})


def test_warmup_cannot_exceed_max_steps():
    with pytest.raises(ValueError, match="exceeds max_steps"):
        TrainingConfig(warmup_steps=500, max_steps=100)


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        TrainingConfig(optimizer="sgd")


def test_betas_from_yaml_list_become_a_tuple():
    assert TrainingConfig(betas=[0.9, 0.99]).betas == (0.9, 0.99)


# ----------------------------------------------------------------- derived


def test_ffn_hidden_keeps_swiglu_and_gelu_comparable():
    """SwiGLU's third matrix must not make it a bigger model."""
    from minime.model import build_model

    common = dict(vocab_size=256, d_model=384, n_heads=6, n_layers=4, context_length=64)
    gelu = build_model(ModelConfig(**common, ffn="gelu")).num_parameters(non_embedding=True)
    swiglu = build_model(ModelConfig(**common, ffn="swiglu")).num_parameters(non_embedding=True)
    assert abs(swiglu - gelu) / gelu < 0.05


def test_tokens_per_step_and_budget():
    cfg = Config(
        model=ModelConfig(context_length=512),
        training=TrainingConfig(batch_size=32, gradient_accumulation=8, max_steps=100),
    )
    assert cfg.tokens_per_step == 32 * 8 * 512
    assert cfg.token_budget == cfg.tokens_per_step * 100


def test_normalised_mixture_is_scale_invariant():
    a = DataConfig(mixture={"prose": 7, "code": 3}).normalised_mixture()
    b = DataConfig(mixture={"prose": 70, "code": 30}).normalised_mixture()
    assert a == b == pytest.approx({"prose": 0.7, "code": 0.3})


def test_empty_mixture_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        DataConfig(mixture={"prose": 0}).normalised_mixture()


def test_fingerprint_changes_only_with_the_config():
    base = Config(model=ModelConfig(n_layers=6))
    same = Config(model=ModelConfig(n_layers=6))
    other = Config(model=ModelConfig(n_layers=7))
    assert base.fingerprint() == same.fingerprint()
    assert base.fingerprint() != other.fingerprint()


def test_config_saves_and_reloads(tmp_path):
    cfg = Config(
        name="roundtrip",
        model=ModelConfig(n_layers=3),
        training=TrainingConfig(max_steps=7, warmup_steps=2),
    )
    path = tmp_path / "saved.yaml"
    cfg.save(path)
    assert load_config(path).fingerprint() == cfg.fingerprint()


# ------------------------------------------------------------- cli parsing


def test_cli_overrides_build_nested_dicts():
    parsed = parse_cli_overrides(["training.max_steps=50", "model.n_layers=2", "name=smoke"])
    assert parsed == {"training": {"max_steps": 50}, "model": {"n_layers": 2}, "name": "smoke"}


def test_cli_override_parses_yaml_scalars():
    parsed = parse_cli_overrides([
        "training.learning_rate=1e-4",     # YAML 1.1 calls this a string
        "training.max_steps=50",
        "model.tie_embeddings=false",
        "name=smoke",
    ])
    assert parsed["training"]["learning_rate"] == 1e-4
    assert isinstance(parsed["training"]["learning_rate"], float)
    assert parsed["training"]["max_steps"] == 50
    assert parsed["model"]["tie_embeddings"] is False
    assert parsed["name"] == "smoke"


@pytest.mark.parametrize("text,expected", [
    ("1e-4", 1e-4), ("3E-5", 3e-5), ("1.0e-3", 1e-3), ("42", 42),
    ("0.5", 0.5), ("true", True), ("rmsnorm", "rmsnorm"),
])
def test_cli_scalar_coercion(text, expected):
    parsed = parse_cli_overrides([f"k={text}"])["k"]
    assert parsed == expected
    assert type(parsed) is type(expected)


def test_cli_override_without_equals_is_rejected():
    with pytest.raises(ValueError, match="key=value"):
        parse_cli_overrides(["max_steps"])


def test_cli_overrides_apply_on_top_of_a_file(tmp_path):
    path = write(tmp_path, "c.yaml", {"training": {"max_steps": 1000, "warmup_steps": 10}})
    cfg = load_config(path, overrides=parse_cli_overrides(["training.max_steps=50"]))
    assert cfg.training.max_steps == 50
    assert cfg.training.warmup_steps == 10


def test_shortening_a_run_below_its_warmup_is_rejected(tmp_path):
    """A schedule whose warmup never completes is a broken experiment, not a
    short one -- so the error names both numbers."""
    path = write(tmp_path, "c.yaml", {"training": {"max_steps": 1000, "warmup_steps": 200}})
    with pytest.raises(ValueError, match=r"warmup_steps \(200\) exceeds max_steps \(5\)"):
        load_config(path, overrides=parse_cli_overrides(["training.max_steps=5"]))


# -------------------------------------------------------- shipped configs


def _shipped(repo_root, name: str):
    # The flagship's LR is null until the M10 sweep picks it (tests/test_m10.py).
    overrides = parse_cli_overrides(["training.learning_rate=1.6e-3"]) if name == "flagship" else None
    return load_config(repo_root / "configs" / f"{name}.yaml", overrides=overrides)


@pytest.mark.parametrize("name", ["tiny", "probe", "flagship"])
def test_shipped_configs_load_and_build(name, repo_root):
    from minime.model import build_model

    cfg = _shipped(repo_root, name)
    model = build_model(cfg.model)
    assert model.num_parameters() > 0
    assert cfg.training.max_steps > 0
    assert cfg.tokens_per_step > 0


def test_shipped_config_sizes_match_the_documented_plan(repo_root):
    """Guards the parameter counts quoted in the README and the cost table."""
    sizes = {name: _shipped(repo_root, name) for name in ("tiny", "probe", "flagship")}
    from minime.model import build_model

    millions = {k: build_model(v.model).num_parameters() / 1e6 for k, v in sizes.items()}
    assert millions["tiny"] == pytest.approx(1.38, abs=0.05)
    assert millions["probe"] == pytest.approx(16.9, abs=0.2)
    assert millions["flagship"] == pytest.approx(49.3, abs=0.5)
    assert sizes["probe"].token_budget == pytest.approx(301e6, rel=0.02)
    assert sizes["flagship"].token_budget == pytest.approx(904e6, rel=0.02)


def test_the_default_mixture_is_the_general_corpus_in_its_own_proportions(repo_root):
    """Every probe run (and the E4 sweep) samples the shards the corpus build writes."""
    from minime.data.sources import GENERAL_SLICES

    mixture = load_config(repo_root / "configs" / "probe.yaml").data.normalised_mixture()
    assert set(mixture) == set(GENERAL_SLICES)
    total = sum(s.target_tokens for s in GENERAL_SLICES.values())
    for name, weight in mixture.items():
        assert weight == pytest.approx(GENERAL_SLICES[name].target_tokens / total)


# ------------------------------------------------- the LR and the role


def test_a_null_learning_rate_is_refused_when_the_config_loads(tmp_path):
    """E1's main configs leave the LR null until the pilot picks it."""
    path = tmp_path / "waits.yaml"
    path.write_text("training:\n  learning_rate: null\n")
    with pytest.raises(ValueError, match="learning_rate is null"):
        load_config(path)
    assert load_config(path, overrides={"training": {"learning_rate": 3e-4}}).training.learning_rate == 3e-4


def test_a_string_learning_rate_is_refused(tmp_path):
    path = tmp_path / "yaml11.yaml"
    path.write_text("training:\n  learning_rate: 1e-4\n")   # YAML 1.1: a string
    with pytest.raises(ValueError, match="must be a number"):
        load_config(path)


def test_the_role_is_a_label_not_a_variable(tmp_path):
    from minime.training.lr_sweep import _UNCONTROLLED
    assert Config().role == ""
    assert "role" in _UNCONTROLLED
