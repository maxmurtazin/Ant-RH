import torch
from core.ncg_braid_spectral import BraidNCGConfig, enumerate_braid_words, build_dirac_operator, safe_eigvalsh


def test_operator_is_symmetric():
    cfg = BraidNCGConfig(dim=32, n_strands=3, max_word_len=5)
    words = enumerate_braid_words(cfg.n_strands, cfg.max_word_len, cfg.dim)
    D = build_dirac_operator(words, cfg)
    assert torch.allclose(D, D.T, atol=1e-10)


def test_eigs_finite():
    cfg = BraidNCGConfig(dim=32, n_strands=3, max_word_len=5)
    words = enumerate_braid_words(cfg.n_strands, cfg.max_word_len, cfg.dim)
    D = build_dirac_operator(words, cfg)
    eigs = safe_eigvalsh(D)
    assert torch.isfinite(eigs).all()
