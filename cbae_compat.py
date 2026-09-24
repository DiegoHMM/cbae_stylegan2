"""Compatibilidade das ops customizadas do StyleGAN neste ambiente Windows.

Esta copia do dnnlib/torch_utils (linhagem StyleGAN3) levanta RuntimeError quando
nao encontra MSVC/CUDA Toolkit para compilar bias_act/upfirdn2d/filtered_lrelu, em
vez de cair na implementacao de referencia como o StyleGAN2-ADA fazia.

`use_ref_ops()` restaura esse comportamento: se o plugin nao compilar, as ops usam
a versao em PyTorch puro (mais lenta, mesmos resultados numericos ate tolerancia).

Uso, antes de carregar o pickle do gerador:

    import cbae_compat; cbae_compat.use_ref_ops()

Depois de instalar o VS Build Tools + CUDA Toolkit, basta nao chamar a funcao (ou
chamar use_ref_ops(force=False), que so desativa o plugin se ele realmente falhar).
"""

import warnings

_PLUGIN_MODULES = ("bias_act", "upfirdn2d", "filtered_lrelu")


def use_ref_ops(force=True):
    """Faz as ops customizadas usarem a implementacao de referencia.

    force=True  -> nem tenta compilar o plugin.
    force=False -> tenta compilar; so desativa o plugin se a compilacao falhar.
    """
    import importlib

    disabled = []
    for name in _PLUGIN_MODULES:
        mod = importlib.import_module(f"torch_utils.ops.{name}")
        original_init = mod._init

        if force:
            mod._init = lambda: False
            disabled.append(name)
            continue

        def patched_init(_mod=mod, _original=original_init):
            try:
                return _original()
            except Exception as exc:  # compilador ou CUDA Toolkit ausente
                warnings.warn(
                    f"plugin CUDA de {_mod.__name__} indisponivel ({exc}); "
                    "usando implementacao de referencia (mais lenta).",
                    RuntimeWarning,
                )
                _mod._init = lambda: False
                return False

        mod._init = patched_init

    if disabled:
        warnings.warn(
            "ops customizadas do StyleGAN desativadas (" + ", ".join(disabled) + "); "
            "usando implementacao de referencia (mais lenta).",
            RuntimeWarning,
        )
    return True


def force_fp32_synthesis(G):
    """Desliga os blocos fp16 da synthesis do StyleGAN2.

    Necessario junto com use_ref_ops(): sem o plugin CUDA compilado, os blocos
    fp16 (32x32 em diante, num_fp16_res=4) estouram e a imagem sai toda NaN.
    Com o plugin compilado (maquina de treino), NAO chame isto — o fp16 funciona
    e e mais rapido.

    Retorna a lista de resolucoes que foram convertidas para fp32.
    """
    changed = []
    for res in getattr(G.synthesis, "block_resolutions", []):
        block = getattr(G.synthesis, f"b{res}", None)
        if block is not None and getattr(block, "use_fp16", False):
            block.use_fp16 = False
            changed.append(res)
    if changed:
        warnings.warn(
            "synthesis forcada para fp32 nas resolucoes "
            + ", ".join(str(r) for r in changed)
            + " (ops em modo referencia estouram em fp16).",
            RuntimeWarning,
        )
    return changed
