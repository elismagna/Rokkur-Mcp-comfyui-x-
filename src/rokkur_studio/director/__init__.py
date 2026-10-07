"""The AI director: a cinematography pass over the creative brief and a deterministic prompt
compiler that turns each shot into a diffusion prompt (see docs/director.md).

* ``vocabulary`` – the allowed cinematography terms (the only values the DP pass may pick).
* ``rules`` – the diffusion parsing rules: clean phrases, the action strip-out, token weights.
* ``assets`` – the asset tracker: global look, global negative prompt, named characters.
* ``prompts`` – per-shot prompt assembly and the Batch Prompt Schedule export.
"""
