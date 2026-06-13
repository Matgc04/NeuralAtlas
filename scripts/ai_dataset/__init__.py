"""Generate an AI-regenerated dataset paired 1:1 with a source ``public/<dataset>`` tree.

For every source image we (1) caption it with a vision model, anchored on the known
ImageNet class label, then (2) feed the resulting ``regeneration_prompt`` to an image
model and save the result under a parallel ``public/<target>`` tree with the same
class/index layout. The caption prompt optimises for *faithful reconstruction* so the
regenerated image can be compared against its source (paired mode).

Captioning and image generation are separate roles, each with its own ``--*-provider``,
so they can be mixed (e.g. caption with one backend, generate with another). Per image we
caption, persist the caption, then generate — a generation failure never loses the caption.
Gemini ships for captioning and image generation; Cloudflare Workers AI is also
available as an image-generation provider.

Module layout:
- ``core``       data shapes, role ABCs, prompt/schema, shared IO helpers
- ``gemini``     Gemini captioner + image generator
- ``cloudflare`` Cloudflare Workers AI image generator
- ``generator``  the paired-generation orchestration loop
- ``cli``        argument parsing, provider registry, ``main``
"""
