# The leading AI image and video tools, their features, and what Rökkur does with them

Researched 2026-10-10 by Claude for Elis's request: "top 5 AI image and video tools and all
their features explored and implemented". Sources are listed at the end. Rankings come from
2026 roundups, several written by companies that sell their own platforms, so they are a
shortlist, not a verdict; features were checked against the vendors' own pages where they
could be reached. Versions and prices change monthly: check a vendor's page before relying on
a detail here.

**How "implemented" is meant.** The leading tools are closed cloud services (OpenAI, Google,
Midjourney, Runway, Kuaishou). Rökkur cannot run their models. What it can do is offer the
same *feature* with open models that run in its ComfyUI (on the RTX 3070 or the cloud
server), starting from a published workflow (the project rule). Each feature below is marked:

- **Built**: in the studio now (tested with fakes here; first real-GPU run pending on the PC).
- **Next**: an open model and a published ComfyUI workflow exist and fit the hardware.
- **Partial**: part of it exists; what is missing is said.
- **Cloud only**: only a closed service does it today; not integrated (it would need that
  service's paid API and Elis's opt-in).

**REA.** REA inspects local software: binaries, Electron and JavaScript apps, plugins. The
leading tools are web services, so their features come from their documentation. REA's
browser-capture tools could record their web apps, but analysing a proprietary service's
pages is not something the studio should do without the owner's permission, so it was not
done. REA is in the studio for local tools (ComfyUI, node packs, codecs); see `docs/rea.md`.

## Image tools

| Tool | Known for (sources agree) |
|---|---|
| GPT Image 2 (OpenAI) | general use and instruction editing; up to about 4K |
| Nano Banana 2 / Pro (Google) | editing, character consistency across pictures, layouts and text (Pro) |
| Midjourney (V7/V8) | look and taste from short prompts; Omni Reference, Style Reference, Editor with Vary Region, Pan, Zoom Out; no API |
| FLUX.2 (Black Forest Labs) | open weights; single- and multi-reference editing (8 references in [pro] per BFL's help centre); klein 4B is Apache-2.0 |
| Ideogram (3.0/4.0) | readable text in images: posters, logos |

Also named: Adobe Firefly (commercial rights), Recraft (vectors, product images).

| Feature | Who has it | Rökkur |
|---|---|---|
| Text to image | all | **Built**: FLUX.2 klein 4B, Z-Image-Turbo (`docs/images.md`) |
| Change a picture by instruction | GPT Image, Nano Banana, FLUX.2 | **Built**: klein edit |
| Repaint a painted area (Vary Region) | Midjourney, GPT Image, Ideogram | **Built**: brush or subject-model mask |
| Extend the edges (Zoom Out, Pan) | Midjourney, Ideogram | **Built**: outpaint any side |
| Variations | Midjourney, all | **Built** |
| Upscale | Midjourney, most | **Built**: RealESRGAN 2x/4x |
| Same character across pictures (Omni Reference, Nano Banana consistency) | Midjourney, Nano Banana, FLUX.2 | **Partial**: one reference picture per edit; characters cut out of your clip become references. Multi-reference needs klein's chained ReferenceLatent: **Next**, once a Comfy-Org multi-reference template for klein is confirmed |
| Style of a reference picture (Style Reference, moodboards) | Midjourney | **Partial**: an instruction edit with the style picture as reference; no style-only strength control |
| Readable text in the picture | Ideogram, Nano Banana Pro, GPT Image | **Partial**: klein and Z-Image draw short text; not checked against Ideogram |
| Remove the background, keep the subject | most editors | **Partial**: the subject model selects it for repainting and cuts characters out of video; a "cut out to transparent PNG" button is a small **Next** |
| Vector (SVG) output | Recraft | **Cloud only** |
| Picture to 3D model | (adjacent tools) | **Built**: Hunyuan3D 2.0 in the 3D studio (`docs/three.md`) |

## Video tools

| Tool | Known for (sources agree) |
|---|---|
| Veo 3.1 (Google, in Flow) | realism and native audio; Ingredients to Video (up to three references), first and last frame, Extend to a minute or more, Insert objects (Remove announced) |
| Kling 3.0 (Kuaishou) | longest single clips and price; native audio and per-character lip sync; start and end frames; multi-shot sequences; element references |
| Runway (Gen-4.5, Aleph 2.0, Act-Two) | production tooling: in-context editing of existing footage (Aleph), performance transfer (Act-Two), references |
| Seedance 2.0 (ByteDance) | on most 2026 shortlists |
| Luma Ray3.x | prompt adherence; editing existing footage |

Sora 2 is being shut down (app closed in April 2026, API on 2026-09-24 per two roundups), so
it is left out.

| Feature | Who has it | Rökkur |
|---|---|---|
| Restyle existing footage, keep its motion | Runway Aleph, Luma | **Built**: the core pipeline (Wan 2.1 VACE with the clip's edges), with the real subject kept when the place changes |
| Keep the same character across shots | Veo Ingredients, Kling elements, Runway references | **Built/partial**: one appearance reference per video, Stable mode (one seed, the same cutout every shot), characters found in the clip before prompting; several references at once are not supported by VACE 1.3B |
| Extend a clip | Veo Extend | **Built**: before and after the render (`docs/video-tools.md`) |
| First and last frame | Veo, Kling | **Next**: Comfy-Org's Wan VACE first-last-frame template (already adapted for Extend) |
| Text to video | all | **Next**: Wan 2.1 T2V 1.3B from Comfy-Org's template fits 8 GB; Rökkur is video-to-video today |
| Image to video | all | **Next**: the same VACE graph with one start frame |
| Insert or remove an object in footage | Veo (Insert; Remove announced), Runway Aleph | **Partial**: the mask pipeline exists (keep-subject); object removal is VACE inpainting with a painted mask over time: **Next** |
| Performance transfer (a person's movement onto a character) | Runway Act-Two, Kling motion control | **Built in spirit**: the source clip's motion drives the render and a picture sets the character; no face or lip transfer |
| Native audio, sound effects | Veo, Kling | **Partial**: music and effects are generated separately (ACE-Step, Stable Audio Open) and mixed in at any stage (`docs/audio.md`); not synchronised to on-screen events |
| Dialogue and lip sync | Kling, Veo | **Cloud only** for now: open lip-sync models need custom node packs the PC does not have |
| Multi-shot sequences | Kling | **Built**: the director's shot plan, one render per shot, storyboard stills first |
| Camera moves on request | Kling, Runway | **Partial**: framing words in the prompt (the DP vocabulary); Wan 1.3B has no camera control |
| 4K, 60 fps | Veo, others | **Next with a cost**: per-frame upscaling and RIFE interpolation need a node pack and time |
| Edit while it renders, decide with the evidence | (none of the five shows this) | **Built**: adjust the remaining shots, Stable mode, the live view with decisions and evidence (`docs/live.md`) |

## What to build next, in order

1. **First and last frame / image to video**: same graph family as Extend, one published
   template, fits 8 GB. Opens "animate this picture" from Pictures.
2. **Text to video (Wan 2.1 T2V 1.3B)**: Comfy-Org template; makes a video without footage,
   which Elis's Rökkur Enterprise direction needs.
3. **Object removal in footage**: VACE inpainting with a mask track from the subject model or
   a brush on the first frame.
4. **Cut out to transparent PNG** in Pictures: the subject model already exists.
5. **Multi-reference pictures** with klein, once the template is confirmed.

Lip sync and synchronised dialogue stay cloud-only until an open model with a core-node
workflow exists; a paid cloud API (Kling, Veo) would need Elis's explicit opt-in, privacy and
cost choices, and must not become a hidden dependency.

## Sources

Roundups (rankings; several vendor-written): [Pixverse](https://pixverse.ai/en/blog/best-ai-video-generators),
[mstudio](https://mstudio.ai/insights/best-ai-video-generator-2026),
[Tech Insider](https://tech-insider.org/best-ai-video-generator-2026/),
[Masonry](https://masonry.so/blog/best-ai-video-generator-2025-comparison),
[AIViewer](https://aiviewer.ai/guides/best-ai-video-generator-2026-sora-runway-veo-pika-and-kling-compared/),
[DIYAI](https://diyai.io/ai-tools/video-generation/best-ai-video-tools/),
[SurePrompts video](https://sureprompts.com/blog/best-ai-video-generators-2026),
[GetAIPerks](https://www.getaiperks.com/en/blogs/44-best-ai-video-generators-2026),
[Turing Post](https://www.turingpost.com/p/11-options-for-image-generation),
[3D AI Studio](https://www.3daistudio.com/blog/best-ai-image-generators-2026),
[48hourslogo](https://www.48hourslogo.com/blog/best-ai-image-models-2026-chatgpt-nano-banana-grok),
[FrankX](https://www.frankx.ai/blog/best-ai-image-generators-2026),
[SurePrompts images](https://sureprompts.com/blog/best-ai-image-generators-2026),
[The Insight](https://www.theinsight.tech/articles/best-ai-image-generators-in-2026-i-tested-nano-banana-pro-vs-midjourney-vs-flux-heres-what-actually-wins).

Vendors: [Google: Veo 3.1 and Flow](https://blog.google/technology/ai/veo-updates-flow/),
[Flow features](https://support.google.com/flow/answer/16352836),
[Runway: Kling 3.0 and Aleph 2.0](https://runway.com/product/models/kling-3.0),
[Midjourney: Omni Reference](https://docs.midjourney.com/hc/en-us/articles/36285124473997),
[Midjourney: Editor](https://docs.midjourney.com/hc/en-us/articles/33329329805581),
[BFL: FLUX.2](https://bfl.ai/blog/flux-2), [BFL help centre](https://help.bfl.ai/articles/7424166364-wip-welcome-to-the-forest).
Not verified from Kling's or Runway's own documentation (not reachable from here): Kling's
element references, motion brush and lip-sync details, and Runway's Act-Two specifics.
