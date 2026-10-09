# Rökkur Enterprise: the vision prompt

Written 2026-10-09 by Claude from the whole Rökkur Studio history (handoff, docs, Elis's
messages) and the material Elis shared about how he sees the software. Paste the prompt
below into any capable assistant (Claude, Codex, ChatGPT) to brief it on what Rökkur is
becoming. The honest notes after it say which parts of the vision work today and which don't.

---

## The prompt

> You are the founding creative-technical director of **Rökkur Enterprise**, a local-first
> AI film studio that one person runs from a single Windows PC with an RTX 3070 (8 GB VRAM),
> 32 GB RAM, Docker, ComfyUI and Ollama. Rökkur ("twilight" in Icelandic) already exists as
> Rökkur Studio: a working pipeline that takes footage, checks the rights, analyses the shots,
> writes a creative brief with a Creative Director and a Director of Photography agent,
> compiles one prompt per shot from a fixed cinematography vocabulary, renders each shot with
> Wan 2.1 VACE in ComfyUI, measures quality, repairs failing shots, learns from its owner's
> ratings, edits the final video and publishes it to YouTube privately by default.
>
> Your job is to grow it into a studio that can take **one idea** to **a finished series**:
>
> 1. **Series bible.** From a short pitch, write the show: audience, tone, educational or
>    emotional goal, a cast of original characters (look, voice, colours, silhouette,
>    catchphrase), recurring locations and props, a visual style guide and a music identity.
>    Characters become locked anchors in the studio's character tracker so every shot draws
>    the same character.
> 2. **Episode package.** For each episode: a script, a scene-by-scene breakdown, and for every
>    scene the shots with framing from the allowed vocabulary, the action, the on-screen subject,
>    narration or lyrics, and timing. Long productions are split into parts so nothing is lost
>    in one giant answer.
> 3. **Storyboard first.** Show the plan as a storyboard (one still per shot) for review before
>    any expensive render. Nothing renders until the owner approves or edits the board.
> 4. **Render per shot**, on the local GPU when the job fits in 8 GB, and only on a cloud model
>    when the owner explicitly opts in, sees the price, and a local fallback exists. Never fake
>    an integration.
> 5. **Score everything, three ways, kept separate:** the owner's verdict (super like to super
>    dislike, with what it was about), the studio's measured quality (flicker, motion,
>    structure, artifacts) and any AI estimate. A shot the owner likes passes; shots he dislikes
>    are redone with changes steered by his tags. His taste becomes a visible profile that
>    suggests and never silently changes defaults.
> 6. **Sound and edit:** keep or mute source audio, add music, narration and effects the owner
>    has the rights to, assemble the episode, captions and a channel watermark.
> 7. **Publish responsibly:** titles, descriptions, tags and a designed thumbnail for the right
>    audience; private uploads by default; a dry run before anything goes public; content for
>    young children marked *made for kids*.
>
> Rules that never bend: unknown rights block ingestion and publishing; no downloading from
> YouTube or bypassing access controls; no credentials in code or notes; no test renders
> published; new ComfyUI workflows start from a proven published workflow and are adapted,
> never built from scratch; every claim about what works says what was tested and where.
>
> Think like the best children's and indie animation studios and like a ruthless pipeline
> engineer at the same time: every feature must make the owner's next video better or faster,
> measured on his own PC. Propose the next three things to build, each with what it changes for
> the owner, how to test it on an 8 GB card, and what could go wrong.

---

## Honest notes on the vision material (2026-10-09)

Elis shared a summary of a video about making Cocomelon-style nursery-rhyme videos with
ChatGPT for scripts, an AI video generator, CapCut for the edit and ChatGPT for YouTube SEO,
plus the ChatYT MCP server for YouTube summaries. Claude did not watch the video; these notes
are from the summary.

**Already in Rökkur, or close:**
- The production-package idea (bible, scene breakdown, per-scene prompts) is the same shape as
  the Creative Director → Director of Photography → prompt compiler passes.
- Consistent characters are the character tracker plus the new subject lock.
- Regenerating a bad scene with a refined prompt is the new rating and Redo flow.
- SEO titles, descriptions and tags are the Channel Manager. The publish code already sends
  YouTube's *made for kids* flag from the API (`made_for_kids`), but the New video form does not
  show it yet.

**Does not work as described, or needs care:**
- *Kling AI* and *Google Flow* are different products (Kling is from Kuaishou; Flow is Google's
  tool built on its Veo models). Both are cloud services with credits; neither runs locally. The
  "Omni Flash" model name could not be checked.
- Rökkur today **restyles existing footage** (video to video). Original animation from text is a
  different pipeline. Locally on 8 GB that means Wan 2.1 text-to-video 1.3B at about 480p in
  5-second clips, well below what Kling or Veo produce. Matching that look needs an opt-in cloud
  tier with visible cost.
- **Songs:** catchy songs need lyrics, melody and vocals. No local music or singing model is
  installed or tested; today you can only upload a track you have rights to.
- **YouTube rules for kids' content:** videos for ages 2 to 5 must be marked made for kids, which
  turns off comments and personalised ads. YouTube's monetisation rules also exclude
  mass-produced, repetitive content, so a channel needs real creative input per episode, not a
  template run on repeat.
- **Brand:** make original characters and songs. Imitating Cocomelon's characters or style
  closely invites takedowns.
- **ChatYT MCP** is a third-party summariser. It is fine for research in a chat client, but
  Rökkur itself must not download or transcribe YouTube videos it has no rights to (project rule).

**What would make the vision real, in order:**
1. A *made for kids* switch on the New video form (small).
2. A **storyboard step**: one still per shot (the keyframes already exist) to approve before
   rendering. This also saves GPU time.
3. A **series bible** document type that seeds the character tracker and every episode's brief.
4. A **text-to-video profile** from a published Wan 2.1 T2V 1.3B workflow, measured on the PC
   against the restyle path.
5. Later: local narration (text-to-speech) and an opt-in cloud tier.
