# JCMIL: P3 + Joint on Colab

Vietnamese research runner for the revised common-label CrisisMMD benchmark.
This is a candidate implementation, not a completed GPU reproduction or the original MagMax/TSV-M benchmark.

Open `JCMIL_GitHub.ipynb` in Colab and select a GPU runtime. Run the short cells in order.
The notebook clones a pinned commit. Data, models and outputs remain in `/content`; Google Drive is not mounted.
The code downloads the official annotations and images, verifies the annotation hash, reconstructs the fixed manifest, and checks all image rows before training.

## Recipe

- P3 four sequential stages plus Joint; seed 0; full training split of 13,608 rows.
- Ten epochs per job; batch size 32; FP32; CLIP ViT-B/16 pinned pretrained revision.
- Fixed common-label target, eight known class names, seen-class logits during training.
- Official dev/test each contain 2,237 rows. Exact image overlap exists across splits (82 groups); it is acknowledged explicitly, not removed silently.
- Dev evaluation enabled; test requires an explicit decision after recipe freeze.
- Latest completed checkpoint per job retained; previous tensor payload removed only after validating its successor. All epoch statistics/metadata remain.

## RAM and speed changes

- Hash checkpoint files in 4 MiB blocks rather than loading the whole file into RAM.
- `torch.load(mmap=True, weights_only=True)` avoids eagerly reading unused optimizer tensors during inference/parent loading. Release parent state after copying the model.
- Cache deterministic token IDs/masks once per dataset instance, in bounded batches; keep image decoding on demand.
- Pinned host batches and nonblocking CUDA copies. No multiworker ordering change.
- Log epoch training seconds and peak allocated VRAM; download through resumable aria2 with eight connections.
- FP32 and batch32 remain the reference. AMP, accumulation, gradient checkpointing and worker parallelism are not enabled silently. T4 fit/speed and end-to-end GPU equivalence still require measurement.

These changes target memory and repeated CPU work, not a promised speedup or identical cross-hardware results.
See [PyTorch checkpoint loading](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html),
[performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), and
[aria2 manual](https://aria2.github.io/manual/en/html/aria2c.html).

## Outputs and interruption

`colab.export_evidence()` packages logs, logits, metrics, source revision and metadata without weights.
`colab.export_checkpoint('s4')` packages one final checkpoint; use `s1`, `s2`, `s3`, `s4`, or `joint`.
On interruption, rerun that job to resume from its latest completed epoch in the same environment.
Do not delete the runtime while downloads/checkpoints need preserving. The image directory is shared with earlier notebook versions; unfinished aria2 transfers keep their control file.

## Evidence limits

Software tests and CPU tokenization comparisons are not full training results. GPU training, resume equivalence on T4, and final seed-level comparisons are unverified.
This repository contains source code, documentation, and a reviewed overlap inventory of hashes/IDs. It contains no tweet text, images, pretrained weights, credentials or local research reports.

Dataset sources: [CrisisMMD authors](https://github.com/firojalam/multimodal_social_media), [official dataset page](https://crisisnlp.qcri.org/crisismmd.html).
