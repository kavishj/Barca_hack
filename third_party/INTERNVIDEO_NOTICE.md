# InternVideo2 attribution

The unchanged `internvideo2_source.py` is from OpenGVLab/InternVideo2-Stage2_6B,
Hugging Face revision `fd8ef758daf132d4738325d39546dd76ce26911d`.
SHA256: `9c1ddd97062e971ebd35e1645db3e06e0cdbb66cc1590777d258c67e4be9344f`.

Authors: the InternVideo2/OpenGVLab team. Paper: https://arxiv.org/abs/2403.15377.
Code: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.
Model: https://huggingface.co/OpenGVLab/InternVideo2-Stage2_6B.

The model card declares MIT for the checkpoint. The upstream code repository's
root license is Apache-2.0, reproduced in `INTERNVIDEO_LICENSE`. Preserve both
the source attribution and these license records when distributing this adapter.

Our adapter compiles only reviewed vision definitions, disables optional fused
implementations, uses PyTorch SDPA and computes only projected pooled video
features. Text and distillation-only parameters are omitted; pretrained tensor
values are unchanged apart from the documented FP32-to-FP16 inference conversion.
No downloaded model weights are stored in this repository.
