# PANNs audio encoder attribution

The frozen Cnn6 architecture in temporal_possession/audio.py is adapted from
Qiuqiang Kong's PANNs source, pinned at commit
d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4:
https://github.com/qiuqiangkong/audioset_tagging_cnn

Copyright (c) 2018–2020 Qiuqiang Kong. Source license: MIT, reproduced in
PANNS_LICENSE.MIT. The adaptation removes training-only augmentation/dropout,
uses the original checkpoint's frozen STFT filters and mel matrix, and returns
embeddings and generic AudioSet Whistle/Whistling scores. It does not identify
the referee or certify that play stopped.

The unchanged Cnn6_mAP=0.343.pth checkpoint is distributed separately under
CC BY 4.0 by the authors: https://zenodo.org/records/3987831
MD5: e25e26b84585b14c7754c91e48efc9be
SHA-256: 8a17541944f2623a2a0164e072ac2f915dcedca99968083cc44bfbc7eb682275
The weights are not included in this public source tree.

Reference: Qiuqiang Kong et al., PANNs: Large-Scale Pretrained Audio Neural
Networks for Audio Pattern Recognition, IEEE/ACM TASLP, 2020.
https://arxiv.org/abs/1912.10211
