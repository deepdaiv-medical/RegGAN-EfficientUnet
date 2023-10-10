# RegGAN-EfficientUNet
This project is based on RegGAN [[code](https://github.com/Kid-Liet/Reg-GAN)] [[paper](https://arxiv.org/pdf/2110.06465.pdf)]

# Overview
RegGAN stands as a groundbreaking model crafted specifically to tackle the constraints that have hampered the efficacy of prevailing image-to-image translation models such as Pix2Pix and CycleGAN. The distinguishing feature of RegGAN lies in its innovative treatment of unaligned images within the training data. It perceives these unaligned images as noisy labels and artfully employs a registration network to harmonize them with the disordered noise distribution.   

This novel approach propels RegGAN to outperform Pix2Pix and Cycle-GAN without necessitating the inclusion of "well-aligned image pairs" in the training dataset. What sets RegGAN apart is its remarkable adaptability, seamlessly integrating into pre-existing models without necessitating any modifications to their original architectural design. Furthermore, RegGAN achieves these commendable results while demanding a reduced parameter count, thus enhancing overall performance efficiency.   

Within the context of this research paper, we opted for the utilization of the U-Net as our chosen registration network. While the paper presents this choice without extensive elaboration, it prompts intriguing inquiries. Specifically, we contemplate whether alterations in the depth of the U-Net or the introduction of an Efficient-Unet can yield both a more streamlined model and performance enhancements worth exploring in the future.   

# Training

# Result

## Citation

```
@inproceedings{
kong2021breaking,
title={Breaking the Dilemma of Medical Image-to-image Translation},
author={Lingke Kong and Chenyu Lian and Detian Huang and ZhenJiang Li and Yanle Hu and Qichao Zhou},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=C0GmZH2RnVR}
}
```
