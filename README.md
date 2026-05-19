# CANet
Context-aware multi-view stereo network for efficient edge-preserving depth estimation, IJCV2025

### Scripts
#### 1. train on DTU
```bash
bash scripts/train_dtu.sh
```
#### 2. test on DTU
```bash
bash scripts/test_dtu.sh
```
#### 3. finetune on BlendedMVS
```bash
bash scripts/train_blended.sh
```

#### 4. test on Tanks and Temple
```bash
bash scripts/test_tt.sh
```
#### 5. test on ETH3D
```bash
bash scripts/test_eth.sh
```

### Citation
```bibtex
@article{su2025context,
  title={Context-Aware Multi-view Stereo Network for Efficient Edge-Preserving Depth Estimation},
  author={Su, Wanjuan and Tao, Wenbing},
  journal={International Journal of Computer Vision},
  volume={133},
  pages={3367--3391},
  year={2025},
  publisher={Springer}
}
```

### Acknowledge
Our work is partially based on these opening source work: [CasMvsnet](https://github.com/alibaba/cascade-stereo), [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet), [GeoMVSNet](https://github.com/doubleZ0108/GeoMVSNet), and [XCiT](https://github.com/facebookresearch/xcit). Thanks for their contributions to the community.
