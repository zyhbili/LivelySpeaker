# LivelySpeaker
[ICCV-2023] The official repo for the paper "LivelySpeaker: Towards Semantic-aware Co-Speech Gesture Generation"

[[paper](https://arxiv.org/abs/2309.09294) / video]

## Install the dependencies
```
conda create -n livelyspeaker python=3.7 
conda activate livelyspeaker
pip install -r requirements.txt
```

## Prepare Data
### TED Dataset
Prepare TED following [Gesture-Generation-from-Trimodal-Context
](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context) link and it to ./datasets/ted_dataset. 

```
ln -s path_to_ted ./datasets/ted_dataset
```


### BEAT Dataset
Prepare TED following [BEAT](https://pantomatrix.github.io/BEAT/) link and it to ./datasets/BEAT. 

```
ln -s path_to_ted ./datasets/ted_dataset
```

## Run the code
Take `ZJU-Mocap 313` as an example, other configs files are provided in `configs/{h36m,zju_mocap}`.
### Command Lines 
   Train RAG

```
python scripts/train_RAG.py --exp RAG -b 512
```

Test RAG
```
python scripts/test_RAG_ted.py --model_path ckpts/ted_best.pt 
```

Test LivelySpeaker
```
python scripts/test_LivelySpeaker_ted.py.py --model_path ckpts/ted_best.pt 
```

## Acknowledge
We would like to acknowledge the following third-party repositories we used in this project:
- [[Tinycuda-nn]](https://github.com/NVlabs/tiny-cuda-nn)
- [[Openpose]](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [[ROMP]](https://github.com/Arthur151/ROMP)
- [[Segment-anything]](https://github.com/facebookresearch/segment-anything)

Besides, we used code from:
- [[Anim-NeRf]](https://github.com/JanaldoChen/Anim-NeRF)
- [[SelfRecon]](https://github.com/jby1993/SelfReconCode)
- [[lpips]](https://github.com/richzhang/PerceptualSimilarity)
- [[SMPLX]](https://github.com/vchoutas/smplx)
- [[pytorch3d]](https://github.com/facebookresearch/pytorch3d)





## Citation
```
@article{jiang2022instantavatar,
  author    = {Jiang, Tianjian and Chen, Xu and Song, Jie and Hilliges, Otmar},
  title     = {InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds},
  journal   = {arXiv},
  year      = {2022},
}
```
