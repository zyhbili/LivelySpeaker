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
Prepare TED following [TriModel
](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context)  and link it to ./datasets/ted_dataset. 

```
ln -s path_to_ted ./datasets/ted_dataset
```


### BEAT Dataset
Prepare BEAT following [BEAT](https://pantomatrix.github.io/BEAT/) and link it to ./datasets/BEAT. 

```
ln -s path_to_ted ./datasets/ted_dataset
```

## Run the code
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

## TODO
1.supp video upload. 

2.checkpoints.

3.code on BEAT

...


## Acknowledge
We build our code base from: [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [MDM](https://github.com/GuyTevet/motion-diffusion-model),
[TriModal
](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context), [BEAT](https://pantomatrix.github.io/BEAT/). 





## Citation
```
@InProceedings{Zhi_2023_ICCV,
    author    = {Zhi, Yihao and Cun, Xiaodong and Chen, Xuelin and Shen, Xi and Guo, Wen and Huang, Shaoli and Gao, Shenghua},
    title     = {LivelySpeaker: Towards Semantic-Aware Co-Speech Gesture Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20807-20817}
}
```
