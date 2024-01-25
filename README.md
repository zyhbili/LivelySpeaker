# LivelySpeaker
[ICCV-2023] The official repo for the paper "LivelySpeaker: Towards Semantic-aware Co-Speech Gesture Generation"

[[paper](https://arxiv.org/abs/2309.09294) / [video](https://www.youtube.com/watch?v=arYqydsXM2I)]

## Install the dependencies
```
conda create -n livelyspeaker python=3.7 
conda activate livelyspeaker
pip install -r requirements.txt
```

## Prepare Data
### TED Dataset
Prepare TED following [TriModel
](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context)  and link it to `./datasets/ted_dataset`. 

```
ln -s path_to_ted ./datasets/ted_dataset
```


### BEAT Dataset
Prepare BEAT following [BEAT](https://pantomatrix.github.io/BEAT/) and modify the data path in `./scripts_beat.yaml/configs/beat.yaml`

```
ln -s path_to_ted ./datasets/ted_dataset
```

## Run the code on TED
### Command Lines 
   Train RAG

```
python scripts/train_RAG.py --exp RAG -b 512
```

Test RAG
```
python scripts/test_RAG_ted.py --model_path ckpts/TED/RAG.pt 
```

Test LivelySpeaker
```
python scripts/test_LivelySpeaker_ted.py.py --model_path ckpts/TED/RAG.pt
```

## Run the code on BEAT
### Command Lines 
   Train RAG

```
python train_RAG.py -c ./configs/beat.yaml --exp beat --epochs 1501
```

Test RAG
```
python test_LivelySpeaker_beat.py --model_path ckpts/BEAT/RAG.pt -c configs/beat.yaml
```

Test LivelySpeaker
```
python test_RAG_beat.py --model_path ckpts/BEAT/RAG.pt -c configs/beat.yaml
```



## Model

We provide all checkpoints at [here](https://cuhko365-my.sharepoint.com/:f:/g/personal/223010099_link_cuhk_edu_cn/EsK6Bc_a3A1FhgxvBUjlYN4BsLt1Ur6U4LF5mqo3BQZLwQ?e=bh5UzG). Download and link it to `./ckpts`.

To run the FGD evaluation of TED. You should first download of Encoder weights from [TriModal](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context)





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
