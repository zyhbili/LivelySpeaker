Here we illustrate the steps to process the BEAT. You should do the processing under the env of BEAT.

We only use the sub dataset of speaker 2,4,6,8 for comparison. 

First, download the data from [BEAT](https://pantomatrix.github.io/BEAT/), and run ``preprocess_0.py`` to downsample the raw data into 15fps.

```
Replace your/path/to in line 174~178
python preprocess_0.py
```

Then we use the split rule in [BEAT](https://pantomatrix.github.io/BEAT/) to generate the train, val, test split ``bvh_rot_2_4_6_8_cache``.

```
Replace your/path/to in line 293
python preprocess_1.py

run the datatloader to split the long seqs into 34 frames like:
python train_RAG.py -c ./configs/beat.yaml --exp beat --epochs 1501

```


If you do not need rot6d and mel spectrogram, ignore the following step and change the path in dataloader to``bvh_rot_2_4_6_8_cache`. 

Finally, we patch the cache ``bvh_rot_2_4_6_8_cache`` and generate ``my6d_bvh_rot_2_4_6_8_cache``. Specifically, we converting axis-angle representataion into rot6d represetation and extract mel spectrogram from audio input. 

```
Replace your/path/to in line 13~24
python preprocess_cache.py
```

Notably, we also generate 34-frame clips dubbed ``finaltest`` for testing following TED.