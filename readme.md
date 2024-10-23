## mts env fix

TypeError: forward() got an unexpected keyword argument 'is_causal' (torch 2.x)
site-packpage -> torch(>2.) -> TransformerEncoder -> forward -> for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers) ##remove is_casual=is_casual

###Pre-training 
## Video
Extract faces from videos:
```bash
python preprocess/celebvhq_preprocess.py --data_dir
```
Generate masks:
```bash
python preprocess/ytf_preprocess.py --data_dir
```

train_set.csv example for pre-training (from [YoutubeFaces](https://www.cs.tau.ac.il/~wolf/ytfaces/))
```
path,len
AJ_Cook/0,79
AJ_Cook/2,194
Aaron_Sorkin/0,70
Aaron_Sorkin/3,174
Aaron_Tippin/0,119
Aaron_Tippin/1,83
Abdel_Aziz_Al-Hakim/0,103
Abdel_Aziz_Al-Hakim/1,285
Abdel_Aziz_Al-Hakim/4,141
Abdul_Majeed_Shobokshi/1,624
Abdulaziz_Kamilov/4,195
```

Video pre-training:
```bash
python train.py \
    --config config/pretrain/marlin_vit_base.yaml \
    --data_dir /path/to/youtube_faces \
    --n_gpus 4 \
    --num_workers 8 \
    --batch_size 16 \
    --epochs 2000 \
    --official_pretrained /path/to/videomae/checkpoint.pth
```
##Signal
Directory for .csv
```
├── Data
│   ├── id1
│   │   ├── 1.csv
│   │   ├── ...
│   ├── id2
│   ├── ...
```
Signal pre-training
```bash
cd mvts_transformer
python src/main.py --output_dir experiments --comment "pretraining through imputation" --name $1_pretrained --records_file Imputation_records.xls --data_dir /path/to/$1/ --data_class pain --pattern TRAIN --val_ratio 0.2 --epochs 700 --lr 0.001 --optimizer RAdam --batch_size 32 --pos_encoding learnable --d_model 128
```

## References
Much of the code in this repo is taken from:

Visual branch [MARLIN](https://github.com/ControlNet/MARLIN).

Signal branch [mvts](https://github.com/gzerveas/mvts_transformer).

## Probing
python3 evaluate.py

