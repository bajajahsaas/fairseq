#!/bin/bash
#
#SBATCH --job-name=finetune-bart
#SBATCH --output=logsfinetunebart/bart_%j.txt  # output file
#SBATCH -e logsfinetunebart/bart_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/mnt/nfs/work1/mccallum/abajaj/gs-summ/models/bart.large.cnn/model.pt
PROCESSED_DATA_PATH=/mnt/nfs/work1/mccallum/abajaj/gs-summ/data/gs_data_full/beige_books_rev/data-bin
#PROCESSED_DATA_PATH=/mnt/nfs/work1/mccallum/abajaj/gs-summ/data/gs_data_full/minutes/data-bin
SAVE_DIR=/mnt/nfs/work1/696ds-s20/abajaj/gs-finetune/checkpoints

python train.py $PROCESSED_DATA_PATH \
    --save-dir $SAVE_DIR \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;