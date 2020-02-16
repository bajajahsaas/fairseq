export DATA_PATH=/mnt/nfs/work1/mccallum/abajaj/gs-summ/data/cnn-for-bart

fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "$DATA_PATH/train.bpe" \
  --validpref "$DATA_PATH/val.bpe" \
  --destdir "$DATA_PATH/cnn_dm-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;