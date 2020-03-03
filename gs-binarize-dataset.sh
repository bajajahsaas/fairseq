export DATA_PATH=/mnt/nfs/work1/mccallum/abajaj/gs-summ/data/gs_data_full/beige_books_rev/splits

fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "$DATA_PATH/train.bpe" \
  --validpref "$DATA_PATH/dev.bpe" \
  --destdir "$DATA_PATH/data-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;