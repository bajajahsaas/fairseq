#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

export DATA_PATH=/mnt/nfs/work1/mccallum/abajaj/gs-summ/data/gs_data_full/beige_books_rev

for SPLIT in train
do
  for LANG in src tgt
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$DATA_PATH/$SPLIT.$LANG" \
    --outputs "$DATA_PATH/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done