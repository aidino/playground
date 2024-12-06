#! /bin/bash/
TODAY=`date +%m-%d.%H-%M`

HOME_DIR=/home/dino/Desktop/playground/fairseq-examples/translation
BIN_DIR=${HOME_DIR}/binarized
SAVE_DIR=${HOME_DIR}/models

mkdir -p $SAVE_DIR


fairseq-train \
    $BIN_DIR \
    --arch transformer --share-all-embeddings \
    --encoder-layers 6 --decoder-layers 4 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 5 \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --max-update 100000 \
    --keep-last-epochs 10 2>&1 | tee $SAVE_DIR/train.$TODAY.log

