#! /bin/bash

SRC_LANG=en
TGT_LANG=vi

HOME_DIR=/home/dino/Desktop/playground/fairseq-examples/translation
BIN_DIR=$HOME_DIR/binarized
TEXT=/home/dino/Desktop/playground/fairseq-examples/data

mkdir -p $BIN_DIR

fairseq-preprocess \
--source-lang $SRC_LANG \
--target-lang $TGT_LANG \
--joined-dictionary \
--trainpref $TEXT/train --validpref $TEXT/valid \
--destdir $BIN_DIR \
--workers 20
