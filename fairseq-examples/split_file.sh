#! /bin/bash/

source_file="/home/dino/Desktop/playground/fairseq-examples/data/OpenSubtitles.en-vi.en"
target_file="/home/dino/Desktop/playground/fairseq-examples/data/OpenSubtitles.en-vi.vi"

total_lines=$(wc -l < "$source_file")

split_line=$(((total_lines * 90)/100))

head -n $split_line $source_file > data/train.en
tail -n +$((split_line + 1)) $source_file > data/valid.en

head -n $split_line $target_file > data/train.vi
tail -n +$((split_line + 1)) $target_file > data/valid.vi