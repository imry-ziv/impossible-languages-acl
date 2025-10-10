#!/bin/bash

# Create subdirectory for English
mkdir -p ./data/multilang/english

# Download English files
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/train.txt -O ./data/multilang/english/train.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/test.txt -O ./data/multilang/english/test.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/valid.txt -O ./data/multilang/english/valid.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt -O ./data/multilang/english/vocab.txt


# Create subdirectory for Hebrew
mkdir -p ./data/multilang/hebrew

# Download Hebrew files
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/train.txt -O ./data/multilang/hebrew/train.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/test.txt -O ./data/multilang/hebrew/test.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/valid.txt -O ./data/multilang/hebrew/valid.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Hebrew/vocab.txt -O ./data/multilang/hebrew/vocab.txt


# Create subdirectory for Italian
mkdir -p ./data/multilang/italian

# Download English files
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Italian/train.txt -O ./data/multilang/italian/train.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Italian/test.txt -O ./data/multilang/italian/test.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Italian/valid.txt -O ./data/multilang/italian/valid.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Italian/vocab.txt -O ./data/multilang/italian/vocab.txt


# Create subdirectory for Russian
mkdir -p ./data/multilang/russian

# Download English files
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Russian/train.txt -O ./data/multilang/russian/train.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Russian/test.txt -O ./data/multilang/russian/test.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Russian/valid.txt -O ./data/multilang/russian/valid.txt.original
wget -x https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/Russian/vocab.txt -O ./data/multilang/russian/vocab.txt
