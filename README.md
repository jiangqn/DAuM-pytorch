# DAuM-pytorch

This is a pytorch implementation for [Enhanced Aspect Level Sentiment Classification with Auxiliary Memory][1] (Peisong Zhu, Tieyun Qian, COLING 2018).

## requirements

```
pytorch 1.0.0
numpy
spacy
```

## Quick Start

- Download the 300-dimensional pre-trained word vectors from [Glove][2] and save it in the 'data' folder as 'data/glove.840B.300d.txt'

- Use the following command to run the code.

```
python main.py
```

## Results

| Dataset    | Accuracy |
| ---------- | -------- |
| Laptop     | 69.12   |
| Restaurant | 79.20   |

[1]:http://aclweb.org/anthology/C18-1092
[2]:https://nlp.stanford.edu/projects/glove/