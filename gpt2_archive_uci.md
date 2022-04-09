

## gpt2_archive_uci
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/gpt2_archive_uci.ipynb)

Gpt 2 is a generative model that predicts the next words, given an input sequence so some work needs to be done for gpt 2 to be used for classification.
We use the last token of the input sequence to predict the classification, instead of predicting the next word. 


[gmihaila](https://github.com/gmihaila)
https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/

[archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

```
!pip install -q git+https://github.com/huggingface/transformers.git
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
!unzip smsspamcollection.zip -d spam
```


#### [Open `gpt2_archive_uci.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/gpt2_archive_uci.ipynb)



