This repo contains the scripts that were created for the course Machine Translation Advanced Topics.

Most of the scripts were vibe-coded with ChatGPT with lots of testing and back-and-forth conversation.

# Chapters in the course
You can find the slides of each chapter in the slides directory.

## Chapter 1. Introduction and MT pre-neural history

The course starts with an introduction to MT and a description of what happened before the NMT paradigm.

## Chapter 2. Data Preparation
* Github: https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT26_DataPreparation.ipynb
* Colab: [Data Preparation](https://colab.research.google.com/drive/1KO8hei6DsMTm_bqFUEZnqeri8nuMq_Ik)
  
## Chapter 3. MT Evaluation
Translation through Python with commercial engines and evaluation with most common metrics
* Github:
  * https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT26_Translation%26Evaluation.ipynb
  * https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT2026_BLEURT.ipynb
  * https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT26_COMET.ipynb
    
  (Note that these notebooks are rendered in Github as Invalid Notebooks, but they do run in Google Colab)
* Colab:
  * [Part 1](https://colab.research.google.com/drive/1bnORQp4GsunokbuHcfoNFeaTpg9SVue3)
  * [Bleurt](https://colab.research.google.com/drive/1lHoEoGQM8-Hg6KqCgCbqiwU_RYiO3L8n)
  * [COMET](https://colab.research.google.com/drive/1m65-TU26XJYaXBNRkjlxDd1_oYtAhgLI)

## Chapter 4. RNN Language Modeling
Before we start on MT, we explain RNN language modeling with a toy example and later expand it to a larger language model.

  * Github:
    * [Toy LM](https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT26_RNN_Language_Model.ipynb)
    * [Larger LM](https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT26_RNN_Language_Modeling_Large.ipynb)
  * Colab:
    * [Toy LM](https://colab.research.google.com/drive/1TITxFBtkRwoejC0vXT9AWZjZc7ms-hJM)
    * [Larger LM](https://colab.research.google.com/drive/1rTZJGbs2SSsDWHmgHXRv5OSfdmePWej1)

## Chapter 5. RNN Machine Translation

* Github:
  * [Toy example](https://github.com/VincentCCL/MTAT/blob/main/notebooks/MTAT26_RNN_Encoder_Decoder.ipynb)
  * [Scaling up](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-chapter5-rnns.ipynb)
* Colab / Kaggle:
  * [Toy example](https://colab.research.google.com/drive/1KG8QWkcscaxTU4as5rT1oUzbUYlTpWyQ)
  * [Scaling up in Kaggle](https://www.kaggle.com/code/vincentvandeghinste/mtat26-chapter5-rnns)
 
## Chapter 6. Subwording and Transformers

* Github:
  * [Subwording](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-subwording.ipynb)
  * [Transformers](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-transformers.ipynb)
  * [Multilingual Transformers](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-multilingualtransformers.ipynb)
* Kaggle:
  * [Subwording](https://www.kaggle.com/code/vincentvandeghinste/mtat26-subwording)
  * [Transformers](https://www.kaggle.com/code/vincentvandeghinste/mtat26-transformers)
  * [Multilingual Transformers](https://www.kaggle.com/code/vincentvandeghinste/mtat26-multilingualtransformers)     

## Chapter 7. Pretrained Encoder-Decoders

* Github:
  * [T5](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-t5.ipynb)
  * [mT5](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-mt5.ipynb)
  * [mBART](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-mbart.ipynb)
  * [M2M](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-m2m.ipynb)
  * [NLLB](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-nllb.ipynb)
  * [Madlad 400](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-madlad400.ipynb)
* Kaggle:
  * [T5](https://www.kaggle.com/code/vincentvandeghinste/mtat26-t5)
  * [mT5](https://www.kaggle.com/code/vincentvandeghinste/mtat26-mt5)
  * [mBART](https://www.kaggle.com/code/vincentvandeghinste/mtat26-mbart)
  * [M2M](https://www.kaggle.com/code/vincentvandeghinste/mtat26-m2m)
  * [NLLB](https://www.kaggle.com/code/vincentvandeghinste/mtat26-nllb)
  * [Madlad 400](https://www.kaggle.com/code/vincentvandeghinste/mtat26-madlad400)
 
  ## Chapter 8. Decoders only
  * Github
    * [GPT-2](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-gpt2.ipynb)
    * [Llama](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-llama.ipynb)
    * [Mistral](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-mistral.ipynb)
    * [Qwen](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat26-qwen.ipynb)
    * [Groq](https://github.com/VincentCCL/MTAT/blob/main/notebooks/mtat-groq.ipynb)
  * Kaggle
    * [GPT-2](https://www.kaggle.com/code/vincentvandeghinste/mtat26-gpt2)
    * [Llama](https://www.kaggle.com/code/vincentvandeghinste/mtat26-llama)
    * [Mistral](https://www.kaggle.com/code/vincentvandeghinste/mtat26-mistral)
   
  ## Chapter 9. Conclusions
  There are no hands-on sessions for the conclusions.
    * [Qwen](https://www.kaggle.com/code/vincentvandeghinste/mtat26-qwen)
    * [Groq](https://www.kaggle.com/code/vincentvandeghinste/mtat-groq)

# Current work
We are in the process of integrating and testing all the different transformer encoder-decoder scripts from chapters 6 and 7 into a single script. Current version (unfinished) is in [code/mtat.py]
