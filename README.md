# A Guide to Large Language Models (LLMs)
_Note: knowledge of attention and transformer architecture by the reader is assumed_

## Introduction to Large Language Models
Large language models have taken the world by storm as the pioneering technology of the AI boom of the 2020s, but what are they and what makes them so special? Text generation is not a new phenomena-- we have already learned about statistical methods like n-grams (Brown et al., 1990) and nascent deep learning algorithms like LSTM (Hochreiter & Schmidhuber, et al., 1997)-- yet the advent of LLMs like OpenAI's GPT has propelled artificial intelligence into the global spotlight.

Put plainly, a large language model is a machine learning model trained on vast amounts of text data to understand and generate human language. Typically on the order of billions of _parameters_ (a buzzword meaning nothing more than learnable weights and biases) in size, LLMs are used as chatbots, translators, summarizers, and for all sorts of text-based tasks.

There are many main players in the story of large language models, but perhaps none more important than the marriage of the transformer, the deep learning architecture powered by the attention mechanism, and the graphics processing unit (GPU). Natural language processing was made into a parallelizable matrix multiplication problem with _Attention is All You Need_ in 2017, and NVIDIA's market cap has grown by a factor of thirtyfold since its publishing at the time these notes have been recorded.

Nevertheless, while the invention of transformers was certainly the breakthrough technology that enabled the rise of LLMs, it was not the architecture alone that propelled them to where they are today. A critical realization was brought into focus with the release of OpenAI's landmark _Language Models are Unsupervised Multitask Learners_ in 2019, where researchers observed that performance of GPT models improved _smoothly_ and _predictably_ with scale. It was this apprehension that, as we will come to see, put "large" in "large language model" and has engendered the outpouring of investment into artificial intelligence that we've seen in recent years.

In the first part of these notes, we'll explore the major families of large language models before diving into the challenges and innovations involved in scaling and tuning them for **??*consumer usage**?*.  

## Transformer Models
### BERT
BERT (Bidirectional Encoder Representation) was introduced in 2018 as a transformer-based model developed by Google and offered a major leap in natural language understanding. Before we speak about the specifics of BERT, it's important to clear up a common misunderstanding: 
<div align="center" style="padding-bottom:10px;">
  <strong>GPTs are a subset of LLMs, but they are not interchangeable terms.</strong>
</div>

As mentioned in the introduction, there are a several transformer models that can be considered "large language models," BERT among them. As we'll see, LLMs are not strictly generative, and models like BERT were made not with the intention to produce text but to, as the name suggests, form rich encoder representations as contextual embeddings.

BERT uses only the encoder half of the transformer architecture to produce this output, so the transformer pipeline is essentially halved:

1) Text input (e.g."I like coffee") is fed into token, positional, and sentence embeddings and are added together.
2) The combined vectors are passed through multi-head self-attention and feed-forward layers of the encoder
3) Contextual word embeddings outputted

(Note that (2) is often repeated multiple times in practice and often with LayerNorm and residual connections. This step is often abstracted into what is referred to as a transformer block-- the original BERT paper used 12)

The key innovation of BERT that set it apart from the traditional transformer encoder was its training task. It's crucial to remember that the encoder-decoder transformer model of 2017 was training with the objective of next word prediction, which while suited for that type of task, was suboptimal for a holistic understanding of context. BERT proposed a new objective: <strong>masked lanuage modeling (MLM)</strong>, which benefitted from fully bidirectional context, and better word representation.

[[[[[INSERT FIGURE FOR MASKED LANGUAGE MODELING]]]]]

CONTINUE BERT...


### T5/BART


### GPT
The Generative Pre-trained Transformer (GPT) was introduced by OpenAI in 2018 as a decoder-only transformer architecture-- no encoder, and no cross-attention. In many ways, GPTs depart from the "transforming" of text that the vanilla transformer was made for (translation, most notably) and double-down on text generation through _autoregressive next-word prediction_, which is simply the process of sequentially taking outputs and feeding them back through the model at every iteration. 

Recall the definition of language modelling developed by Claude Shannon in his _Prediction and Entropy of Printed English_ (1951) paper. Given a sequence of tokens $(x_1, x_2, ... x_n)$, we want to maximize $P(x_1, x_2, ... x_n)$. We know by the chain rule of probabilities that

$$ P(x_1, x_2, \dots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)\dots P(x_n|x_1, \dots x_{n-1}) = \prod_{i=1}^{n}P(x_i | x_1, x_2, \dots, x_{i-1}).$$

When there are probabilities, there are probability distributions, and this probability distribution is exactly the one GPTs are attempting to approximate. At their core, they are simply learning a categorical distribution over tokens and sampling from it, one step at a time. This is, of course, exactly the same function that fixed-length n-grams attempt to model. GPTs, however, are able leverage the power of self-attention for this task and have since proven to be indeed remarkably good at it. 

## The Fine-Tuning Paradigm

## Scaling Up

## How to Make a Large Language Model
