# A Guide to Large Language Models (LLMs)
_Note: knowledge of attention and transformer architecture by the reader is assumed_

## Introduction to Large Language Models
Large language models have taken the world by storm as the pioneering technology of the AI boom of the 2020s, but what are they and what makes them so special? Text generation is not a new phenomena-- we have already learned about statistical methods like n-grams (Brown et al., 1990) and nascent deep learning algorithms like LSTM (Hochreiter & Schmidhuber, et al., 1997)-- yet the advent of LLMs like OpenAI's GPT has propelled artificial intelligence into the global spotlight.

There are many main players in the story of large language models, but perhaps none more important than the marriage of the transformer, the deep learning architecture powered by the attention mechanism, and the graphics processing unit (GPU). Natural language processing was made into a parallelizable matrix multiplication problem with _Attention is All You Need_ in 2017, and NVIDIA's market cap has grown by a factor of thirtyfold since its publishing at the time these notes have been recorded.

Nevertheless, while the invention of transformers was certainly the breakthrough technology that enabled the rise of LLMs, it was not the architecture alone that propelled them to where they are today. A critical realization was brought into focus with the release of OpenAI's landmark _Language Models are Unsupervised Multitask Learners_ in 2019, where researchers observed that performance of GPT models improved _smoothly_ and _predictably_ with scale. It was this apprehension that, as we will come to see, put "large" in "large language model" and has engendered the outpouring of investment into artificial intelligence that we've seen in recent years.

In the first part of these notes, we'll explore the major families of large language models before diving into the challenges and innovations involved in scaling and tuning them for **??*consumer usage**?*.

## Transformer Models
### BERT
