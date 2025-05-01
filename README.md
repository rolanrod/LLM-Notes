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

When there are probabilities, there are probability distributions, and this probability distribution is exactly the one GPTs are attempting to approximate. At their core, they are simply learning a categorical distribution over tokens and sampling from it, one step at a time. This is, of course, exactly the same function that fixed-length n-grams attempt to model. GPTs, however, are able leverage the power of self-attention for this task and have indeed proven to be remarkably good at it. 

Why did GPTs ditch encoders? Recall what encoders are used for in the first place: capturing contextual relationships between all tokens bidirectionally(previous and future words). GPTs don't want this, and they explicitly block access to future tokens as they generate in real-time using _masked (causal) self-attention_ in each transformer block. Think of the original task of the transformer of translating text from one language to another, where we first need to understand (encode) what the original language is saying and then produce an output (decode) in another language. With GPTs, we are just generating text and all we need to do this best is the context sequence we already have.

[[[INSERT GPT IMAGE OR COMPARING TRANSLATION TO TEXT-GENERATION]]]

At the heart of the GPT architecture is the aforementioned masked self-attention mechanism, which is just like the vanilla self-attention we have seen previously but without forward connections-- each word can only attend to itself and to words before it. Consider the input \<START> "I like black coffee"; "black" only attends to "I", "like", and "black", not "coffee". This no-looking-ahead mechanism is referred to as a causal mask and, as you might imagine, looks like a triangular matrix with zeros to the right of the diagonal.

Besides the ommission of the encoder and cross-attention, nothing is new about GPT architecture: word and positonal embeddings added together, passed through several decoder layers (masked self-attention and feed-forward networks), through a linear classifier that projects embeddings into wordspace, and finally through a softmax that outputs a categorical distribution over the vocabulary. As soon as the next word is sampled, it is concatenated to the original input and the process repeats.


[[Insert objective function]]


## Training Paradigm: Pretraining + Fine-Tuning
The original GPT was trained in two stages: first, a large-scaled unsupervised language modeling phase, referred to as _pretraining_, and a second supervised fine-tuning phase on specific downstream tasks like question answering. 

### Pretraining
In pretraining, LLMs are trained on a massive corpus of unstructured text data using the causal language modeling objective, which simply minimizes cross-entropy, the negative log-likelihood of the actual words across a dataset, where $x_t$ is the next word:
$$ \mathcal{L}_{pretrain} = -\sum_{t=1}^{n}\log P(x_t | x_1, x_2, \dots, x_t).$$ 

How is this computed in pracitce in training? $n$ is a hyperparameter called the _context window_ or _context length_, and it specifies how many words the model can attend to in a forward pass. Every article, every piece of text that the LLM sees is broken up into these $n$-sized chunks and that is what specifies the size of our input to the model. Recall, however, that we compute the joint distribution $P(x_t | x_1, x_2, \dots, x_t)$ by explicitly expanding it via the chain rule of probabilities-- it's important to remember that we are indeed also computing $P(x_2|x_1), P(x_3|x_1, x_2),$ and so on. Think of the context window as the maximum length of a sequence that the model sees, but that it indeed calculates many other subsequences as it moves from left to right thanks to masked self-attention Therefore, even though our context window might be 1,024 or 8,000 words, we know perfectly well how to deal with short inputs like "I like black coffee".

It's here where the magic of masked self-attention reveals its power during pretraining: it allows the model to process an entire sequence in parallel while still preserving left-to-right structure required for generation. Because the "correct" next token is just the actual next word, the model doesn't need human-provided labels! This is what makes GPTs _self-supervised_-- they learn entirely from raw text with the data as its own supervisor. This is a crucial driver of implementing LLMs at scale that cannot be understated. Human-supervised learning would require assigning "correct" next words for every possible sequence in the English language, a task that would be not just impractical but virtually impossible.

So far, we've been referring to these $x$ vectors as words, but that's actually not how it's done in practice-- instead of operating at the level of full words, LLMs process the $x$ vectors in sequence as _tokens_, which are often subword units often derived using Byte Pair Encoding (BPE). Words are split into smaller, resuable fragments (e.g "un", "break", "able) that collectively cover language morphology. Although perhaps unintuitive to humans, _tokenization_ is hugely advantageous in providing extra granularity, reducing vocabulary size, and even handling rare or unseen words. Consider Oxford's 2022 Word of the Year: "goblinmode." For a model trained on, say, Wikipedia articles up to the year 2018, it's plausible to say that such a word might have never once appeared in the text data. Nevertheless, what tokenization essentially allows us to break down an input like "goblinmode" into "goblin" and "mode" tokens, which, when attended together might tell us about what "goblinmode" means.

[[ INSERT goblin + mode = goblinmode]]

The pretraining phase is the workhorse of building a large language model. By learning to predict the next word over billions of tokens, LLMs acquire a broad understanding of grammar, syntax, word meanings and relationships, facts, and long-range dependencies. The result of a converged pretrained model is the _base model_. While they might have the full knowledge of the dataset they were trained on baked into their weights, it's crucial to understand that <strong>base models are not assistants, nor chatbots.</strong> If you download the weights of a base model like LLaMA-2-7B and provide a text input like, "What's 10 / 5?" you might be surprised to find that you not get an answer like, "2" or "10 / 5 is 2." Instead, you're more likely to get something along the lines of, "We know the answer to this question is 2, this represents a basic mathematical truth. Mathematical truths are an example of objective knowledge that are necessarily true, and it is indeed impossible to think of the answer of being anything other than 2. Kant argued that mathematical truths are synthetic a priori, and that these types of truths are..." and so on. Base models are nothing more than a compression of their datasets, and if you feed them input, they will just keep on sampling from learned probability distributions ad infinitum, certainly with no particular orientation towards helping you solve a problem. 

In the earliest versions of GPT and BERT, this general-purpose knowledge was just the first step. To 

... 
What, then, is so special about GPTs? While the original GPT in 2018 demonstrated that decoder-only transformers set a baseline of feasibility, it was GPT-2 that truly captured the artificial intelligence community. As alluded to in the introduction, OpenAI made the remarkable observation that performance continued to improve predictably by scaling up model size, data size, and compute. 
## Scaling Up

## How to Make a Large Language Model
