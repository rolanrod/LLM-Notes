# A Guide to Large Language Models (LLMs)
_Note: knowledge of attention and transformer architecture by the reader is assumed_

## Introduction to Large Language Models
Large language models have taken the world by storm as the pioneering technology of the AI boom of the 2020s, but what are they and what makes them so special? Text generation is not a new phenomenon-- we have already learned about statistical methods like n-grams (Brown et al., 1990) and nascent deep learning algorithms like LSTM (Hochreiter & Schmidhuber, et al., 1997)-- yet the advent of LLMs like OpenAI's GPT has propelled artificial intelligence into the global spotlight.

Put plainly, a large language model is a machine learning model trained on vast amounts of text data to understand and generate human language. Typically on the order of billions of _parameters_ (a word meaning nothing more than learnable weights and biases) in size, LLMs are used as chatbots, translators, summarizers, and for all sorts of text-based tasks.

There are many main players in the story of large language models, but perhaps none more important than the marriage of the transformer, the deep learning architecture powered by the attention mechanism, and the graphics processing unit (GPU). Natural language processing was made into a parallelizable matrix multiplication problem with _Attention is All You Need_ in 2017, and NVIDIA's market cap has grown by a factor of thirtyfold since its publishing at the time these notes have been compiled.

Nevertheless, while the invention of transformers was certainly the breakthrough technology that enabled the rise of LLMs, it was not the architecture alone that propelled them to where they are today. A critical realization was brought into focus with the release of OpenAI's landmark _Language Models are Unsupervised Multitask Learners_ in 2019, where researchers observed that performance of GPT models improved _smoothly_ and _predictably_ with scale. It was this apprehension that, as we will come to see, put "large" in "large language model" and has engendered the outpouring of investment into artificial intelligence that we've seen in recent years.

In the first part of these notes, we'll explore the major families of large language models before diving into the challenges and innovations involved in scaling and tuning them for widespread usability.

## Transformer Models
### BERT
BERT (Bidirectional Encoder Representation) was introduced in 2018 as a transformer-based model developed by Google and offered a major leap in natural language understanding. Before we speak about the specifics of BERT, it's important to clear up a common misunderstanding: 
<div align="center" style="padding-bottom:10px;">
  <strong>GPTs are a subset of LLMs, but they are not interchangeable terms.</strong>
</div>

As mentioned in the introduction, there are several transformer models that can be considered "large language models," BERT among them. As we'll see, LLMs are not strictly generative, and models like BERT were made not with the intention to produce text but to, as the name suggests, form rich encoder representations as contextual embeddings.

BERT uses only the encoder half of the transformer architecture to produce this output, so the transformer pipeline is essentially halved:

1) Text input (e.g."I like coffee") is fed into word, positional, and sentence embeddings are added together.
2) The combined vectors are passed through multi-head self-attention and feed-forward layers of the encoder
3) Contextual word embeddings outputted

(Note that (2) is often repeated multiple times in practice and often with LayerNorm and residual connections. This step is often abstracted into what is referred to as a transformer block-- the original BERT paper used 12)

The key innovation of BERT that set it apart from the traditional transformer encoder was its training task. It's crucial to remember that the encoder-decoder transformer model of 2017 was training with the objective of next word prediction, which while suited for that type of task, was suboptimal for a holistic understanding of context. BERT proposed a new objective: <strong>masked lanuage modeling (MLM)</strong>, which benefitted from fully bidirectional context, and better word representation. With MLM, BERT is trained by randomly masking out a subset of input words (typically ~15%) and asking the model to predict the original words based on the full, unmasked context. For example, the sentence _"I like [MASK] coffee"_ might appear in training-- the model is tasked with recovering the missing word ("black") by attending to the left ("I like") and right ("coffee") contexts. This is what we mean by "bidirectional." The loss function of MLM is simply cross-entropy applied only at the masked positions.

![IMG_EBF0FD24F981-1](https://github.com/user-attachments/assets/26b90613-232e-42ff-b58f-d1be2135426e)

In addition to MLM, the original BERT paper also introduced a next stentence prediction (NSP) objective in which the model is given two segments of text and must predict whether the second segment follows the first in the original corpus. While its use has since been debated (and often removed in later models), it was initially introduced to help BERT model relationships more macroscopically at the sentence level. 

Once trained on a large corpus, BERT is _fine-tuned_ on specific tasks by adding a small classificaiton head on top of the final encoder output. During this phase, the network is updated using labeled data for tasks. More on this pipeline later.

![IMG_D0056EF8F59B-1](https://github.com/user-attachments/assets/e6c5091d-ed47-43b7-ac95-785ceb230a0b)


### T5/BART
Google Research released T5 (Text-to-Text Transfer Transformer) in 2019 and was based on the original encoder-decoder model of the original transformer.

![IMG_CE90673FB270-1](https://github.com/user-attachments/assets/5feb9404-beb5-440e-a675-5c0e7c3d38b4)

### GPT
The Generative Pre-trained Transformer (GPT) was introduced by OpenAI in 2018 as a decoder-only transformer architecture-- no encoder, and no cross-attention. In many ways, GPTs depart from the "transforming" of text that the vanilla transformer was made for (translation, most notably) and double-down on text generation through _autoregressive next-word prediction_, which is simply the process of sequentially taking outputs and feeding them back through the model at every iteration. 

Recall the definition of language modelling developed by Claude Shannon in his _Prediction and Entropy of Printed English_ (1951) paper. Given a sequence of words $(x_1, x_2, ... x_n)$, we want to maximize $P(x_1, x_2, ... x_n)$. We know by the chain rule of probabilities that

$$ P(x_1, x_2, \dots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)\dots P(x_n|x_1, \dots x_{n-1}) = \prod_{i=1}^{n}P(x_i | x_1, x_2, \dots, x_{i-1}).$$

When there are probabilities, there are probability distributions, and this probability distribution is exactly the one GPTs are attempting to approximate. At their core, they are simply learning a categorical distribution over words and sampling from it, one step at a time. This is, of course, exactly the same function that fixed-length n-grams attempt to model. GPTs, however, are able leverage the power of self-attention for this task and have indeed proven to be remarkably good at it. 

Why did GPTs ditch encoders? Recall what encoders are used for in the first place: capturing contextual relationships between all words bidirectionally (previous and future words). GPTs don't want this, and they explicitly block access to future words as they generate in real-time using _masked (causal) self-attention_ in each transformer block. Think of the original task of the transformer of translating text from one language to another, where we first need to understand (encode) what the original language is saying and then produce an output (decode) in another language. With GPTs, we are just generating text and all we need to do this best is the context sequence we already have.

![IMG_44C267C705EE-1](https://github.com/user-attachments/assets/1c99fd69-488a-4361-91ed-ce8c382dcefa)

At the heart of the GPT architecture is the aforementioned masked self-attention mechanism, which is just like the vanilla self-attention we have seen previously but without forward connections-- each word can only attend to itself and to words before it. Consider the input \<START> "I like black coffee"; "black" only attends to "I", "like", and "black", not "coffee". This no-looking-ahead mechanism is referred to as a causal mask and, as you might imagine, looks like a triangular matrix with zeros to the right of the diagonal.

Besides the omission of the encoder and cross-attention, nothing is new about GPT architecture: word and positonal embeddings added together, passed through several decoder layers (masked self-attention and feed-forward networks), through a linear classifier that projects embeddings into wordspace, and finally through a softmax that outputs a categorical distribution over the vocabulary. As soon as the next word is sampled, it is concatenated to the original input and the process repeats.

[[Insert objective function]]

## Scaling Up: GPT-2 and GPT-3
At this point, you might be asking what's so special about GPTs. While it was certainly a bold and interesting view to disregard the encoder in its architecture, GPT did not invent the decoder. While the original 2018 GPT demonstrated that decoder-only transformers set a baseline of feasibility, it was the release of GPT-2 in the following year that truly captured the artificial intelligence community. 

Part of what made GPT-2 truly remarkable was not merely its scale (ten times the size of the original GPT) but the emergence of capabilities it had not been explicitly trained for without any direct fine-tuning or task-specific supervision. This "_zero-shot learning_" phenomenon was especially peculiar to researchers, who noticed that if one provided input carefully enough to GPT-2, the model could generalize into performing all sorts of different tasks. Inputs that asked something like, "Translate from English to French: 'The book is on the table.'" would often yield a fluent French sentence, despite GPT-2 never being fine-tuned on a translation dataset. This surprising generalization hinted at the emergence of in-context learning-- the model was showing an emergent ability to adapt its behavior based on patterns present in the input. This suggested that, at sufficient scale, language models could learn not just facts and grammar, but how to actually do tasks from text alone.

This observation formed the basis of what are now called _scaling laws_, empirical trends that demonstrated that as the number of parameters, size of training set, and amount of compute were increased, a model's performance improved smoothly and predictably. It suddenly became clear that sheer scale was a primary driver of emergent capability.

[[INSERT SCALING LAWS IMAGE AND POWER LAW FORMULAS]]

OpenAI put everything they had behind these findings and cranked up scale wildly. GPT-3, released in 2020, was over one-hundred times larger than GPT-2 at 175 billion parameters and over 300 billion tokens of data (more on tokens later). Emergent behavior began to appear with greater regularity: GPT-3 could perform multi-step arithmetic, solve analogy problems, write code, and complete SAT-style reading comprehension questions with minimal prompting. The ability to interpret and act on a task description with minimal instruction (zero- or one-shot) gave the model a special kind of flexibility that allowed it to adapt to new problems with nothing more than the right prompt phrasing needed. GPT-3 had not mastered each task individually, but it had been so relentlessly trained to predict the next token correctly across a vast range of contexts that it implicitly had to learn the abstract patterns of logic and instruction-following needed to solve different types of problems.

![IMG_E5369BB463C6-1](https://github.com/user-attachments/assets/3c61e95c-7396-409c-bc37-c0360b1aff86)

You might have been surprised to learn that GPT-3 was released in 2020 given that it was the model behind ChatGPT, which really caught on in late 2022 and early 2023. But the original GPT-3, powerful as it was, was still a base-model, trained purely on predicting the next token in a sequence with no explicit goal beyond minimizing language modeling loss. While the model showed that it didn't need any fine-tuning, the gap between usability and capability brought the need for _post-training_, which helped modify behavior that made a powerful but clumsy base model into a genuinely useful tool.

## Pre-Training + Post-Training: How to Build a Large Language Model
The original GPT was trained in two stages: first, a large-scaled unsupervised language modeling phase, referred to as _pre-training_, and a second supervised post-training phase on specific downstream tasks like question answering. 

### Pre-Training
In pre-training, LLMs are trained on a massive corpus of unstructured text data using the causal language modeling objective, which simply minimizes cross-entropy, the negative log-likelihood of the actual words across a dataset, where $x_t$ is the next word:

$$𝓛_{pre-train} = -∑_{t=1}^n \log P(x_t | x_1, x_2, ..., x_t)$$

How is this computed in practice during training? $n$ is a hyperparameter called the _context window_ or _context length_, and it specifies how many words the model can attend to in a forward pass. Every article, every piece of text that the LLM sees is broken up into these $n$-sized chunks and that is what specifies the size of our input to the model. Recall, however, that we compute the joint distribution $P(x_t | x_1, x_2, \dots, x_t)$ by explicitly expanding it via the chain rule of probabilities-- it's important to remember that we are indeed also computing $P(x_2|x_1), P(x_3|x_1, x_2),$ and so on. Think of the context window as the maximum length of a sequence that the model sees, but that it indeed calculates many other subsequences as it moves from left to right thanks to masked self-attention Therefore, even though our context window might be 1,024 or 8,000 words, we know perfectly well how to deal with short inputs like "I like black coffee".

It's here where the magic of masked self-attention reveals its power during pre-training: it allows the model to process an entire sequence in parallel while still preserving left-to-right structure required for generation. Because the "correct" next word is just the actual next word, the model doesn't need human-provided labels! This is what makes GPTs _self-supervised_-- they learn entirely from raw text with the data as its own supervisor. This is a crucial driver of implementing LLMs at scale that cannot be understated. Human-supervised learning would require assigning "correct" next words for every possible sequence in the English language, a task that would be not just impractical but virtually impossible. Other LLMs like BERT and T5 are perfectly capable of self-supervision too by giving the data the driver's seat.

So far, we've been referring to these $x$ vectors as words, but that's actually not how it's done in practice-- instead of operating at the level of full words, LLMs process the $x$ vectors in sequence as _tokens_, which are often subword units often derived using Byte Pair Encoding (BPE). Words are split into smaller, reusable fragments (e.g "un", "break", "able") that collectively cover language morphology. Although perhaps unintuitive to humans, _tokenization_ is hugely advantageous in providing extra granularity, reducing vocabulary size, and even handling rare or unseen words. Consider Oxford's 2022 Word of the Year: "goblinmode." For a model trained on, say, Wikipedia articles up to the year 2018, it's plausible to say that such a word might have never once appeared in the text data. Nevertheless, what tokenization essentially allows us to break down an input like "goblinmode" into "goblin" and "mode" tokens, which, when attended together might tell us about what "goblinmode" means. 

![IMG_8880EF7BE6B7-1](https://github.com/user-attachments/assets/fd9ace6c-b1d5-4887-b852-27dac7280418)

The pre-training phase is the workhorse of building a large language model. By learning to predict the next word over billions of tokens, LLMs acquire a broad understanding of grammar, syntax, word meanings and relationships, facts, and long-range dependencies. The result of a converged pretrained model is the _base model_. While they might have the full knowledge of the dataset they were trained on baked into their weights, it's crucial to understand that <strong>base models are not assistants, nor chatbots.</strong> If you download the weights of a base model like LLaMA-2-7B and provide a text input like, "What's 10 / 5?" you might be surprised to find that you not get an answer like, "2" or "10 / 5 is 2." Instead, you're more likely to get something along the lines of, "We know the answer to this question is 2, this represents a basic mathematical truth. Mathematical truths are an example of objective knowledge that are necessarily true, and it is indeed impossible to think of the answer of being anything other than 2. Kant argued that mathematical truths are synthetic a priori, and that these types of truths are..." and so on. Base models are nothing more than a compression of their datasets, and if you feed them input, they will just keep on sampling from learned probability distributions ad infinitum, certainly with no particular orientation towards helping you solve a problem. 


Consider the following scenario, taken directly from OpenAI's user-testing on GPT-3: 
```python
# Prompt:
# What is the purpose of the list C in the code below?

def binomial_coefficient(n, r):
    C = [0 for i in range(r + 1)]
    C[0] = 1
    for i in range(1, n + 1):
        j = min(i, r)
        while j > 0:
            C[j] += C[j - 1]
            j -= 1
    return C[r]
```
Of course, the answer that the prompter wants here is an explanation detailing that $C$ is storing binomial coefficients as we iterate through $n$ and $r$. Nevertheless, GPT-3's output was rather underwhelming:

```
A. to store the value of C[0]  
B. to store the value of C[1]  
C. to store the value of C[i]  
D. to store the value of C[i - 1].
```

What happened here is that GPT-3 says to itself, "Ah, I've seen this exact question on the 2014 AP Computer Science A exam; I know what comes next!" and it proceeds to spit out what indeed came next, but not what we want from it. Base-models this big can memorize very well, and actions need to be taken to move them away from this behavior.


### Fine-Tuning and Post-Training
In the earliest versions of GPT and BERT, this general-purpose knowledge was just the first step. To 


## Social Ramifications of Large Language Models
