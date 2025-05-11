# A Guide to Large Language Models (LLMs)
_Note: knowledge of attention and transformer architecture by the reader is assumed_

## Introduction to Large Language Models
Large language models have taken the world by storm as the pioneering technology of the AI boom of the 2020s, but what are they and what makes them so special? Text generation is not a new phenomenon-- we have already learned about statistical methods like n-grams (Brown et al., 1990) and nascent deep learning algorithms like LSTM (Hochreiter & Schmidhuber, et al., 1997)-- yet the advent of LLMs like OpenAI's GPT has propelled artificial intelligence into the global spotlight.

Put plainly, a large language model is a machine learning model trained on vast amounts of text data to understand and generate natural language. Typically on the order of billions of _parameters_ (a word synonymous with learnable weights and biases) in size, LLMs are used as chatbots, translators, summarizers, and for all sorts of text-based tasks.

There are many main players in the story of large language models, but perhaps none more important than the marriage of the transformer, the deep learning architecture powered by the attention mechanism, and the graphics processing unit (GPU). Natural language processing was made into a parallelizable matrix multiplication problem with _Attention is All You Need_ in 2017, and NVIDIA's market cap has grown by a factor of thirtyfold since its publishing at the time these notes have been compiled.

Nevertheless, while the invention of transformers was certainly the breakthrough technology that enabled the rise of LLMs, it was not the architecture alone that propelled them to where they are today. A critical realization was brought into focus with the release of OpenAI's landmark _Language Models are Unsupervised Multitask Learners_ in 2019, where researchers observed that performance of GPT models improved _smoothly_ and _predictably_ with scale. It was this apprehension that, as we will come to see, put "large" in "large language model" and has engendered the outpouring of investment into artificial intelligence that we've seen in recent years.

In the first part of these notes, we'll explore the major families of large language models before diving into the challenges and innovations involved in training, scaling, and tuning them for widespread usability.

## Transformer Models
### BERT
BERT (Bidirectional Encoder Representation) was introduced in 2018 as a transformer-based model developed by Google and offered a major leap in natural language understanding. Before we speak about the specifics of BERT, it's important to clear up a common misunderstanding: *GPTs are a subset of LLMs, but they are not interchangeable terms.*

As mentioned in the introduction, there are several transformer models that can be considered "large language models," BERT among them. As we'll see, LLMs are not strictly generative, and models like BERT were made not with the intention to produce text but to, as the name suggests, form rich encoder representations as contextual embeddings.

BERT uses only the encoder half of the transformer architecture to produce this output, so the transformer pipeline is essentially halved:

1) Text input (e.g."I like coffee") is fed into word, positional, and sentence embeddings that are added together.
2) The combined vectors are passed through multi-head self-attention and feed-forward layers of the encoder
3) Contextual word embeddings are outputted

(Note that (2) is often repeated multiple times in practice and often with LayerNorm and residual connections. This step is often abstracted into what is referred to as a transformer block-- the original BERT paper used 12)

The key innovation of BERT that set it apart from the traditional transformer encoder was its training task. It's crucial to remember that the encoder-decoder transformer model of 2017 was training with the objective of next word prediction, which while suited for that type of task, was suboptimal for a holistic understanding of context. BERT proposed a new objective: *masked language modeling (MLM)*, which benefitted from fully bidirectional context, and better word representation. With MLM, BERT is trained by randomly masking out a subset of input words (typically ~15%) and asking the model to predict the original words based on the full, unmasked context. For example, the sentence _"I like [MASK] coffee"_ might appear in training-- the model is tasked with recovering the missing word ("black") by attending to the left ("I like") and right ("coffee") contexts. This is what we mean by "bidirectional." The loss function of MLM is simply cross-entropy applied only at the masked positions.

<p align="center">
    <img src="https://github.com/user-attachments/assets/92c47ac8-c3c1-40ca-9e8f-286823918b86" width="500px">
</p>

In addition to MLM, the original BERT paper also introduced a next sentence prediction (NSP) objective in which the model is given two segments of text and must predict whether the second segment follows the first in the original corpus. While its use has since been debated (and often removed in later models), it was initially introduced to help BERT model relationships more macroscopically at the sentence level. 

BERT’s true impact came not just from its architecture, but from the pre-train + fine-tune paradigm it popularized. Once pretrained on massive corpora (Wikipedia and BookCorpus), BERT could be _fine-tuned_ with minimal task-specific data by attaching lightweight heads to the model for classification, question answering, or word-level tasks like named entity recognition. This transfer learning approach led to state-of-the-art results on benchmarks like GLUE and SQuAD at the time of release.

While BERT set a new standard, it had limitations: it was not generative and the NSP objective was later shown to be of limited value. Successors like RoBERTa removed NSP and used dynamic masking, DistilBERT compressed BERT via knowledge distillation for speed, and ALBERT reduced parameter count by sharing weights across layers.

Together, these developments solidified BERT’s legacy as a foundation of modern natural language understanding, even as generative models like GPT and encoder-decoder models like T5 expanded the frontier.

<p align="center">
    <img src="https://github.com/user-attachments/assets/c2828162-7684-4860-8e4c-542d2e4bbd72" width="500px">
</p>

### T5/BART
Google Research released T5 (Text-to-Text Transfer Transformer), an encoder-decoder model similar to the original transformer in 2019. It's important to remember that transformers required task-specific architectures for these things: if we wanted to use classification, we might add a softmax head over a [CLS] word embedding, or if we wanted to translate, we'd use a sequence-to-sequence decoder trained on bilingual pairs. T5 proposed a generalization: reframing everything (translation, classification, Q&A, summarization) as a text sequence generation problem. For example, classification on movie reviews are converted to text generation by having the model output labels as text (e.g., "This movie review has a negative sentiment").

Unlike the original transformer, which had different heads and output formats for each task it was used for, T5 normalized all NLP tasks into *text input --> text output*. This required a carefully aligned training strategy. Instead of next-word prediction, T5 trained with a _span corruption_ objective, which was similar to MLM with some key differences. Instead of masking individual words, span corruption masks contiguous sequences of words and replaces them with "sentinel words," which the model is trained to generate in order. For instance,
- Input: "The \<extra_id_0> sat on the \<extra_id_1>"
- Output: \<extra_id_0> cat \<extra_id_1>  mat"

This objective encourages the model not just to guess masked tokens but to reconstruct coherent, semantically meaningful spans. In a sense, it's applying the intuition behind BERT in a generative setting. 

<p align="center">
    <img src="https://github.com/user-attachments/assets/8939d725-15f9-490f-acbe-f331b6080586" width="500px">
</p>

Around the same time, Facebook AI released BART (Bidirectional and Auto-Regressive Transformers), which also adopted encoder-decoder transformer architecture with a distinct approach to training. BART employed a more diverse set of corruption techniques that included masking, sentence permutation, and word deletion and insertion.

What made T5 and BART so portable and streamlined was how efficiently they could be fine-tuned. Each of their pre-training objectives created models that already possessed strong general language understanding and generation capabilities, requiring only small adjustments to excel at downstream tasks. 

<p align="center">
    <img src="https://github.com/user-attachments/assets/8f7db6c8-6508-480b-9b84-da7456702aa9" width="600px">
</p>
<p align="center"><i>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</i>, Raffel et. al 2019</p>


### GPT
The Generative Pre-trained Transformer (GPT) was introduced by OpenAI in 2018 as a decoder-only transformer architecture-- no encoder, and no cross-attention. In many ways, GPTs depart from the "transforming" of text that the vanilla transformer was made for (translation, most notably) and double-down on text generation through _autoregressive next-word prediction_, which is simply the process making predictions and sequentially feeding them back through the model at every iteration, yielding a sequence that grows from left to right. 

Recall the definition of language modelling developed by Claude Shannon in his _Prediction and Entropy of Printed English_ (1951) paper. Given a sequence of words $(x_1, x_2, \dots, x_n)$, we want to maximize $P(x_1, x_2, \dots, x_n)$. We know by the chain rule of probabilities that

$$ P(x_1, x_2, \dots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)\dots P(x_n|x_1, \dots, x_{n-1}) = \prod_{i=1}^{n}P(x_i | x_1, x_2, \dots, x_{i-1}).$$

When there are probabilities, there are probability distributions, and this distribution is exactly the one GPTs are attempting to approximate. At their core, they are simply learning a categorical distribution over words and sampling from it, one step at a time. This is, of course, exactly the same function that fixed-length n-grams attempt to model. GPTs, however, are able leverage the power of self-attention for this task and have indeed proven to be remarkably good at it. 

Why did GPTs ditch encoders? Recall what encoders are used for in the first place: capturing contextual relationships between all words bidirectionally (previous and future words). GPTs don't want this, and they explicitly block access to future words as they generate in real-time using _masked (causal) self-attention_ in each transformer block. Think of the original task of the transformer of translating text from one language to another, where we first need to understand (encode) what the original language is saying and then produce an output (decode) in another language. With GPTs, we are just generating text and all we need is the context sequence we already have.

<p align="center">
    <img src="https://github.com/user-attachments/assets/ac1b1827-124c-423a-bfa7-58fb25c4fe0e" width="500px">
</p>

At the heart of the GPT architecture is the aforementioned masked self-attention mechanism, which is just like the vanilla self-attention we have seen previously but without forward connections-- each word can only attend to itself and to words before it. Consider the input "I like black coffee"; "black" only attends to "I", "like", and "black", not "coffee". To implement this no-looking-ahead mechanism, recall the attention formula $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} \right) V$. If we add a causal mask $M$, defined as 

$$
M = 
\begin{bmatrix}
0 & -\infty & \cdots & -\infty \\
0 & 0 & \cdots & -\infty \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix},
$$

to $\frac{QK^\top}{\sqrt{d_k}} $, we can effectively remove any bidirectionality and enforce a backwards-only context (remember that softmaxes flatten large negative numbers to 0).

<p align="center">
    <img src="https://github.com/user-attachments/assets/3ad356e5-f259-450e-8aab-4d9082c2a890" width="500px">
</p>

Besides the omission of the encoder and cross-attention, nothing is new about GPT architecture: word and positonal embeddings added together, passed through several decoder layers (masked self-attention and feed-forward networks), through a linear classifier that projects embeddings into wordspace, and finally through a softmax that outputs a categorical distribution over the vocabulary. As soon as the next word is sampled, it is concatenated to the original input and the process repeats.


## Scaling Up: GPT-2 and GPT-3
At this point, you might be asking what's so special about GPTs. It was certainly a bold and interesting decision to disregard the encoder in its architecture, but GPT did not invent the decoder. While the original 2018 GPT demonstrated that decoder-only transformers set a baseline of feasibility, it was the release of GPT-2 in the following year that truly captured the artificial intelligence community. 

Part of what made GPT-2 truly remarkable was not merely its scale (ten times the size of the original GPT) but the emergence of capabilities it had not been explicitly trained for without any direct fine-tuning or task-specific supervision. This "_zero-shot learning_" phenomenon was especially peculiar to researchers, who noticed that, given carefully-worded input, GPT-2 could generalize into performing all sorts of different tasks. Inputs that asked something like, "Translate from English to French: 'The book is on the table.'" would often yield a fluent French sentence, despite GPT-2 never being fine-tuned on a translation dataset. This surprising generalization hinted at the emergence of in-context learning-- the model was showing an emergent ability to adapt its behavior based on patterns present in the input. This suggested that, at sufficient scale, language models could learn not just facts and grammar, but how to actually do tasks from text alone.

This observation formed the basis of what are now called _scaling laws_, empirical trends that demonstrated that as the number of parameters, size of training set, and amount of compute were increased, a model's performance improved smoothly and predictably. It suddenly became clear that sheer scale was a primary driver of emergent capability.

Two relationships have been proposed to capture these scaling trends. The first is a formula for compute cost:

$$ C = C_0ND,$$

where $C$ is the total compute required, $N$ is the number of model parameters, $D$ is the number of training tokens (our data), and a constant $C_0$ that absorbs architectural and hardware-specific efficiency. The formula tells us that compute cost grows linearly with both model size and dataset size, meaning that if you want to double your model and double your data, you'll need roughly four times the compute.

The second is a power-law approximation for model loss:
$$ L = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_0. $$

Here, $L$ is the test-loss, and the formula shows diminishing returns: as you increase $N$ and $D$, the model's loss decreases following an inverse power-law. The constants $A$, $B$, and $L_0$ capture irreducible error and dataset/model specifics, while $\alpha$ and $\beta$  (usually ~0.07-0.1) describe how efficiently performance improves with scale.

These relationships revealed that model loss could be predictably reduced by scaling up either the number of parameters or the amount of training data. While the returns eventually slow, the formulas made it clear that with enough compute, one could always drive the loss lower by increasing either model size or dataset size.

<p align="center">
    <img src="https://github.com/user-attachments/assets/3586b06c-e27e-48d7-9ea8-2741516a5683" width="600">
</p>
<p align="center"><i>Scaling Laws for Neural Language Models</i>, Kaplan et. al 2020</p>

OpenAI put everything they had behind these findings and cranked up scale wildly. GPT-3, released in 2020, was over one-hundred times larger than GPT-2 at 175 billion parameters and was trained on over 300 billion tokens of data (more on tokens later). Emergent behavior began to appear with greater regularity: GPT-3 could perform multi-step arithmetic, solve analogy problems, write code, and complete SAT-style reading comprehension questions with minimal prompting. The ability to interpret and act on a task description with minimal instruction (zero- or one-shot) gave the model a special kind of flexibility that allowed it to adapt to new problems with nothing more than the right prompt phrasing needed. GPT-3 had not mastered each task individually, but it had been so relentlessly trained to predict the next token correctly across a vast range of contexts that it implicitly had to learn the abstract patterns of logic and instruction-following needed to solve different types of problems.

<p align="center">
    <img src="https://github.com/user-attachments/assets/d0530bf3-46db-4010-b19d-b720c9d7c1f8" width="500px">
</p>


## Pre-Training + Post-Training: How to Build a Large Language Model
The original GPT was trained in two stages: first, a large-scaled unsupervised language modeling phase, referred to as _pre-training_, and a second supervised _post-training_ phase on specific downstream tasks like question answering. 

### Pre-Training
In pre-training, LLMs are trained on a massive corpus of unstructured text data using the causal language modeling objective, which simply minimizes cross-entropy, the negative log-likelihood of the actual words across a dataset, where $x_t$ is the next word:

$$𝓛_{pre-train} = -∑_{t=1}^n \log P(x_t | x_1, x_2, ..., x_{t-1})$$

How is this computed in practice during training? $n$ is a hyperparameter called the _context window_ or _context length_, and it specifies how many words the model can attend to in a forward pass. Every piece of text that the LLM sees is broken up into these $n$-sized chunks, and $n$ is what specifies the maximum size of input to the model. Recall, however, that we compute the joint distribution $P(x_t | x_1, x_2, \dots, x_{t-1})$ by explicitly expanding it via the chain rule of probabilities-- it's important to remember that we are indeed also computing $P(x_2|x_1), P(x_3|x_1, x_2),$ and so on. While the context window is the maximum length of a sequence that the model sees, it indeed still calculates many other subsequences as it moves from left to right thanks to masked self-attention. Therefore, even though our context window might be 1,024 or 8,000 words, we know perfectly well how to deal with short inputs like "I like black coffee".

It's here where the magic of masked self-attention reveals its power during pre-training: it allows the model to process an entire sequence in parallel while still preserving the left-to-right structure required for generation. Because the "correct" next word is just the actual next word, the model doesn't need human-provided labels! This is what makes GPTs _self-supervised_-- they learn entirely from raw text with the data as its own supervisor. This is a crucial driver of implementing LLMs at scale that cannot be understated. Human-supervised learning would require assigning "correct" next words for every possible sequence in the English language, a task that would be not just impractical but virtually impossible. Other LLMs like BERT and T5 are perfectly capable of self-supervision too by giving the data the driver's seat.

So far, we've been referring to these $x$ vectors as words, but that's actually not how it's done in practice. Instead of operating at the level of full words, LLMs process the $x$ vectors in sequence as _tokens_, which are subword units often derived using Byte Pair Encoding (BPE). Words are split into smaller, reusable fragments (e.g. "un", "break", "able") that collectively cover language morphology. Although perhaps unintuitive to humans, _tokenization_ is hugely advantageous in providing extra granularity, reducing vocabulary size, and even handling rare or unseen words. Consider Oxford's 2022 Word of the Year: "goblinmode." For a model trained on, say, Wikipedia articles up to the year 2018, it's plausible to say that such a word might have never once appeared in the text data. Nevertheless, what tokenization essentially allows us to break down an input like "goblinmode" into "goblin" and "mode" tokens, which, when attended together might tell us about what "goblinmode" means. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd9ace6c-b1d5-4887-b852-27dac7280418" width="600">
</p>

The pre-training phase is the workhorse of building a large language model. By learning to predict the next word over billions of tokens, LLMs acquire a broad understanding of grammar, syntax, word meanings and relationships, facts, and long-range dependencies. The result of a converged pretrained model is the _base model_. While they might have the full knowledge of the dataset they were trained on baked into their weights, it's crucial to understand that *base models are not assistants nor chatbots.* If you download the weights of a base model like LLaMA-2-7B and provide a text input like, "What's 10 / 5?" you might be surprised to find that you won't get an answer like, "2" or "10 / 5 is 2." Instead, you're more likely to get something along the lines of, "We know the answer to this question is 2, this represents a basic mathematical truth. Mathematical truths are an example of objective knowledge that are necessarily true, and it is indeed impossible to think of the answer being anything other than 2. Kant argued that mathematical truths are synthetic a priori, and that these types of truths are..." and so on. Base models are nothing more than a compression of their datasets, and if you feed them input, they will just keep on sampling from learned probability distributions ad infinitum, certainly with no particular orientation towards helping you solve a problem. 


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

What happened here is that GPT-3 says to itself, "Ah, I've seen this exact question on the 2014 AP Computer Science A exam; I know what comes next!" and it proceeds to spit out what indeed came next, but not what we want from it. Base models this big can memorize very well, and actions need to be taken to move them away from this behavior.

You might have been surprised earlier to learn that GPT-3 was released in 2020 given that it was the model behind ChatGPT, which really caught on in late 2022 and early 2023. But the original GPT-3, powerful as it was, was still a base model, trained purely on predicting the next token in a sequence with no explicit goal beyond minimizing language modeling loss. While the model showed that it didn't need any fine-tuning, the gap between usability and capability brought the need for _post-training_, which helped modify behavior that made a powerful but clumsy base model into a genuinely useful tool.

### Post-Training
Post-training is the set of techniques applied after pre-training to make a model useful, aligned, and safe for human interaction. While zero-shot and one-shot learning largely did away with having to fine-tune to acquire different capabilities, there were still a lot of rough edges to attend to, as we saw with the coding question scenario.

_Instruction tuning_ was popularized by InstructGPT, a 2022 variant of GPT-3, that was fine-tuned on a dataset of just 12,000 samples of instruction-response pairs, simply a dataset of questions and answers. This was an important step in eliminating the unhelpful completions like literal A-B-C-D multiple choice outputs or continuations that ignored the user's intent. Instead, the model learned to interpret a wide range of natural prompts and provide more helpful responses.

Nevertheless, it was just a first step, and much more was needed to get toward responses that humans were satisfied with. Even when trained on good examples, the model could still produce technically plausible but subtly wrong answers, or respond in ways that were verbose and misaligned with user preferences. Tasks like language modeling are inherently subtle and ambiguous, and ultimately rely on human preference-- something difficult to quantify. How do you measure the quality of word choice or if an explanation is clear or condescending? There aren't objective metrics for these things, but humans can generally agree that phrases like, "Now this might be a bit hard for you to understand, but..." are indeed pretty condescending.

And so, humans were brought directly into the training process with _reinforcement learning from human feedback_ (RLHF), which allowed for preference to be formalized in a reinforcement learning setting. Human annotators ranked muliple completions for the same prompt that ultimately guided a reward model that guided the LLM toward outputs more consistent with human taste, helpfulness, and factuality-- this is exactly what LLMs are asking for if you ever recall being asked to choose between two outputted answers. Reinforcement learning and RLHF will be covered in more detail in the latter half of the course.

<p align="center">
    <img width="500" src="https://github.com/user-attachments/assets/6eecf2ad-ec3d-49b7-800c-963f0fd46b8d"/>
</p>
<p align="center"> 
    <i>InstructGPT</i>, Ouyang, et. al 2022</p>

RLHF laid the groundwork for post-training to build off of, and it continued to get better and better. Researchers realized that how you prompt the model could make a surprising difference in reasoning performance. In a technique called chain-of-thought prompting, simply adding intermediate steps—- like saying, “Let’s think step-by-step”—caused models to dramatically improve on arithmetic, logic, and common-sense reasoning tasks. These cues help the model unfold its internal "reasoning" across multiple tokens, rather than jumping straight to a potentially flawed conclusion. Arithmetic could still be clunky and prone to error in early LLMs until researchers made what in retrospect seemed an obvious move of just allowing models to use calculators; instead of trying to predict what followed "77 + 33 = ", those tokens were just replaced with an API call to a calculator. This strategy, dubbed _Toolformer_, was eventually extended to other external tools like search engines and translation APIs. Instead of trying to memorize everything, the model learns to delegate too.

Despite these advances, _hallucinations_ remain a major challenge. LLMs sometimes generate text that is syntactically and stylistically correct but factually wrong, such as inventing citations, quoting fake laws, or providing incorrect historical dates. These hallucinations stem from the fact that the model is fundamentally trained to predict likely sequences, not to verify facts. Even post-training cannot fully eliminate hallucinations, though techniques like retrieval-augmented generation, better RLHF reward models, and external tool use help reduce them.
