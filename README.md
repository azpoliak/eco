Efficient, Compositional, Order-Sensitive n-gram Embeddings
---

A suite for creating & evaluating phrasal embeddings via the `ECO` model based on [Efficient, Compositional, Order-Sensitive n-gram Embeddings](http://www.cs.jhu.edu/~apoliak1/papers/ECO--EACL-2017.pdf)



Directories:
-----------
1. `data`: location of the data used to create and evaluate the ECO embeddings
⋅⋅1. The Skip-Embeddings can be downloaded from ...  
2. `evaluations`: data and scripts for different evaluation tasks to evaluate the embeddings.
1. `skipEmbeds`: the script used to generate the `ECO Skip-Embeddings` and vanilla `word2vec` embeddings.
⋅⋅1. We extended Debora Sujono's [python version of word2vec](https://github.com/deborausujono/word2vecpy).
⋅⋅2. We also have a local C version that is not tested.
⋅⋅3. The embeddings used in the paper and released were created using the python version. 


