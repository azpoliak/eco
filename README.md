#Efficient, Compositional, Order-Sensitive n-gram Embeddings

A suite for creating & evaluating phrasal embeddings via the `ECO` model based on [Efficient, Compositional, Order-Sensitive n-gram Embeddings](http://www.cs.jhu.edu/~apoliak1/papers/ECO--EACL-2017.pdf)

Directories:
-----------
1. `data`: location of the data used to create and evaluate the ECO embeddings
2. `evaluations`: data and scripts for different evaluation tasks to evaluate the embeddings.
1. `genWordEmbeds`: the scripts used to generate the `ECO Skip-Embeddings` and vanilla `word2vec` embeddings.
⋅⋅* We have an implementation in python and C. The python version is an extension of Debora Sujono's [python version of word2vec](https://github.com/deborausujono/word2vecpy) and the C version is an extension of the original word2vec. The C version sometimes seg faults and we have not fixed it. The embeddings released were created using the python version. 

4. h
5. 


