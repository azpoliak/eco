# Efficient, Compositional, Order-Sensitive n-gram Embeddings


A suite for creating & evaluating phrasal embeddings via the `ECO` model based on [Efficient, Compositional, Order-Sensitive n-gram Embeddings](https://www.cs.jhu.edu/~apoliak1/papers/ECO--EACL-2017.pdf)



### Directories:
1. `data`: location of the data used to create and evaluate the ECO embeddings
⋅⋅1. The Skip-Embeddings can be downloaded from ...  
2. `evaluations`: data and scripts for different evaluation tasks to evaluate the embeddings.
1. `skipEmbeds`: the script used to generate the `ECO Skip-Embeddings` and vanilla `word2vec` embeddings.
⋅⋅1. We extended Debora Sujono's [python version of word2vec](https://github.com/deborausujono/word2vecpy).
⋅⋅2. We also have a local C version that is not tested.
⋅⋅3. The embeddings used in the paper and released were created using the python version. 


### Citation:

If you use our changes to the code or our skip-embeddings, please cite us:

@inproceedings{Poliak:2017EACL,
Title = {Efficient, Compositional, Order-sensitive n-gram Embeddings},
 Author = {Poliak, Adam and Rastogi, Pushpendre and Martin, M. Patrick and Van Durme, Benjamin},
 booktitle = {Proceedings of the 15th Conference of the European Chapter of the 
 Association for Computational Linguistics},
 Year = {2017},
 Publisher = {Association for Computational Linguistics},
 location = {Valencia, Spain}
}

 ### Errata:
 
 There is a typo in equations (6) and (7) in the EACL proceedings. The version found at https://www.cs.jhu.edu/~apoliak1/papers/ECO--EACL-2017.pdf has the correct equations.
