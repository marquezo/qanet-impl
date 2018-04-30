# QANet
Pytorch implementation of QAnet

**Authors (equal contribution)** : Patrice Béchard, Orlando Marquez, Benjamin Rosa, Nicholas Vachon

### Structure of QANet

The various modules presented below are found in the **qanet** folder. The various classes can be found in their respective file i.e. **InputEmbedding** can be found in the file **input_embedding.py**

* QANet
    - InputEmbedding
        + WordEmbedding
        + CharaterEmbedding
        + Highway
    - EmbeddingEncoder
        + EncoderBlock
    - ContextQueryAttention
    - ModelEncoder
        + EncoderBlock
    - Output

