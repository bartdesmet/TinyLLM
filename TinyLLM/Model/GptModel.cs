using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// A toy decoder-only Transformer (a.k.a. "GPT" style) language model.
///
/// Forward(ids[T]) returns logits[T, vocab]: at every position t the model
/// predicts the distribution over the next token, conditioned on ids[0..t].
///
/// Because the autograd engine allocates one heap node per scalar op, the
/// hyperparameters here should be kept very small (e.g. embedDim=16,
/// numHeads=2, numLayers=1, blockSize=8).
///
/// Use <see cref="ModelCheckpoint"/> to persist the architecture, tokenizer recipe,
/// and parameter Data values for inference in another process.
/// </summary>
public sealed class GptModel : Module
{
    public readonly int VocabSize;
    public readonly int BlockSize;
    public readonly int EmbedDim;
    public readonly int NumHeads;
    public readonly int NumLayers;

    public readonly Embedding TokEmbedding;
    public readonly Embedding PosEmbedding;
    public readonly TransformerBlock[] Blocks;
    public readonly LayerNorm FinalLn;
    public readonly Linear LmHead;

    public GptModel(
        int vocabSize,
        int blockSize,
        int embedDim,
        int numHeads,
        int numLayers,
        Random rng)
    {
        VocabSize = vocabSize;
        BlockSize = blockSize;
        EmbedDim = embedDim;
        NumHeads = numHeads;
        NumLayers = numLayers;

        TokEmbedding = new Embedding(vocabSize, embedDim, rng);
        PosEmbedding = new Embedding(blockSize, embedDim, rng);

        Blocks = new TransformerBlock[numLayers];
        for (var i = 0; i < numLayers; i++)
        {
            Blocks[i] = new TransformerBlock(embedDim, numHeads, rng);
        }

        FinalLn = new LayerNorm(embedDim);
        LmHead = new Linear(embedDim, vocabSize, rng, useBias: false);
    }

    /// <summary>
    /// Forward pass over a single sequence of token ids of length T &lt;= BlockSize.
    /// Returns logits as a 2D array of shape [T, VocabSize].
    /// </summary>
    public Value[,] Forward(int[] ids)
    {
        var T = ids.Length;
        if (T == 0)
        {
            throw new ArgumentException("ids must contain at least one token.", nameof(ids));
        }
        if (T > BlockSize)
        {
            throw new ArgumentException($"Sequence length {T} exceeds block size {BlockSize}.", nameof(ids));
        }

        //
        // 1) Token + positional embeddings.
        //

        var x = new Value[T][];

        for (var t = 0; t < T; t++)
        {
            var tok = TokEmbedding.Forward(ids[t]);
            var pos = PosEmbedding.Forward(t);

            var row = new Value[EmbedDim];

            for (var d = 0; d < EmbedDim; d++)
            {
                row[d] = tok[d] + pos[d];
            }

            x[t] = row;
        }

        //
        // 2) Transformer blocks.
        //

        foreach (var block in Blocks)
        {
            x = block.Forward(x);
        }

        //
        // 3) Final LayerNorm + LM head.
        //

        var logits = new Value[T, VocabSize];

        for (var t = 0; t < T; t++)
        {
            //
            // Final layer normalization
            //  - Normalizes each token's hidden representation across its features.
            //

            var normed = FinalLn.Forward(x[t]);

            //
            // "Language Modeling Head"
            //  - Linear projection from hidden representation to vocabulary logits.
            //

            var row = LmHead.Forward(normed);

            //
            // Logits
            //  - Raw scores for each vocabulary token, before softmax to produce probabilities.
            //

            for (var v = 0; v < VocabSize; v++)
            {
                logits[t, v] = row[v];
            }
        }

        return logits;
    }

    public override IEnumerable<Value> Parameters()
    {
        foreach (var p in TokEmbedding.Parameters()) yield return p;
        foreach (var p in PosEmbedding.Parameters()) yield return p;

        foreach (var block in Blocks)
        {
            foreach (var p in block.Parameters()) yield return p;
        }

        foreach (var p in FinalLn.Parameters()) yield return p;
        foreach (var p in LmHead.Parameters()) yield return p;
    }
}
