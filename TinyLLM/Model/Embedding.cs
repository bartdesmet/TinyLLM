using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// Lookup table from integer id -> learnable vector of length <see cref="EmbedDim"/>.
/// Used for both token embeddings and (separately) positional embeddings.
/// </summary>
public sealed class Embedding : Module
{
    public readonly int NumEmbeddings;
    public readonly int EmbedDim;
    public readonly Value[,] Table; // [numEmbeddings, embedDim]

    public Embedding(int numEmbeddings, int embedDim, Random rng, double initStdDev = 0.02)
    {
        NumEmbeddings = numEmbeddings;
        EmbedDim = embedDim;
        Table = new Value[numEmbeddings, embedDim];

        for (var i = 0; i < numEmbeddings; i++)
        {
            for (var j = 0; j < embedDim; j++)
            {
                Table[i, j] = new Value(rng.NextGaussian(0.0, initStdDev), "emb");
            }
        }
    }

    /// <summary>Returns a fresh array referencing the row for <paramref name="id"/>.</summary>
    public Value[] Forward(int id)
    {
        if ((uint)id >= (uint)NumEmbeddings)
        {
            throw new ArgumentOutOfRangeException(nameof(id), id, "Id outside embedding table.");
        }

        var row = new Value[EmbedDim];

        for (var j = 0; j < EmbedDim; j++)
        {
            row[j] = Table[id, j];
        }

        return row;
    }

    public override IEnumerable<Value> Parameters()
    {
        for (var i = 0; i < NumEmbeddings; i++)
        {
            for (var j = 0; j < EmbedDim; j++)
            {
                yield return Table[i, j];
            }
        }
    }
}
