using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// One transformer block, GPT-style "pre-LayerNorm" variant:
///
///   x = x + CausalSelfAttention(LayerNorm1(x))
///   x = x + FeedForward(LayerNorm2(x))
///
/// Illustration:
/// 
///   Input token vectors: x
///          │
///          ├──────────────────────┐
///          ▼                      │
///      LayerNorm                  │
///          │                      │
///          ▼                      │
///      Causal Self-Attention      │
///          │                      │
///          ▼                      │
///         Add ◄───────────────────┘
///          │
///          ├──────────────────────┐
///          ▼                      │
///      LayerNorm                  │
///          │                      │
///          ▼                      │
///      Feed-Forward               │
///          │                      │
///          ▼                      │
///         Add ◄───────────────────┘
///          │
///          ▼
///  Output token vectors
///
/// Note:
///   Pre-LN is more numerically stable at small scale than the original
///   post-LN formulation from "Attention Is All You Need".
/// </summary>
public sealed class TransformerBlock : Module
{
    public readonly LayerNorm Ln1;
    public readonly CausalSelfAttention Attn;
    public readonly LayerNorm Ln2;
    public readonly FeedForward Ffn;

    public TransformerBlock(int embedDim, int numHeads, Random rng)
    {
        Ln1 = new LayerNorm(embedDim);
        Attn = new CausalSelfAttention(embedDim, numHeads, rng);
        Ln2 = new LayerNorm(embedDim);
        Ffn = new FeedForward(embedDim, rng);
    }

    public Value[][] Forward(Value[][] x)
    {
        var T = x.Length;
        var embedDim = x[0].Length;

        // Sub-block 1: attention with residual.
        var lnX = new Value[T][];
        for (var t = 0; t < T; t++)
        {
            lnX[t] = Ln1.Forward(x[t]);
        }
        var attnOut = Attn.Forward(lnX);

        var afterAttn = new Value[T][];
        for (var t = 0; t < T; t++)
        {
            var row = new Value[embedDim];
            for (var d = 0; d < embedDim; d++)
            {
                row[d] = x[t][d] + attnOut[t][d];
            }
            afterAttn[t] = row;
        }

        // Sub-block 2: FFN with residual.
        var afterFfn = new Value[T][];
        for (var t = 0; t < T; t++)
        {
            var lnXt = Ln2.Forward(afterAttn[t]);
            var ffnOut = Ffn.Forward(lnXt);
            var row = new Value[embedDim];
            for (var d = 0; d < embedDim; d++)
            {
                row[d] = afterAttn[t][d] + ffnOut[d];
            }
            afterFfn[t] = row;
        }

        return afterFfn;
    }

    public override IEnumerable<Value> Parameters()
    {
        foreach (var p in Ln1.Parameters()) yield return p;
        foreach (var p in Attn.Parameters()) yield return p;
        foreach (var p in Ln2.Parameters()) yield return p;
        foreach (var p in Ffn.Parameters()) yield return p;
    }
}
