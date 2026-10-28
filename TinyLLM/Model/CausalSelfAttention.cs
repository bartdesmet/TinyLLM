using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// Causal multi-head self-attention.
///
/// Intuition:
///
///   1. Causal self-attention lets each token gather information from itself and earlier tokens,
///      not future ones.
///   2. Multiple attention heads learn different ways of relating tokens. E.g.
///      a. Pronoun -> referent, e.g. "it" -> "dog"
///      b. Verb -> subject, e.g. "barked" -> "dog"
///      c. Matching punctuation, e.g. "(" -> ")"
///      d. Descriptive context, e.g. "dog" -> "furry"
///
/// Implementation:
///
///   For each token position t = 0..T-1 and each head h:
///     Query: q_h = Q(x_t)
///     Key:   k_h = K(x_s)  for s = 0..t  (causal)
///     Value: v_h = V(x_s)  for s = 0..t  (causal)
///
///     scores[s] = (q_h . k_h) / sqrt(headDim)
///     weights = softmax(scores)
///
///     head_out_t = sum_s weights[s] * v_h(s)
///
///   Then concat across heads and apply a final output projection.
/// </summary>
public sealed class CausalSelfAttention : Module
{
    public readonly int EmbedDim;
    public readonly int NumHeads;
    public readonly int HeadDim;

    //
    // Q = Query (the vector we dot against keys to get attention scores)
    //
    // "What am I looking for?" is the query.
    // E.g. "it" might be looking for a noun to resolve its antecedent.
    //
    private readonly Linear _qProj;

    //
    // K = Key   (the vectors we dot against query to get attention scores)
    //
    // "What I have" is the key.
    // E.g. "the cat" might be a key that matches the query "it".
    //
    private readonly Linear _kProj;

    //
    // V = Value (the vectors we will take a weighted sum of)
    //
    // "What I have" is also the value.
    // E.g. "the cat" might provide a value that matches the query "it".
    //
    private readonly Linear _vProj;

    //
    // Formula:
    //
    //  head_out_t = sum_s softmax((q_h . k_h) / sqrt(headDim))[s] * v_h(s)
    //
    private readonly Linear _outProj;
    private readonly double _scale;

    public CausalSelfAttention(int embedDim, int numHeads, Random rng)
    {
        if (embedDim % numHeads != 0)
        {
            throw new ArgumentException($"embedDim ({embedDim}) must be divisible by numHeads ({numHeads}).");
        }

        EmbedDim = embedDim;
        NumHeads = numHeads;
        HeadDim = embedDim / numHeads;
        _scale = 1.0 / Math.Sqrt(HeadDim);

        _qProj = new Linear(embedDim, embedDim, rng, useBias: false);
        _kProj = new Linear(embedDim, embedDim, rng, useBias: false);
        _vProj = new Linear(embedDim, embedDim, rng, useBias: false);
        _outProj = new Linear(embedDim, embedDim, rng, useBias: false);
    }

    /// <summary>
    /// <paramref name="x"/> is the sequence of token vectors:
    /// x[t] has length <see cref="EmbedDim"/>. Returns a same-shaped sequence.
    /// </summary>
    public Value[][] Forward(Value[][] x)
    {
        var T = x.Length;

        // 1) Project every position once into Q, K, V (shape [T, EmbedDim]).
        var q = new Value[T][];
        var k = new Value[T][];
        var v = new Value[T][];
        for (var t = 0; t < T; t++)
        {
            q[t] = _qProj.Forward(x[t]);
            k[t] = _kProj.Forward(x[t]);
            v[t] = _vProj.Forward(x[t]);
        }

        // 2) For each head and each query position, compute attention output.
        // We'll fill outConcat[t] of length EmbedDim, then run output projection.
        var outConcat = new Value[T][];
        for (var t = 0; t < T; t++)
        {
            outConcat[t] = new Value[EmbedDim];
        }

        for (var h = 0; h < NumHeads; h++)
        {
            var headOffset = h * HeadDim;

            for (var t = 0; t < T; t++)
            {
                // a) raw scores against all causal positions s = 0..t
                var scores = new Value[t + 1];
                for (var s = 0; s <= t; s++)
                {
                    Value dot = 0.0;
                    for (var d = 0; d < HeadDim; d++)
                    {
                        dot = dot + q[t][headOffset + d] * k[s][headOffset + d];
                    }
                    scores[s] = dot * _scale;
                }

                // b) numerically stable softmax: subtract the (data-only) max
                //    so that exp() doesn't overflow. We treat the max as a
                //    constant (the gradient through "max" is just an indicator,
                //    and shifting all logits by a constant doesn't change the
                //    softmax mathematically, so detaching is safe).
                var maxData = scores[0].Data;
                for (var s = 1; s <= t; s++)
                {
                    if (scores[s].Data > maxData) maxData = scores[s].Data;
                }

                var expScores = new Value[t + 1];
                Value denom = 0.0;
                for (var s = 0; s <= t; s++)
                {
                    var shifted = scores[s] - maxData;
                    var e = shifted.Exp();
                    expScores[s] = e;
                    denom = denom + e;
                }
                var invDenom = denom.Pow(-1.0);

                // c) weighted sum of V into the head slice of outConcat[t].
                for (var d = 0; d < HeadDim; d++)
                {
                    Value sum = 0.0;
                    for (var s = 0; s <= t; s++)
                    {
                        var w = expScores[s] * invDenom;
                        sum = sum + w * v[s][headOffset + d];
                    }
                    outConcat[t][headOffset + d] = sum;
                }
            }
        }

        // 3) Final output projection (per position).
        var y = new Value[T][];
        for (var t = 0; t < T; t++)
        {
            y[t] = _outProj.Forward(outConcat[t]);
        }
        return y;
    }

    public override IEnumerable<Value> Parameters()
    {
        foreach (var p in _qProj.Parameters()) yield return p;
        foreach (var p in _kProj.Parameters()) yield return p;
        foreach (var p in _vProj.Parameters()) yield return p;
        foreach (var p in _outProj.Parameters()) yield return p;
    }
}
