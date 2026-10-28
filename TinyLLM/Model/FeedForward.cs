using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// Position-wise feed-forward network (FFN) used inside each transformer block:
///
///   y = Linear2(GELU(Linear1(x)))
///
/// with an inner hidden size of 4 * embedDim (the standard GPT ratio).
/// 
/// Illustration:
///
///        Input
///          │
///          ▼
///       Linear
///          ▼
///        GELU
///          ▼
///       Linear
///          │
///          ▼
///       Output
///
/// 1. Each Linear operation is a learned matrix multiplication plus an optional bias: (Wx+b).
///
/// 2. GELU (Gaussian Error Linear Unit) is an "activation function" that smoothly suppresses
///    negative values while largely preserving positive ones.
///
/// Without the nonlinearity of GELU, the two linear layers could collapse into a single linear 
/// transformation—the extra layer wouldn’t add expressive power.
///
/// Intuition:
///   Attention gathers relevant context from other tokens; the feed-forward network processes
///   the resulting information at each token. The same network weights are used at every position,
///   but each transformer block has its own weights.
/// </summary>
public sealed class FeedForward : Module
{
    public readonly Linear Fc1;
    public readonly Linear Fc2;

    public FeedForward(int embedDim, Random rng, int hiddenMultiplier = 4)
    {
        var hidden = embedDim * hiddenMultiplier;

        // x * W1 + b1, where W1 is [embedDim, hidden]
        Fc1 = new Linear(inFeatures: embedDim, outFeatures: hidden, rng);

        // h * W2 + b2, where W2 is [hidden, embedDim]
        Fc2 = new Linear(inFeatures: hidden, outFeatures: embedDim, rng);
    }

    public Value[] Forward(Value[] x)
    {
        // h = x * W1 + b1, where W1 is [embedDim, hidden]
        var h = Fc1.Forward(x);

        // h = GELU(h) elementwise, where GELU is the Gaussian Error Linear Unit activation function.
        for (var i = 0; i < h.Length; i++)
        {
            h[i] = h[i].Gelu();
        }

        // y = h * W2 + b2, where W2 is [hidden, embedDim]
        return Fc2.Forward(h);
    }

    public override IEnumerable<Value> Parameters()
    {
        foreach (var p in Fc1.Parameters()) yield return p;
        foreach (var p in Fc2.Parameters()) yield return p;
    }
}
