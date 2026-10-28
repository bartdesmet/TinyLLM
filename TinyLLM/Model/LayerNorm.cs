using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// Per-token layer normalization.
///
/// Given an input vector x of length <c>dim</c>, it
///   - recenters features around their mean, and
///   - scales them by their spread,
/// keeping activations on a more consistent scale as they pass through the network.
///
/// Learnable gamma and beta then rescale and shift each feature so the model can
/// adapt the result.
/// 
/// Intuition:
///   Successive attention and feed-forward layers can change the magnitude of the numbers.
///   Layer norm gives the next computation a more predictable input scale, so it can focus
///   on the pattern of features rather than accommodate wildly varying magnitudes.
///
/// Computes y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta.
///
/// All operations go through <see cref="Value"/> so the mean, variance, and
/// normalization terms participate in the autograd graph.
/// </summary>
public sealed class LayerNorm : Module
{
    public readonly int Dim;
    public readonly Value[] Gamma;
    public readonly Value[] Beta;
    private readonly double _eps;

    public LayerNorm(int dim, double eps = 1e-5)
    {
        Dim = dim;
        _eps = eps;
        Gamma = new Value[dim];
        Beta = new Value[dim];
        for (var i = 0; i < dim; i++)
        {
            //
            // Initial scaling/shift are the identity.
            //

            Gamma[i] = new Value(1.0, "gamma");
            Beta[i] = new Value(0.0, "beta");
        }
    }

    public Value[] Forward(Value[] x)
    {
        if (x.Length != Dim)
        {
            throw new ArgumentException($"Expected input of length {Dim}, got {x.Length}.", nameof(x));
        }

        // mean
        Value mean = 0.0;
        for (var i = 0; i < Dim; i++)
        {
            mean = mean + x[i];
        }
        mean = mean * (1.0 / Dim);

        // variance (biased, matching the usual LN convention)
        Value var = 0.0;
        var centered = new Value[Dim];
        for (var i = 0; i < Dim; i++)
        {
            var d = x[i] - mean;
            centered[i] = d;
            var = var + d * d;
        }
        var = var * (1.0 / Dim);

        // invStd = 1 / sqrt(var + eps) = (var + eps)^-0.5
        var invStd = (var + _eps).Pow(-0.5);

        var y = new Value[Dim];
        for (var i = 0; i < Dim; i++)
        {
            y[i] = Gamma[i] * (centered[i] * invStd) + Beta[i];
        }

        return y;
    }

    public override IEnumerable<Value> Parameters()
    {
        for (var i = 0; i < Dim; i++)
        {
            yield return Gamma[i];
        }

        for (var i = 0; i < Dim; i++)
        {
            yield return Beta[i];
        }
    }
}
