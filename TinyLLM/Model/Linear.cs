using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// Fully-connected layer: y = x * W + b.
/// Stored as <c>W[in, out]</c> and <c>b[out]</c> so that <c>Forward(x[in])</c>
/// returns a fresh array <c>y[out]</c>.
/// </summary>
public sealed class Linear : Module
{
    public readonly int InFeatures;
    public readonly int OutFeatures;
    public readonly Value[,] W; // [in, out]
    public readonly Value[] B;  // [out]
    public readonly bool UseBias;

    public Linear(int inFeatures, int outFeatures, Random rng, bool useBias = true, double initStdDev = 0.02)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        UseBias = useBias;

        W = new Value[inFeatures, outFeatures];
        for (var i = 0; i < inFeatures; i++)
        {
            for (var j = 0; j < outFeatures; j++)
            {
                W[i, j] = new Value(rng.NextGaussian(0.0, initStdDev), "W");
            }
        }

        B = new Value[outFeatures];
        for (var j = 0; j < outFeatures; j++)
        {
            B[j] = new Value(0.0, "b");
        }
    }

    public Value[] Forward(Value[] x)
    {
        if (x.Length != InFeatures)
        {
            throw new ArgumentException($"Expected input of length {InFeatures}, got {x.Length}.", nameof(x));
        }

        var y = new Value[OutFeatures];

        //
        // This is matrix-vector multiplication y = x * W, followed by bias addition:
        //
        //   y[j] = sum_i x[i] * W[i, j] + b[j]
        //
        // The scalar loops build the autograd graph.
        //

        for (var j = 0; j < OutFeatures; j++)
        {
            Value sum = UseBias ? B[j] : (Value)0.0;

            for (var i = 0; i < InFeatures; i++)
            {
                sum = sum + x[i] * W[i, j];
            }

            y[j] = sum;
        }

        return y;
    }

    public override IEnumerable<Value> Parameters()
    {
        for (var i = 0; i < InFeatures; i++)
        {
            for (var j = 0; j < OutFeatures; j++)
            {
                yield return W[i, j];
            }
        }

        if (UseBias)
        {
            for (var j = 0; j < OutFeatures; j++)
            {
                yield return B[j];
            }
        }
    }
}
