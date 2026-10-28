using TinyLLM.Autograd;

namespace TinyLLM.Training;

/// <summary>
/// AdamW optimizer (Loshchilov &amp; Hutter, 2017).
///
/// Adam stands for Adaptive Moment Estimation; W refers to weight decay.
///
/// It tracks moving averages of gradients (first moment, providing momentum)
/// and squared gradients (second moment, adapting the step size per parameter).
///
/// Unlike plain SGD, updates use this gradient history rather than just the
/// current gradient. Weight decay is decoupled: it shrinks parameters toward
/// zero separately, without mixing the decay term into the moment estimates.
/// </summary>
public sealed class AdamW(
    IList<Value> parameters,
    double learningRate = 1e-3,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double eps = 1e-8,
    double weightDecay = 0.0) : IOptimizer
{
    // First moment (moving averages of gradients, i.e. mean).
    private readonly double[] _m = new double[parameters.Count];

    // Second moment (moving averages of squared gradients, i.e. uncentered variance).
    private readonly double[] _v = new double[parameters.Count];

    // Time step counter for bias correction of the moment estimates.
    private int _t = 0;

    public void Step()
    {
        _t++;
        var biasCorr1 = 1.0 - Math.Pow(beta1, _t);
        var biasCorr2 = 1.0 - Math.Pow(beta2, _t);

        for (var i = 0; i < parameters.Count; i++)
        {
            var p = parameters[i];
            var g = p.Grad;

            // Smooth the gradient and its square; larger betas retain more history.
            _m[i] = beta1 * _m[i] + (1.0 - beta1) * g;
            _v[i] = beta2 * _v[i] + (1.0 - beta2) * g * g;

            // Correct the early estimates for their zero initialization.
            var mHat = _m[i] / biasCorr1;
            var vHat = _v[i] / biasCorr2;

            // Decoupled weight decay.
            if (weightDecay != 0.0)
            {
                p.Data -= learningRate * weightDecay * p.Data;
            }

            // Move opposite the smoothed gradient, scaled by its recent magnitude.
            // Epsilon keeps the denominator nonzero when the second moment is zero.
            p.Data -= learningRate * mHat / (Math.Sqrt(vHat) + eps);
        }
    }

    public void ZeroGrad()
    {
        for (var i = 0; i < parameters.Count; i++)
        {
            parameters[i].Grad = 0.0;
        }
    }
}
