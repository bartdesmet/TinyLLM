using TinyLLM.Autograd;

namespace TinyLLM.Training;

/// <summary>
/// Plain stochastic gradient descent: p -= learningRate * grad.
/// The trainer supplies gradients from randomly sampled token windows rather
/// than the whole dataset; this sampling is what makes the updates stochastic.
/// All parameters use the same learning rate, with no momentum or adaptive scaling.
/// </summary>
public sealed class Sgd(IList<Value> parameters, double learningRate) : IOptimizer
{
    public void Step()
    {
        //
        // Move opposite the loss gradient:
        //   - positive Grad decreases the parameter;
        //   - negative Grad increases it.
        //
        // The learning rate controls the step size:
        //   - too large can overshoot or diverge;
        //   - too small makes training slow.
        //
        for (var i = 0; i < parameters.Count; i++)
        {
            parameters[i].Data -= learningRate * parameters[i].Grad;
        }
    }

    public void ZeroGrad()
    {
        //
        // Backward() accumulates gradients with +=. Reset before each new backward
        // pass so the next Step() uses only the current sample's gradients.
        //
        for (var i = 0; i < parameters.Count; i++)
        {
            parameters[i].Grad = 0.0;
        }
    }
}
