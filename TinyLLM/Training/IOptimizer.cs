using TinyLLM.Autograd;

namespace TinyLLM.Training;

public interface IOptimizer
{
    /// <summary>Apply one update step using current gradients.</summary>
    void Step();

    /// <summary>Reset all parameter gradients to zero.</summary>
    void ZeroGrad();
}
