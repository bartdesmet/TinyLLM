using TinyLLM.Autograd;

namespace TinyLLM.Model;

/// <summary>
/// Base class for any neural-network building block. The contract is just:
/// "tell me all of your learnable scalar parameters". The optimizer takes the
/// flat list returned by <see cref="Parameters"/> on the root module and walks
/// it on every training step.
/// </summary>
public abstract class Module
{
    /// <summary>Enumerates every learnable <see cref="Value"/> owned by this module (recursively).</summary>
    public abstract IEnumerable<Value> Parameters();

    /// <summary>Convenience: zero out the gradient on every parameter.</summary>
    public void ZeroGrad()
    {
        foreach (var p in Parameters())
        {
            p.Grad = 0.0;
        }
    }
}
