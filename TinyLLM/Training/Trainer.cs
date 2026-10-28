using TinyLLM.Model;

namespace TinyLLM.Training;

/// <summary>
/// Drives the training loop:
///   for step in 1..numSteps:
///     pick a random contiguous window of (blockSize + 1) tokens
///     inputs  = window[0..blockSize-1]
///     targets = window[1..blockSize]      (next-token prediction)
///     loss = CrossEntropy(model.Forward(inputs), targets)
///     loss.Backward()
///     optimizer.Step()
///     optimizer.ZeroGrad()
/// </summary>
public sealed class Trainer
{
    private readonly GptModel _model;
    private readonly IOptimizer _optimizer;
    private readonly int[] _tokens;
    private readonly int _blockSize;
    private readonly Random _rng;

    public Trainer(GptModel model, IOptimizer optimizer, int[] tokens, Random rng)
    {
        if (tokens.Length < model.BlockSize + 1)
        {
            throw new ArgumentException(
                $"Need at least blockSize+1 ({model.BlockSize + 1}) tokens; got {tokens.Length}.",
                nameof(tokens));
        }
        _model = model;
        _optimizer = optimizer;
        _tokens = tokens;
        _blockSize = model.BlockSize;
        _rng = rng;
    }

    /// <summary>
    /// Runs <paramref name="numSteps"/> training updates and invokes
    /// <paramref name="onStep"/> after each step with (stepIndex, lossValue).
    /// </summary>
    public void Train(int numSteps, Action<int, double>? onStep = null)
    {
        var inputs = new int[_blockSize];
        var targets = new int[_blockSize];

        for (var step = 1; step <= numSteps; step++)
        {
            //
            // Pick a random contiguous window of (blockSize + 1) tokens from the training data.
            // The first blockSize tokens are the inputs, and the last blockSize tokens are the targets.
            //
            // Example:
            //
            //   BlockSize     = 3
            //   Character IDs = [' '=0, 'S'=1, 'T'=2, 'a'=3, 'e'=4, 'h'=5, 'i'=6, 'n'=7, 'p'=8, 'r'=9]
            //   Training data = "T h e   r a i n   i n   S p a i n"
            //   Tokens        = [2 5 4 0 9 3 6 7 0 6 7 0 1 8 3 6 7]
            //
            // Randomly pick a window of 4 tokens:
            //
            //   Tokens        = [2 5 4 0 9 3 6 7 0 6 7 0 1 8 3 6 7]
            //   Window        =         [9 3 6 7]
            //   Inputs        =          r a i
            //   Targets       =            a i n
            //
            var maxStart = _tokens.Length - _blockSize - 1;
            var start = _rng.Next(0, maxStart + 1);

            for (var i = 0; i < _blockSize; i++)
            {
                inputs[i] = _tokens[start + i];
                targets[i] = _tokens[start + i + 1];
            }

            //
            // Forward pass: compute the logits and loss for the current window of tokens.
            //
            var logits = _model.Forward(inputs);
            var loss = CrossEntropyLoss.Compute(logits, targets);

            //
            // Backward pass: compute gradients of the loss w.r.t. all model parameters.
            //
            _optimizer.ZeroGrad();
            loss.Backward();
            _optimizer.Step();

            onStep?.Invoke(step, loss.Data);
        }
    }
}
