using TinyLLM.Autograd;

namespace TinyLLM.Training;

/// <summary>
/// Average cross-entropy loss over a sequence of next-token predictions.
///
/// Given logits of shape [T, V] and integer targets of length T, computes
///   loss = -(1/T) * sum_t log(softmax(logits[t])[target[t]])
/// using the numerically stable log-sum-exp trick.
/// </summary>
public static class CrossEntropyLoss
{
    public static Value Compute(Value[,] logits, int[] targets)
    {
        var T = logits.GetLength(0);
        var V = logits.GetLength(1);
        if (targets.Length != T)
        {
            throw new ArgumentException(
                $"targets length ({targets.Length}) must match logits time dim ({T}).",
                nameof(targets));
        }

        //
        // Total negative log-likelihood over the sequence; divide by T at the end to get average.
        //

        Value totalNll = 0.0;

        for (var t = 0; t < T; t++)
        {
            var target = targets[t];
            if ((uint)target >= (uint)V)
            {
                throw new ArgumentOutOfRangeException(nameof(targets), target, "target id out of vocab range");
            }

            totalNll += ComputeTokenNll(logits, t, target, V);
        }

        //
        // Average negative log-likelihood over the sequence.
        //
        // NB: Using 1.0 / T is slightly more efficient than using / on Value which uses T.Pow(-1),
        //     but 1.0 / T is a constant, so we don't need a node in the graph for this.
        //

        return totalNll * (1.0 / T);
    }

    private static Value ComputeTokenNll(Value[,] logits, int timeStep, int target, int vocabSize)
    {
        //
        // Find max over this row (use Data only; shifting all logits by a
        // constant is mathematically a no-op for softmax).
        //

        var maxData = logits[timeStep, 0].Data;
        for (var v = 1; v < vocabSize; v++)
        {
            if (logits[timeStep, v].Data > maxData) maxData = logits[timeStep, v].Data;
        }

        //
        // sumExp = sum_v exp(logits[t,v] - maxData)
        //

        Value sumExp = 0.0;
        for (var v = 0; v < vocabSize; v++)
        {
            sumExp = sumExp + (logits[timeStep, v] - maxData).Exp();
        }
        var logSumExp = sumExp.Log() + maxData;
        
        //
        // -log p(target) = -(logits[target] - logSumExp) = logSumExp - logits[target]
        //

        return logSumExp - logits[timeStep, target];
    }
}
