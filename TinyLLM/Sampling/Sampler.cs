using TinyLLM.Model;
using TinyLLM.Tokenization;

namespace TinyLLM.Sampling;

/// <summary>
/// Autoregressive text sampler for a trained <see cref="GptModel"/>.
/// Repeatedly feeds the last <c>BlockSize</c> tokens through the model, picks
/// the next token from the resulting distribution (temperature + optional
/// top-k), appends it, and streams the decoded character.
/// </summary>
public static class Sampler
{
    /// <summary>
    /// Generates up to <paramref name="maxNewTokens"/> characters after
    /// <paramref name="prompt"/>.
    /// </summary>
    /// <param name="model">Trained model.</param>
    /// <param name="tokenizer">Tokenizer used during training.</param>
    /// <param name="prompt">Seed text (may be empty; an empty prompt is treated as a single newline if available, or the first vocab char).</param>
    /// <param name="maxNewTokens">How many tokens to emit.</param>
    /// <param name="temperature">Softens (>1) or sharpens (&lt;1) the distribution. Must be &gt; 0.</param>
    /// <param name="topK">If &gt; 0, restricts sampling to the top-k highest-probability tokens.</param>
    /// <param name="rng">RNG used for sampling.</param>
    /// <param name="onToken">Optional per-token streaming callback (e.g. <c>Console.Write</c>). Receives the decoded text for each newly generated token; for BPE this may be multiple characters at a time.</param>
    /// <returns>The concatenation of prompt + generated text.</returns>
    public static string Generate(
        GptModel model,
        ITokenizer tokenizer,
        string prompt,
        int maxNewTokens,
        double temperature,
        int topK,
        Random rng,
        Action<string>? onToken = null)
    {
        if (temperature <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "temperature must be > 0");
        }

        //
        // Running example
        //
        //   Character vocabulary in token ID order:
        //     ID:       0    1    2    3    4    5
        //     Text:    ' '  'T'  'e'  'h'  'n'  'r'
        //
        //   prompt = "The", BlockSize = 3, temperature = 0.5, topK = 3.
        //
        //   Encoding gives ids = [1, 3, 2]. IDs are lookup keys, not scores.
        //
        var ids = new List<int>(prompt.Length + maxNewTokens);
        if (prompt.Length > 0)
        {
            ids.AddRange(tokenizer.Encode(prompt));
        }
        else
        {
            // Seed with id 0 so the model has at least one input token.
            ids.Add(0);
        }

        for (var step = 0; step < maxNewTokens; step++)
        {
            //
            // Context is the text the model can see: at most the last BlockSize tokens.
            //
            //   Example: ids = [1, 3, 2] -> ctx = [1, 3, 2] ("The").
            //
            // Only ctx is cropped; ids retains the full sequence for the final output.
            //
            var ctxStart = Math.Max(0, ids.Count - model.BlockSize);
            var ctxLen = ids.Count - ctxStart;
            var ctx = new int[ctxLen];
            for (var i = 0; i < ctxLen; i++)
            {
                ctx[i] = ids[ctxStart + i];
            }

            //
            // Forward returns [context length, vocabulary size] logits: here [3, 6].
            //
            // A logit is a raw next-token score, not a probability.
            //
            // Example values, with columns in token ID order:
            //
            //              ' '    'T'    'e'    'h'    'n'    'r'
            //   "T"     [  0.0,  -1.0,   0.5,   3.0,  -0.5,   0.0 ]
            //   "Th"    [  0.5,  -1.0,   3.0,   0.0,   0.0,   0.5 ]
            //   "The"   [  2.0,   0.0,  -1.0,   0.5,   1.5,   1.0 ]
            //
            // Rows 0, 1, 2 predict after "T", "Th", "The" respectively; no row sees ahead.
            //
            // We need only the final row: [2.0, 0.0, -1.0, 0.5, 1.5, 1.0].
            //
            var logits = model.Forward(ctx);

            //
            // Temperature controls how adventurous sampling is, without changing rankings.
            //
            // Divide by 0.5: [2.0, 0.0, -1.0, 0.5, 1.5, 1.0] -> [4.0, 0.0, -2.0,  1.0, 3.0 , 2.0].
            // Divide by 2.0: [2.0, 0.0, -1.0, 0.5, 1.5, 1.0] -> [1.0, 0.0, -0.5, 0.25, 0.75, 0.5].
            //
            // For example, the gap between IDs 0 and 4 starts at 2.0 - 1.5 = 0.5:
            //
            //   temperature 0.5 doubles it to 1.0 ;
            //   temperature 2.0 halves  it to 0.25.
            //
            // Below 1 magnifies score gaps (more predictable); above 1 shrinks them
            // (more varied). At 1, we use the model's original distribution.
            //
            // The remaining running example continues with temperature = 0.5.
            //
            var V = model.VocabSize;
            var lastLogits = new double[V];
            for (var v = 0; v < V; v++)
            {
                lastLogits[v] = logits[ctxLen - 1, v].Data / temperature;
            }

            //
            // Top-k restricts sampling to the strongest candidates; 0 disables filtering.
            //
            // Example:
            //
            //   [4.0, 0.0, -2.0, 1.0, 3.0, 2.0]
            //
            //   with topK = 3 -> cutoff = 2
            //
            //   [4.0, -inf, -inf, -inf, 3.0, 2.0].
            //
            // IDs 0, 4, 5 survive: space, 'n', 'r'. exp(-infinity) will be zero.
            //
            //              ' '    'T'    'e'    'h'    'n'    'r'
            //   "The"   [  4.0,  -inf,  -inf,  -inf,   3.0,   2.0  ]
            //
            // Notes:
            //   1. The helper sorts a copy; lastLogits stays in token ID order, not score order.
            //   2. Ties at the cutoff can leave more than k candidates.
            //
            if (topK > 0 && topK < V)
            {
                var threshold = ComputeTopKThreshold(lastLogits, topK);
                for (var v = 0; v < V; v++)
                {
                    if (lastLogits[v] < threshold)
                    {
                        lastLogits[v] = double.NegativeInfinity;
                    }
                }
            }

            //
            // Softmax turns scores into probabilities that sum to 1:
            //
            //   p[v] = exp(score[v] - maxScore) / sum(exp(score - maxScore)).
            //
            // Subtracting the maximum avoids overflow without changing probabilities.
            //
            // Example (input after temperature scaling and top-k filtering):
            //   Original:     [4     ,  -inf,  -inf,  -inf,      3,      2]; maxScore = 4
            //   Shifted (-4): [0     ,  -inf,  -inf,  -inf,     -1,     -2]
            //   exp:          [1     ,     0,     0,     0, 0.3679, 0.1353]; sum = 1.5032
            //   p = exp / sum:[0.6652,     0,     0,     0, 0.2447, 0.0900] (rounded).
            //
            // The array is reused in place: lastLogits now becomes probabilities.
            //
            var maxL = double.NegativeInfinity;
            for (var v = 0; v < V; v++)
            {
                if (lastLogits[v] > maxL) maxL = lastLogits[v];
            }
            var sum = 0.0;
            for (var v = 0; v < V; v++)
            {
                lastLogits[v] = Math.Exp(lastLogits[v] - maxL);
                sum += lastLogits[v];
            }
            for (var v = 0; v < V; v++)
            {
                lastLogits[v] /= sum;
            }

            //
            // Multinomial sampling: a weighted random choice (one categorical draw).
            //
            // Think of a roulette wheel with segment widths equal to probabilities.
            //
            // The cumulative sums divide [0, 1) into ranges, in token ID order:
            //
            //   ID 0, space: [0.0000, 0.6652) -> about 66.5% of draws
            //   ID 4, 'n':   [0.6652, 0.9100) -> about 24.5% of draws
            //   ID 5, 'r':   [0.9100, 1.0000) -> about  9.0% of draws
            //
            // Example: r = 0.80 -> acc reaches r at ID 4 -> choose 'n'.
            //
            // Notes:
            // 1. No sorting is needed: range width, not position, determines likelihood.
            // 2. Unlike greedy decoding (pick the largest logit), favorites do not always win.
            //
            var r = rng.NextDouble();
            var acc = 0.0;
            var nextId = V - 1;
            for (var v = 0; v < V; v++)
            {
                acc += lastLogits[v];
                if (r <= acc)
                {
                    nextId = v;
                    break;
                }
            }

            //
            // Append ID 4: ids = [1, 3, 2, 4], decoding to "Then".
            //
            // Next iteration, BlockSize = 3 gives ctx = [3, 2, 4] ("hen").
            //
            // The output retains "T", but the model can no longer see it.
            //
            ids.Add(nextId);
            onToken?.Invoke(tokenizer.Decode([nextId]));
        }

        return tokenizer.Decode(ids);
    }

    private static double ComputeTopKThreshold(double[] logits, int k)
    {
        // Copy, partial-sort by descending value, return the k-th largest.
        var copy = (double[])logits.Clone();
        Array.Sort(copy);
        Array.Reverse(copy);
        return copy[k - 1];
    }
}
