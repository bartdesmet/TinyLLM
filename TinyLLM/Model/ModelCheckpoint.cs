using System.Text.Json;
using TinyLLM.Tokenization;

namespace TinyLLM.Model;

/// <summary>
/// Saves inference checkpoints: architecture, tokenizer recipe, and parameter values.
/// The embedded corpus reproduces the deterministic tokenizer without an external file.
/// Optimizer moments, gradients, and RNG state are not saved; this is not training resumption.
/// </summary>
public static class ModelCheckpoint
{
    private const int FormatVersion = 1;

    public static void Save(string path, GptModel model, ITokenizer tokenizer, string corpus)
    {
        var tokenizerKind = tokenizer switch
        {
            CharTokenizer => nameof(CharTokenizer),
            BytePairTokenizer => nameof(BytePairTokenizer),
            _ => throw new ArgumentException("Unsupported tokenizer type.", nameof(tokenizer)),
        };
        if (model.VocabSize != tokenizer.VocabSize)
        {
            throw new ArgumentException("Model and tokenizer vocabulary sizes must match.", nameof(tokenizer));
        }

        var data = new CheckpointData(
            FormatVersion, model.BlockSize, model.EmbedDim, model.NumHeads, model.NumLayers,
            model.VocabSize, tokenizerKind, corpus, model.Parameters().Select(p => p.Data).ToArray());
        var rebuiltTokenizer = CreateTokenizer(data);
        if (rebuiltTokenizer.DebugString() != tokenizer.DebugString())
        {
            throw new ArgumentException("The corpus does not reproduce the supplied tokenizer.", nameof(corpus));
        }
        if (data.Weights.Any(w => !double.IsFinite(w)))
        {
            throw new InvalidDataException("Cannot save non-finite model weights.");
        }

        // Serialize before opening the destination so serialization failures cannot truncate it.
        var json = JsonSerializer.Serialize(data);
        File.WriteAllText(path, json);
    }

    public static (GptModel Model, ITokenizer Tokenizer) Load(string path)
    {
        var data = JsonSerializer.Deserialize<CheckpointData>(File.ReadAllText(path))
            ?? throw new InvalidDataException("Checkpoint is empty.");
        if (data.Version != FormatVersion)
        {
            throw new InvalidDataException($"Unsupported checkpoint version: {data.Version}.");
        }
        if (data.BlockSize <= 0 || data.EmbedDim <= 0 || data.NumHeads <= 0 ||
            data.NumLayers <= 0 || data.VocabSize <= 0 || data.EmbedDim % data.NumHeads != 0)
        {
            throw new InvalidDataException("Checkpoint contains invalid model dimensions.");
        }
        if (string.IsNullOrEmpty(data.Corpus) || data.Weights is null ||
            data.Weights.Any(w => !double.IsFinite(w)))
        {
            throw new InvalidDataException("Checkpoint is missing its corpus or contains invalid weights.");
        }

        var tokenizer = CreateTokenizer(data);
        if (tokenizer.VocabSize != data.VocabSize)
        {
            throw new InvalidDataException("Checkpoint tokenizer vocabulary size does not match the model.");
        }

        var model = new GptModel(data.VocabSize, data.BlockSize, data.EmbedDim,
            data.NumHeads, data.NumLayers, new Random(0));
        var parameters = model.Parameters().ToArray();
        if (parameters.Length != data.Weights.Length)
        {
            throw new InvalidDataException(
                $"Checkpoint has {data.Weights.Length} weights; expected {parameters.Length}.");
        }
        for (var i = 0; i < parameters.Length; i++)
        {
            parameters[i].Data = data.Weights[i];
        }
        return (model, tokenizer);
    }

    private static ITokenizer CreateTokenizer(CheckpointData data) => data.TokenizerKind switch
    {
        nameof(CharTokenizer) => CharTokenizer.Build(data.Corpus),
        nameof(BytePairTokenizer) => BytePairTokenizer.Train(data.Corpus, data.VocabSize),
        _ => throw new InvalidDataException($"Unknown checkpoint tokenizer: {data.TokenizerKind}."),
    };

    private sealed record CheckpointData(
        int Version,
        int BlockSize,
        int EmbedDim,
        int NumHeads,
        int NumLayers,
        int VocabSize,
        string TokenizerKind,
        string Corpus,
        double[] Weights);
}
