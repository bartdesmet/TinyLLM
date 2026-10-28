namespace TinyLLM.Tokenization;

/// <summary>
/// Minimal contract every tokenizer must satisfy so the model, sampler, and
/// trainer can be agnostic to the underlying tokenization scheme
/// (character-level, byte-level BPE, etc.).
/// </summary>
public interface ITokenizer
{
    /// <summary>Number of distinct token ids the tokenizer can produce.</summary>
    int VocabSize { get; }

    /// <summary>Encodes <paramref name="text"/> into a sequence of token ids.</summary>
    /// <exception cref="ArgumentException">Thrown if <paramref name="text"/> contains symbols outside the vocabulary.</exception>
    int[] Encode(string text);

    /// <summary>Decodes a sequence of token ids back into the original text.</summary>
    string Decode(IEnumerable<int> ids);

    /// <summary>
    /// Returns one ID-to-text mapping per line, ordered by ID, with quoted text
    /// and escaped whitespace. Tokens that are not complete UTF-8 sequences
    /// are displayed as hexadecimal bytes instead.
    /// </summary>
    string DebugString();
}
