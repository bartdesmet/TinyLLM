namespace TinyLLM.Tokenization;

/// <summary>
/// The simplest possible tokenizer: one id per distinct character in the
/// training corpus. Tokens are deterministic across runs (sorted by char code)
/// so a model trained in one process can be re-used in another with the same
/// corpus.
/// </summary>
public sealed class CharTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _charToId;
    private readonly char[] _idToChar;

    private CharTokenizer(Dictionary<char, int> charToId, char[] idToChar)
    {
        _charToId = charToId;
        _idToChar = idToChar;
    }

    public int VocabSize => _idToChar.Length;

    /// <inheritdoc />
    public string DebugString()
    {
        var sb = new System.Text.StringBuilder();
        for (var id = 0; id < VocabSize; id++)
        {
            sb.AppendLine($"{id} -> {TokenizerDebugFormatter.Quote(_idToChar[id].ToString())}");
        }
        return sb.ToString();
    }

    /// <summary>Returns the character corresponding to the given id.</summary>
    public char IdToChar(int id) => _idToChar[id];

    /// <summary>Builds a tokenizer from the unique characters of <paramref name="corpus"/>.</summary>
    public static CharTokenizer Build(string corpus)
    {
        ArgumentNullException.ThrowIfNull(corpus);

        var sorted = corpus.Distinct().OrderBy(c => c).ToArray();
        var charToId = new Dictionary<char, int>(sorted.Length);
        for (var i = 0; i < sorted.Length; i++)
        {
            charToId[sorted[i]] = i;
        }
        return new CharTokenizer(charToId, sorted);
    }

    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var ids = new int[text.Length];
        for (var i = 0; i < text.Length; i++)
        {
            if (!_charToId.TryGetValue(text[i], out var id))
            {
                throw new ArgumentException(
                    $"Character '{text[i]}' (U+{(int)text[i]:X4}) is not in the vocabulary.",
                    nameof(text));
            }
            ids[i] = id;
        }
        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);

        var sb = new System.Text.StringBuilder();
        foreach (var id in ids)
        {
            if ((uint)id >= (uint)_idToChar.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(ids), id, "Token id is outside the vocabulary range.");
            }
            sb.Append(_idToChar[id]);
        }
        return sb.ToString();
    }
}
