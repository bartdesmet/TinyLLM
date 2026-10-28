using System.Text;
using System.Text.RegularExpressions;

namespace TinyLLM.Tokenization;

/// <summary>
/// Byte-level Byte-Pair Encoding (Sennrich, Haddow &amp; Birch, 2016), trained
/// from scratch on a corpus.
///
/// Algorithm (training):
///   1. Determine the base alphabet (every distinct byte appearing in the
///      corpus). These get the first ids, sorted by byte value.
///   2. Pre-tokenize the corpus into "words" (runs of whitespace OR non-
///      whitespace), and represent each as a list of base ids. This pre-
///      tokenization prevents BPE from merging across word boundaries, which
///      keeps merges meaningful.
///   3. Repeatedly find the most frequent adjacent pair of ids across all
///      word-sequences, assign it the next free id (= baseCount + rank), and
///      rewrite every sequence in place. Stop when the target vocab size is
///      reached or no pair appears more than once.
///
/// Encoding new text mirrors the training rewrite, but using the recorded
/// merges in priority order (lowest rank = highest priority = was the most
/// common pair during training).
///
/// Decoding is a lookup: each id maps to a fixed byte sequence; concatenate
/// the bytes and UTF-8 decode.
///
/// Example: train on "the in the in the" with targetVocabSize = 9.
/// These ASCII characters each start as one byte token. Base IDs, sorted by byte:
///   space=0, e=1, h=2, i=3, n=4, t=5.
/// Below, brackets group tokens within a word; each | represents a separate space
/// token (ID 0). Pairs are counted within each group, never across spaces.
///
///   Start:  [t h e] | [i n] | [t h e] | [i n] | [t h e]
///   Pass 1: (h, e) and (t, h) both occur 3 times; (i, n) occurs 2 times.
///           The ID-pair tie-break picks (h, e): (2, 1) precedes (5, 2).
///           Add ID 6 = "he" and replace all (h, e) pairs:
///           [t he] | [i n] | [t he] | [i n] | [t he]
///   Pass 2: Recount: (t, he) occurs 3 times; (i, n) occurs 2 times.
///           Add ID 7 = "the" and replace all (t, he) pairs:
///           [the] | [i n] | [the] | [i n] | [the]
///   Pass 3: Recount: (i, n) occurs 2 times and is the only remaining pair.
///           Add ID 8 = "in" and replace both occurrences:
///           [the] | [in] | [the] | [in] | [the]
///
/// The final encoding is [7, 0, 8, 0, 7, 0, 8, 0, 7]. Both "the" and "in" now
/// have their own tokens; the original character tokens remain in the vocabulary.
///
/// Note: for an ASCII-only corpus, "byte-level" and "char-level" base
/// alphabets are identical. The "byte-level" framing matters when the corpus
/// (or prompts) contain non-ASCII text, because BPE then works on bytes
/// rather than codepoints.
/// </summary>
public sealed class BytePairTokenizer : ITokenizer
{
    // base id range: [0, _baseCount)   = distinct bytes that appear in the corpus
    // merge id range: [_baseCount, _baseCount + _merges.Count)
    private readonly int _baseCount;
    private readonly Dictionary<byte, int> _byteToId;
    private readonly byte[][] _idToBytes;
    private readonly (int A, int B)[] _merges; // recorded in order of training rank
    private readonly Dictionary<(int A, int B), int> _mergeRank; // pair -> training rank (0 = highest priority)

    // Pre-tokenization regex: split into runs of whitespace OR runs of non-whitespace.
    private static readonly Regex PreTokenRegex = new(@"\s+|\S+", RegexOptions.Compiled);

    private BytePairTokenizer(
        int baseCount,
        Dictionary<byte, int> byteToId,
        byte[][] idToBytes,
        (int A, int B)[] merges,
        Dictionary<(int A, int B), int> mergeRank)
    {
        _baseCount = baseCount;
        _byteToId = byteToId;
        _idToBytes = idToBytes;
        _merges = merges;
        _mergeRank = mergeRank;
    }

    public int VocabSize => _idToBytes.Length;
    public int BaseCount => _baseCount;
    public int MergeCount => _merges.Length;

    /// <inheritdoc />
    public string DebugString()
    {
        var sb = new StringBuilder();
        for (var id = 0; id < VocabSize; id++)
        {
            var bytes = _idToBytes[id];
            var text = System.Text.Unicode.Utf8.IsValid(bytes)
                ? TokenizerDebugFormatter.Quote(Encoding.UTF8.GetString(bytes))
                : $"<partial UTF-8: {string.Join(" ", bytes.Select(b => b.ToString("X2")))}>";
            sb.AppendLine($"{id} -> {text}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Train a tokenizer on <paramref name="corpus"/>. The resulting vocab size
    /// is at most <paramref name="targetVocabSize"/>; it may be smaller if the
    /// corpus has fewer than that many learnable merges.
    /// </summary>
    public static BytePairTokenizer Train(string corpus, int targetVocabSize)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetVocabSize);

        // 1) Base alphabet.
        var corpusBytes = Encoding.UTF8.GetBytes(corpus);
        var uniqueBytes = corpusBytes.Distinct().OrderBy(b => b).ToArray();
        var baseCount = uniqueBytes.Length;

        var byteToId = new Dictionary<byte, int>(baseCount);
        var idToBytes = new List<byte[]>(Math.Max(baseCount, targetVocabSize));
        for (var i = 0; i < baseCount; i++)
        {
            byteToId[uniqueBytes[i]] = i;
            idToBytes.Add([uniqueBytes[i]]);
        }

        // 2) Pre-tokenize the corpus into sequences of base ids.
        var sequences = new List<List<int>>();
        foreach (Match m in PreTokenRegex.Matches(corpus))
        {
            var bytes = Encoding.UTF8.GetBytes(m.Value);
            var seq = new List<int>(bytes.Length);
            foreach (var b in bytes)
            {
                seq.Add(byteToId[b]);
            }
            sequences.Add(seq);
        }

        // 3) Iteratively merge the most frequent adjacent pair.
        var merges = new List<(int A, int B)>();
        var mergeRank = new Dictionary<(int A, int B), int>();
        var maxMerges = Math.Max(0, targetVocabSize - baseCount);

        for (var m = 0; m < maxMerges; m++)
        {
            // Count adjacent pairs.
            var pairCounts = new Dictionary<(int A, int B), int>();
            foreach (var seq in sequences)
            {
                for (var i = 0; i + 1 < seq.Count; i++)
                {
                    var pair = (seq[i], seq[i + 1]);
                    pairCounts.TryGetValue(pair, out var c);
                    pairCounts[pair] = c + 1;
                }
            }
            if (pairCounts.Count == 0) break;

            // Pick highest count; tie-break by (a, b) lex order for determinism.
            (int A, int B) bestPair = default;
            var bestCount = 0;
            foreach (var kv in pairCounts)
            {
                var c = kv.Value;
                if (c > bestCount ||
                    (c == bestCount &&
                     (kv.Key.A < bestPair.A || (kv.Key.A == bestPair.A && kv.Key.B < bestPair.B))))
                {
                    bestCount = c;
                    bestPair = kv.Key;
                }
            }
            if (bestCount < 2) break; // nothing worth merging anymore

            // Record the merge and emit a new id whose bytes are bytes(A) + bytes(B).
            var newId = idToBytes.Count;
            merges.Add(bestPair);
            mergeRank[bestPair] = m;

            var aBytes = idToBytes[bestPair.A];
            var bBytes = idToBytes[bestPair.B];
            var merged = new byte[aBytes.Length + bBytes.Length];
            Buffer.BlockCopy(aBytes, 0, merged, 0, aBytes.Length);
            Buffer.BlockCopy(bBytes, 0, merged, aBytes.Length, bBytes.Length);
            idToBytes.Add(merged);

            // Rewrite all sequences, replacing every occurrence of the pair.
            foreach (var seq in sequences)
            {
                ApplyMergeInPlace(seq, bestPair.A, bestPair.B, newId);
            }
        }

        return new BytePairTokenizer(
            baseCount,
            byteToId,
            idToBytes.ToArray(),
            merges.ToArray(),
            mergeRank);
    }

    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var result = new List<int>();
        foreach (Match m in PreTokenRegex.Matches(text))
        {
            var bytes = Encoding.UTF8.GetBytes(m.Value);
            var seq = new List<int>(bytes.Length);
            foreach (var b in bytes)
            {
                if (!_byteToId.TryGetValue(b, out var id))
                {
                    throw new ArgumentException(
                        $"Byte 0x{b:X2} is not in the tokenizer's base alphabet (was not present in the training corpus).",
                        nameof(text));
                }
                seq.Add(id);
            }
            ApplyMergesGreedy(seq);
            result.AddRange(seq);
        }
        return result.ToArray();
    }

    public string Decode(IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);

        // Collect into a byte buffer, then UTF-8 decode at the end. For
        // ASCII-only corpora every merged id resolves to a valid ASCII span;
        // for general UTF-8 corpora a single id may decode to a partial code
        // point, which is fine because we concatenate first.
        var bytes = new List<byte>();
        foreach (var id in ids)
        {
            if ((uint)id >= (uint)_idToBytes.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(ids), id, "Token id is outside the vocabulary range.");
            }
            bytes.AddRange(_idToBytes[id]);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>Replaces every adjacent (a, b) pair in <paramref name="seq"/> with <paramref name="newId"/>, in place.</summary>
    private static void ApplyMergeInPlace(List<int> seq, int a, int b, int newId)
    {
        var write = 0;
        var read = 0;
        while (read < seq.Count)
        {
            if (read + 1 < seq.Count && seq[read] == a && seq[read + 1] == b)
            {
                seq[write++] = newId;
                read += 2;
            }
            else
            {
                seq[write++] = seq[read++];
            }
        }
        if (write < seq.Count)
        {
            seq.RemoveRange(write, seq.Count - write);
        }
    }

    /// <summary>
    /// Encode-time greedy merge: repeatedly find the adjacent pair with the
    /// LOWEST training rank (= the highest training priority) and merge it,
    /// until no pair in the sequence is a recorded merge.
    ///
    /// O(n * merges_applied) per pre-token, which is fine for short words.
    /// </summary>
    private void ApplyMergesGreedy(List<int> seq)
    {
        while (seq.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIndex = -1;
            for (var i = 0; i + 1 < seq.Count; i++)
            {
                if (_mergeRank.TryGetValue((seq[i], seq[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;

            var newId = _baseCount + bestRank;
            seq[bestIndex] = newId;
            seq.RemoveAt(bestIndex + 1);
        }
    }
}
