// TinyLLM — a toy GPT-style language model written from scratch in C#,
// using only the .NET BCL. Educational, not fast.
//
// What the demo does when you run it:
//   1. Loads Data/corpus.txt (Shakespeare sonnets, public domain).
//   2. Builds a vocabulary using the selected tokenizer.
//   3. Constructs a tiny decoder-only Transformer (see hyperparameters below).
//   4. Trains it for a small number of AdamW steps, printing the loss
//      periodically and showing a sample generation mid-run.
//   5. Enters a REPL: type a prompt, see the model continue it. Empty line exits.
//
// IMPORTANT: every scalar in the network is an autograd `Value` node. A single
// forward pass through this tiny transformer allocates many thousands of them,
// so each training step takes noticeable wall-clock time. Keep all sizes small.
//
// Run with --train-save <path> to train and save without a REPL, or
// --load-repl <path> to load a checkpoint and enter the REPL without model training.

using TinyLLM.Model;
using TinyLLM.Sampling;
using TinyLLM.Tokenization;
using TinyLLM.Training;

const int Seed = 1337;

// ---- Hyperparameters (intentionally tiny so scalar autograd is feasible) ----
const int BlockSize = 16;
const int EmbedDim = 16;
const int NumHeads = 2;
const int NumLayers = 1;
const int TrainSteps = 2000;
const int LogEvery = 50;
const int MidRunSampleAt = 1000;
const int MidRunSampleTokens = 80;
const double LearningRate = 3e-3;
const double WeightDecay = 0.0;

const int ReplMaxTokens = 200;
const double ReplTemperature = 1.0;
const int ReplTopK = 0; // 0 = disabled

// ---- Tokenizer selection ----
// Switch this to compare tokenizers without touching the model code.
//   CharTokenizer: 1 id per distinct character. Tiny vocab, long sequences.
//   BytePair    : byte-level BPE trained on the corpus. Larger vocab, shorter sequences,
//                 better at reusing common substrings ("the", "ing", "th", ...).
const TokenizerKind UseTokenizer = TokenizerKind.BytePair;
const int BpeTargetVocabSize = 256; // upper bound; BPE may stop earlier if no useful merges remain.

var mode = RunMode.TrainRepl;
var checkpointPath = "tinyllm.checkpoint.json";
var trainSteps = TrainSteps;
var stepsSpecified = false;
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--help":
            PrintUsage();
            return 0;
        case "--train-save":
        case "--load-repl":
            if (mode != RunMode.TrainRepl || i + 1 >= args.Length ||
                string.IsNullOrWhiteSpace(args[i + 1]) || args[i + 1].StartsWith("--"))
            {
                Console.Error.WriteLine("Specify one mode and a checkpoint file path.");
                PrintUsage();
                return 1;
            }
            mode = args[i] == "--train-save" ? RunMode.TrainSave : RunMode.LoadRepl;
            checkpointPath = args[++i];
            break;
        case "--steps":
            if (stepsSpecified || i + 1 >= args.Length ||
                !int.TryParse(args[++i], out trainSteps) || trainSteps <= 0)
            {
                Console.Error.WriteLine("--steps requires a positive integer and may be specified once.");
                return 1;
            }
            stepsSpecified = true;
            break;
        default:
            Console.Error.WriteLine($"Unknown argument: {args[i]}");
            PrintUsage();
            return 1;
    }
}
if (mode == RunMode.LoadRepl && stepsSpecified)
{
    Console.Error.WriteLine("--steps cannot be used with --load-repl.");
    return 1;
}

var rng = new Random(Seed);
GptModel model;
ITokenizer tokenizer;
try
{
    var loadCheckpoint = mode == RunMode.LoadRepl ||
        (mode == RunMode.TrainRepl && File.Exists(checkpointPath) &&
            AskYesNo($"Found {checkpointPath}. Load it instead of training?"));
    if (loadCheckpoint)
    {
        (model, tokenizer) = ModelCheckpoint.Load(checkpointPath);
        Console.WriteLine($"Loaded checkpoint: {checkpointPath}");
        Console.WriteLine($"Tokenizer: {tokenizer.GetType().Name}");
        Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");
        Console.WriteLine($"Model parameters: {model.Parameters().Count():N0}");
    }
    else
    {
        var corpusPath = Path.Combine(AppContext.BaseDirectory, "Data", "corpus.txt");
        var corpus = File.ReadAllText(corpusPath);
        Console.WriteLine($"Loaded corpus: {corpus.Length} characters.");

        tokenizer = UseTokenizer switch
        {
            TokenizerKind.Char => CharTokenizer.Build(corpus),
            TokenizerKind.BytePair => BytePairTokenizer.Train(corpus, BpeTargetVocabSize),
            _ => throw new InvalidOperationException($"Unknown tokenizer kind: {UseTokenizer}"),
        };
        var tokens = tokenizer.Encode(corpus);
        Console.WriteLine($"Tokenizer: {UseTokenizer}");
        Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");
        Console.WriteLine($"Token sequence length: {tokens.Length} (corpus has {corpus.Length} chars)");
        Console.WriteLine("Token ID -> text (whitespace escaped):");
        Console.Write(tokenizer.DebugString());

        model = new GptModel(
            vocabSize: tokenizer.VocabSize,
            blockSize: BlockSize,
            embedDim: EmbedDim,
            numHeads: NumHeads,
            numLayers: NumLayers,
            rng: rng);
        var parameters = model.Parameters().ToList();
        Console.WriteLine($"Model parameters: {parameters.Count:N0}");
        var optimizer = new AdamW(parameters, learningRate: LearningRate, weightDecay: WeightDecay);

        Console.WriteLine();
        Console.WriteLine($"Training for {trainSteps} steps (this will be slow — scalar autograd)...");
        Console.WriteLine();

        var trainer = new Trainer(model, optimizer, tokens, rng);
        trainer.Train(trainSteps, (step, loss) =>
        {
            if (step % LogEvery == 0 || step == 1)
            {
                Console.WriteLine($"  step {step,4}  loss {loss:F4}");
            }
            if (step == MidRunSampleAt)
            {
                Console.WriteLine();
                Console.WriteLine($"--- mid-run sample at step {step} ---");
                Sampler.Generate(
                    model,
                    tokenizer,
                    prompt: "Shall ",
                    maxNewTokens: MidRunSampleTokens,
                    temperature: ReplTemperature,
                    topK: ReplTopK,
                    rng: rng,
                    onToken: s => Console.Write(s));
                Console.WriteLine();
                Console.WriteLine("--- resuming training ---");
                Console.WriteLine();
            }
        });

        Console.WriteLine();
        Console.WriteLine("Training complete.");
        Console.WriteLine();
        if (mode == RunMode.TrainSave || AskYesNo($"Save the trained model to {checkpointPath} (replace if it exists)?"))
        {
            ModelCheckpoint.Save(checkpointPath, model, tokenizer, corpus);
            Console.WriteLine($"Saved checkpoint: {checkpointPath}");
        }
        if (mode == RunMode.TrainSave)
        {
            return 0;
        }
    }
}
catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or
    System.Text.Json.JsonException or InvalidDataException)
{
    Console.Error.WriteLine($"Unable to complete model setup or checkpoint operation: {ex.Message}");
    return 1;
}

// ---- REPL ----
Console.WriteLine("Enter a prompt (empty line to quit):");
while (true)
{
    Console.Write("> ");
    var line = Console.ReadLine();
    if (string.IsNullOrEmpty(line))
    {
        break;
    }

    // Make sure every prompt symbol is in vocab; otherwise warn and skip.
    try
    {
        _ = tokenizer.Encode(line);
    }
    catch (ArgumentException ex)
    {
        Console.WriteLine($"  ({ex.Message} — try characters that appear in the corpus.)");
        continue;
    }

    Console.Write(line);
    Sampler.Generate(
        model,
        tokenizer,
        prompt: line,
        maxNewTokens: ReplMaxTokens,
        temperature: ReplTemperature,
        topK: ReplTopK,
        rng: rng,
        onToken: s => Console.Write(s));
    Console.WriteLine();
}

return 0;

static bool AskYesNo(string question)
{
    while (true)
    {
        Console.Write($"{question} (y/n): ");
        var answer = Console.ReadLine();
        if (answer is null)
        {
            Console.WriteLine();
            return false;
        }
        switch (answer.Trim().ToLowerInvariant())
        {
            case "y":
            case "yes":
                return true;
            case "n":
            case "no":
                return false;
            default:
                Console.WriteLine("Please enter y or n.");
                break;
        }
    }
}

static void PrintUsage()
{
    Console.WriteLine("Usage: TinyLLM [--train-save <path> | --load-repl <path>] [--steps <count>]");
    Console.WriteLine("No mode: offer to load tinyllm.checkpoint.json; otherwise train and offer to save, then REPL.");
    Console.WriteLine("--steps overrides training steps (training modes only). End of input answers no to prompts.");
}

enum RunMode
{
    TrainRepl,
    TrainSave,
    LoadRepl,
}

enum TokenizerKind
{
    Char,
    BytePair,
}

