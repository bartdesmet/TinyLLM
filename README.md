# TinyLLM

A toy GPT-style language model written from scratch in C#, using only the
.NET standard libraries. The goal is **clarity**, not speed: every concept
behind a modern transformer LM is visible and step-debuggable.

## What's in here

```
TinyLLM/
├── Autograd/
│   └── Value.cs              # one heap node per scalar op + reverse-mode autograd
├── Tokenization/
│   └── CharTokenizer.cs      # char ↔ id maps built from the training corpus
├── Model/
│   ├── Module.cs             # base class: IEnumerable<Value> Parameters()
│   ├── Linear.cs             # y = x W + b
│   ├── Embedding.cs          # lookup table of learnable vectors
│   ├── LayerNorm.cs          # per-token normalization
│   ├── CausalSelfAttention.cs# multi-head, masked
│   ├── FeedForward.cs        # Linear → GELU → Linear
│   ├── TransformerBlock.cs   # pre-LN residual block
│   └── GptModel.cs           # tok+pos embeddings + N blocks + LM head
├── Training/
│   ├── CrossEntropyLoss.cs   # stable log-sum-exp
│   ├── IOptimizer.cs
│   ├── Sgd.cs
│   ├── AdamW.cs              # decoupled weight decay, bias-corrected moments
│   └── Trainer.cs            # random window → forward → backward → step
├── Sampling/
│   └── Sampler.cs            # temperature + optional top-k autoregressive sampler
├── Data/
│   └── corpus.txt            # Shakespeare sonnets (1609, public domain)
└── Program.cs                # end-to-end demo: train, sample mid-run, REPL
```

## Running

```pwsh
dotnet run --project TinyLLM
```

By default, the app checks for `tinyllm.checkpoint.json` in the current working
directory and asks whether to load it (`y/n`). If no checkpoint exists or you
answer no, it trains a new model, then asks whether to save it to that same path
(replacing an existing file). Either path then enters the REPL. Prompts accept
`y`/`yes` and `n`/`no`, retry invalid answers, and treat end of input as no.

To separate training from inference without confirmation prompts:

```pwsh
# Train, save a checkpoint, and exit (no REPL).
dotnet run --project TinyLLM -- --train-save .\tinyllm.checkpoint.json

# Load the checkpoint and enter the REPL without model training.
dotnet run --project TinyLLM -- --load-repl .\tinyllm.checkpoint.json

# Optional: override the number of training steps, for example for a smoke test.
dotnet run --project TinyLLM -- --train-save .\tinyllm.checkpoint.json --steps 2
```

Checkpoint paths are relative to the current working directory unless absolute.
Saving replaces an existing file; its parent directory must already exist.
Checkpoints are versioned JSON containing model dimensions, parameter values, and
an embedded copy of the training corpus plus tokenizer kind and vocabulary size.
Loading rebuilds the deterministic tokenizer from that embedded corpus, preserving
its token IDs (including BPE merges), so it needs neither the external corpus file
nor the current model/tokenizer constants in `Program.cs`. BPE tokenizer training
is repeated on load, but neural-network training is skipped. Do not edit the
embedded corpus or metadata independently of the weights.
These are **inference checkpoints**, not resumable training sessions: optimizer
moments, gradients, and random-number-generator state are not persisted.
Missing, malformed, incompatible, or unwritable checkpoints report an error and
exit with a nonzero status. Use `--help` for command-line usage.

When training in the default mode, you'll see:

1. Corpus + vocabulary size reported, followed by token ID-to-text mappings.
2. 2000 AdamW training steps by default, with the loss printed on step 1 and every 50 steps.
3. A short sample generation at step 1000.
4. A prompt to save the model, then a REPL prompt. Type any text built from
   characters present in the corpus and the model will continue it.
   Press Enter on an empty REPL line to exit.

Both tokenizers expose `DebugString()` through `ITokenizer`. It returns one
mapping per line in ID order, such as `0 -> "\n"` or `42 -> "the"`.
Text is quoted, with whitespace escaped (`\n`, `\r`, `\t`, and `\u0020` for
space), so token boundaries stay visible. BPE tokens that contain only part of
a UTF-8 character show their exact hexadecimal bytes, e.g.
`0 -> <partial UTF-8: C3>`. The demo prints this view before training; switch
`UseTokenizer` in `Program.cs` to compare character-level and byte-pair vocabularies.

## Visualizing the building blocks

Small, hand-written conceptual diagrams are available in `Graphs`:

- `dot-product.dot`: scalar multiply and add nodes for a 3-element vector dot product.
- `matrix-multiply.dot`: x [1x2] * W [2x3] = y [1x3]; each boxed column of W is dotted with the whole input row to produce one output element.
- `linear.dot`: matrix multiplication and optional bias (2 inputs, 3 outputs).
- `layer-norm.dot`: centering, variance, normalization, learned scale and shift.
- `feed-forward.dot`: expansion, GELU, and projection back to embedding size.
- `attention.dot`: Q/K/V projections, one repeated causal head, softmax, and output projection.
- `transformer-block.dot`: pre-LayerNorm sub-blocks and both residual paths.
- `model.dot`: embedding lookups, transformer stack, LM head, and cross-entropy loss.

These diagrams show grouped operations, tensor shapes, and parameter inputs,
not individual scalar nodes or measured graph counts. Their visual convention is:
blue ellipses are inputs; green ellipses are outputs; orange hexagons are learned
parameters (one tensor per node); gray boxes/circles are operations; white ellipses
are intermediate values; yellow diamonds are constants. Dashed edges mark residual
skips or data-only maximum shifts, as labeled. In the transformer-block overview,
learned parameters live inside the attention, LayerNorm, and feed-forward sublayers;
see their individual diagrams to inspect those parameters. `D` is embedding size,
`T` sequence length, `V` vocabulary size, `H` head count, and `d = D/H`. The
feed-forward diagrams use the default 4x hidden size. `Graphs\model.dot` is a static
overview; training does not export the full scalar graph.

For example, render a small block diagram locally with Graphviz:

```pwsh
dot -Tsvg .\Graphs\transformer-block.dot -o .\Graphs\transformer-block.svg
```

## How it works (one paragraph per concept)

- **Scalar autograd (`Value`)** — every arithmetic operation on a `Value`
  produces a new `Value` node that remembers its inputs and a backward closure
  for the local gradient. `Backward()` builds a topological order of all
  ancestors, seeds the root gradient to 1, and runs each closure in reverse,
  populating `.Grad` on every leaf parameter.
- **Tokenization** — character-level: each distinct character in the corpus
  becomes an id. Simple, no external vocab files.
- **Embeddings** — token ids and absolute positions are both looked up in
  learnable tables of shape `[vocab|blockSize, embedDim]` and summed
  element-wise.
- **Causal self-attention** — for each query position `t` and each head, we
  compute scaled dot products with the **previous** key positions `s ≤ t`
  (the causal mask is just the loop bound), apply a numerically stable
  softmax, and weight the corresponding values.
- **Feed-forward** — per-position `Linear(embed → 4·embed) → GELU → Linear`.
- **Pre-LN residual blocks** — `x ← x + Attn(LN(x))` then `x ← x + FFN(LN(x))`.
- **Loss** — cross-entropy via log-sum-exp on the LM-head logits against the
  next-token targets.
- **Optimizer** — `AdamW` with decoupled weight decay; SGD is also provided.
- **Sampling** — apply temperature to the final logits, optionally restrict to
  top-k, softmax, draw a multinomial sample.

## Speed expectations

Every scalar op (each individual multiply, add, exp, etc.) allocates a heap
node. A single forward pass through the default tiny configuration
(`blockSize=8`, `embedDim=16`, `numHeads=2`, `numLayers=1`, `vocab≈50`)
allocates roughly tens of thousands of `Value` nodes. Expect each training
step to take a noticeable fraction of a second on a typical machine, and
expect `200` steps to be enough to start producing Shakespeare-ish character
patterns but **not** coherent text.

This is the deliberate tradeoff: you can put a breakpoint anywhere — inside
softmax, inside the LayerNorm variance calculation, inside the GELU backward —
and inspect the exact `Value` graph that produced the number.

## Tuning knobs (`Program.cs` constants)

| Constant            | Default | Meaning                                                |
|---------------------|---------|--------------------------------------------------------|
| `Seed`              | 1337    | RNG seed for weight init and sampling                  |
| `BlockSize`         | 8       | Context length                                         |
| `EmbedDim`          | 16      | Token/position embedding size                          |
| `NumHeads`          | 2       | Attention heads (must divide `EmbedDim`)               |
| `NumLayers`         | 1       | Stacked transformer blocks                             |
| `TrainSteps`        | 200     | Number of optimizer updates                            |
| `LogEvery`          | 10      | Loss-print cadence                                     |
| `LearningRate`      | 3e-3    | AdamW learning rate                                    |
| `ReplTemperature`   | 1.0     | Higher = more random; lower = more conservative        |
| `ReplTopK`          | 0       | 0 disables; otherwise sample from the top-K candidates |

## What's deliberately missing (good follow-up exercises)

- **Training resumption.** Inference checkpoints save weights and model/tokenizer
  configuration, but not optimizer moments or RNG state for continuing a training run.
- **Batching.**
  gradients over several sequences would smooth the loss curve.
- **Byte-pair tokenization.** Character-level is simplest; BPE is a natural
  next step.
- **Faster backend.** Replace the scalar engine with `float[]`-tensor ops
  (loop-fused or SIMD via `System.Numerics.Tensors`) once the educational
  version is well understood.
