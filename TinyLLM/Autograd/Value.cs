namespace TinyLLM.Autograd;

/// <summary>
/// A single scalar in the computation graph. Inspired by Karpathy's micrograd:
/// every arithmetic operation produces a new <see cref="Value"/> that remembers
/// its inputs and how to back-propagate the gradient through itself.
///
/// This is intentionally one-scalar-per-node so that every concept in the
/// network (a weight, a bias, an attention score, a softmax probability, the
/// loss) is a visible, debuggable object. The price is speed: a single forward
/// pass through even a tiny transformer allocates many thousands of these.
/// </summary>
public sealed class Value
{
    /// <summary>The current value (forward pass).</summary>
    public double Data;

    /// <summary>The accumulated d(loss)/d(this) gradient.</summary>
    public double Grad;

    /// <summary>Receives this node's gradient and adds the local gradient contribution to each parent's <see cref="Grad"/>.</summary>
    private readonly Action<double> _backward;

    /// <summary>Direct parents in the DAG (inputs to the op that produced this value).</summary>
    private readonly Value[] _prev;

    /// <summary>Short tag (e.g. "+", "*", "tanh") to aid debugging.</summary>
    public string Op { get; }

    private Value(double data, Value[] prev, string op, Action<double> backward)
    {
        Data = data;
        _prev = prev;
        Op = op;
        _backward = backward;
    }

    /// <summary>Leaf node (a learnable parameter or a fixed input).</summary>
    public Value(double data, string op = "leaf")
        : this(data, [], op, static _ => { })
    {
    }

    public static Value Constant(double d) => new(d, "const");

    public static implicit operator Value(double d) => new(d, "const");

    // ----- binary ops ---------------------------------------------------

    //
    // Examples below use incoming output gradient g = d(loss)/d(output) = 3.
    //
    // Notation:
    //
    //   d/da (a + b) means differentiate with respect to a, holding b fixed (a partial derivative)
    //   Unary derivatives use the same notation with one input.
    //
    // Chain rule for output y: d(loss)/da = d(loss)/dy * dy/da.
    // Add each path's g * local slope to Grad; do not overwrite other contributions.
    // Backward() seeds the final loss with 1; intermediate gradients can differ.
    //

    //
    // Addition has slope +1 for each input: increasing either increases the output.
    //
    //   d/da (a + b) = 1
    //   d/db (a + b) = 1
    //
    // Example:
    //   a = 2, b = 4 -> output = 6
    // ->
    //   add 3 to a.Grad
    //   add 3 to b.Grad
    //
    public static Value operator +(Value a, Value b)
    {
        return new Value(a.Data + b.Data, [a, b], "+", gradient =>
        {
            a.Grad += gradient;
            b.Grad += gradient;
        });
    }

    //
    // Subtraction has slopes +1 for a and -1 for b: increasing b lowers the output.
    //
    //   d/da (a - b) = 1
    //   d/db (a - b) = -1
    //
    // Example:
    //   a = 2, b = 4 -> output = -2
    // ->
    //   add  3 to a.Grad
    //   add -3 to b.Grad
    //
    public static Value operator -(Value a, Value b)
    {
        return new Value(a.Data - b.Data, [a, b], "-", gradient =>
        {
            a.Grad += gradient;
            b.Grad += -gradient;
        });
    }

    //
    // Multiplication's slope for each input is the other input's value.
    //
    //   d/da (a * b) = b
    //   d/db (a * b) = a
    //
    // Example:
    //   a = 2, b = 4 -> output = 8
    // ->
    //   add 4 * 3 = 12 to a.Grad
    //   add 2 * 3 =  6 to b.Grad
    //
    public static Value operator *(Value a, Value b)
    {
        return new Value(a.Data * b.Data, [a, b], "*", gradient =>
        {
            a.Grad += b.Data * gradient;
            b.Grad += a.Data * gradient;
        });
    }

    //
    // Division composes multiplication and Pow, so their backward rules do the work.
    //
    //   d/da (a / b) = 1/b
    //   d/db (a / b) = -a/b^2 (b must be nonzero)
    //
    // Example:
    //   a = 6, b = 2 -> output = 3
    // ->
    //   add  (1/2) * 3 =  1.5 to a.Grad
    //   add (-6/4) * 3 = -4.5 to b.Grad
    //
    public static Value operator /(Value a, Value b)
    {
        // a / b == a * b^-1
        return a * b.Pow(-1.0);
    }

    //
    // Negation has slope -1: increasing the input decreases the output.
    //
    //   d/da (-a) = -1
    //
    // Example:
    //   a = 2 -> output = -2
    // ->
    //   add -1 * 3 = -3 to a.Grad
    //
    public static Value operator -(Value a)
    {
        return new Value(-a.Data, [a], "neg", gradient => a.Grad += -gradient);
    }

    // ----- unary ops ----------------------------------------------------

    //
    // Power:
    //
    //   d/dx (x^n) = n * x^(n-1), wherever this real derivative is defined.
    //
    //   The exponent n is fixed, not a Value to differentiate.
    //
    // Example:
    //   x = 2, n = 3 -> output = 8
    // ->
    //   add (3 * 2^2) * 3 = 36 to x.Grad.
    //
    public Value Pow(double exponent)
    {
        var self = this;
        return new Value(Math.Pow(Data, exponent), [self], $"^{exponent}", gradient =>
        {
            // d/dx (x^n) = n * x^(n-1)
            self.Grad += exponent * Math.Pow(self.Data, exponent - 1) * gradient;
        });
    }

    //
    // exp(x)
    //
    //   d/dx exp(x) = exp(x), so reuse the computed forward value.
    //
    // Example:
    //   x = 1 -> output = e
    // ->
    //   add e * 3, approximately 8.1548, to x.Grad
    //
    public Value Exp()
    {
        var self = this;
        var e = Math.Exp(Data);
        return new Value(e, [self], "exp", gradient => self.Grad += e * gradient);
    }

    //
    // Natural logarithm:
    //
    //   d/dx ln(x) = 1/x, for x > 0.
    //
    // Example:
    //   x = 2 -> output = ln(2)
    // ->
    //   add (1/2) * 3 = 1.5 to x.Grad
    //
    public Value Log()
    {
        var self = this;
        return new Value(Math.Log(Data), [self], "log",
            gradient => self.Grad += (1.0 / self.Data) * gradient);
    }

    //
    // Hyperbolic tangent squashes the input to (-1, 1). Its slope is steepest at 0 and flattens out near +/-1.
    //
    //   d/dx tanh(x) = 1 - tanh(x)^2.
    //
    //   Saturated outputs near +/-1 pass little gradient.
    //
    // Example:
    //   x = 1 -> output approximately 0.7616
    // ->
    //   add (1 - tanh(1)^2) * 3, approximately 1.2599, to x.Grad
    //
    public Value Tanh()
    {
        var self = this;
        var t = Math.Tanh(Data);
        return new Value(t, [self], "tanh", gradient => self.Grad += (1.0 - t * t) * gradient);
    }

    //
    // ReLU (Rectified Linear Unit) passes positive inputs unchanged and blocks negative inputs.
    //
    //   d/dx max(0, x) = 1         if x > 0;
    //                    0         if x < 0;
    //                    undefined if x = 0.
    //
    // Examples:
    //   x =  2 -> output = 2  ->  add 1 * 3 = 3 to x.Grad;
    //   x = -2 -> output = 0  ->  add 0 * 3 = 0 to x.Grad;
    //
    public Value ReLU()
    {
        var self = this;
        var r = Data > 0 ? Data : 0;
        return new Value(r, [self], "relu",
            gradient => self.Grad += (self.Data > 0 ? 1.0 : 0.0) * gradient);
    }

    //
    // Gaussian Error Linear Unit (tanh approximation used in GPT-2).
    //
    // GELU smoothly gates the input; backward multiplies g by the full local slope dy.
    //
    //  let c = 0.044715
    //  let k = sqrt(2/pi)
    //  let u = k * (x + c*x^3)
    //
    //        GELU(x) = 0.5 * x * (1 + tanh(u))
    //
    //   d/dx GELU(x) = 0.5 * (1 + tanh(u))
    //                + 0.5 * x * (1 - tanh(u)^2) * k * (1 + 3*c*x^2)
    //
    // Example:
    //   x = 0 -> output = 0 but dy = 0.5
    // ->
    //   add 0.5 * 3 = 1.5 to x.Grad
    //
    public Value Gelu()
    {
        const double k = 0.7978845608028654; // sqrt(2 / pi)
        const double c = 0.044715;
        var self = this;
        var x = Data;
        var x3 = x * x * x;
        var inner = k * (x + c * x3);
        var tanhInner = Math.Tanh(inner);
        var y = 0.5 * x * (1.0 + tanhInner);
        return new Value(y, [self], "gelu", gradient =>
        {
            // d/dx GELU = 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh^2(inner)) * k * (1 + 3*c*x^2)
            var dInner = k * (1.0 + 3.0 * c * x * x);
            var sech2 = 1.0 - tanhInner * tanhInner;
            var dy = 0.5 * (1.0 + tanhInner) + 0.5 * x * sech2 * dInner;
            self.Grad += dy * gradient;
        });
    }

    // ----- backward pass -----------------------------------------------

    /// <summary>
    /// Builds a topological order of all ancestors and invokes their stored
    /// backward closures in reverse, propagating gradients from this node
    /// (seeded to 1.0) back to every leaf.
    /// </summary>
    public void Backward()
    {
        var topo = BuildTopo(this);

        Grad = 1.0;
        for (var i = topo.Count - 1; i >= 0; i--)
        {
            var node = topo[i];
            node._backward(node.Grad);
        }
    }

    private static List<Value> BuildTopo(Value root)
    {
        //
        // Iterative post-order DFS to avoid stack overflow on deep graphs
        // (e.g. long chains of additions in the cross-entropy reduction).
        //
        // Example: x = a + b; loss = x + a; root = loss.
        // Forward computation, top to bottom; arrowheads point at the input terms:
        //
        //      a---------+--------------------+
        //                |                    |
        //                a + b = x--------x + a = loss (root)
        //                    |
        //      b-------------+
        //
        // Starting at loss, DFS follows _prev back to its inputs, left to right.
        //
        // Returned order: [a, b, x, loss]; each node follows all of its inputs.
        //
        // Although a is used twice, visited ensures it appears only once.
        //
        // Backward reverses this order: [loss, x, b, a], so a receives contributions
        // from both loss and x before its own backward function runs.
        //
        var topo = new List<Value>();
        var visited = new HashSet<Value>(ReferenceEqualityComparer.Instance)
        {
            root
        };

        var stack = new Stack<(Value node, int childIdx)>();
        stack.Push((root, 0));

        while (stack.Count > 0)
        {
            var (current, idx) = stack.Pop();
            if (idx < current._prev.Length)
            {
                stack.Push((current, idx + 1));
                var child = current._prev[idx];
                if (visited.Add(child))
                {
                    stack.Push((child, 0));
                }
            }
            else
            {
                topo.Add(current);
            }
        }

        return topo;
    }

    public override string ToString() => $"Value(data={Data:F4}, grad={Grad:F4}, op={Op})";
}
