using System.Text;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// Minimal ONNX protobuf writer for single-node test graphs (fp32 / int64 tensors, int / ints / float / string
/// attributes). Only the fields the engine's reader consumes are emitted.
/// </summary>
internal sealed class OnnxBuilder
{
    private readonly List<byte[]> _nodes = [], _inits = [], _inputs = [], _outputs = [];
    private int _opset = 13;

    /// <summary>Default-domain opset version written to the model (13 unless set).</summary>
    public OnnxBuilder Opset(int version) { _opset = version; return this; }

    public OnnxBuilder Input(string name, params int[] shape) { _inputs.Add(ValueInfo(name, shape)); return this; }
    public OnnxBuilder Output(string name) { _outputs.Add(ValueInfo(name, null)); return this; }

    public OnnxBuilder Init(string name, float[] data, params int[] dims)
    {
        var t = new Pb();
        foreach (int d in dims) t.Varint(1, (ulong)d);
        t.Varint(2, 1);                      // FLOAT
        t.String(8, name);
        byte[] raw = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        t.Bytes(9, raw);
        _inits.Add(t.ToArray());
        return this;
    }

    public OnnxBuilder InitI64(string name, long[] data, params int[] dims)
    {
        var t = new Pb();
        foreach (int d in dims) t.Varint(1, (ulong)d);
        t.Varint(2, 7);                      // INT64
        t.String(8, name);
        byte[] raw = new byte[data.Length * 8];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        t.Bytes(9, raw);
        _inits.Add(t.ToArray());
        return this;
    }

    public OnnxBuilder Node(string op, string[] inputs, string[] outputs, params (string name, object value)[] attrs)
    {
        var n = new Pb();
        foreach (string i in inputs) n.String(1, i);
        foreach (string o in outputs) n.String(2, o);
        n.String(3, op + "_" + _nodes.Count);
        n.String(4, op);
        foreach (var (name, value) in attrs)
        {
            var a = new Pb();
            a.String(1, name);
            switch (value)
            {
                case float f: a.Fixed32(2, BitConverter.SingleToInt32Bits(f)); a.Varint(20, 1); break;
                case int i: a.Varint(3, (ulong)(long)i); a.Varint(20, 2); break;
                case int[] ints: foreach (int v in ints) a.Varint(8, (ulong)(long)v); a.Varint(20, 7); break;
                case string str: a.String(4, str); a.Varint(20, 3); break;
                default: throw new ArgumentException(name);
            }
            n.Message(5, a.ToArray());
        }
        _nodes.Add(n.ToArray());
        return this;
    }

    public byte[] Build()
    {
        var g = new Pb();
        foreach (byte[] n in _nodes) g.Message(1, n);
        g.String(2, "test");
        foreach (byte[] t in _inits) g.Message(5, t);
        foreach (byte[] v in _inputs) g.Message(11, v);
        foreach (byte[] v in _outputs) g.Message(12, v);
        var opset = new Pb();
        opset.String(1, "");
        opset.Varint(2, (ulong)_opset);
        var m = new Pb();
        m.Varint(1, 8);
        m.Message(7, g.ToArray());
        m.Message(8, opset.ToArray());
        return m.ToArray();
    }

    private static byte[] ValueInfo(string name, int[]? shape)
    {
        var tensor = new Pb();
        tensor.Varint(1, 1);                 // FLOAT
        if (shape is not null)
        {
            var sh = new Pb();
            foreach (int d in shape)
            {
                var dim = new Pb();
                dim.Varint(1, (ulong)d);
                sh.Message(1, dim.ToArray());
            }
            tensor.Message(2, sh.ToArray());
        }
        var type = new Pb();
        type.Message(1, tensor.ToArray());
        var v = new Pb();
        v.String(1, name);
        v.Message(2, type.ToArray());
        return v.ToArray();
    }

    private sealed class Pb
    {
        private readonly MemoryStream _s = new();
        private void Raw(ulong v) { while (v >= 0x80) { _s.WriteByte((byte)(v | 0x80)); v >>= 7; } _s.WriteByte((byte)v); }
        public void Varint(int field, ulong v) { Raw((ulong)field << 3); Raw(v); }
        public void Fixed32(int field, int v) { Raw((ulong)field << 3 | 5); _s.Write(BitConverter.GetBytes(v), 0, 4); }
        public void Bytes(int field, byte[] b) { Raw((ulong)field << 3 | 2); Raw((ulong)b.Length); _s.Write(b, 0, b.Length); }
        public void String(int field, string s) => Bytes(field, Encoding.UTF8.GetBytes(s));
        public void Message(int field, byte[] m) => Bytes(field, m);
        public byte[] ToArray() => _s.ToArray();
    }
}
