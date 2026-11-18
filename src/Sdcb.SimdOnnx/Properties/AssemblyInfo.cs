using System.Runtime.CompilerServices;

// Consumers compile the engine's source into their own assembly (Sdcb.SimdOnnx.Sources.props), where
// their own InternalsVisibleTo grants cover their test / bench assemblies. This development build only
// serves the engine's tests and bench.
[assembly: InternalsVisibleTo("Sdcb.SimdOnnx.Tests")]
[assembly: InternalsVisibleTo("Sdcb.SimdOnnx.Bench")]
