# Published-alpha compatibility fixtures

`v2.0.0a1.json` contains eight serialized payloads captured by importing source
from release tag `v2.0.0a1` at commit
`88b110f79922ceef7e4ae3bc92392c0bdbdfa2ba`. Capture used Python 3.12.14,
NumPy 2.5.3 and Pandas 3.0.6; library versions are retained in the file. The
source tag, not the installed editable distribution's metadata, identifies the
producer. Preserve the stored strings; tests must not regenerate them using the
current writer.

The HMAC key is a public test-only string. No pickle or executable model payloads
are included. The integration tests describe the original values and expected legacy
reconstruction; use the historical source when investigating fixture changes.

Legacy collection values were written as lists. Legacy scalar tags did not
record scalar width and continue to reconstruct int64/float64/complex128. Empty
array shape metadata was present in a1 even though its reader ignored it; the
current reader applies that metadata. A basic numeric DataFrame does not prove
compatibility for every index, extension dtype or library release.
