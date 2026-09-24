Added ``zarr.integrity``, a module for computing saveable per-chunk integrity
manifests of array data. Digests are computed from decoded values with
canonical rules for NaN payloads, byte order, and negative zero, so they are
independent of the compressor chain, shard layout, and host NumPy memory
layout. Manifests record which chunks were implicit fill and can be verified
later against the full array or restricted to the chunks touched by a basic
selection, distinguishing changed data, missing chunks, and incompatible
array metadata.
