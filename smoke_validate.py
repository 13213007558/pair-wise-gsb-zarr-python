import numpy as np
from zarr.codecs.sharding import _ShardIndex, MAX_UINT_64
idx = _ShardIndex.create_empty((2,))
idx.set_chunk_slice((0,), slice(100, 150))
idx.set_chunk_slice((1,), slice(150, 200))
idx.validate(data_section=(100, 300), allow_empty_chunks=False)
print('valid ok')
bad = _ShardIndex.create_empty((2,))
bad.offsets_and_lengths[(0,)] = (MAX_UINT_64, 5)
try:
    bad.validate(data_section=(0, 100), allow_empty_chunks=False)
except ValueError as e:
    print('half-marker:', e)
bad2 = _ShardIndex.create_empty((2,))
bad2.set_chunk_slice((0,), slice(50, 150))
try:
    bad2.validate(data_section=(0, 100), allow_empty_chunks=False)
except ValueError as e:
    print('out-of-region:', e)
bad3 = _ShardIndex.create_empty((2,))
bad3.offsets_and_lengths[(0,)] = (2**64 - 10, 20)
try:
    bad3.validate(data_section=(0, 100), allow_empty_chunks=False)
except ValueError as e:
    print('overflow:', e)
bad4 = _ShardIndex.create_empty((3,))
bad4.set_chunk_slice((0,), slice(0, 50))
bad4.set_chunk_slice((2,), slice(40, 60))
try:
    bad4.validate(data_section=(0, 100), allow_empty_chunks=False)
except ValueError as e:
    print('overlap:', e)
bad5 = _ShardIndex.create_empty((2,))
bad5.offsets_and_lengths[(1,)] = (10, 0)
try:
    bad5.validate(data_section=(0, 100), allow_empty_chunks=False)
except ValueError as e:
    print('zero-len:', e)
bad5.validate(data_section=(0, 100), allow_empty_chunks=True)
print('zero-len allowed ok')
bad6 = _ShardIndex.create_empty((2, 2))
bad6.set_chunk_slice((0, 1), slice(0, 50))
bad6.set_chunk_slice((1, 0), slice(10, 60))
try:
    bad6.validate(data_section=(0, 100), allow_empty_chunks=False)
except ValueError as e:
    print('overlap-2d:', e)
