"""Independent Python implementation of rand 0.9's seeded ChaCha12 masks.

Used only when producing the official PyTorch development reference. The
runtime training path remains Rust/Burn. The generator is intentionally pinned
to the rand version in Cargo.lock; a sampling change should break the parity test.
"""

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def _rotate_left(value, count):
    return ((value << count) | (value >> (32 - count))) & MASK32


def _quarter(state, a, b, c, d):
    state[a] = (state[a] + state[b]) & MASK32
    state[d] = _rotate_left(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & MASK32
    state[b] = _rotate_left(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & MASK32
    state[d] = _rotate_left(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & MASK32
    state[b] = _rotate_left(state[b] ^ state[c], 7)


def _seed_words(seed):
    words = []
    for _ in range(8):
        seed = (seed * 6364136223846793005 + 11634580027462260723) & MASK64
        shifted = (((seed >> 18) ^ seed) >> 27) & MASK32
        rotate = seed >> 59
        words.append(((shifted >> rotate) | (shifted << ((-rotate) & 31))) & MASK32)
    return words


def words_from_key(key):
    counter = 0
    while True:
        initial = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574] + key + [
            counter & MASK32, counter >> 32, 0, 0
        ]
        state = initial.copy()
        for _ in range(6):
            for indices in ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
                            (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
                _quarter(state, *indices)
        yield from ((a + b) & MASK32 for a, b in zip(state, initial))
        counter += 1


def selection_and_masks(frame_count, books, seed):
    n = frame_count - 1
    selected = sorted({0, n // 6, n // 3, n // 2, 2 * n // 3, 5 * n // 6, n - 1})
    words = words_from_key(_seed_words(seed))
    masks = []
    for _ in selected:
        ratio = next(words) >> 8
        mask = [(next(words) >> 8) < ratio for _ in range(books)]
        masks.append(mask if any(mask) else [True] * books)
    return selected, masks


if __name__ == "__main__":
    # rand 0.9.5's upstream StdRng construction test, interpreted as LE u64s.
    words = words_from_key([1, 23, 456, 7890, 0, 0, 0, 0])
    first = next(words) | (next(words) << 32)
    second_rng = words_from_key([next(words) for _ in range(8)])
    second = next(second_rng) | (next(second_rng) << 32)
    assert [first, second] == [10719222850664546238, 14064965282130556830]
    print(selection_and_masks(12, 12, 159))
