// Standalone transfer tests. Include the bridge to inspect physical IOSurface
// contents independently of writeHalf/readHalf; no ANE program is executed.
#include "Direct.m"

static uint16_t pattern(size_t i, unsigned phase) {
    static const uint16_t special[] = {0x0000, 0x8000, 0x7e01, 0xfeff, 0x7c00, 0xfc00, 0x0001, 0x03ff};
    return i % 11 < 8 ? special[(i + phase) % 8] : (uint16_t)(i * 193 + phase);
}

static void canonical(const uint16_t *logical, uint16_t *physical, NSArray *shape, size_t count) {
    if (shape.count == 2 || shape.count == 3) {
        size_t b = shape.count == 3 ? [shape[0] unsignedLongLongValue] : 1;
        size_t s = [shape[shape.count - 2] unsignedLongLongValue];
        size_t c = [shape.lastObject unsignedLongLongValue];
        for (size_t batch = 0; batch < b; ++batch)
            for (size_t channel = 0; channel < c; ++channel)
                for (size_t row = 0; row < s; ++row)
                    physical[(batch * c + channel) * s + row] = logical[(batch * s + row) * c + channel];
    } else {
        memcpy(physical, logical, count * 2);
    }
}

static void verify(NSArray *shape, size_t b, size_t c, size_t h, size_t w, size_t padding) {
    size_t count = b * c * h * w;
    size_t rowStride = w * 2 + padding;
    size_t planeStride = h * rowStride + padding * 2;
    size_t batchStride = c * planeStride + padding * 3;
    NSDictionary *layout = @{@"Type": @"Float16", @"Channels": @(c), @"Height": @(h), @"Width": @(w),
                             @"Batches": @(b), @"Depth": @1, @"RowStride": @(rowStride),
                             @"PlaneStride": @(planeStride), @"BatchStride": @(batchStride)};
    BurnAneSurface *s = [[BurnAneSurface alloc] initWithLayout:layout
        expected:@{@"shape": @[@(b), @(c), @(h), @(w)], @"logical": shape, @"offset": @13}];
    NSCAssert(s->offset == 13 && s->elements == count, @"I/O descriptor mismatch");
    uint16_t *source = malloc(count * 2);
    uint16_t *physical = malloc(count * 2);
    uint16_t *output = malloc((count + 2) * 2);
    NSCAssert(source && physical && output, @"allocation failed");
    for (size_t i = 0; i < count; ++i) source[i] = pattern(i, 0);
    canonical(source, physical, shape, count);
    [s writeHalf:source];
    NSCAssert(IOSurfaceLock(s->surface, 0, NULL) == kIOReturnSuccess, @"lock failed");
    uint8_t *base = IOSurfaceGetBaseAddress(s->surface);
    size_t i = 0;
    for (size_t batch = 0; batch < b; ++batch)
        for (size_t channel = 0; channel < c; ++channel)
            for (size_t row = 0; row < h; ++row)
                for (size_t column = 0; column < w; ++column) {
                    uint16_t value;
                    memcpy(&value, base + batch * batchStride + channel * planeStride + row * rowStride + column * 2, 2);
                    NSCAssert(value == physical[i++], @"physical write mismatch for %@, padding %zu", shape, padding);
                }
    // Supply independently packed, changed bits to verify the read direction.
    for (size_t j = 0; j < count; ++j) source[j] = pattern(j, 3);
    canonical(source, physical, shape, count);
    i = 0;
    for (size_t batch = 0; batch < b; ++batch)
        for (size_t channel = 0; channel < c; ++channel)
            for (size_t row = 0; row < h; ++row) {
                memcpy(base + batch * batchStride + channel * planeStride + row * rowStride, physical + i, w * 2);
                i += w;
            }
    IOSurfaceUnlock(s->surface, 0, NULL);
    for (size_t j = 0; j < count + 2; ++j) output[j] = 0xa55a;
    [s readHalf:output + 1];
    NSCAssert(output[0] == 0xa55a && output[count + 1] == 0xa55a, @"output guard overwritten");
    NSCAssert(memcmp(source, output + 1, count * 2) == 0, @"logical read mismatch for %@", shape);
    free(source); free(physical); free(output);
}

static void shape4(NSArray *shape, size_t result[4]) {
    for (size_t d = 0; d < 4; ++d) result[d] = 1;
    if (shape.count == 1) result[1] = [shape[0] unsignedLongLongValue];
    else if (shape.count == 2 || shape.count == 3) {
        result[0] = shape.count == 3 ? [shape[0] unsignedLongLongValue] : 1;
        result[1] = [shape.lastObject unsignedLongLongValue];
        result[3] = [shape[shape.count - 2] unsignedLongLongValue];
    } else if (shape.count == 4) {
        for (size_t d = 0; d < 4; ++d) result[d] = [shape[d] unsignedLongLongValue];
    }
}

static BurnAneSurface *testSurface(NSArray *shape, size_t padding, BOOL flatten) {
    size_t dims[4]; shape4(shape, dims);
    if (flatten) { dims[3] *= dims[0] * dims[1] * dims[2]; dims[0] = dims[1] = dims[2] = 1; }
    size_t row = dims[3] * 2 + padding, plane = dims[2] * row + padding * 2, batch = dims[1] * plane + padding * 3;
    return [[BurnAneSurface alloc] initWithLayout:@{@"Type": @"Float16", @"Batches": @(dims[0]), @"Channels": @(dims[1]),
        @"Height": @(dims[2]), @"Width": @(dims[3]), @"Depth": @1, @"RowStride": @(row), @"PlaneStride": @(plane), @"BatchStride": @(batch)}
        expected:@{@"shape": @[@(dims[0]), @(dims[1]), @(dims[2]), @(dims[3])], @"logical": shape, @"offset": @0}];
}

// Independent scatter in logical Burn order, deliberately poisoning source
// capacity holes so a whole-buffer copy would fail even if reads agree.
static uint16_t *expectedView(NSArray *shape, size_t axis, const BurnAneSegment *segments, size_t segmentCount,
                              BOOL poison, NSArray **paddedShape, size_t *paddedCount) {
    NSMutableArray *padded = [shape mutableCopy];
    if (axis < shape.count) {
        size_t length = 0;
        for (size_t s = 0; s < segmentCount; ++s) length += segments[s].capacity;
        padded[axis] = @(length);
    }
    size_t validCount = 1; *paddedCount = 1;
    for (size_t d = 0; d < shape.count; ++d) {
        validCount *= [shape[d] unsignedLongLongValue];
        *paddedCount *= [padded[d] unsignedLongLongValue];
    }
    uint16_t *values = malloc(*paddedCount * 2);
    NSCAssert(values, @"test allocation failed");
    for (size_t i = 0; i < *paddedCount; ++i) values[i] = poison ? 0xa55a : 0;
    for (size_t i = 0; i < validCount; ++i) {
        size_t remaining = i, flat = 0, stride = 1;
        for (size_t dd = shape.count; dd > 0; --dd) {
            size_t d = dd - 1, extent = [shape[d] unsignedLongLongValue];
            size_t coord = remaining % extent; remaining /= extent;
            if (d == axis) {
                size_t start = 0;
                for (size_t s = 0; s < segmentCount; ++s) {
                    if (coord < segments[s].valid) { coord += start; break; }
                    coord -= segments[s].valid; start += segments[s].capacity;
                }
            }
            flat += coord * stride;
            stride *= [padded[d] unsignedLongLongValue];
        }
        values[flat] = pattern(i, 5);
    }
    *paddedShape = padded;
    return values;
}

static size_t physicalAxis(size_t rank, size_t axis) {
    if (axis == rank) return 4;
    if (rank == 1) return 1;
    if (rank == 2) return axis == 0 ? 3 : 1;
    if (rank == 3) return axis == 0 ? 0 : axis == 1 ? 3 : 1;
    return axis;
}

static void verifyTransfer(NSArray *shape, size_t sourceAxis, size_t targetAxis, size_t padding) {
    size_t sourceLength = sourceAxis < shape.count ? [shape[sourceAxis] unsignedLongLongValue] : 1;
    size_t targetLength = targetAxis < shape.count ? [shape[targetAxis] unsignedLongLongValue] : 1;
    BurnAneSegment sourceSegments[] = {{sourceLength - 1, sourceLength + 3}, {1, 2}};
    BurnAneSegment targetSegments[] = {{targetLength, targetLength + 5}};
    BurnAneView sourceView = {.axis = physicalAxis(shape.count, sourceAxis), .segments = sourceSegments,
                             .segmentCount = sourceAxis < shape.count ? 2 : 0};
    BurnAneView targetView = {.axis = physicalAxis(shape.count, targetAxis), .segments = targetSegments,
                             .segmentCount = targetAxis < shape.count ? 1 : 0};
    shape4(shape, sourceView.shape); shape4(shape, targetView.shape);
    NSArray *sourceShape, *targetShape;
    size_t sourceCount, targetCount;
    uint16_t *sourceData = expectedView(shape, sourceAxis, sourceSegments, sourceView.segmentCount, YES, &sourceShape, &sourceCount);
    uint16_t *expected = expectedView(shape, targetAxis, targetSegments, targetView.segmentCount, NO, &targetShape, &targetCount);
    BurnAneSurface *source = testSurface(sourceShape, padding, padding == 3);
    BurnAneSurface *target = testSurface(targetShape, padding + 3, padding == 6);
    [source writeHalf:sourceData];
    // Prime target memory with nonzero data to exercise destination clearing.
    uint16_t *actual = malloc(MAX(sourceCount, targetCount) * 2);
    NSCAssert(actual, @"test allocation failed");
    for (size_t i = 0; i < targetCount; ++i) actual[i] = 0xdead;
    [target writeHalf:actual];
    copySurface(source, &sourceView, target, &targetView);
    [target readHalf:actual];
    NSCAssert(memcmp(expected, actual, targetCount * 2) == 0, @"native transfer mismatch %@ axes %zu/%zu padding %zu", shape, sourceAxis, targetAxis, padding);
    [source readHalf:actual];
    NSCAssert(memcmp(sourceData, actual, sourceCount * 2) == 0, @"native source was modified");
    free(sourceData); free(expected); free(actual);
}

int main(void) {
    @autoreleasepool {
        for (size_t padding = 0; padding <= 6; padding += 3) {
            verify(@[], 1, 1, 1, 1, padding);
            verify(@[@37], 1, 37, 1, 1, padding);
            verify(@[@1, @37], 1, 37, 1, 1, padding);
            verify(@[@37, @1], 1, 1, 1, 37, padding);
            verify(@[@33, @37], 1, 37, 1, 33, padding);
            verify(@[@2, @33, @37], 2, 37, 1, 33, padding);
            verify(@[@2, @3, @5, @7], 2, 3, 5, 7, padding);
            verify(@[@2, @33, @37], 1, 37, 3, 22, padding);
            verify(@[@2, @33, @37], 1, 1, 1, 2442, padding);
        }
        puts("27 native I/O cases passed (independent physical write/read, bit patterns, strided layouts, guards)");
        size_t transfers = 0;
        for (NSArray *shape in @[@[], @[@7], @[@5, @7], @[@2, @5, @7], @[@2, @3, @5, @7]])
            for (size_t sourceAxis = 0; sourceAxis <= shape.count; ++sourceAxis)
                for (size_t targetAxis = 0; targetAxis <= shape.count; ++targetAxis)
                    for (size_t padding = 0; padding <= 6; padding += 3) {
                        verifyTransfer(shape, sourceAxis, targetAxis, padding); ++transfers;
                    }
        printf("%zu native surface transfers passed (segmented padding, independent axes, poisoned holes, changed strides, alternate layouts)\n", transfers);
    }
    return 0;
}
