// Experimental private ANE ABI, isolated from Burn and the public Core ML path.
// Interface research: maderix/ANE and Epistates/pmetal (see docs/ane-backend-research.md).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

// Declarations only: classes are resolved dynamically so the backend still loads
// on systems without these private interfaces. Never link the private framework.
@protocol BurnAnePrivate
+ (id)modelWithMILText:(NSData *)text weights:(NSDictionary *)weights optionsPlist:(id)options;
+ (id)inMemoryModelWithDescriptor:(id)descriptor;
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices
                outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices
          weightsBuffer:(id)weights perfStats:(id)stats procedureIndex:(NSNumber *)index;
- (NSString *)hexStringIdentifier;
- (BOOL)compiledModelExists;
- (BOOL)compileWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)loadWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)unloadWithQoS:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithQoS:(unsigned int)qos options:(NSDictionary *)options request:(id)request error:(NSError **)error;
- (id)model;
- (NSDictionary *)modelAttributes;
- (void)setDescriptor:(id)descriptor;
@end

static void fail(NSString *reason) {
    @throw [NSException exceptionWithName:@"BurnDirectANE" reason:reason userInfo:nil];
}
static void check(BOOL ok, NSString *operation, NSError *error) {
    if (!ok) fail([NSString stringWithFormat:@"%@: %@", operation, error ?: @"failed"]);
}
static size_t product(size_t a, size_t b) {
    size_t result;
    if (__builtin_mul_overflow(a, b, &result)) fail(@"ANE layout size overflow");
    return result;
}
static void errorText(NSException *error, char *out, size_t capacity) {
    if (out && capacity) snprintf(out, capacity, "%s", error.reason.UTF8String ?: "ANE exception");
}
static double now(void) { return NSProcessInfo.processInfo.systemUptime * 1000.; }

// Copy bits between logical [S,C] and physical [C,S] views in cache-sized tiles.
// Strides are in uint16_t elements; no FP16 arithmetic or float32 staging occurs.
static void matrixCopy(const uint16_t *source, size_t sourceRow, size_t sourceColumn,
                       uint16_t *target, size_t targetRow, size_t targetColumn,
                       size_t rows, size_t columns) {
    for (size_t r0 = 0; r0 < rows; r0 += 32)
        for (size_t c0 = 0; c0 < columns; c0 += 32)
            for (size_t r = r0; r < MIN(r0 + 32, rows); ++r)
                for (size_t c = c0; c < MIN(c0 + 32, columns); ++c)
                    target[r * targetRow + c * targetColumn] = source[r * sourceRow + c * sourceColumn];
}

@interface BurnAneSurface : NSObject {
@public IOSurfaceRef surface;
    size_t channels, height, width, depth, batches, rowStride, planeStride, batchStride, elements;
    size_t logicalRows, logicalChannels, offset;
    size_t canonicalShape[4];
    NSDictionary *layoutDescription, *ioDescription;
}
- (instancetype)initWithLayout:(NSDictionary *)layout expected:(NSDictionary *)expected;
- (void)writeHalf:(const uint16_t *)values;
- (void)readHalf:(uint16_t *)values;
@end
@implementation BurnAneSurface
- (instancetype)initWithLayout:(NSDictionary *)l expected:(NSDictionary *)expected {
    self = [super init];
    if (!self) return nil;
    if (![l[@"Type"] isEqual:@"Float16"]) fail(@"Direct ANE requires reported Float16 I/O");
    channels = [l[@"Channels"] unsignedLongLongValue];
    height = [l[@"Height"] unsignedLongLongValue];
    width = [l[@"Width"] unsignedLongLongValue];
    depth = MAX(1, [l[@"Depth"] unsignedLongLongValue]);
    batches = MAX(1, [l[@"Batches"] unsignedLongLongValue]);
    rowStride = [l[@"RowStride"] unsignedLongLongValue];
    planeStride = [l[@"PlaneStride"] unsignedLongLongValue];
    batchStride = [l[@"BatchStride"] unsignedLongLongValue];
    elements = product(product(product(product(channels, height), width), depth), batches);
    size_t expectedCount = 1;
    for (NSNumber *dim in expected[@"shape"]) expectedCount = product(expectedCount, dim.unsignedLongLongValue);
    NSArray *logical = expected[@"logical"];
    size_t logicalCount = 1;
    for (NSNumber *dim in logical) logicalCount = product(logicalCount, dim.unsignedLongLongValue);
    if (!logical || logicalCount != expectedCount) fail(@"Invalid ANE logical I/O shape");
    layoutDescription = l;
    ioDescription = expected;
    for (size_t d = 0; d < 4; ++d) canonicalShape[d] = 1;
    if (logical.count == 1) canonicalShape[1] = [logical[0] unsignedLongLongValue];
    else if (logical.count == 2 || logical.count == 3) {
        canonicalShape[0] = logical.count == 3 ? [logical[0] unsignedLongLongValue] : 1;
        canonicalShape[1] = [logical.lastObject unsignedLongLongValue];
        canonicalShape[3] = [logical[logical.count - 2] unsignedLongLongValue];
    } else if (logical.count == 4) {
        for (size_t d = 0; d < 4; ++d) canonicalShape[d] = [logical[d] unsignedLongLongValue];
    }
    logicalRows = logical.count == 2 ? [logical[0] unsignedLongLongValue]
                : logical.count == 3 ? [logical[1] unsignedLongLongValue] : 1;
    logicalChannels = logical.count == 2 ? [logical[1] unsignedLongLongValue]
                    : logical.count == 3 ? [logical[2] unsignedLongLongValue] : 1;
    offset = [expected[@"offset"] unsignedLongLongValue];
    // Depth > 1 needs a separately reported depth stride. Reject rather than
    // guessing the memory layout of a rank-five model boundary.
    if (!elements || elements != expectedCount || depth != 1 ||
        rowStride < product(width, 2) || planeStride < product(height, rowStride) ||
        (batches > 1 && batchStride < product(channels, planeStride))) {
        fail([NSString stringWithFormat:@"Unsupported ANE I/O layout for %@: %@", expected, l]);
    }
    size_t bytes = MAX(product(batchStride, batches), product(product(planeStride, channels), batches));
    if (!bytes || bytes > 4ULL * 1024 * 1024 * 1024) fail(@"ANE I/O allocation exceeds 4 GiB");
    size_t page = (size_t)getpagesize();
    size_t allocationBytes = (bytes + page - 1) / page * page;
    surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(allocationBytes), (id)kIOSurfacePixelFormat: @0
    });
    if (!surface) fail(@"IOSurface allocation failed");
    return self;
}
- (void)writeHalf:(const uint16_t *)values {
    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) fail(@"IOSurface input lock failed");
    uint8_t *base = IOSurfaceGetBaseAddress(surface);
    memset(base, 0, IOSurfaceGetAllocSize(surface));
    BOOL transpose = logicalRows > 1 && logicalChannels > 1;
    if (transpose && height == 1 && width == logicalRows && channels == logicalChannels &&
        planeStride % 2 == 0 && batchStride % 2 == 0) {
        for (size_t b = 0; b < batches; ++b)
            matrixCopy(values + b * logicalRows * logicalChannels, logicalChannels, 1,
                       (uint16_t *)(base + b * batchStride), 1, planeStride / 2,
                       logicalRows, logicalChannels);
        IOSurfaceUnlock(surface, 0, NULL);
        return;
    }
    if (!transpose && rowStride == width * 2 && planeStride == height * rowStride &&
        (batches == 1 || batchStride == channels * planeStride)) {
        memcpy(base, values, elements * sizeof(uint16_t));
        IOSurfaceUnlock(surface, 0, NULL);
        return;
    }
    size_t i = 0;
    for (size_t b = 0; b < batches; ++b)
        for (size_t c = 0; c < channels; ++c)
            for (size_t h = 0; h < height; ++h) {
                uint8_t *row = base + b * batchStride + c * planeStride + h * rowStride;
                if (transpose) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t p = i + w;
                        size_t logical = (p / (logicalRows * logicalChannels) * logicalRows + p % logicalRows)
                                       * logicalChannels + (p / logicalRows) % logicalChannels;
                        memcpy(row + w * sizeof(uint16_t), values + logical, sizeof(uint16_t));
                    }
                } else {
                    memcpy(row, values + i, width * sizeof(uint16_t));
                }
                i += width;
            }
    IOSurfaceUnlock(surface, 0, NULL);
}
- (void)readHalf:(uint16_t *)values {
    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) fail(@"IOSurface output lock failed");
    const uint8_t *base = IOSurfaceGetBaseAddress(surface);
    BOOL transpose = logicalRows > 1 && logicalChannels > 1;
    if (transpose && height == 1 && width == logicalRows && channels == logicalChannels &&
        planeStride % 2 == 0 && batchStride % 2 == 0) {
        for (size_t b = 0; b < batches; ++b)
            matrixCopy((const uint16_t *)(base + b * batchStride), 1, planeStride / 2,
                       values + b * logicalRows * logicalChannels, logicalChannels, 1,
                       logicalRows, logicalChannels);
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        return;
    }
    if (!transpose && rowStride == width * 2 && planeStride == height * rowStride &&
        (batches == 1 || batchStride == channels * planeStride)) {
        memcpy(values, base, elements * sizeof(uint16_t));
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        return;
    }
    size_t i = 0;
    for (size_t b = 0; b < batches; ++b)
        for (size_t c = 0; c < channels; ++c)
            for (size_t h = 0; h < height; ++h) {
                const uint8_t *row = base + b * batchStride + c * planeStride + h * rowStride;
                if (transpose) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t p = i + w;
                        size_t logical = (p / (logicalRows * logicalChannels) * logicalRows + p % logicalRows)
                                       * logicalChannels + (p / logicalRows) % logicalChannels;
                        memcpy(values + logical, row + w * sizeof(uint16_t), sizeof(uint16_t));
                    }
                } else {
                    memcpy(values + i, row, width * sizeof(uint16_t));
                }
                i += width;
            }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}
- (void)dealloc { if (surface) CFRelease(surface); }
@end

typedef struct { size_t valid, capacity; } BurnAneSegment;
typedef struct {
    size_t shape[4]; // Valid extents in canonical [B,C,H,W] order.
    size_t axis; // 4 means no padding.
    const BurnAneSegment *segments;
    size_t segmentCount;
} BurnAneView;
typedef struct {
    void *surface; // Retained by Rust for the entire synchronous prediction.
    const uint16_t *host;
    size_t hostCount;
    BurnAneView source, target;
} BurnAneInput;

static void validateView(BurnAneSurface *s, const BurnAneView *v) {
    if (v->axis > 4 || ((v->axis == 4) != (v->segmentCount == 0))) fail(@"Invalid ANE view axis");
    size_t valid = 0, capacity = 0;
    for (size_t i = 0; i < v->segmentCount; ++i) {
        BurnAneSegment segment = v->segments[i];
        if (segment.valid > segment.capacity ||
            __builtin_add_overflow(valid, segment.valid, &valid) ||
            __builtin_add_overflow(capacity, segment.capacity, &capacity)) fail(@"Invalid ANE view segments");
    }
    if (v->axis < 4 && valid != v->shape[v->axis]) fail(@"ANE view logical extent mismatch");
    for (size_t d = 0; d < 4; ++d)
        if (!v->shape[d] || (d == v->axis ? capacity : v->shape[d]) != s->canonicalShape[d])
            fail(@"ANE view physical extent mismatch");
}

// Map one valid coordinate to its padded coordinate and contiguous run length.
static size_t viewPosition(const BurnAneView *v, size_t dim, size_t index, size_t *run) {
    *run = v->shape[dim] - index;
    if (dim != v->axis) return index;
    size_t physical = 0;
    for (size_t i = 0; i < v->segmentCount; ++i) {
        BurnAneSegment segment = v->segments[i];
        if (index < segment.valid) { *run = segment.valid - index; return physical + index; }
        index -= segment.valid;
        physical += segment.capacity;
    }
    fail(@"ANE view coordinate out of bounds");
    return 0;
}

// The ANE compiler may report a different reshape of the same canonical data.
// Flat physical order is stable, while byte strides come from that report.
static size_t surfacePosition(BurnAneSurface *s, size_t b, size_t c, size_t h, size_t w, size_t *run) {
    size_t flat = ((b * s->canonicalShape[1] + c) * s->canonicalShape[2] + h) * s->canonicalShape[3] + w;
    size_t column = flat % s->width;
    flat /= s->width;
    size_t row = flat % s->height;
    flat /= s->height;
    size_t channel = flat % s->channels;
    size_t batch = flat / s->channels;
    *run = MIN(*run, s->width - column);
    return batch * s->batchStride + channel * s->planeStride + row * s->rowStride + column * 2;
}

static void copySurface(BurnAneSurface *source, const BurnAneView *sv,
                        BurnAneSurface *target, const BurnAneView *tv) {
    validateView(source, sv); validateView(target, tv);
    if (source == target) fail(@"ANE input and output views must not alias");
    for (size_t d = 0; d < 4; ++d)
        if (sv->shape[d] != tv->shape[d]) fail(@"ANE transfer logical shape mismatch");
    if (IOSurfaceLock(source->surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
        fail(@"ANE source surface lock failed");
    @try {
        if (IOSurfaceLock(target->surface, 0, NULL) != kIOReturnSuccess) fail(@"ANE target surface lock failed");
        @try {
            const uint8_t *src = IOSurfaceGetBaseAddress(source->surface);
            uint8_t *dst = IOSurfaceGetBaseAddress(target->surface);
            memset(dst, 0, IOSurfaceGetAllocSize(target->surface));
            size_t unused;
            for (size_t b = 0; b < sv->shape[0]; ++b) {
                size_t sb = viewPosition(sv, 0, b, &unused), tb = viewPosition(tv, 0, b, &unused);
                for (size_t c = 0; c < sv->shape[1]; ++c) {
                    size_t sc = viewPosition(sv, 1, c, &unused), tc = viewPosition(tv, 1, c, &unused);
                    for (size_t h = 0; h < sv->shape[2]; ++h) {
                        size_t sh = viewPosition(sv, 2, h, &unused), th = viewPosition(tv, 2, h, &unused);
                        for (size_t w = 0; w < sv->shape[3];) {
                            size_t sr, tr;
                            size_t sw = viewPosition(sv, 3, w, &sr), tw = viewPosition(tv, 3, w, &tr);
                            size_t so = surfacePosition(source, sb, sc, sh, sw, &sr);
                            size_t to = surfacePosition(target, tb, tc, th, tw, &tr);
                            size_t count = MIN(sr, tr);
                            memcpy(dst + to, src + so, count * 2);
                            w += count;
                        }
                    }
                }
            }
        } @finally { IOSurfaceUnlock(target->surface, 0, NULL); }
    } @finally { IOSurfaceUnlock(source->surface, kIOSurfaceLockReadOnly, NULL); }
}

@interface BurnDirectGraph : NSObject {
@public id<BurnAnePrivate> model;
    id descriptor;
    id request;
    NSArray<BurnAneSurface *> *inputs, *outputs;
    NSArray<BurnAneSurface *> *boundInputs;
    BOOL loaded, cacheHit;
    double compileMs, loadMs;
}
@end
@implementation BurnDirectGraph
- (void)dealloc {
    if (loaded) {
        @try { NSError *error = nil; [model unloadWithQoS:21 error:&error]; }
        @catch (NSException *exception) { /* No exception may cross the Rust ABI. */ }
    }
}
@end

static void stageCompile(id<BurnAnePrivate> model, NSString *path, NSData *mil, NSData *weights) {
    NSFileManager *fm = NSFileManager.defaultManager;
    NSError *error = nil;
    check([fm createDirectoryAtPath:[path stringByAppendingPathComponent:@"weights"]
       withIntermediateDirectories:YES attributes:nil error:&error], @"create ANE source directory", error);
    check([mil writeToFile:[path stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&error], @"write MIL", error);
    if (weights.length) check([weights writeToFile:[path stringByAppendingPathComponent:@"weights/weight.bin"] options:NSDataWritingAtomic error:&error], @"write weights", error);
    BOOL ok = [model compileWithQoS:21 options:@{} error:&error];
    check(ok, @"ANE compile", error);
}

// Immutable packed blobs can be shared by descriptors for different row/KV
// shapes. NSData owns the allocation independently of every Rust source buffer.
void *burn_direct_weights_create(const uint8_t *bytes, size_t length, char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            NSData *data = [[NSData alloc] initWithBytes:bytes length:length];
            if (!data) fail(@"Cannot allocate ANE weight data");
            return (__bridge_retained void *)data;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return NULL; }
    }
}
void burn_direct_weights_release(void *handle) {
    @autoreleasepool { __unused NSData *data = (__bridge_transfer NSData *)handle; }
}

void *burn_direct_open(const uint8_t *milBytes, size_t milLen, void *weightHandle,
                       const uint8_t *ioBytes, size_t ioLen, char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            static void *framework;
            static dispatch_once_t once;
            dispatch_once(&once, ^{ framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL); });
            if (!framework) fail(@"AppleNeuralEngine framework unavailable");
            Class<BurnAnePrivate> descriptorClass = (Class)NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class<BurnAnePrivate> modelClass = (Class)NSClassFromString(@"_ANEInMemoryModel");
            Class<BurnAnePrivate> surfaceClass = (Class)NSClassFromString(@"_ANEIOSurfaceObject");
            Class<BurnAnePrivate> requestClass = (Class)NSClassFromString(@"_ANERequest");
            if (!descriptorClass || !modelClass || !surfaceClass || !requestClass) fail(@"Direct ANE classes unavailable");
            NSData *mil = [NSData dataWithBytes:milBytes length:milLen];
            NSData *weights = weightHandle ? (__bridge NSData *)weightHandle : [NSData data];
            NSDictionary *io = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:ioBytes length:ioLen] options:0 error:nil];
            if (!io) fail(@"Invalid ANE I/O descriptor");
            NSDictionary *weightDict = weights.length ? @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}} : @{};
            BurnDirectGraph *graph = [BurnDirectGraph new];
            graph->descriptor = [descriptorClass modelWithMILText:mil weights:weightDict optionsPlist:nil];
            graph->model = [modelClass inMemoryModelWithDescriptor:graph->descriptor];
            if (!graph->model) fail(@"Cannot create direct ANE model");
            NSString *hex = [graph->model hexStringIdentifier];
            NSCharacterSet *invalid = [[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-"] invertedSet];
            if (!hex.length || [hex rangeOfCharacterFromSet:invalid].location != NSNotFound) fail([NSString stringWithFormat:@"Invalid ANE cache identifier: %@",hex]);
            NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
            NSString *lockPath = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"burn-ane-" stringByAppendingString:[hex stringByAppendingString:@".lock"]]];
            int fd = open(lockPath.fileSystemRepresentation, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (fd < 0) fail(@"Cannot open ANE compilation lock");
            BOOL locked = NO;
            @try {
                if (flock(fd, LOCK_EX)) fail(@"Cannot acquire ANE compilation lock");
                locked = YES;
                graph->cacheHit = [graph->model compiledModelExists];
                if (!graph->cacheHit) {
                    double start = now();
                    stageCompile(graph->model, path, mil, weights);
                    graph->compileMs += now() - start;
                }
                NSError *loadError = nil;
                double start = now();
                BOOL ok = [graph->model loadWithQoS:21 options:@{} error:&loadError];
                graph->loadMs += now() - start;
                BOOL exhausted = !ok && [loadError.domain isEqualToString:@"com.apple.appleneuralengine"] && loadError.code == 54;
                if (exhausted) fail([@"ANE_RESOURCE_EXHAUSTED: " stringByAppendingString:loadError.description]);
                if (!ok && graph->cacheHit) {
                    graph->cacheHit = NO;
                    start = now();
                    stageCompile(graph->model, path, mil, weights);
                    graph->compileMs += now() - start;
                    start = now();
                    ok = [graph->model loadWithQoS:21 options:@{} error:&loadError];
                    graph->loadMs += now() - start;
                }
                if (!ok && [loadError.domain isEqualToString:@"com.apple.appleneuralengine"] && loadError.code == 54)
                    fail([@"ANE_RESOURCE_EXHAUSTED: " stringByAppendingString:loadError.description]);
                check(ok, @"ANE load", loadError);
                graph->loaded = YES;
            } @finally {
                // aned owns its compiled cache. This directory holds source copies only.
                if (locked) [NSFileManager.defaultManager removeItemAtPath:path error:nil];
                close(fd);
            }
            id<BurnAnePrivate> inner = [graph->model model];
            NSDictionary *attributes = [inner modelAttributes];
            NSArray *networks = attributes[@"NetworkStatusList"];
            if (networks.count != 1) fail(@"Unsupported ANE network layout");
            NSDictionary *network = networks[0];
            NSMutableArray *objects[2] = {[NSMutableArray new], [NSMutableArray new]};
            NSMutableArray *indices[2] = {[NSMutableArray new], [NSMutableArray new]};
            NSArray *layouts[2] = {network[@"LiveInputList"], network[@"LiveOutputList"]};
            NSArray *expected[2] = {io[@"inputs"], io[@"outputs"]};
            for (size_t kind = 0; kind < 2; ++kind) {
                if (!layouts[kind] || layouts[kind].count != expected[kind].count) fail([NSString stringWithFormat:@"ANE I/O count mismatch: %@", network]);
                NSMutableArray *buffers = [NSMutableArray new];
                for (NSUInteger i = 0; i < layouts[kind].count; ++i) {
                    BurnAneSurface *buffer = [[BurnAneSurface alloc] initWithLayout:layouts[kind][i] expected:expected[kind][i]];
                    [buffers addObject:buffer];
                    [objects[kind] addObject:[surfaceClass objectWithIOSurface:buffer->surface]];
                    [indices[kind] addObject:@(i)];
                }
                if (kind == 0) graph->inputs = buffers; else graph->outputs = buffers;
            }
            // perfStats takes a collection on macOS 27. nil works across versions.
            graph->request = [requestClass requestWithInputs:objects[0] inputIndices:indices[0]
                outputs:objects[1] outputIndices:indices[1] weightsBuffer:nil perfStats:nil procedureIndex:@0];
            if (!graph->request) fail(@"Cannot create ANE evaluation request");
            [graph->model setDescriptor:nil];
            return (__bridge_retained void *)graph;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return NULL; }
    }
}

// Keep the compiled description and I/O request, but release the model instance:
// unload alone does not free all per-instance ANE resources on the local OS.
int burn_direct_set_loaded(void *handle, int loaded, char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            BurnDirectGraph *graph = (__bridge BurnDirectGraph *)handle;
            if (graph->loaded == (loaded != 0)) return 0;
            id<BurnAnePrivate> model = graph->model;
            if (loaded) {
                Class<BurnAnePrivate> modelClass = (Class)NSClassFromString(@"_ANEInMemoryModel");
                model = [modelClass inMemoryModelWithDescriptor:graph->descriptor];
                if (!model) fail(@"Cannot recreate direct ANE model");
            }
            NSError *failure = nil;
            BOOL ok = loaded
                ? [model loadWithQoS:21 options:@{} error:&failure]
                : [model unloadWithQoS:21 error:&failure];
            if (loaded && !ok && [failure.domain isEqualToString:@"com.apple.appleneuralengine"] && failure.code == 54)
                fail([@"ANE_RESOURCE_EXHAUSTED: " stringByAppendingString:failure.description]);
            check(ok, loaded ? @"ANE reload" : @"ANE unload", failure);
            if (loaded) [model setDescriptor:nil];
            graph->model = loaded ? model : nil;
            graph->loaded = loaded != 0;
            return 0;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return -1; }
    }
}

// Rust owns logical Burn-order IEEE binary16 buffers. I/O offsets retain holes
// left by dead inputs. Copy and layout conversion share one IOSurface traversal.
int burn_direct_predict_f16(void *handle, const uint16_t *input, size_t inputCount,
                            uint16_t *output, size_t outputCount, double *timings,
                            char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            BurnDirectGraph *graph = (__bridge BurnDirectGraph *)handle;
            if (!graph->loaded) fail(@"ANE prediction requires a loaded program");
            for (BurnAneSurface *s in graph->inputs)
                if (s->offset > inputCount || s->elements > inputCount - s->offset)
                    fail(@"ANE input buffer size mismatch");
            for (BurnAneSurface *s in graph->outputs)
                if (s->offset > outputCount || s->elements > outputCount - s->offset)
                    fail(@"ANE output buffer size mismatch");
            double start = now();
            for (BurnAneSurface *s in graph->inputs) [s writeHalf:input + s->offset];
            timings[0] = now() - start;
            NSError *failure = nil;
            start = now();
            BOOL ok = [graph->model evaluateWithQoS:21 options:@{} request:graph->request error:&failure];
            timings[1] = now() - start;
            check(ok, @"ANE evaluate", failure);
            start = now();
            for (BurnAneSurface *s in graph->outputs) [s readHalf:output + s->offset];
            timings[2] = now() - start;
            return 0;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return -1; }
    }
}

void *burn_direct_output_retain(void *handle, size_t index) {
    BurnDirectGraph *graph = (__bridge BurnDirectGraph *)handle;
    return index < graph->outputs.count ? (__bridge_retained void *)graph->outputs[index] : NULL;
}

void *burn_direct_input_retain(void *handle, size_t index) {
    BurnDirectGraph *graph = (__bridge BurnDirectGraph *)handle;
    if (index >= graph->inputs.count) return NULL;
    return (__bridge_retained void *)graph->inputs[index];
}

int burn_direct_surface_info(void *handle, size_t *values, char *error, size_t capacity) {
    @autoreleasepool { @try {
        BurnAneSurface *s = (__bridge BurnAneSurface *)handle;
        if (!s || !values) fail(@"Missing shared surface");
        size_t info[13] = {s->batches, s->channels, s->height, s->width,
            s->batchStride, s->planeStride, s->rowStride, 2,
            s->canonicalShape[0], s->canonicalShape[1], s->canonicalShape[2], s->canonicalShape[3],
            IOSurfaceGetAllocSize(s->surface)};
        memcpy(values, info, sizeof(info));
        return 0;
    } @catch (NSException *exception) { errorText(exception, error, capacity); return -1; } }
}

// The Metal buffer owns a Rust Arc lease as well as the IOSurface. Releasing the
// Rust Tensor while Metal commands are pending cannot recycle the ANE output.
void *burn_direct_surface_metal_buffer(void *handle, void *deviceHandle, void *owner,
                                      void (*releaseOwner)(void *), char *error, size_t capacity) {
    @autoreleasepool { @try {
        BurnAneSurface *s = (__bridge BurnAneSurface *)handle;
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        if (!device.hasUnifiedMemory || !s) fail(@"Shared ANE tensors require a unified-memory Metal device");
        void *base = IOSurfaceGetBaseAddress(s->surface);
        size_t bytes = IOSurfaceGetAllocSize(s->surface), page = (size_t)getpagesize();
        if ((uintptr_t)base % page || bytes % page) fail(@"IOSurface allocation is not page aligned");
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:base length:bytes
            options:MTLResourceStorageModeShared
            deallocator:^(void *pointer, NSUInteger length) {
                (void)pointer; (void)length; (void)s; releaseOwner(owner);
            }];
        if (!buffer) fail(@"Cannot wrap IOSurface as a Metal buffer");
        return (__bridge_retained void *)buffer;
    } @catch (NSException *exception) { errorText(exception, error, capacity); return NULL; } }
}
void *burn_direct_surface_clone(void *handle, char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            BurnAneSurface *source = (__bridge BurnAneSurface *)handle;
            BurnAneSurface *copy = [[BurnAneSurface alloc] initWithLayout:source->layoutDescription expected:source->ioDescription];
            return (__bridge_retained void *)copy;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return NULL; }
    }
}
int burn_direct_surface_read(void *handle, uint16_t *output, size_t count, char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            BurnAneSurface *s = (__bridge BurnAneSurface *)handle;
            if (count != s->elements) fail(@"ANE surface read size mismatch");
            [s readHalf:output];
            return 0;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return -1; }
    }
}
void burn_direct_surface_release(void *handle) {
    @autoreleasepool { __unused BurnAneSurface *s = (__bridge_transfer BurnAneSurface *)handle; }
}

int burn_direct_predict_surfaces(void *handle, const BurnAneInput *bindings, size_t inputCount,
                                 void *const *outputHandles, size_t outputCount, double *timings,
                                 char *error, size_t capacity) {
    @autoreleasepool {
        @try {
            BurnDirectGraph *graph = (__bridge BurnDirectGraph *)handle;
            if (!graph->loaded || inputCount != graph->inputs.count || outputCount != graph->outputs.count)
                fail(@"ANE surface prediction state mismatch");
            double start = now();
            BOOL changed = NO;
            NSMutableArray *selectedInputs = [NSMutableArray arrayWithCapacity:inputCount];
            for (size_t i = 0; i < inputCount; ++i) {
                const BurnAneInput *binding = &bindings[i];
                BurnAneSurface *target = graph->inputs[i];
                BurnAneSurface *source = (__bridge BurnAneSurface *)binding->surface;
                BurnAneSurface *selected = target;
                if (source) {
                    validateView(source, &binding->source);
                    validateView(target, &binding->target);
                    BOOL same = [source->layoutDescription isEqual:target->layoutDescription]
                        && !memcmp(source->canonicalShape, target->canonicalShape, sizeof(source->canonicalShape))
                        && !memcmp(binding->source.shape, binding->target.shape, sizeof(binding->source.shape))
                        && binding->source.axis == binding->target.axis
                        && binding->source.segmentCount == binding->target.segmentCount;
                    for (size_t j = 0; same && j < binding->source.segmentCount; ++j)
                        same = binding->source.segments[j].valid == binding->target.segments[j].valid
                            && binding->source.segments[j].capacity == binding->target.segments[j].capacity;
                    if (same) selected = source;
                }
                [selectedInputs addObject:selected];
                changed |= !graph->boundInputs || selected != graph->boundInputs[i];
            }
            NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:outputCount];
            for (size_t i = 0; i < outputCount; ++i) {
                BurnAneSurface *s = (__bridge BurnAneSurface *)outputHandles[i];
                if (!s || ![s->layoutDescription isEqual:graph->outputs[i]->layoutDescription] ||
                    ![s->ioDescription isEqual:graph->outputs[i]->ioDescription]) fail(@"ANE output surface layout mismatch");
                changed |= s != graph->outputs[i];
                [outputs addObject:s];
            }
            if (changed) {
                Class<BurnAnePrivate> surfaceClass = (Class)NSClassFromString(@"_ANEIOSurfaceObject");
                Class<BurnAnePrivate> requestClass = (Class)NSClassFromString(@"_ANERequest");
                NSMutableArray *objects[2] = {[NSMutableArray new], [NSMutableArray new]};
                NSMutableArray *indices[2] = {[NSMutableArray new], [NSMutableArray new]};
                NSArray *buffers[2] = {selectedInputs, outputs};
                for (size_t k = 0; k < 2; ++k)
                    for (BurnAneSurface *s in buffers[k]) {
                        [objects[k] addObject:[surfaceClass objectWithIOSurface:s->surface]];
                        [indices[k] addObject:@(indices[k].count)];
                    }
                id request = [requestClass requestWithInputs:objects[0] inputIndices:indices[0]
                    outputs:objects[1] outputIndices:indices[1] weightsBuffer:nil perfStats:nil procedureIndex:@0];
                if (!request) fail(@"Cannot rebind ANE output surfaces");
                graph->outputs = outputs;
                graph->boundInputs = selectedInputs;
                graph->request = request;
            }
            for (size_t i = 0; i < inputCount; ++i) {
                const BurnAneInput *binding = &bindings[i];
                BurnAneSurface *target = graph->inputs[i];
                if (binding->surface && (__bridge BurnAneSurface *)binding->surface == selectedInputs[i]) {
                    // Same IOSurface: the completed producer hands ownership to ANE.
                    continue;
                } else if (binding->surface) {
                    copySurface((__bridge BurnAneSurface *)binding->surface, &binding->source, target, &binding->target);
                } else {
                    if (!binding->host || binding->hostCount != target->elements) fail(@"ANE host input size mismatch");
                    [target writeHalf:binding->host];
                }
            }
            timings[0] = now() - start;
            NSError *failure = nil;
            start = now();
            BOOL ok = [graph->model evaluateWithQoS:21 options:@{} request:graph->request error:&failure];
            timings[1] = now() - start;
            check(ok, @"ANE evaluate", failure);
            timings[2] = 0;
            return 0;
        } @catch (NSException *exception) { errorText(exception, error, capacity); return -1; }
    }
}
double burn_direct_info(void *handle, int kind) {
    BurnDirectGraph *graph = (__bridge BurnDirectGraph *)handle;
    switch (kind) { case 0: return graph->cacheHit; case 1: return graph->compileMs; case 2: return graph->loadMs; default: return 0; }
}
void burn_direct_close(void *handle) {
    @autoreleasepool { __unused BurnDirectGraph *graph = (__bridge_transfer BurnDirectGraph *)handle; }
}
