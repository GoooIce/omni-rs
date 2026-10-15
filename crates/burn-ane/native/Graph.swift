import CoreML
import Foundation
import Darwin
import CryptoKit

private func failure(_ message: String) -> NSError {
    NSError(domain: "burn.ane", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

private func writeError(_ error: Error, _ output: UnsafeMutablePointer<CChar>?, _ capacity: Int) {
    guard let output, capacity > 0 else { return }
    let bytes = Array(error.localizedDescription.utf8.prefix(capacity - 1))
    for (i, byte) in bytes.enumerated() { output[i] = CChar(bitPattern: byte) }
    output[bytes.count] = 0
}

// Separate OS builds and hardware families. The digest supplied by Rust includes
// every graph field and weight byte; v3 identifies this runtime/compute policy.
private let cacheNamespace: String = {
    var length = 0
    sysctlbyname("hw.model", nil, &length, nil, 0)
    var model = [CChar](repeating: 0, count: max(length, 1))
    _ = model.withUnsafeMutableBufferPointer { sysctlbyname("hw.model", $0.baseAddress, &length, nil, 0) }
    let tag = "v3-\(ProcessInfo.processInfo.operatingSystemVersionString)-\(String(cString: model))"
    return tag.replacingOccurrences(of: "/", with: "_")
}()

private let manifestName = ".burn-ane-checksums.json"
private func checksums(_ root: URL) throws -> [String: String] {
    guard let files = FileManager.default.enumerator(atPath: root.path) else { throw failure("cannot enumerate cached graph") }
    var result: [String: String] = [:]
    for case let relative as String in files {
        if relative == manifestName { continue }
        let url = root.appendingPathComponent(relative)
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true else { throw failure("unexpected link in cached graph") }
        guard values.isRegularFile == true else { continue }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while let data = try file.read(upToCount: 1024 * 1024), !data.isEmpty { digest.update(data: data) }
        result[relative] = digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    return result
}

private func validateCache(_ root: URL) throws {
    let expected = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: root.appendingPathComponent(manifestName)))
    // Missing/truncated Core ML bundles can throw an uncaught C++ exception inside
    // Apple's loader. Validate the entire private copy before calling Core ML.
    let actual = try checksums(root)
    guard expected["coremldata.bin"] != nil, expected == actual else {
        throw failure("cached graph checksum mismatch (expected: \(expected.keys.sorted()), actual: \(actual.keys.sorted()))")
    }
}

private func bundleSize(_ url: URL) -> UInt64 {
    guard let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
    return files.compactMap { $0 as? URL }.reduce(0) {
        $0 + UInt64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
}

// Keep lock files outside bundles: replacing a bundle must not replace its lock
// inode. The namespace lock serializes publication/eviction; shared entry leases
// protect live models across devices, threads and processes.
private final class FileLease {
    private let fd: Int32
    init(_ url: URL, _ operation: Int32) throws {
        let descriptor = Darwin.open(url.path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw failure("cannot open cache lease: \(url.path)") }
        if flock(descriptor, operation) != 0 {
            Darwin.close(descriptor)
            throw failure("cache entry is in use")
        }
        fd = descriptor
    }
    func share() throws {
        guard flock(fd, LOCK_SH) == 0 else { throw failure("cannot share cache lease") }
    }
    deinit { Darwin.close(fd) }
}

private func leaseURL(_ bundle: URL) -> URL { bundle.appendingPathExtension("lock") }

// The budget is soft while models are live. Never remove their source packages:
// Core ML may lazily read them after MLModel initialization.
private func trimCache(_ root: URL, _ budget: UInt64) {
    let fm = FileManager.default
    do {
        let entries = try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.contentModificationDateKey])
            .filter { $0.pathExtension == "mlmodelc" }
            .map { ($0, bundleSize($0), (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
            .sorted { $0.2 < $1.2 }
        var total = entries.reduce(UInt64(0)) { $0 + $1.1 }
        for (url, size, _) in entries where total > budget {
            if let lease = try? FileLease(leaseURL(url), LOCK_EX | LOCK_NB) {
                try withExtendedLifetime(lease) {
                    try fm.removeItem(at: url)
                    total -= size
                }
            }
        }
    } catch {} // A read-only/full cache must not fail inference.
}

private final class GraphTimings {
    var compile = 0.0
    var prepare = 0.0
    var plan = 0.0
    var load = 0.0
}

private func now() -> Double { ProcessInfo.processInfo.systemUptime * 1000 }

private final class CompiledBundle {
    let url: URL
    let diskHit: Bool
    private let lease: FileLease?
    private let temporary: Bool

    init(_ url: URL, _ lease: FileLease?, _ diskHit: Bool, temporary: Bool = false) {
        self.url = url; self.lease = lease; self.diskHit = diskHit; self.temporary = temporary
    }
    deinit {
        if temporary { try? FileManager.default.removeItem(at: url) }
    }

    static func prepare(_ source: URL, _ destination: URL?, _ budget: UInt64, _ timings: GraphTimings) throws -> CompiledBundle {
        let fm = FileManager.default
        let started = now()
        defer { timings.prepare = now() - started - timings.compile }
        func compile() throws -> URL {
            let start = now()
            defer { timings.compile += now() - start }
            return try MLModel.compileModel(at: source)
        }
        if let destination {
            let root = destination.deletingLastPathComponent()
            // Cache I/O is optional. Compile/load errors themselves still propagate.
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
            if let namespace = try? FileLease(root.appendingPathComponent(".lock"), LOCK_EX) {
                let result: CompiledBundle? = try withExtendedLifetime(namespace) {
                    if fm.fileExists(atPath: destination.path),
                       let lease = try? FileLease(leaseURL(destination), LOCK_SH),
                       (try? validateCache(destination)) != nil {
                        try? fm.setAttributes([.modificationDate: Date()], ofItemAtPath: destination.path)
                        trimCache(root, budget)
                        return CompiledBundle(destination, lease, true)
                    }
                    // A corrupt entry can be replaced only when nobody is using it.
                    guard let lease = try? FileLease(leaseURL(destination), LOCK_EX | LOCK_NB) else { return nil }
                    return try withExtendedLifetime(lease) {
                        try? fm.removeItem(at: destination)
                        let fresh = try compile()
                        var keep = false
                        defer { if !keep { try? fm.removeItem(at: fresh) } }
                        let bytes = bundleSize(fresh)
                        guard bytes > 0, bytes <= budget else {
                            keep = true
                            return CompiledBundle(fresh, nil, false, temporary: true)
                        }
                        let stage = root.appendingPathComponent(".\(UUID().uuidString).tmp")
                        defer { try? fm.removeItem(at: stage) }
                        do {
                            try fm.copyItem(at: fresh, to: stage)
                            try JSONEncoder().encode(checksums(stage)).write(to: stage.appendingPathComponent(manifestName), options: .atomic)
                            try fm.moveItem(at: stage, to: destination)
                            try lease.share()
                            trimCache(root, budget)
                            // Publish BEFORE the first MLComputePlan/MLModel load.
                            // Device specialization is keyed by this stable path.
                            return CompiledBundle(destination, lease, false)
                        } catch {
                            keep = true
                            return CompiledBundle(fresh, nil, false, temporary: true)
                        }
                    }
                }
                if let result { return result }
            }
        }
        return CompiledBundle(try compile(), nil, false, temporary: true)
    }
}

// A model-independent runtime. Graphs, tensor names and shapes come from Burn IR.
private final class Graph {
    private var model: MLModel!
    private let bundle: CompiledBundle
    let inputs: [(String, MLMultiArray)]
    let outputs: [String]
    let aneOperations: Int
    let cpuOperations: Int
    let aneComputeOperations: Int
    let diskCacheHit: Bool
    let assignmentsJSON: [UInt8]
    let timings: GraphTimings

    init(path: String, cache: String?, cacheBudget: UInt64) throws {
        let destination = cache.map { value -> URL in
            let key = URL(fileURLWithPath: value)
            return key.deletingLastPathComponent().appendingPathComponent(cacheNamespace)
                .appendingPathComponent(key.lastPathComponent + ".mlmodelc")
        }
        let timings = GraphTimings()
        self.timings = timings
        bundle = try CompiledBundle.prepare(URL(fileURLWithPath: path), destination, cacheBudget, timings)
        let (loadedModel, ane, cpu, aneCompute, assignments) = try Graph.load(bundle.url, timings)
        model = loadedModel
        inputs = try model.modelDescription.inputDescriptionsByName.sorted { $0.key < $1.key }.map { name, desc in
            guard let constraint = desc.multiArrayConstraint, constraint.dataType == .float32 else {
                throw failure("expected float32 graph input: \(name)")
            }
            return (name, try MLMultiArray(shape: constraint.shape, dataType: .float32))
        }
        outputs = model.modelDescription.outputDescriptionsByName.keys.sorted()
        aneOperations = ane; cpuOperations = cpu; aneComputeOperations = aneCompute
        diskCacheHit = bundle.diskHit
        assignmentsJSON = Array(try JSONSerialization.data(withJSONObject: assignments, options: [.sortedKeys]))
    }

    private static func load(_ url: URL, _ timings: GraphTimings) throws -> (MLModel, Int, Int, Int, [String: Int]) {
        let configuration = MLModelConfiguration()
        // No GPU execution can be mistaken for ANE execution in this runtime.
        configuration.computeUnits = .cpuAndNeuralEngine
        let planStart = now()
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<MLComputePlan, Error>?
        Task.detached {
            do { result = .success(try await MLComputePlan.load(contentsOf: url, configuration: configuration)) }
            catch { result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        timings.plan = now() - planStart
        let plan = try result!.get()
        var ane = 0, cpu = 0, aneCompute = 0
        var assignments: [String: Int] = [:]
        func count(_ usage: MLComputePlan.DeviceUsage?, _ kind: String) {
            guard let usage else { return }
            switch usage.preferred {
            case .neuralEngine:
                assignments["ane:\(kind)", default: 0] += 1
                ane += 1
                if ["convolution", "inner_product", "batch_matmul", "conv", "linear", "matmul"].contains(kind) {
                    aneCompute += 1
                }
            case .cpu: cpu += 1; assignments["cpu:\(kind)", default: 0] += 1
            default: break
            }
        }
        switch plan.modelStructure {
        case .neuralNetwork(let network):
            for layer in network.layers { count(plan.deviceUsage(for: layer), layer.type) }
        case .program(let program):
            func visit(_ block: MLModelStructure.Program.Block) {
                for op in block.operations {
                    count(plan.deviceUsage(for: op), op.operatorName)
                    for child in op.blocks { visit(child) }
                }
            }
            for function in program.functions.values { visit(function.block) }
        default: break
        }
        let stableAssignments = assignments.keys.sorted().map { "\($0)=\(assignments[$0]!)" }.joined(separator: ",")
        guard aneCompute > 0 else { throw failure("Core ML assigned no matrix multiplication or convolution to ANE: \(stableAssignments)") }
        let loadStart = now()
        let model = try MLModel(contentsOf: url, configuration: configuration)
        timings.load = now() - loadStart
        return (model, ane, cpu, aneCompute, assignments)
    }

    // Release Core ML before the bundle's shared cache lease.
    deinit { model = nil }

    // Bulk-copy contiguous buffers; only strided outputs need index conversion.
    // Cache NSNumber conversions outside the hot loop (important for streaming).
    private func copy(_ source: UnsafePointer<Float>, _ destination: UnsafeMutablePointer<Float>,
                      _ array: MLMultiArray, intoArray: Bool) {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        var contiguous = true, expected = 1
        for dim in (0..<shape.count).reversed() {
            if shape[dim] > 1 && strides[dim] != expected { contiguous = false }
            expected *= shape[dim]
        }
        if contiguous { destination.update(from: source, count: array.count); return }
        for flat in 0..<array.count {
            var index = flat, offset = 0
            for dim in (0..<shape.count).reversed() {
                offset += (index % shape[dim]) * strides[dim]
                index /= shape[dim]
            }
            if intoArray { destination[offset] = source[flat] }
            else { destination[flat] = source[offset] }
        }
    }

    func predict(_ source: UnsafePointer<Float>, _ inputCount: Int,
                 _ destination: UnsafeMutablePointer<Float>, _ outputCount: Int) throws {
        guard inputs.reduce(0, { $0 + $1.1.count }) == inputCount else { throw failure("graph input length mismatch") }
        var consumed = 0
        for (_, array) in inputs {
            let data = array.dataPointer.assumingMemoryBound(to: Float.self)
            copy(source.advanced(by: consumed), data, array, intoArray: true)
            consumed += array.count
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: Dictionary(uniqueKeysWithValues: inputs))
        let prediction = try model.prediction(from: provider)
        var written = 0
        for name in outputs {
            guard let array = prediction.featureValue(for: name)?.multiArrayValue,
                  array.dataType == .float32, array.count <= outputCount - written else {
                throw failure("invalid graph output: \(name)")
            }
            let data = array.dataPointer.assumingMemoryBound(to: Float.self)
            copy(UnsafePointer(data), destination.advanced(by: written), array, intoArray: false)
            written += array.count
        }
        guard written == outputCount else { throw failure("graph output length mismatch") }
    }
}

@_cdecl("burn_ane_open")
public func openGraph(_ path: UnsafePointer<CChar>, _ cache: UnsafePointer<CChar>?, _ cacheBudget: UInt64,
                      _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: Int) -> UnsafeMutableRawPointer? {
    autoreleasepool {
        do { return Unmanaged.passRetained(try Graph(path: String(cString: path), cache: cache.map { String(cString: $0) }, cacheBudget: cacheBudget)).toOpaque() }
        catch { writeError(error, errorBuffer, capacity); return nil }
    }
}

@_cdecl("burn_ane_predict")
public func predictGraph(_ handle: UnsafeMutableRawPointer, _ input: UnsafePointer<Float>, _ inputCount: Int,
                         _ output: UnsafeMutablePointer<Float>, _ outputCount: Int,
                         _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: Int) -> Int32 {
    autoreleasepool {
        do {
            try Unmanaged<Graph>.fromOpaque(handle).takeUnretainedValue().predict(input, inputCount, output, outputCount)
            return 0
        } catch { writeError(error, errorBuffer, capacity); return -1 }
    }
}

@_cdecl("burn_ane_plan_count")
public func planCount(_ handle: UnsafeMutableRawPointer, _ kind: Int32) -> Int {
    let graph = Unmanaged<Graph>.fromOpaque(handle).takeUnretainedValue()
    if kind == 3 { return graph.diskCacheHit ? 1 : 0 }
    if kind == 2 { return graph.aneComputeOperations }
    return kind == 1 ? graph.cpuOperations : graph.aneOperations
}

@_cdecl("burn_ane_close")
public func closeGraph(_ handle: UnsafeMutableRawPointer) { Unmanaged<Graph>.fromOpaque(handle).release() }

@_cdecl("burn_ane_assignments")
public func assignments(_ handle: UnsafeMutableRawPointer, _ output: UnsafeMutablePointer<UInt8>?, _ capacity: Int) -> Int {
    let bytes = Unmanaged<Graph>.fromOpaque(handle).takeUnretainedValue().assignmentsJSON
    if let output, capacity >= bytes.count { bytes.withUnsafeBufferPointer { output.update(from: $0.baseAddress!, count: bytes.count) } }
    return bytes.count
}

@_cdecl("burn_ane_timing")
public func timing(_ handle: UnsafeMutableRawPointer, _ kind: Int32) -> Double {
    let t = Unmanaged<Graph>.fromOpaque(handle).takeUnretainedValue().timings
    switch kind {
    case 0: return t.compile
    case 1: return t.prepare
    case 2: return t.plan
    default: return t.load
    }
}

