import CoreML
import Foundation

private func writeError(_ error: Error, _ output: UnsafeMutablePointer<CChar>?, _ capacity: Int) {
    guard let output, capacity > 0 else { return }
    let bytes = Array(error.localizedDescription.utf8.prefix(capacity - 1))
    for (i, byte) in bytes.enumerated() { output[i] = CChar(bitPattern: byte) }
    output[bytes.count] = 0
}

private func failure(_ message: String) -> NSError {
    NSError(domain: "omni-r2t2.coreml", code: 1,
            userInfo: [NSLocalizedDescriptionKey: message])
}

private final class Encoder {
    let model: MLModel
    let input: MLMultiArray
    let mask: MLMultiArray
    let capacity: Int
    let width: Int
    let outputWidth: Int
    let aneOperations: Int
    // The Rust owner serializes predictions; buffers and the model stay resident.
    init(path: String, capacity: Int, width: Int, outputWidth: Int) throws {
        let url = URL(fileURLWithPath: path)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let semaphore = DispatchSemaphore(value: 0)
        var planResult: Result<MLComputePlan, Error>?
        Task.detached {
            do { planResult = .success(try await MLComputePlan.load(contentsOf: url, configuration: configuration)) }
            catch { planResult = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        let plan = try planResult!.get()
        var count = 0
        func visit(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                if let usage = plan.deviceUsage(for: op), case .neuralEngine = usage.preferred {
                    count += 1
                }
                for child in op.blocks { visit(child) }
            }
        }
        if case let .program(program) = plan.modelStructure {
            for function in program.functions.values { visit(function.block) }
        }
        guard count > 0 else { throw failure("Core ML did not assign any encoder operations to ANE") }
        self.aneOperations = count
        self.model = try MLModel(contentsOf: url, configuration: configuration)
        self.capacity = capacity
        self.width = width
        self.outputWidth = outputWidth
        self.input = try MLMultiArray(shape: [1, NSNumber(value: width), 1, NSNumber(value: capacity)], dataType: .float32)
        self.mask = try MLMultiArray(shape: [1, NSNumber(value: capacity), 1, 1], dataType: .float32)
        let expected = ["hidden": input.shape, "mask": mask.shape]
        for (name, shape) in expected {
            guard let constraint = model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint,
                  constraint.shape == shape, constraint.dataType == .float32 else {
                throw failure("Core ML input \(name) does not match manifest shape/type")
            }
        }
    }

    func predict(_ source: UnsafePointer<Float>, tokens: Int, destination: UnsafeMutablePointer<Float>) throws {
        guard tokens > 0 && tokens <= capacity else { throw failure("invalid encoder token count") }
        let inputBuffer = input.dataPointer.assumingMemoryBound(to: Float.self)
        let maskBuffer = mask.dataPointer.assumingMemoryBound(to: Float.self)
        let stride = input.strides[1].intValue
        let timeStride = input.strides[3].intValue
        for c in 0..<width {
            for t in 0..<capacity { inputBuffer[c * stride + t * timeStride] = t < tokens ? source[t * width + c] : 0 }
        }
        for t in 0..<capacity { maskBuffer[t * mask.strides[1].intValue] = t < tokens ? 0 : -10000 }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["hidden": input, "mask": mask])
        let prediction = try model.prediction(from: provider)
        guard let output = prediction.featureValue(for: "embeddings")?.multiArrayValue,
              output.shape == [1, NSNumber(value: outputWidth), 1, NSNumber(value: capacity)],
              output.dataType == .float32 else { throw failure("invalid Core ML encoder output") }
        let outputBuffer = output.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = output.strides[1].intValue
        let tokenStride = output.strides[3].intValue
        for t in 0..<tokens {
            for c in 0..<outputWidth {
                let value = outputBuffer[c * channelStride + t * tokenStride]
                guard value.isFinite else { throw failure("Core ML encoder produced NaN or infinity") }
                destination[t * outputWidth + c] = value
            }
        }
    }
}

@_cdecl("r2t2_coreml_open")
public func openEncoder(_ path: UnsafePointer<CChar>, _ capacity: Int, _ width: Int,
                        _ outputWidth: Int, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ errorCapacity: Int) -> UnsafeMutableRawPointer? {
    autoreleasepool {
        do {
            let encoder = try Encoder(path: String(cString: path), capacity: capacity, width: width, outputWidth: outputWidth)
            return Unmanaged.passRetained(encoder).toOpaque()
        } catch { writeError(error, errorBuffer, errorCapacity); return nil }
    }
}

@_cdecl("r2t2_coreml_predict")
public func predictEncoder(_ handle: UnsafeMutableRawPointer, _ input: UnsafePointer<Float>, _ tokens: Int,
                           _ output: UnsafeMutablePointer<Float>, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ errorCapacity: Int) -> Int32 {
    autoreleasepool {
        do {
            try Unmanaged<Encoder>.fromOpaque(handle).takeUnretainedValue().predict(input, tokens: tokens, destination: output)
            return 0
        } catch { writeError(error, errorBuffer, errorCapacity); return -1 }
    }
}

@_cdecl("r2t2_coreml_ane_operations")
public func aneOperations(_ handle: UnsafeMutableRawPointer) -> Int {
    Unmanaged<Encoder>.fromOpaque(handle).takeUnretainedValue().aneOperations
}

@_cdecl("r2t2_coreml_close")
public func closeEncoder(_ handle: UnsafeMutableRawPointer) {
    Unmanaged<Encoder>.fromOpaque(handle).release()
}
