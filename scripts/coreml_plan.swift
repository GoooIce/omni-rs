// Inspect actual Core ML scheduling; merely setting computeUnits is not proof of ANE use.
import CoreML
import Foundation

@main
struct InspectPlan {
    static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let compiled: URL
        if source.pathExtension == "mlmodelc" {
            compiled = source
        } else {
            compiled = try await MLModel.compileModel(at: source)
            if CommandLine.arguments.count > 2 {
                let destination = URL(fileURLWithPath: CommandLine.arguments[2])
                try FileManager.default.copyItem(at: compiled, to: destination)
            }
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
        var devices: [String: Int] = [:]
        var operators: [String: [String: Int]] = [:]
        var aneCost: Double = 0
        var totalCost: Double = 0
        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                if let usage = plan.deviceUsage(for: operation) {
                    let name: String
                    switch usage.preferred {
                    case .neuralEngine: name = "ane"
                    case .cpu: name = "cpu"
                    case .gpu: name = "gpu"
                    @unknown default: name = "unknown"
                    }
                    devices[name, default: 0] += 1
                    operators[operation.operatorName, default: [:]][name, default: 0] += 1
                    if let cost = plan.estimatedCost(of: operation) {
                        totalCost += cost.weight
                        if name == "ane" { aneCost += cost.weight }
                    }
                }
                for inner in operation.blocks { visit(inner) }
            }
        }
        if case let .program(program) = plan.modelStructure {
            for function in program.functions.values { visit(function.block) }
        }
        let result: [String: Any] = ["compiled": compiled.path, "devices": devices,
            "operators": operators, "ane_estimated_cost_fraction": totalCost > 0 ? aneCost / totalCost : 0]
        print(String(data: try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8)!)
        if devices["ane", default: 0] == 0 {
            throw NSError(domain: "r2t2.coreml", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model has no operations assigned to ANE"])
        }
    }
}
