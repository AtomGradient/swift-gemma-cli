import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

@main
struct GemmaCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gemma-cli",
        abstract: "CLI tool for running Gemma 3 VLM inference on Apple Silicon with MLX"
    )

    @Argument(help: "Path to the local model directory")
    var modelPath: String

    @Option(name: .long, help: "Path to an image file")
    var image: String?

    @Option(name: .shortAndLong, help: "Prompt text")
    var prompt: String = "Hello, how are you?"

    @Option(name: .shortAndLong, help: "Maximum tokens to generate")
    var maxTokens: Int = 100

    @Option(name: .shortAndLong, help: "Sampling temperature (0 = greedy)")
    var temperature: Float = 0.6

    @Option(name: .long, help: "Top-p nucleus sampling")
    var topP: Float = 1.0

    @Option(name: .long, help: "Repetition penalty")
    var repetitionPenalty: Float?

    @Option(name: .long, help: "Repetition context size")
    var repetitionContextSize: Int = 20

    func run() async throws {
        let modelURL = URL(filePath: modelPath)
        print("Loading model from: \(modelPath)...")

        let config = ModelConfiguration(
            directory: modelURL,
            extraEOSTokens: ["<end_of_turn>"]
        )

        let modelContainer = try await VLMModelFactory.shared.loadContainer(
            configuration: config
        ) { progress in
            print("  \(progress.fractionCompleted * 100, specifier: "%.0f")%", terminator: "\r")
            fflush(stdout)
        }

        print("Model loaded.")

        // Build user input
        var messages: [Chat.Message] = []
        if let imagePath = image {
            let imageURL = URL(filePath: imagePath)
            messages.append(.user(prompt, images: [.url(imageURL)]))
        } else {
            messages.append(.user(prompt))
        }

        let userInput = UserInput(chat: messages)

        // Prepare LM input
        let lmInput = try await modelContainer.prepare(input: userInput)

        // Generate
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize
        )

        print("\nPrompt: \(prompt)")
        print("------")

        Memory.peakMemory = 0

        var completionInfo: GenerateCompletionInfo?
        let stream = try await modelContainer.generate(input: lmInput, parameters: params)
        for try await item in stream {
            switch item {
            case .chunk(let text):
                print(text, terminator: "")
                fflush(stdout)
            case .info(let info):
                completionInfo = info
            default:
                break
            }
        }

        print("\n------")

        if let info = completionInfo {
            print(info.summary())
        }

        let peakMB = Memory.peakMemory / 1024 / 1024
        print("Peak memory: \(peakMB) MB")
    }
}

extension DefaultStringInterpolation {
    mutating func appendInterpolation(_ value: Double, specifier: String) {
        appendInterpolation(String(format: specifier, value))
    }
}
