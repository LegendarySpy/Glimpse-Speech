import Foundation

let bridgeSchemaVersion = 1

enum BridgeError: Error, Equatable {
    case invalidPayload(String)
    case invalidConfig(String)
    case unsupportedPlatform(String)
    case modelNotFound(String)
    case fluidUnavailable(String)
    case internalFailure(String)

    var code: String {
        switch self {
        case .invalidPayload:
            return "invalid_payload"
        case .invalidConfig:
            return "invalid_config"
        case .unsupportedPlatform:
            return "unsupported_platform"
        case .modelNotFound:
            return "model_not_found"
        case .fluidUnavailable:
            return "fluid_unavailable"
        case .internalFailure:
            return "internal_failure"
        }
    }

    var message: String {
        switch self {
        case let .invalidPayload(text),
             let .invalidConfig(text),
             let .unsupportedPlatform(text),
             let .modelNotFound(text),
             let .fluidUnavailable(text),
             let .internalFailure(text):
            return text
        }
    }
}

func toBridgeError(_ error: Error) -> BridgeError {
    if let bridge = error as? BridgeError {
        return bridge
    }
    return .internalFailure(String(describing: error))
}

struct BridgeConfig: Codable, Equatable {
    let schemaVersion: Int
    let asrModelDir: String
    let diarizationModelDir: String?
    let runtimeMacOSMajor: Int

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case asrModelDir = "asr_model_dir"
        case diarizationModelDir = "diarization_model_dir"
        case runtimeMacOSMajor = "runtime_macos_major"
    }
}

struct BridgeTranscribeOptions: Codable, Equatable {
    let schemaVersion: Int
    let languageHint: String?
    let vocabulary: [String]
    let timestamps: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case languageHint = "language_hint"
        case vocabulary
        case timestamps
    }
}

struct BridgeDiarizeOptions: Codable, Equatable {
    let schemaVersion: Int
    let speakerCount: Int?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case speakerCount = "speaker_count"
    }
}

struct BridgeSegment: Codable, Equatable {
    let startMs: UInt64
    let endMs: UInt64
    let text: String

    enum CodingKeys: String, CodingKey {
        case startMs = "start_ms"
        case endMs = "end_ms"
        case text
    }
}

struct BridgeWord: Codable, Equatable {
    let startMs: UInt64
    let endMs: UInt64
    let text: String
    let segmentIndex: Int?

    enum CodingKeys: String, CodingKey {
        case startMs = "start_ms"
        case endMs = "end_ms"
        case text
        case segmentIndex = "segment_index"
    }
}

struct BridgeTranscript: Codable, Equatable {
    let schemaVersion: Int
    let engine: String
    let text: String
    let segments: [BridgeSegment]
    let words: [BridgeWord]?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case engine
        case text
        case segments
        case words
    }
}

struct BridgeSpeakerTurn: Codable, Equatable {
    let startMs: UInt64
    let endMs: UInt64
    let speaker: String

    enum CodingKeys: String, CodingKey {
        case startMs = "start_ms"
        case endMs = "end_ms"
        case speaker
    }
}

struct BridgeDiarization: Codable, Equatable {
    let schemaVersion: Int
    let turns: [BridgeSpeakerTurn]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case turns
    }
}

struct BridgeErrorPayload: Codable, Equatable {
    let code: String
    let message: String
}

struct BridgeEnvelope<T: Codable>: Codable {
    let schemaVersion: Int
    let ok: Bool
    let data: T?
    let error: BridgeErrorPayload?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case ok
        case data
        case error
    }
}

enum Serialization {
    private static let decoder = JSONDecoder()
    private static let encoder = JSONEncoder()

    static func decodeConfig(from bytes: Data) throws -> BridgeConfig {
        do {
            return try decoder.decode(BridgeConfig.self, from: bytes)
        } catch {
            throw BridgeError.invalidPayload("invalid create config payload: \(error)")
        }
    }

    static func decodeTranscribeOptions(from bytes: Data) throws -> BridgeTranscribeOptions {
        do {
            return try decoder.decode(BridgeTranscribeOptions.self, from: bytes)
        } catch {
            throw BridgeError.invalidPayload("invalid transcribe options payload: \(error)")
        }
    }

    static func decodeDiarizeOptions(from bytes: Data) throws -> BridgeDiarizeOptions {
        do {
            return try decoder.decode(BridgeDiarizeOptions.self, from: bytes)
        } catch {
            throw BridgeError.invalidPayload("invalid diarize options payload: \(error)")
        }
    }

    static func encodeSuccess<T: Codable>(_ payload: T) -> Data {
        let envelope = BridgeEnvelope(
            schemaVersion: bridgeSchemaVersion,
            ok: true,
            data: payload,
            error: nil
        )
        // If encoding fails we still return deterministic JSON.
        return (try? encoder.encode(envelope))
            ?? Data(#"{"schema_version":1,"ok":false,"error":{"code":"internal_failure","message":"encoding failure"}}"#.utf8)
    }

    static func encodeError(_ error: BridgeError) -> Data {
        let envelope = BridgeEnvelope<String>(
            schemaVersion: bridgeSchemaVersion,
            ok: false,
            data: nil,
            error: BridgeErrorPayload(code: error.code, message: error.message)
        )
        return (try? encoder.encode(envelope))
            ?? Data(#"{"schema_version":1,"ok":false,"error":{"code":"internal_failure","message":"encoding failure"}}"#.utf8)
    }
}

enum ModelPaths {
    static func validate(config: BridgeConfig) throws {
        if config.schemaVersion != bridgeSchemaVersion {
            throw BridgeError.invalidConfig(
                "schema_version must be \(bridgeSchemaVersion), got \(config.schemaVersion)"
            )
        }

        if config.runtimeMacOSMajor < 14 {
            throw BridgeError.unsupportedPlatform(
                "Fluid backend requires macOS 14+, got \(config.runtimeMacOSMajor)"
            )
        }

        try validateDirectory(path: config.asrModelDir, label: "ASR")
        if let diar = config.diarizationModelDir, !diar.isEmpty {
            try validateDirectory(path: diar, label: "diarization")
        }
    }

    private static func validateDirectory(path: String, label: String) throws {
        guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BridgeError.invalidConfig("\(label) model directory is empty")
        }

        var isDirectory = ObjCBool(false)
        if !FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            throw BridgeError.modelNotFound("\(label) model directory not found: \(path)")
        }
    }
}
