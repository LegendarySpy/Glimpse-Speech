import Foundation
import XCTest
@testable import GlimpseSpeechFluidBridge

final class ModelPathsTests: XCTestCase {
    func testValidateAcceptsExistingDirectories() throws {
        let root = makeTempDir(name: "models_ok")
        let asr = root.appendingPathComponent("asr")
        let diar = root.appendingPathComponent("diar")
        try FileManager.default.createDirectory(at: asr, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: diar, withIntermediateDirectories: true)

        let config = BridgeConfig(
            schemaVersion: 1,
            asrModelDir: asr.path,
            diarizationModelDir: diar.path,
            runtimeMacOSMajor: 14
        )

        XCTAssertNoThrow(try ModelPaths.validate(config: config))
    }

    func testValidateRejectsMissingAsrDirectory() throws {
        let config = BridgeConfig(
            schemaVersion: 1,
            asrModelDir: "/this/does/not/exist",
            diarizationModelDir: nil,
            runtimeMacOSMajor: 14
        )

        XCTAssertThrowsError(try ModelPaths.validate(config: config)) { error in
            guard case let BridgeError.modelNotFound(message) = error else {
                XCTFail("expected modelNotFound, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("ASR"))
        }
    }

    func testValidateRejectsOldMacOSVersion() {
        let config = BridgeConfig(
            schemaVersion: 1,
            asrModelDir: "/tmp",
            diarizationModelDir: nil,
            runtimeMacOSMajor: 13
        )

        XCTAssertThrowsError(try ModelPaths.validate(config: config)) { error in
            guard case BridgeError.unsupportedPlatform = error else {
                XCTFail("expected unsupportedPlatform, got \(error)")
                return
            }
        }
    }

    private func makeTempDir(name: String) -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("glimpse_swift_bridge_\(name)_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
