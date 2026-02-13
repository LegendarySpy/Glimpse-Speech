import Foundation
import XCTest
@testable import GlimpseSpeechFluidBridge

final class SerializationTests: XCTestCase {
    func testDecodeConfigRoundtrip() throws {
        let config = BridgeConfig(
            schemaVersion: 1,
            asrModelDir: "/tmp/asr",
            diarizationModelDir: "/tmp/diar",
            runtimeMacOSMajor: 14
        )
        let data = try JSONEncoder().encode(config)
        let decoded = try Serialization.decodeConfig(from: data)
        XCTAssertEqual(decoded, config)
    }

    func testEncodeErrorEnvelope() throws {
        let data = Serialization.encodeError(.invalidPayload("bad input"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(json?["ok"] as? Bool, false)
        XCTAssertEqual(json?["schema_version"] as? Int, 1)
        let error = json?["error"] as? [String: String]
        XCTAssertEqual(error?["code"], "invalid_payload")
        XCTAssertEqual(error?["message"], "bad input")
    }
}
