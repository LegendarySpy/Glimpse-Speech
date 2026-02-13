import XCTest
@testable import GlimpseSpeechFluidBridge

final class ErrorMappingTests: XCTestCase {
    func testBridgeErrorCodesAreDeterministic() {
        XCTAssertEqual(BridgeError.invalidPayload("x").code, "invalid_payload")
        XCTAssertEqual(BridgeError.invalidConfig("x").code, "invalid_config")
        XCTAssertEqual(BridgeError.unsupportedPlatform("x").code, "unsupported_platform")
        XCTAssertEqual(BridgeError.modelNotFound("x").code, "model_not_found")
        XCTAssertEqual(BridgeError.fluidUnavailable("x").code, "fluid_unavailable")
        XCTAssertEqual(BridgeError.internalFailure("x").code, "internal_failure")
    }

    func testUnknownErrorMapsToInternalFailure() {
        enum LocalError: Error { case bad }
        let mapped = toBridgeError(LocalError.bad)

        guard case let BridgeError.internalFailure(message) = mapped else {
            XCTFail("expected internalFailure")
            return
        }
        XCTAssertTrue(message.contains("bad"))
    }
}
