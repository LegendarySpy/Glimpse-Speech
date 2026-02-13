import XCTest
@testable import GlimpseSpeechFluidBridge
#if canImport(FluidAudio)
import FluidAudio
#endif

final class FluidFacadeHelpersTests: XCTestCase {
    func testNormalizeVocabularyTermsTrimsAndDeduplicatesCaseInsensitive() {
        let normalized = FluidFacade.normalizeVocabularyTerms([
            "  Alpha ",
            "alpha",
            "",
            "Beta",
            "beta ",
            "Gamma"
        ])

        XCTAssertEqual(normalized, ["Alpha", "Beta", "Gamma"])
    }

    func testToMillisecondsRoundsAndSaturates() {
        XCTAssertEqual(FluidFacade.toMilliseconds(0), 0)
        XCTAssertEqual(FluidFacade.toMilliseconds(0.001), 1)
        XCTAssertEqual(FluidFacade.toMilliseconds(1.499), 1_499)
        XCTAssertEqual(FluidFacade.toMilliseconds(1.5), 1_500)
        XCTAssertEqual(FluidFacade.toMilliseconds(Double.greatestFiniteMagnitude), UInt64.max)
    }

    #if canImport(FluidAudio)
    func testToBridgeWordsCollapsesTokenTimingsIntoLexicalWords() {
        let timings = [
            TokenTiming(token: "▁Hi", tokenId: 0, startTime: 0.00, endTime: 0.08, confidence: 0.9),
            TokenTiming(token: ",", tokenId: 1, startTime: 0.08, endTime: 0.10, confidence: 0.9),
            TokenTiming(token: "▁I", tokenId: 2, startTime: 0.12, endTime: 0.18, confidence: 0.9),
            TokenTiming(token: "'", tokenId: 3, startTime: 0.18, endTime: 0.20, confidence: 0.9),
            TokenTiming(token: "m", tokenId: 4, startTime: 0.20, endTime: 0.24, confidence: 0.9),
            TokenTiming(token: "▁G", tokenId: 5, startTime: 0.28, endTime: 0.33, confidence: 0.9),
            TokenTiming(token: "aren", tokenId: 6, startTime: 0.33, endTime: 0.40, confidence: 0.9),
        ]

        let words = FluidFacade.toBridgeWords(from: timings)
        XCTAssertEqual(words?.map(\.text), ["Hi,", "I'm", "Garen"])
        XCTAssertEqual(words?.count, 3)
    }

    func testToBridgeSegmentsSplitsLongRunsByPunctuationAndGap() {
        let words = [
            BridgeWord(startMs: 0, endMs: 400, text: "Hello", segmentIndex: nil),
            BridgeWord(startMs: 420, endMs: 900, text: "world.", segmentIndex: nil),
            BridgeWord(startMs: 2_000, endMs: 2_300, text: "Next", segmentIndex: nil),
            BridgeWord(startMs: 2_320, endMs: 2_700, text: "block", segmentIndex: nil),
        ]

        let segments = FluidFacade.toBridgeSegments(from: words)
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].text, "Hello world.")
        XCTAssertEqual(segments[0].startMs, 0)
        XCTAssertEqual(segments[0].endMs, 900)
        XCTAssertEqual(segments[1].text, "Next block")
        XCTAssertEqual(segments[1].startMs, 2_000)
        XCTAssertEqual(segments[1].endMs, 2_700)
    }

    func testAssignWordSegmentsIndexesWordsBySegmentBoundaries() {
        let segments = [
            BridgeSegment(startMs: 0, endMs: 1_000, text: "A"),
            BridgeSegment(startMs: 1_000, endMs: 2_000, text: "B"),
        ]
        let words = [
            BridgeWord(startMs: 100, endMs: 200, text: "first", segmentIndex: nil),
            BridgeWord(startMs: 1_100, endMs: 1_300, text: "second", segmentIndex: nil),
        ]

        let indexed = FluidFacade.assignWordSegments(words, to: segments)
        XCTAssertEqual(indexed.map(\.segmentIndex), [0, 1])
    }

    func testToBridgeWordsRecognizesGptStyleBoundaryMarkers() {
        let timings = [
            TokenTiming(token: "ĠHello", tokenId: 0, startTime: 0.00, endTime: 0.10, confidence: 0.9),
            TokenTiming(token: "Ġworld", tokenId: 1, startTime: 0.12, endTime: 0.20, confidence: 0.9),
            TokenTiming(token: "!", tokenId: 2, startTime: 0.20, endTime: 0.23, confidence: 0.9),
        ]

        let words = FluidFacade.toBridgeWords(from: timings)
        XCTAssertEqual(words?.map(\.text), ["Hello", "world!"])
        XCTAssertEqual(words?.count, 2)
    }

    func testToBridgeWordsStartsNewWordOnLargeTokenGap() {
        let timings = [
            TokenTiming(token: "hello", tokenId: 0, startTime: 0.00, endTime: 0.10, confidence: 0.9),
            TokenTiming(token: "there", tokenId: 1, startTime: 0.50, endTime: 0.64, confidence: 0.9),
        ]

        let words = FluidFacade.toBridgeWords(from: timings)
        XCTAssertEqual(words?.map(\.text), ["hello", "there"])
        XCTAssertEqual(words?.count, 2)
    }

    func testToBridgeSegmentsFromTextSplitsCoarseDurations() {
        let text = Array(repeating: "alpha", count: 90).joined(separator: " ")
        let segments = FluidFacade.toBridgeSegmentsFromText(text, durationMs: 300_000)

        XCTAssertGreaterThan(segments.count, 10)
        XCTAssertEqual(segments.first?.startMs, 0)
        XCTAssertEqual(segments.last?.endMs, 300_000)
        XCTAssertTrue(
            segments.allSatisfy { $0.endMs > $0.startMs && $0.endMs - $0.startMs <= 15_000 }
        )
    }
    #endif
}
