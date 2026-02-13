import Foundation

#if canImport(FluidAudio)
import FluidAudio
#endif

final class FluidFacade: @unchecked Sendable {
    private let config: BridgeConfig

    #if canImport(FluidAudio)
    private let asrManager: AsrManager
    private let ctcModelDirectory: URL?
    private let ctcModelVariant: CtcModelVariant?
    private var cachedCtcModels: CtcModels?
    private var cachedCtcTokenizer: CtcTokenizer?
    private var configuredVocabularyTerms: [String] = []
    private let bridgeLock = NSLock()
    #endif

    init(config: BridgeConfig) throws {
        try ModelPaths.validate(config: config)
        self.config = config

        #if canImport(FluidAudio)
        let asrResolution = try Self.resolveAsrDirectory(from: config.asrModelDir)

        let ctcResolution = Self.resolveCtcDirectory(near: asrResolution.directory)
        self.ctcModelDirectory = ctcResolution?.directory
        self.ctcModelVariant = ctcResolution?.variant

        self.asrManager = AsrManager(config: .default)
        try Self.blocking {
            let models = try await AsrModels.load(
                from: asrResolution.directory,
                version: asrResolution.version
            )
            try await self.asrManager.initialize(models: models)
        }
        #endif
    }

    func transcribe(wavPath: String, options: BridgeTranscribeOptions) throws -> BridgeTranscript {
        if options.schemaVersion != bridgeSchemaVersion {
            throw BridgeError.invalidPayload(
                "schema_version must be \(bridgeSchemaVersion), got \(options.schemaVersion)"
            )
        }

        #if canImport(FluidAudio)
        let wavURL = try Self.existingFileURL(path: wavPath, label: "wav")
        let wavName = wavURL.lastPathComponent
        NSLog(
            "[GlimpseSpeechFluidBridge] transcribe start wav=\(wavName) vocab_terms=\(options.vocabulary.count) timestamps=\(options.timestamps)"
        )
        return try bridgeLock.withLock {
            try Self.blocking {
                do {
                    try await self.configureVocabularyBoosting(vocabulary: options.vocabulary)
                } catch {
                    // Vocabulary boosting is optional; fall back to plain ASR if CTC setup fails.
                    self.asrManager.disableVocabularyBoosting()
                    self.configuredVocabularyTerms = []
                    NSLog("[GlimpseSpeechFluidBridge] vocabulary boosting unavailable: \(error)")
                }
                let result: ASRResult
                do {
                    result = try await self.asrManager.transcribe(wavURL, source: .system)
                } catch {
                    if Self.isTokenizerMissingError(error) {
                        self.asrManager.disableVocabularyBoosting()
                        self.configuredVocabularyTerms = []
                        NSLog(
                            "[GlimpseSpeechFluidBridge] tokenizer unavailable during decode; retrying without vocabulary boosting: \(error)"
                        )
                        result = try await self.asrManager.transcribe(wavURL, source: .system)
                    } else {
                        throw error
                    }
                }
                let tokenTimingCount = result.tokenTimings?.count ?? 0
                NSLog(
                    "[GlimpseSpeechFluidBridge] transcribe result text_len=\(result.text.count) token_timings=\(tokenTimingCount)"
                )
                return Self.toBridgeTranscript(from: result, timestampPreference: options.timestamps)
            }
        }
        #else
        throw BridgeError.fluidUnavailable("FluidAudio module is not linked in this build")
        #endif
    }

    func diarize(wavPath: String, options: BridgeDiarizeOptions) throws -> BridgeDiarization {
        if options.schemaVersion != bridgeSchemaVersion {
            throw BridgeError.invalidPayload(
                "schema_version must be \(bridgeSchemaVersion), got \(options.schemaVersion)"
            )
        }

        #if canImport(FluidAudio)
        let wavURL = try Self.existingFileURL(path: wavPath, label: "wav")
        guard let diarizationModelDir = config.diarizationModelDir else {
            throw BridgeError.invalidConfig("diarization_model_dir is required for diarization")
        }

        let diarizationBaseDirectory = try Self.resolveOfflineDiarizationBaseDirectory(
            from: diarizationModelDir
        )

        return try Self.blocking {
            try await Self.runOfflineDiarization(
                modelBaseDirectory: diarizationBaseDirectory,
                wavURL: wavURL,
                speakerCount: options.speakerCount
            )
        }
        #else
        throw BridgeError.fluidUnavailable("FluidAudio module is not linked in this build")
        #endif
    }
}

#if canImport(FluidAudio)
extension FluidFacade {
    static let maxTimestampMs: Double = Double(UInt64.max)
    static let ctcTokenizerBaseURL = URL(
        string: "https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml/resolve/main"
    )!
    static let requiredTokenizerFiles = ["tokenizer.json", "tokenizer_config.json"]
    static let optionalTokenizerFiles = ["special_tokens_map.json", "config.json"]

    struct AsrResolution {
        let directory: URL
        let version: AsrModelVersion
    }

    struct CtcResolution {
        let directory: URL
        let variant: CtcModelVariant
    }

    static func resolveAsrDirectory(from configuredPath: String) throws -> AsrResolution {
        let root = URL(fileURLWithPath: configuredPath, isDirectory: true).standardizedFileURL
        let v3 = Repo.parakeet.folderName
        let v2 = Repo.parakeetV2.folderName

        let candidates: [(directory: URL, version: AsrModelVersion)] = [
            (root, .v3),
            (root, .v2),
            (root.appendingPathComponent(v3, isDirectory: true), .v3),
            (root.appendingPathComponent(v2, isDirectory: true), .v2),
        ]

        for candidate in candidates {
            let repoDirectory = asrRepositoryDirectory(from: candidate.directory, version: candidate.version)
            if asrModelsExist(at: repoDirectory, version: candidate.version) {
                return AsrResolution(directory: repoDirectory, version: candidate.version)
            }
        }

        throw BridgeError.modelNotFound(
            "ASR models not found in \(configuredPath). Expected a '\(v3)' or '\(v2)' directory with \(ModelNames.ASR.requiredModels.joined(separator: ", ")) and \(ModelNames.ASR.vocabularyFile)"
        )
    }

    static func resolveCtcDirectory(near asrDirectory: URL) -> CtcResolution? {
        let parent = asrDirectory.deletingLastPathComponent()
        let ctc110m = parent.appendingPathComponent(Repo.parakeetCtc110m.folderName, isDirectory: true)
        let ctc06b = parent.appendingPathComponent(Repo.parakeetCtc06b.folderName, isDirectory: true)
        let ctcSubdir = asrDirectory.appendingPathComponent("ctc", isDirectory: true)

        let candidates: [CtcResolution] = [
            CtcResolution(directory: ctc110m, variant: .ctc110m),
            CtcResolution(directory: ctc06b, variant: .ctc06b),
            CtcResolution(directory: ctcSubdir, variant: .ctc110m),
            CtcResolution(directory: asrDirectory, variant: .ctc110m),
        ]

        var seen = Set<String>()
        for candidate in candidates {
            if !seen.insert(candidate.directory.path).inserted {
                continue
            }
            if CtcModels.modelsExist(at: candidate.directory) {
                return candidate
            }
        }

        return nil
    }

    static func resolveOfflineDiarizationBaseDirectory(from configuredPath: String) throws -> URL {
        let provided = URL(fileURLWithPath: configuredPath, isDirectory: true).standardizedFileURL
        let repoName = Repo.diarizer.folderName

        let baseDirectory: URL
        let repoDirectory: URL
        if provided.lastPathComponent == repoName {
            baseDirectory = provided.deletingLastPathComponent()
            repoDirectory = provided
        } else {
            baseDirectory = provided
            repoDirectory = provided.appendingPathComponent(repoName, isDirectory: true)
        }

        let required = ModelNames.OfflineDiarizer.requiredModels
        guard directoryContainsAll(repoDirectory, names: required) else {
            throw BridgeError.modelNotFound(
                "Offline diarization models not found under \(repoDirectory.path). Missing one or more of: \(required.joined(separator: ", "))"
            )
        }

        let pldaCandidates = [
            baseDirectory.appendingPathComponent("plda-parameters.json", isDirectory: false),
            baseDirectory.appendingPathComponent("speaker-diarization-coreml/plda-parameters.json", isDirectory: false),
            baseDirectory.appendingPathComponent("speaker-diarization-offline/plda-parameters.json", isDirectory: false),
        ]

        if !pldaCandidates.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            throw BridgeError.modelNotFound(
                "PLDA parameters file not found for diarization at \(baseDirectory.path)"
            )
        }

        return baseDirectory
    }

    func configureVocabularyBoosting(vocabulary: [String]) async throws {
        let terms = Self.normalizeVocabularyTerms(vocabulary)
        guard !terms.isEmpty else {
            asrManager.disableVocabularyBoosting()
            configuredVocabularyTerms = []
            return
        }

        if configuredVocabularyTerms == terms, cachedCtcModels != nil, cachedCtcTokenizer != nil {
            return
        }

        guard
            let ctcModelDirectory,
            let ctcModelVariant
        else {
            // Vocabulary boosting is best-effort. If CTC models are not present,
            // keep transcription working without failing the request.
            asrManager.disableVocabularyBoosting()
            configuredVocabularyTerms = []
            return
        }

        let tokenizerDirectory = CtcModels.defaultCacheDirectory(for: ctcModelVariant)
        try Self.maybeInstallTokenizerFiles(
            from: ctcModelDirectory,
            to: tokenizerDirectory
        )
        if !Self.hasTokenizerFiles(in: tokenizerDirectory) {
            do {
                try await Self.maybeDownloadTokenizerFiles(to: tokenizerDirectory)
            } catch {
                NSLog("[GlimpseSpeechFluidBridge] tokenizer download failed: \(error)")
            }
        }
        guard Self.hasTokenizerFiles(in: tokenizerDirectory) else {
            asrManager.disableVocabularyBoosting()
            configuredVocabularyTerms = []
            NSLog(
                "[GlimpseSpeechFluidBridge] tokenizer files missing in cache directory; disabling vocabulary boosting"
            )
            return
        }

        let ctcModels = try await loadCtcModels(
            from: ctcModelDirectory,
            variant: ctcModelVariant
        )
        let tokenizer = try await loadCtcTokenizer(from: tokenizerDirectory)
        let customTerms = Self.tokenizedVocabularyTerms(terms, tokenizer: tokenizer)
        guard !customTerms.isEmpty else {
            asrManager.disableVocabularyBoosting()
            configuredVocabularyTerms = []
            NSLog("[GlimpseSpeechFluidBridge] vocabulary terms produced no CTC tokens; disabling vocabulary boosting")
            return
        }

        let context = CustomVocabularyContext(terms: customTerms)
        try await asrManager.configureVocabularyBoosting(vocabulary: context, ctcModels: ctcModels)
        configuredVocabularyTerms = terms
    }

    private func loadCtcModels(
        from ctcModelDirectory: URL,
        variant ctcModelVariant: CtcModelVariant
    ) async throws -> CtcModels {
        if let cachedCtcModels {
            NSLog("[GlimpseSpeechFluidBridge] using cached CTC models")
            return cachedCtcModels
        }

        NSLog("[GlimpseSpeechFluidBridge] loading CTC models")
        let loaded = try await CtcModels.loadDirect(
            from: ctcModelDirectory,
            variant: ctcModelVariant
        )
        cachedCtcModels = loaded
        return loaded
    }

    private func loadCtcTokenizer(from tokenizerDirectory: URL) async throws -> CtcTokenizer {
        if let cachedCtcTokenizer {
            return cachedCtcTokenizer
        }

        let tokenizer = try await CtcTokenizer.load(from: tokenizerDirectory)
        cachedCtcTokenizer = tokenizer
        return tokenizer
    }

    private static func tokenizedVocabularyTerms(
        _ terms: [String],
        tokenizer: CtcTokenizer
    ) -> [CustomVocabularyTerm] {
        var tokenizedTerms: [CustomVocabularyTerm] = []
        tokenizedTerms.reserveCapacity(terms.count)

        for term in terms {
            let tokenIds = tokenizer.encode(term)
            guard !tokenIds.isEmpty else {
                NSLog("[GlimpseSpeechFluidBridge] skipping un-tokenizable vocabulary term: \(term)")
                continue
            }
            tokenizedTerms.append(
                CustomVocabularyTerm(
                    text: term,
                    weight: 10.0,
                    aliases: nil,
                    tokenIds: nil,
                    ctcTokenIds: tokenIds
                )
            )
        }

        return tokenizedTerms
    }

    private static func hasTokenizerFiles(in directory: URL) -> Bool {
        let fileManager = FileManager.default
        let tokenizer = directory.appendingPathComponent("tokenizer.json").path
        let tokenizerConfig = directory.appendingPathComponent("tokenizer_config.json").path
        return
            fileManager.fileExists(atPath: tokenizer)
            && fileManager.fileExists(atPath: tokenizerConfig)
    }

    private static func maybeInstallTokenizerFiles(from source: URL, to destination: URL) throws {
        if hasTokenizerFiles(in: destination) {
            return
        }
        if !hasTokenizerFiles(in: source) {
            return
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destination,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let filesToCopy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ]

        for file in filesToCopy {
            let sourceFile = source.appendingPathComponent(file)
            guard fileManager.fileExists(atPath: sourceFile.path) else {
                continue
            }

            let destinationFile = destination.appendingPathComponent(file)
            if fileManager.fileExists(atPath: destinationFile.path) {
                continue
            }

            try fileManager.copyItem(at: sourceFile, to: destinationFile)
        }

        if hasTokenizerFiles(in: destination) {
            NSLog("[GlimpseSpeechFluidBridge] installed tokenizer files in cache directory")
        }
    }

    private static func maybeDownloadTokenizerFiles(to destination: URL) async throws {
        if hasTokenizerFiles(in: destination) {
            return
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destination,
            withIntermediateDirectories: true,
            attributes: nil
        )

        for file in requiredTokenizerFiles + optionalTokenizerFiles {
            let destinationFile = destination.appendingPathComponent(file)
            if fileManager.fileExists(atPath: destinationFile.path) {
                continue
            }

            let remoteURL = ctcTokenizerBaseURL.appendingPathComponent(file)
            do {
                let (data, response) = try await URLSession.shared.data(from: remoteURL)
                guard isSuccessfulHTTP(response) else {
                    if requiredTokenizerFiles.contains(file) {
                        throw BridgeError.modelNotFound(
                            "Required tokenizer file \(file) not available at \(remoteURL.absoluteString)"
                        )
                    }
                    continue
                }
                try data.write(to: destinationFile, options: [.atomic])
            } catch {
                if requiredTokenizerFiles.contains(file) {
                    throw error
                }
            }
        }

        if hasTokenizerFiles(in: destination) {
            NSLog("[GlimpseSpeechFluidBridge] downloaded tokenizer files to cache directory")
        }
    }

    private static func isSuccessfulHTTP(_ response: URLResponse) -> Bool {
        guard let http = response as? HTTPURLResponse else {
            return false
        }
        return (200...299).contains(http.statusCode)
    }

    private static func isTokenizerMissingError(_ error: Error) -> Bool {
        let message = String(describing: error).lowercased()
        return
            message.contains("tokenizernotfound")
            || message.contains("tokenizer.json not found")
            || message.contains("missing required file 'tokenizer.json'")
    }

    static func runOfflineDiarization(
        modelBaseDirectory: URL,
        wavURL: URL,
        speakerCount: Int?
    ) async throws -> BridgeDiarization {
        guard #available(macOS 14.0, *) else {
            throw BridgeError.unsupportedPlatform("Offline diarization requires macOS 14+")
        }
        return try await runOfflineDiarizationMac14(
            modelBaseDirectory: modelBaseDirectory,
            wavURL: wavURL,
            speakerCount: speakerCount
        )
    }

    @available(macOS 14.0, *)
    static func runOfflineDiarizationMac14(
        modelBaseDirectory: URL,
        wavURL: URL,
        speakerCount: Int?
    ) async throws -> BridgeDiarization {
        var diarizationConfig = OfflineDiarizerConfig.default
        if let speakerCount {
            guard speakerCount > 0 else {
                throw BridgeError.invalidPayload("speaker_count must be greater than zero")
            }
            diarizationConfig.clustering.numSpeakers = speakerCount
        }

        let manager = OfflineDiarizerManager(config: diarizationConfig)
        let models = try await OfflineDiarizerModels.load(from: modelBaseDirectory)
        manager.initialize(models: models)

        let result = try await manager.process(wavURL)
        return toBridgeDiarization(from: result)
    }

    static func toBridgeTranscript(
        from result: ASRResult,
        timestampPreference: String
    ) -> BridgeTranscript {
        let includeWords = timestampPreference != "segments_only"
        let words = toBridgeWords(from: result.tokenTimings)

        let resolvedText: String = {
            let raw = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !raw.isEmpty {
                return raw
            }

            guard let words, !words.isEmpty else {
                return ""
            }

            return words.map(\.text).joined(separator: " ")
        }()

        let durationMs = toMilliseconds(result.duration)
        let segments: [BridgeSegment]
        if let words, !words.isEmpty {
            segments = toBridgeSegments(from: words)
        } else if resolvedText.isEmpty {
            segments = []
        } else {
            segments = toBridgeSegmentsFromText(resolvedText, durationMs: durationMs)
        }

        let wordsWithSegment = includeWords
            ? words.map { assignWordSegments($0, to: segments) }
            : nil

        return BridgeTranscript(
            schemaVersion: bridgeSchemaVersion,
            engine: "fluid",
            text: resolvedText,
            segments: segments,
            words: wordsWithSegment
        )
    }

    static func toBridgeDiarization(from result: DiarizationResult) -> BridgeDiarization {
        let sorted = result.segments.sorted {
            if $0.startTimeSeconds == $1.startTimeSeconds {
                return $0.endTimeSeconds < $1.endTimeSeconds
            }
            return $0.startTimeSeconds < $1.startTimeSeconds
        }

        var turns: [BridgeSpeakerTurn] = []
        var previousEndMs: UInt64 = 0

        for segment in sorted {
            var startMs = toMilliseconds(TimeInterval(segment.startTimeSeconds))
            var endMs = toMilliseconds(TimeInterval(segment.endTimeSeconds))
            if endMs <= startMs {
                continue
            }

            if startMs < previousEndMs {
                startMs = previousEndMs
            }
            if endMs <= startMs {
                endMs = startMs + 1
            }

            let speaker = segment.speakerId.trimmingCharacters(in: .whitespacesAndNewlines)
            if speaker.isEmpty {
                continue
            }

            turns.append(
                BridgeSpeakerTurn(
                    startMs: startMs,
                    endMs: endMs,
                    speaker: speaker
                )
            )
            previousEndMs = endMs
        }

        return BridgeDiarization(
            schemaVersion: bridgeSchemaVersion,
            turns: turns
        )
    }

    static func toBridgeWords(from timings: [TokenTiming]?) -> [BridgeWord]? {
        guard let timings, !timings.isEmpty else {
            return nil
        }

        var words: [BridgeWord] = []
        words.reserveCapacity(max(1, timings.count / 2))

        var currentText = ""
        var currentStartMs: UInt64?
        var currentEndMs: UInt64 = 0

        func flushCurrentWord() {
            guard let startMs = currentStartMs, !currentText.isEmpty else {
                return
            }

            var endMs = currentEndMs
            if endMs <= startMs {
                endMs = startMs + 1
            }

            words.append(
                BridgeWord(
                    startMs: startMs,
                    endMs: endMs,
                    text: currentText,
                    segmentIndex: nil
                )
            )

            currentText = ""
            currentStartMs = nil
            currentEndMs = 0
        }

        for timing in timings {
            let tokenGapBreakMs: UInt64 = 220
            let rawToken = timing.token
            let cleanedToken = stripWordBoundaryPrefix(rawToken)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if cleanedToken.isEmpty {
                continue
            }

            let startMs = toMilliseconds(timing.startTime)
            var endMs = toMilliseconds(timing.endTime)
            if endMs <= startMs {
                endMs = startMs + 1
            }

            let gapBreak = !currentText.isEmpty && startMs > currentEndMs + tokenGapBreakMs
            let startsNewWord = tokenStartsWordBoundary(rawToken) || currentText.isEmpty || gapBreak
            if startsNewWord {
                flushCurrentWord()
                currentStartMs = startMs
                currentText = cleanedToken
                currentEndMs = endMs
            } else {
                currentText += cleanedToken
                currentEndMs = endMs
            }
        }

        flushCurrentWord()
        return words.isEmpty ? nil : words
    }

    static func toBridgeSegments(from words: [BridgeWord]) -> [BridgeSegment] {
        let gapBreakMs: UInt64 = 750
        let maxSegmentMs: UInt64 = 6_000
        let maxWordsPerSegment = 14

        if words.isEmpty {
            return []
        }

        var segments: [BridgeSegment] = []
        var currentWords: [BridgeWord] = []
        currentWords.reserveCapacity(maxWordsPerSegment)
        var previousEnd: UInt64 = 0

        func flushSegment() {
            guard
                let start = currentWords.first?.startMs,
                let last = currentWords.last
            else {
                return
            }

            var end = last.endMs
            if end <= start {
                end = start + 1
            }

            let text = currentWords
                .map(\.text)
                .joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if text.isEmpty {
                currentWords.removeAll(keepingCapacity: true)
                return
            }

            var startClamped = start
            if startClamped < previousEnd {
                startClamped = previousEnd
            }
            if end <= startClamped {
                end = startClamped + 1
            }

            segments.append(
                BridgeSegment(
                    startMs: startClamped,
                    endMs: end,
                    text: text
                )
            )
            previousEnd = end
            currentWords.removeAll(keepingCapacity: true)
        }

        for word in words {
            if let previous = currentWords.last {
                let gap = saturatingSubtract(word.startMs, previous.endMs)
                if gap > gapBreakMs {
                    flushSegment()
                }
            }

            currentWords.append(word)

            guard
                let start = currentWords.first?.startMs,
                let last = currentWords.last
            else {
                continue
            }

            let duration = saturatingSubtract(last.endMs, start)
            let punctuationBreak = wordEndsSentence(last.text)
            let lengthBreak = duration >= maxSegmentMs || currentWords.count >= maxWordsPerSegment

            if punctuationBreak || lengthBreak {
                flushSegment()
            }
        }

        if !currentWords.isEmpty {
            flushSegment()
        }

        return segments
    }

    static func toBridgeSegmentsFromText(_ text: String, durationMs: UInt64) -> [BridgeSegment] {
        let tokens = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)

        guard !tokens.isEmpty else {
            return []
        }

        let targetSegmentMs: UInt64 = 10_000
        let averageWordMs = max(1, durationMs / UInt64(tokens.count))
        let wordsByDuration = Int(targetSegmentMs / averageWordMs)
        let chunkWords = max(4, min(24, wordsByDuration))

        var ranges: [Range<Int>] = []
        var index = 0
        while index < tokens.count {
            let upper = min(tokens.count, index + chunkWords)
            ranges.append(index ..< upper)
            index = upper
        }

        if ranges.count == 1 {
            let endMs = max(1, durationMs)
            return [BridgeSegment(startMs: 0, endMs: endMs, text: String(text))]
        }

        let totalWords = max(1, tokens.count)
        var previousEnd: UInt64 = 0
        var segments: [BridgeSegment] = []
        segments.reserveCapacity(ranges.count)

        for (chunkIndex, range) in ranges.enumerated() {
            var startMs = durationMs * UInt64(range.lowerBound) / UInt64(totalWords)
            var endMs = chunkIndex + 1 == ranges.count
                ? durationMs
                : durationMs * UInt64(range.upperBound) / UInt64(totalWords)

            if startMs < previousEnd {
                startMs = previousEnd
            }
            if endMs <= startMs {
                endMs = startMs + 1
            }

            let chunkText = tokens[range].map(String.init).joined(separator: " ")
            segments.append(
                BridgeSegment(
                    startMs: startMs,
                    endMs: endMs,
                    text: chunkText
                )
            )
            previousEnd = endMs
        }

        return segments
    }

    static func assignWordSegments(
        _ words: [BridgeWord],
        to segments: [BridgeSegment]
    ) -> [BridgeWord] {
        guard !segments.isEmpty else {
            return words.map { word in
                BridgeWord(
                    startMs: word.startMs,
                    endMs: word.endMs,
                    text: word.text,
                    segmentIndex: nil
                )
            }
        }

        var output: [BridgeWord] = []
        output.reserveCapacity(words.count)
        var cursor = 0

        for word in words {
            while
                cursor + 1 < segments.count,
                word.startMs >= segments[cursor].endMs
            {
                cursor += 1
            }

            var segmentIndex: Int? = nil
            if cursor < segments.count {
                let segment = segments[cursor]
                if word.startMs >= segment.startMs && word.endMs <= segment.endMs {
                    segmentIndex = cursor
                }
            }

            if segmentIndex == nil {
                segmentIndex = max(0, min(cursor, segments.count - 1))
            }

            output.append(
                BridgeWord(
                    startMs: word.startMs,
                    endMs: word.endMs,
                    text: word.text,
                    segmentIndex: segmentIndex
                )
            )
        }

        return output
    }

    static func wordEndsSentence(_ token: String) -> Bool {
        guard let scalar = token.unicodeScalars.last else {
            return false
        }
        switch scalar {
        case ".", "!", "?", ";", ":":
            return true
        default:
            return false
        }
    }

    static func saturatingSubtract(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        lhs >= rhs ? lhs - rhs : 0
    }

    static func isWordBoundaryScalar(_ scalar: UnicodeScalar) -> Bool {
        CharacterSet.whitespacesAndNewlines.contains(scalar)
            || scalar == "▁"
            || scalar == "Ġ"
            || scalar == "Ċ"
    }

    private static func tokenStartsWordBoundary(_ token: String) -> Bool {
        guard let scalar = token.unicodeScalars.first else {
            return false
        }
        return isWordBoundaryScalar(scalar)
    }

    private static func stripWordBoundaryPrefix(_ token: String) -> String {
        var view = token.unicodeScalars
        while let scalar = view.first, isWordBoundaryScalar(scalar) {
            view.removeFirst()
        }
        return String(view)
    }

    static func normalizeVocabularyTerms(_ terms: [String]) -> [String] {
        var seen = Set<String>()
        var output: [String] = []
        output.reserveCapacity(terms.count)

        for term in terms {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                continue
            }
            let key = trimmed.lowercased()
            if seen.insert(key).inserted {
                output.append(trimmed)
            }
        }

        return output
    }

    static func asrModelsExist(at directory: URL, version: AsrModelVersion) -> Bool {
        let repoDirectory = asrRepositoryDirectory(from: directory, version: version)

        let required = ModelNames.ASR.requiredModels.union([ModelNames.ASR.vocabularyFile])
        return directoryContainsAll(repoDirectory, names: required)
    }

    static func asrRepositoryDirectory(from directory: URL, version: AsrModelVersion) -> URL {
        let repoName: String = switch version {
        case .v2:
            Repo.parakeetV2.folderName
        case .v3:
            Repo.parakeet.folderName
        }

        if directory.lastPathComponent == repoName {
            return directory
        }
        return directory.appendingPathComponent(repoName, isDirectory: true)
    }

    static func directoryContainsAll(_ directory: URL, names: Set<String>) -> Bool {
        let fileManager = FileManager.default
        return names.allSatisfy { name in
            fileManager.fileExists(atPath: directory.appendingPathComponent(name).path)
        }
    }

    static func existingFileURL(path: String, label: String) throws -> URL {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw BridgeError.invalidPayload("\(label) path is empty")
        }

        let url = URL(fileURLWithPath: trimmed, isDirectory: false).standardizedFileURL
        var isDirectory = ObjCBool(false)
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        if !exists || isDirectory.boolValue {
            throw BridgeError.modelNotFound("\(label) file not found: \(url.path)")
        }

        return url
    }

    static func toMilliseconds(_ seconds: TimeInterval) -> UInt64 {
        if seconds <= 0 {
            return 0
        }

        let raw = seconds * 1_000.0
        if raw >= maxTimestampMs {
            return UInt64.max
        }
        return UInt64(raw.rounded())
    }

    static func blocking<T>(
        _ operation: @escaping @Sendable () async throws -> T
    ) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = AsyncResultBox<T>()

        Task {
            do {
                box.result = Swift.Result<T, Error>.success(try await operation())
            } catch {
                box.result = Swift.Result<T, Error>.failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        guard let result = box.result else {
            throw BridgeError.internalFailure("async operation completed without a result")
        }
        return try result.get()
    }
}

private final class AsyncResultBox<T>: @unchecked Sendable {
    var result: Swift.Result<T, Error>?

    init() {}
}

private extension NSLock {
    func withLock<T>(_ operation: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try operation()
    }
}
#endif
