import Foundation

final class BridgeHandle {
    let facade: FluidFacade

    init(facade: FluidFacade) {
        self.facade = facade
    }
}

@inline(__always)
private func buffer(from data: Data, outLen: UnsafeMutablePointer<Int>?) -> UnsafeMutablePointer<UInt8>? {
    if data.isEmpty {
        outLen?.pointee = 0
        return nil
    }

    let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
    data.copyBytes(to: ptr, count: data.count)
    outLen?.pointee = data.count
    return ptr
}

@inline(__always)
private func errorBuffer(_ error: Error, outLen: UnsafeMutablePointer<Int>?) -> UnsafeMutablePointer<UInt8>? {
    let mapped = toBridgeError(error)
    return buffer(from: Serialization.encodeError(mapped), outLen: outLen)
}

@inline(__always)
private func resolveHandle(_ handlePtr: UnsafeMutableRawPointer?) throws -> BridgeHandle {
    guard let handlePtr else {
        throw BridgeError.invalidPayload("bridge handle is null")
    }
    return Unmanaged<BridgeHandle>.fromOpaque(handlePtr).takeUnretainedValue()
}

@inline(__always)
private func resolveCString(
    _ raw: UnsafePointer<CChar>?,
    label: String
) throws -> String {
    guard let raw else {
        throw BridgeError.invalidPayload("\(label) is null")
    }
    return String(cString: raw)
}

@inline(__always)
private func resolvePayloadBytes(
    _ bytes: UnsafePointer<UInt8>?,
    _ length: Int,
    label: String
) throws -> Data {
    guard let bytes, length > 0 else {
        throw BridgeError.invalidPayload("\(label) payload is empty")
    }
    return Data(bytes: bytes, count: length)
}

@_cdecl("glimpse_fluid_create")
public func glimpse_fluid_create(
    _ configBytes: UnsafePointer<UInt8>?,
    _ configLen: Int
) -> UnsafeMutableRawPointer? {
    do {
        guard let configBytes, configLen > 0 else {
            throw BridgeError.invalidPayload("create config payload is empty")
        }

        let data = Data(bytes: configBytes, count: configLen)
        let config = try Serialization.decodeConfig(from: data)
        let facade = try FluidFacade(config: config)
        let handle = BridgeHandle(facade: facade)
        return Unmanaged.passRetained(handle).toOpaque()
    } catch {
        NSLog("[GlimpseSpeechFluidBridge] create failed: \(error)")
        return nil
    }
}

@_cdecl("glimpse_fluid_destroy")
public func glimpse_fluid_destroy(_ handlePtr: UnsafeMutableRawPointer?) {
    guard let handlePtr else {
        return
    }
    Unmanaged<BridgeHandle>.fromOpaque(handlePtr).release()
}

@_cdecl("glimpse_fluid_transcribe_wav")
public func glimpse_fluid_transcribe_wav(
    _ handlePtr: UnsafeMutableRawPointer?,
    _ wavPath: UnsafePointer<CChar>?,
    _ optionsBytes: UnsafePointer<UInt8>?,
    _ optionsLen: Int,
    _ outLen: UnsafeMutablePointer<Int>?
) -> UnsafeMutablePointer<UInt8>? {
    do {
        let handle = try resolveHandle(handlePtr)
        let path = try resolveCString(wavPath, label: "wav path")
        let optionsData = try resolvePayloadBytes(
            optionsBytes,
            optionsLen,
            label: "transcribe options"
        )
        let options = try Serialization.decodeTranscribeOptions(from: optionsData)
        let transcript = try handle.facade.transcribe(wavPath: path, options: options)
        return buffer(from: Serialization.encodeSuccess(transcript), outLen: outLen)
    } catch {
        NSLog("[GlimpseSpeechFluidBridge] transcribe failed: \(error)")
        return errorBuffer(error, outLen: outLen)
    }
}

@_cdecl("glimpse_fluid_diarize_wav")
public func glimpse_fluid_diarize_wav(
    _ handlePtr: UnsafeMutableRawPointer?,
    _ wavPath: UnsafePointer<CChar>?,
    _ optionsBytes: UnsafePointer<UInt8>?,
    _ optionsLen: Int,
    _ outLen: UnsafeMutablePointer<Int>?
) -> UnsafeMutablePointer<UInt8>? {
    do {
        let handle = try resolveHandle(handlePtr)
        let path = try resolveCString(wavPath, label: "wav path")
        let optionsData = try resolvePayloadBytes(
            optionsBytes,
            optionsLen,
            label: "diarize options"
        )
        let options = try Serialization.decodeDiarizeOptions(from: optionsData)
        let diarization = try handle.facade.diarize(wavPath: path, options: options)
        return buffer(from: Serialization.encodeSuccess(diarization), outLen: outLen)
    } catch {
        NSLog("[GlimpseSpeechFluidBridge] diarize failed: \(error)")
        return errorBuffer(error, outLen: outLen)
    }
}

@_cdecl("glimpse_fluid_free_buffer")
public func glimpse_fluid_free_buffer(_ ptr: UnsafeMutablePointer<UInt8>?, _ len: Int) {
    guard let ptr, len > 0 else {
        return
    }
    ptr.deallocate()
}
