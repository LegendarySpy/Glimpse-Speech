// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GlimpseSpeechFluidBridge",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "GlimpseSpeechFluidBridge",
            type: .dynamic,
            targets: ["GlimpseSpeechFluidBridge"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.9.1")
    ],
    targets: [
        .target(
            name: "GlimpseSpeechFluidBridge",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            path: "Sources/GlimpseSpeechFluidBridge"
        ),
        .testTarget(
            name: "GlimpseSpeechFluidBridgeTests",
            dependencies: ["GlimpseSpeechFluidBridge"],
            path: "Tests/GlimpseSpeechFluidBridgeTests"
        )
    ]
)
