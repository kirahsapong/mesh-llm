// swift-tools-version: 5.9
import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let ffiXCFrameworkRelativePath = "Generated/mesh_ffiFFI.xcframework"
let ffiXCFrameworkPath = "\(packageRoot)/\(ffiXCFrameworkRelativePath)"
let hasFFIXCFramework = FileManager.default.fileExists(atPath: ffiXCFrameworkPath)

var meshLLMDependencies: [Target.Dependency] = []
var packageTargets: [Target] = []

if hasFFIXCFramework {
    meshLLMDependencies.append("mesh_ffiFFI")
    packageTargets.append(
        .binaryTarget(
            name: "mesh_ffiFFI",
            path: ffiXCFrameworkRelativePath
        )
    )
}

let package = Package(
    name: "MeshLLM",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "MeshLLM",
            targets: ["MeshLLM"]
        ),
    ],
    targets: [
        .target(
            name: "MeshLLM",
            dependencies: meshLLMDependencies,
            path: "Sources/MeshLLM",
            exclude: hasFFIXCFramework ? [] : ["Generated"],
            linkerSettings: [
                .linkedFramework("SystemConfiguration"),
            ]
        ),
        .testTarget(
            name: "MeshLLMTests",
            dependencies: ["MeshLLM"],
            path: "Tests/MeshLLMTests"
        ),
    ] + packageTargets
)
