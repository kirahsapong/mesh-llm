// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MeshSmokeApp",
    platforms: [
        .macOS(.v13),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "MeshSmokeApp",
            dependencies: [
                .product(name: "MeshLLM", package: "swift"),
            ],
            path: "Sources/MeshSmokeApp"
        ),
    ]
)
