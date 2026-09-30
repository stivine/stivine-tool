// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PhotoSorter",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "PhotoSorter", targets: ["PhotoSorter"])],
    targets: [
        .executableTarget(name: "PhotoSorter"),
        .testTarget(name: "PhotoSorterTests", dependencies: ["PhotoSorter"])
    ]
)
