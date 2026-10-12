// swift-tools-version: 6.0
import Foundation
import PackageDescription

// mozu (鵙): Hawk（RG-LRU 線形再帰）を幹にした日本語音声合成。
// macOS では MLX で学習し、推論は macOS / Linux とも Pure Swift（Accelerate があれば利用）で動く。
#if os(Linux)
let package = Package(
    name: "Mozu",
    products: [
        .library(name: "Mozu", targets: ["Mozu"]),
        .library(name: "MozuWeb", targets: ["MozuWeb"]),
        .executable(name: "mozu-web", targets: ["mozu-web"]),
        .executable(name: "synthesize", targets: ["synthesize"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "Mozu", dependencies: [], path: "Sources/Mozu", exclude: ["MLX"]),
        .target(name: "MozuWeb", dependencies: ["Mozu"], path: "Sources/MozuWeb"),
        .executableTarget(name: "mozu-web", dependencies: ["Mozu", "MozuWeb"], path: "script/web"),
        .executableTarget(name: "synthesize", dependencies: ["Mozu"], path: "script/synthesize"),
    ]
)
#else
let mlxProducts: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXOptimizers", package: "mlx-swift"),
]

var products: [Product] = [
    .library(name: "Mozu", targets: ["Mozu"]),
    .library(name: "MozuWeb", targets: ["MozuWeb"]),
    .executable(name: "synthesize", targets: ["synthesize"]),
    .executable(name: "train", targets: ["train"]),
    .executable(name: "mozu-web", targets: ["mozu-web"]),
]

var targets: [Target] = [
    .target(name: "Mozu", dependencies: mlxProducts, path: "Sources/Mozu"),
    .target(name: "MozuWeb", dependencies: ["Mozu"], path: "Sources/MozuWeb"),
    .executableTarget(name: "synthesize", dependencies: ["Mozu"], path: "script/synthesize"),
    .executableTarget(name: "train", dependencies: ["Mozu"] + mlxProducts, path: "script/train"),
    .executableTarget(name: "mozu-web", dependencies: ["Mozu", "MozuWeb"], path: "script/web"),
    .testTarget(name: "MozuTests", dependencies: ["Mozu", "MozuWeb"] + mlxProducts, path: "Tests/MozuTests"),
]

// 手元の計測用スクリプト（script/probe/main.swift、リポジトリには含めない）があるときだけ probe ターゲットを追加する
if FileManager.default.fileExists(atPath: "script/probe/main.swift") {
    products.append(.executable(name: "probe", targets: ["probe"]))
    targets.append(.executableTarget(name: "probe", dependencies: ["Mozu"] + mlxProducts, path: "script/probe"))
}

let package = Package(
    name: "Mozu",
    platforms: [.macOS(.v14)],
    products: products,
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
    ],
    targets: targets
)
#endif
