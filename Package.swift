// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "62af7587090c7afefef7b08869bc20c74c1186e108d5654a56ccd86d4a70e8e3",
    "sha256" + debug_suffix: "8812f6578d92fd1ea2f9b68e01ddc81968fc492d7cd30e70e574a3a07cdfb664",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "354244a9996b663721f6637c50b0ec93f07596f169fc3d5bd70a04afd119d3af",
    "sha256" + debug_suffix: "95ffde66829108afef8c13ee29e97c5790ad6ca0f3957c620e5268e0df1a749c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2c96b55b8c45892a933df8fa145398b085b4cb5f182e7b23e265ccdfd69dd24f",
    "sha256" + debug_suffix: "48eab359b236565c61ae3b2f11f68b13afe29396ffa1657b96ea8a89fc4763c5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1a6c7d61e947c982c724c7bedf123c2c59dd10011ba6150770ff2163b05c8f92",
    "sha256" + debug_suffix: "21e50c79ae3421944d2d16b0f4c98e313ba2dbdb81629d4fa0d989746e12886f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "77f79477d90a7924a5c3f8f1758220dcf4180b786cddf74cd1a1c7a6e5db83c2",
    "sha256" + debug_suffix: "e76c3eb109147069f08aea6ed4ecb04f96f689d4f65858fa75012404d71b4929",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "50b57bfe81f3c338f4fd5764b8da806868abce6962f248d0330b1765960c6a4f",
    "sha256" + debug_suffix: "0f994878d770c35cd731ea790300da3d0bc89dd92b57509fe16492df9fcc8502",
  ],
  "kernels_optimized": [
    "sha256": "67c8929d94bd289536e078d03471238a9ccb5e917c4470881801bd8eb34bb54b",
    "sha256" + debug_suffix: "43cee7b2e1f4f976a1d5f0d4a41c4e3b64ddfece72993aeb45fee9f357638730",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b964d25953ef245c926851f34772f21b9d2eecb19db020c6bac4dd25340c63d5",
    "sha256" + debug_suffix: "a5c54e79b11747f1213464ff8b11cdc476fd13e35a44e75b77175136fec41b31",
  ],
  "kernels_torchao": [
    "sha256": "7d4a88722f2c81af85a7ef930349a09a2316c3a7ae181ac4bbc02afea5ff5280",
    "sha256" + debug_suffix: "d63af0d216113dac16d9349156a324c929eb1725dd3dea3ef7cd2de1166003e9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2cbe2384d1f80d061ff2023e2bbb2215f6327963ff7d75a7439fe4e8f1e2916f",
    "sha256" + debug_suffix: "c16221a59ef6b79aa4db77af8ea073cf1d36f44b582d4872753634dc0bac27bd",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
