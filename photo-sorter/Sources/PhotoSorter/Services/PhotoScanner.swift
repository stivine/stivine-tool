import Foundation

enum PhotoScannerError: LocalizedError {
    case notDirectory

    var errorDescription: String? { "选择的位置不是文件夹。" }
}

struct PhotoScanner {
    private struct Builder {
        var displayBase: String
        var jpg: URL?
        var raw: URL?
        var edited: [URL] = []
    }

    static let editedSuffixes = [
        "_retouched", "-retouched", "_edited", "-edited",
        "_final", "-final", "_edit", "-edit", "_精修", "-精修", "_修图", "-修图"
    ]
    private static let editedSuffixRegex = try! NSRegularExpression(
        pattern: "(?:_|-)(?:retouched|edited|final|edit|精修|修图)(?:(?:_|-)(?:v)?[0-9]+)?$",
        options: [.caseInsensitive]
    )

    func scan(directory: URL) throws -> [PhotoGroup] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw PhotoScannerError.notDirectory
        }

        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        let urls = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )
        var builders: [String: Builder] = [:]

        for url in urls {
            let values = try? url.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile == true else { continue }
            let ext = url.pathExtension.lowercased()
            guard ["jpg", "jpeg", "arw", "tif", "tiff", "psd"].contains(ext) else { continue }
            let fileBase = url.deletingPathExtension().lastPathComponent
            let editedBase = editedOriginalBase(fileBase, extension: ext)
            let base = editedBase ?? fileBase
            let id = base.lowercased()
            var builder = builders[id] ?? Builder(displayBase: base, jpg: nil, raw: nil)
            if editedBase != nil || ["tif", "tiff", "psd"].contains(ext) {
                builder.edited.append(url)
            } else if ext == "arw" {
                builder.raw = preferred(existing: builder.raw, candidate: url)
            } else {
                builder.jpg = preferred(existing: builder.jpg, candidate: url)
            }
            builders[id] = builder
        }

        return builders.map { id, builder in
            PhotoGroup(
                id: id,
                baseName: builder.displayBase,
                jpgURL: builder.jpg,
                rawURL: builder.raw,
                jpgSize: fileSize(builder.jpg),
                rawSize: fileSize(builder.raw),
                editedURLs: builder.edited.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending },
                editedSize: builder.edited.compactMap(fileSize).reduce(0, +)
            )
        }.sorted {
            $0.baseName.localizedStandardCompare($1.baseName) == .orderedAscending
        }
    }

    private func editedOriginalBase(_ base: String, extension ext: String) -> String? {
        guard ext != "arw" else { return nil }
        let fullRange = NSRange(base.startIndex..<base.endIndex, in: base)
        guard let match = Self.editedSuffixRegex.firstMatch(in: base, range: fullRange),
              match.range.location > 0,
              let range = Range(match.range, in: base) else { return nil }
        return String(base[..<range.lowerBound])
    }

    private func preferred(existing: URL?, candidate: URL) -> URL {
        guard let existing else { return candidate }
        return candidate.lastPathComponent.localizedStandardCompare(existing.lastPathComponent) == .orderedAscending
            ? candidate : existing
    }

    private func fileSize(_ url: URL?) -> Int64? {
        guard let url else { return nil }
        return try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize.map(Int64.init)
    }
}
