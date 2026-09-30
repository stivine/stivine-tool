import AppKit
import Foundation

enum ExternalEditorError: LocalizedError {
    case noFile
    case photoshopNotFound

    var errorDescription: String? {
        switch self {
        case .noFile: return "这一组没有可交给 Photoshop 的文件。"
        case .photoshopNotFound: return "未找到 Adobe Photoshop。请先安装 Photoshop，或将应用放在“应用程序”文件夹中。"
        }
    }
}

struct ExternalEditorService {
    func openInPhotoshop(group: PhotoGroup, completion: @escaping (Error?) -> Void) {
        guard let fileURL = group.rawURL ?? group.jpgURL ?? group.editedURLs.first else {
            completion(ExternalEditorError.noFile)
            return
        }
        guard let photoshopURL = photoshopApplicationURL() else {
            completion(ExternalEditorError.photoshopNotFound)
            return
        }

        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        NSWorkspace.shared.open([fileURL], withApplicationAt: photoshopURL, configuration: configuration) { _, error in
            completion(error)
        }
    }

    private func photoshopApplicationURL() -> URL? {
        if let registered = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.adobe.Photoshop") {
            return registered
        }
        let roots = [URL(fileURLWithPath: "/Applications"),
                     FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications")]
        for root in roots {
            guard let candidates = try? FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
            ) else { continue }
            if let app = candidates.first(where: {
                $0.pathExtension.lowercased() == "app" && $0.lastPathComponent.lowercased().contains("photoshop")
            }) {
                return app
            }
            for folder in candidates where folder.lastPathComponent.lowercased().contains("photoshop") {
                guard let nested = try? FileManager.default.contentsOfDirectory(
                    at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
                ) else { continue }
                if let app = nested.first(where: {
                    $0.pathExtension.lowercased() == "app" && $0.lastPathComponent.lowercased().contains("photoshop")
                }) {
                    return app
                }
            }
        }
        return nil
    }
}
