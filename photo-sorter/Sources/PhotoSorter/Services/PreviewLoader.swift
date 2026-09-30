import AppKit
import Foundation
import ImageIO
import QuickLookThumbnailing

final class PreviewLoader {
    static let shared = PreviewLoader()
    private let cache = NSCache<NSString, NSImage>()

    func image(for group: PhotoGroup, selectedURL: URL? = nil, fullResolution: Bool = false, maxSize: CGSize = CGSize(width: 2400, height: 1600)) async -> NSImage? {
        guard let url = selectedURL ?? group.previewURL else { return nil }
        let cacheKey = "\(url.path)|\(fullResolution ? "full" : "fit")" as NSString
        if let cached = cache.object(forKey: cacheKey) { return cached }

        let image: NSImage?
        if url.pathExtension.lowercased() != "arw" {
            image = await Task.detached(priority: .userInitiated) {
                fullResolution
                    ? Self.fullImage(url: url)
                    : Self.imageIOThumbnail(url: url, maxPixel: max(maxSize.width, maxSize.height))
            }.value
        } else {
            let requestedSize = fullResolution ? CGSize(width: 7000, height: 7000) : maxSize
            image = try? await quickLookThumbnail(url: url, size: requestedSize)
        }
        if let image { cache.setObject(image, forKey: cacheKey) }
        return image
    }

    func preload(_ groups: [PhotoGroup]) {
        for group in groups where group.previewURL?.pathExtension.lowercased() != "arw" {
            Task(priority: .utility) { _ = await image(for: group, maxSize: CGSize(width: 1600, height: 1200)) }
        }
    }

    private static func imageIOThumbnail(url: URL, maxPixel: CGFloat) -> NSImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: Int(maxPixel),
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return NSImage(cgImage: cgImage, size: .zero)
    }

    private static func fullImage(url: URL) -> NSImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { return nil }
        return NSImage(cgImage: cgImage, size: .zero)
    }

    private func quickLookThumbnail(url: URL, size: CGSize) async throws -> NSImage {
        let request = QLThumbnailGenerator.Request(fileAt: url, size: size, scale: 2, representationTypes: .thumbnail)
        return try await withCheckedThrowingContinuation { continuation in
            QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { representation, error in
                if let image = representation?.nsImage { continuation.resume(returning: image) }
                else { continuation.resume(throwing: error ?? CocoaError(.fileReadUnknown)) }
            }
        }
    }
}
