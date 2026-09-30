import Foundation
import ImageIO

actor MetadataLoader {
    static let shared = MetadataLoader()

    private var cache: [String: PhotoMetadata] = [:]
    private var failed: Set<String> = []

    func metadata(for group: PhotoGroup) -> PhotoMetadata? {
        guard let url = group.previewURL else { return nil }
        if let cached = cache[url.path] { return cached }
        if failed.contains(url.path) { return nil }

        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let rawProperties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil),
              let properties = rawProperties as? [CFString: Any]
        else {
            failed.insert(url.path)
            return nil
        }

        let metadata = Self.parse(properties: properties)
        if metadata.hasProfessionalData { cache[url.path] = metadata }
        else { failed.insert(url.path) }
        return metadata.hasProfessionalData ? metadata : nil
    }

    static func parse(properties: [CFString: Any]) -> PhotoMetadata {
        let exif = dictionary(properties[kCGImagePropertyExifDictionary])
        let tiff = dictionary(properties[kCGImagePropertyTIFFDictionary])
        let exifAux = dictionary(properties[kCGImagePropertyExifAuxDictionary])

        let make = text(tiff[kCGImagePropertyTIFFMake])
        let model = text(tiff[kCGImagePropertyTIFFModel])
        let camera = joinedUnique(make, model)
        let lens = text(exif[kCGImagePropertyExifLensModel])
            ?? text(exifAux[kCGImagePropertyExifAuxLensModel])

        let focal = number(exif[kCGImagePropertyExifFocalLength]).map { format($0, maximumFractionDigits: 1) + " mm" }
        let aperture = number(exif[kCGImagePropertyExifFNumber]).map { "ƒ/" + format($0, maximumFractionDigits: 1) }
        let shutter = number(exif[kCGImagePropertyExifExposureTime]).map(formatShutter)
        let iso = isoValue(exif[kCGImagePropertyExifISOSpeedRatings]).map { "ISO \($0)" }
        let bias = number(exif[kCGImagePropertyExifExposureBiasValue]).map(formatBias)
        let date = text(exif[kCGImagePropertyExifDateTimeOriginal]) ?? text(tiff[kCGImagePropertyTIFFDateTime])
        let capturedAt = date.map(formatDate)

        let width = integer(properties[kCGImagePropertyPixelWidth])
        let height = integer(properties[kCGImagePropertyPixelHeight])
        let dimensions: String?
        if let width, let height {
            let megapixels = Double(width * height) / 1_000_000
            dimensions = "\(width) × \(height)  ·  \(format(megapixels, maximumFractionDigits: 1)) MP"
        } else {
            dimensions = nil
        }

        return PhotoMetadata(camera: camera, lens: lens, focalLength: focal,
                             aperture: aperture, shutterSpeed: shutter, iso: iso,
                             exposureBias: bias, capturedAt: capturedAt, dimensions: dimensions)
    }

    private static func dictionary(_ value: Any?) -> [CFString: Any] {
        value as? [CFString: Any] ?? [:]
    }

    private static func text(_ value: Any?) -> String? {
        guard let value = value as? String else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func number(_ value: Any?) -> Double? {
        if let number = value as? NSNumber { return number.doubleValue }
        if let string = value as? String { return Double(string) }
        return nil
    }

    private static func integer(_ value: Any?) -> Int? {
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String { return Int(string) }
        return nil
    }

    private static func isoValue(_ value: Any?) -> Int? {
        if let values = value as? [NSNumber] { return values.first?.intValue }
        return integer(value)
    }

    private static func joinedUnique(_ first: String?, _ second: String?) -> String? {
        switch (first, second) {
        case let (first?, second?) where second.localizedCaseInsensitiveContains(first): return second
        case let (first?, second?): return "\(first) \(second)"
        case let (first?, nil): return first
        case let (nil, second?): return second
        default: return nil
        }
    }

    private static func formatShutter(_ seconds: Double) -> String {
        guard seconds > 0 else { return "0 s" }
        if seconds < 0.8 {
            return "1/\(max(Int((1 / seconds).rounded()), 1)) s"
        }
        return format(seconds, maximumFractionDigits: 1) + " s"
    }

    private static func formatBias(_ value: Double) -> String {
        if abs(value) < 0.05 { return "0 EV" }
        return (value > 0 ? "+" : "") + format(value, maximumFractionDigits: 1) + " EV"
    }

    private static func format(_ value: Double, maximumFractionDigits: Int) -> String {
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = maximumFractionDigits
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    private static func formatDate(_ value: String) -> String {
        let parser = DateFormatter()
        parser.locale = Locale(identifier: "en_US_POSIX")
        parser.dateFormat = "yyyy:MM:dd HH:mm:ss"
        guard let date = parser.date(from: value) else { return value }
        let output = DateFormatter()
        output.locale = Locale(identifier: "zh_CN")
        output.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return output.string(from: date)
    }
}
