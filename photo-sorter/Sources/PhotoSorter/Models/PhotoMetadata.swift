import Foundation

struct PhotoMetadata: Equatable {
    let camera: String?
    let lens: String?
    let focalLength: String?
    let aperture: String?
    let shutterSpeed: String?
    let iso: String?
    let exposureBias: String?
    let capturedAt: String?
    let dimensions: String?

    var hasProfessionalData: Bool {
        [camera, lens, focalLength, aperture, shutterSpeed, iso,
         exposureBias, capturedAt, dimensions].contains { $0 != nil }
    }
}
