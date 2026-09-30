import Foundation
import ImageIO
import XCTest
@testable import PhotoSorter

final class PhotoSorterTests: XCTestCase {
    private var temporaryDirectories: [URL] = []

    func testScannerPairsCaseInsensitivelyAndNaturallySorts() throws {
        let directory = try makeDirectory()
        try makeFile("DSC10.JPEG", in: directory, bytes: 10)
        try makeFile("dsc10.aRw", in: directory, bytes: 20)
        try makeFile("DSC10_edit.TIF", in: directory, bytes: 40)
        try makeFile("DSC10_edited_1.JPG", in: directory, bytes: 41)
        try makeFile("DSC2.jpg", in: directory, bytes: 2)
        try makeFile("ONLYRAW.ARW", in: directory, bytes: 30)
        try makeFile("ignored.png", in: directory, bytes: 1)

        let groups = try PhotoScanner().scan(directory: directory)

        XCTAssertEqual(groups.map(\.id), ["dsc2", "dsc10", "onlyraw"])
        XCTAssertNotNil(groups[1].jpgURL)
        XCTAssertNotNil(groups[1].rawURL)
        XCTAssertEqual(groups[1].editedURLs.map(\.lastPathComponent), ["DSC10_edit.TIF", "DSC10_edited_1.JPG"])
        XCTAssertEqual(groups[1].previewCandidates.map(\.lastPathComponent),
                       ["DSC10.JPEG", "dsc10.aRw", "DSC10_edit.TIF", "DSC10_edited_1.JPG"])
        XCTAssertNotNil(groups[0].jpgURL)
        XCTAssertNil(groups[0].rawURL)
        XCTAssertNil(groups[2].jpgURL)
        XCTAssertNotNil(groups[2].rawURL)
    }

    func testAllClassificationsProduceExpectedPlanAndUnprocessedStaysPut() throws {
        let directory = try makeDirectory()
        let groups = [
            group("trash", directory, jpg: true, raw: true),
            group("all", directory, jpg: true, raw: true),
            group("jpg", directory, jpg: true, raw: true),
            group("raw", directory, jpg: true, raw: true),
            group("waiting", directory, jpg: true, raw: true)
        ]
        let classes: [String: Classification] = [
            "trash": .trash, "all": .keepAll, "jpg": .keepJPG, "raw": .keepRAW
        ]

        let plan = OrganizationPlanner().makePlan(groups: groups, classifications: classes)

        XCTAssertEqual(plan.operations.map { $0.url.lastPathComponent }.sorted(),
                       ["jpg.ARW", "raw.JPG", "trash.ARW", "trash.JPG"])
        XCTAssertEqual(plan.trashGroups, 1)
        XCTAssertEqual(plan.keepAllGroups, 1)
        XCTAssertEqual(plan.keepJPGGroups, 1)
        XCTAssertEqual(plan.keepRAWGroups, 1)
        XCTAssertEqual(plan.unprocessedGroups, 1)
        XCTAssertFalse(plan.operations.contains { $0.groupID == "waiting" })
    }

    func testEditedFilesAreNeverAddedToTrashPlan() {
        let directory = URL(fileURLWithPath: "/tmp/photo-sorter-plan")
        let edited = directory.appendingPathComponent("A_edit.psd")
        let group = PhotoGroup(id: "a", baseName: "A",
                               jpgURL: directory.appendingPathComponent("A.JPG"),
                               rawURL: directory.appendingPathComponent("A.ARW"),
                               jpgSize: 10, rawSize: 20, editedURLs: [edited], editedSize: 100)
        let plan = OrganizationPlanner().makePlan(groups: [group], classifications: ["a": .trash])
        XCTAssertEqual(plan.operations.count, 2)
        XCTAssertFalse(plan.operations.contains { $0.url == edited })
    }

    func testProfessionalMetadataParsingAndFormatting() {
        let metadata = MetadataLoader.parse(properties: [
            kCGImagePropertyPixelWidth: 6192,
            kCGImagePropertyPixelHeight: 4128,
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFMake: "SONY",
                kCGImagePropertyTIFFModel: "ILCE-6700"
            ],
            kCGImagePropertyExifDictionary: [
                kCGImagePropertyExifFocalLength: 35.0,
                kCGImagePropertyExifFNumber: 2.8,
                kCGImagePropertyExifExposureTime: 0.002,
                kCGImagePropertyExifISOSpeedRatings: [800],
                kCGImagePropertyExifExposureBiasValue: 0.7
            ]
        ])
        XCTAssertEqual(metadata.camera, "SONY ILCE-6700")
        XCTAssertEqual(metadata.focalLength, "35 mm")
        XCTAssertEqual(metadata.aperture, "ƒ/2.8")
        XCTAssertEqual(metadata.shutterSpeed, "1/500 s")
        XCTAssertEqual(metadata.iso, "ISO 800")
        XCTAssertEqual(metadata.exposureBias, "+0.7 EV")
        XCTAssertEqual(metadata.dimensions, "6192 × 4128  ·  25.6 MP")
    }

    func testStateRoundTrip() throws {
        let directory = try makeDirectory()
        let expected = SavedState(currentGroupID: "dsc2", classifications: ["dsc2": .keepRAW],
                                  starredGroupIDs: ["dsc2"], notes: ["dsc2": "精修"],
                                  updatedAt: Date(timeIntervalSince1970: 1_700_000_000))
        try StateStore().save(expected, to: directory)
        guard case .loaded(let actual) = StateStore().load(from: directory) else {
            return XCTFail("State did not load")
        }
        XCTAssertEqual(actual, expected)
    }

    @MainActor
    func testUndoRedoUpdatesClassificationAndCounts() throws {
        let directory = try makeDirectory()
        try makeFile("A.JPG", in: directory, bytes: 1)
        try makeFile("A.ARW", in: directory, bytes: 1)
        let session = PhotoSession()
        session.open(directory: directory)

        session.toggleStar()
        session.setNote("调整高光", for: session.groups[0])
        XCTAssertEqual(session.starredCount, 1)
        XCTAssertEqual(session.note(for: session.groups[0]), "调整高光")
        XCTAssertTrue(session.isStarred(session.groups[0]))

        XCTAssertTrue(session.classify(.trash, autoAdvance: false))
        XCTAssertEqual(session.classifiedCount, 1)
        session.undo()
        XCTAssertEqual(session.classifiedCount, 0)
        XCTAssertEqual(session.unprocessedCount, 1)
        session.redo()
        XCTAssertEqual(session.classifications["a"], .trash)
        XCTAssertEqual(session.classifiedCount, 1)
        XCTAssertEqual(session.starredCount, 1)
    }

    @MainActor
    func testFilteringNavigationAndPersistence() throws {
        let directory = try makeDirectory()
        try makeFile("A.JPG", in: directory, bytes: 1)
        try makeFile("B.JPG", in: directory, bytes: 1)
        try makeFile("C.ARW", in: directory, bytes: 1)
        let session = PhotoSession()
        session.open(directory: directory)
        session.toggleStar()
        _ = session.classify(.trash, autoAdvance: false)

        session.setFilter(.unprocessed)
        XCTAssertEqual(session.filteredCount, 2)
        XCTAssertEqual(session.currentGroup?.id, "b")
        _ = session.classify(.keepAll)
        XCTAssertEqual(session.filteredCount, 1)
        XCTAssertEqual(session.currentGroup?.id, "c")

        session.setFilter(.starred)
        XCTAssertEqual(session.currentGroup?.id, "a")
        session.toggleStar()
        XCTAssertEqual(session.filteredCount, 0)
        XCTAssertNil(session.currentGroup)

        session.setFilter(.trash)
        XCTAssertEqual(session.currentGroup?.id, "a")
        let restored = PhotoSession()
        restored.open(directory: directory)
        XCTAssertEqual(restored.selectedFilter, .trash)
        XCTAssertEqual(restored.currentGroup?.id, "a")
    }

    func testLegacyStateWithoutStarsStillLoads() throws {
        let directory = try makeDirectory()
        let json = """
        {"version":1,"currentGroupID":"a","classifications":{"a":"keepAll"},"updatedAt":"2026-09-28T10:00:00Z"}
        """
        try Data(json.utf8).write(to: directory.appendingPathComponent(StateStore.fileName))
        guard case .loaded(let state) = StateStore().load(from: directory) else {
            return XCTFail("Legacy state did not load")
        }
        XCTAssertTrue(state.starredGroupIDs.isEmpty)
        XCTAssertTrue(state.notes.isEmpty)
        XCTAssertEqual(state.selectedFilter, .all)
        XCTAssertEqual(state.classifications["a"], .keepAll)
    }

    func testExecutorRejectsAnythingOutsideDirectoryOrScanSnapshot() throws {
        let directory = try makeDirectory()
        let outside = try makeDirectory()
        let scannedURL = try makeFile("A.JPG", in: directory, bytes: 1)
        let unscannedURL = try makeFile("B.JPG", in: directory, bytes: 1)
        let outsideURL = try makeFile("C.JPG", in: outside, bytes: 1)
        let scanned = [PhotoGroup(id: "a", baseName: "A", jpgURL: scannedURL, rawURL: nil, jpgSize: 1, rawSize: nil)]
        let operations = [
            FileOperation(groupID: "a", url: scannedURL),
            FileOperation(groupID: "b", url: unscannedURL),
            FileOperation(groupID: "c", url: outsideURL)
        ]
        let plan = barePlan(operations)
        let mover = RecordingTrashMover()

        let result = OrganizationExecutor(trashMover: mover).execute(plan: plan, directory: directory, scannedGroups: scanned)

        XCTAssertEqual(mover.urls, [scannedURL])
        XCTAssertEqual(result.succeeded.count, 1)
        XCTAssertEqual(result.failures.count, 2)
    }

    func testSystemTrashMoverUsesMacOSTrashForTemporaryFile() throws {
        let directory = try makeDirectory()
        let file = try makeFile("safe-trash-test-\(UUID().uuidString).JPG", in: directory, bytes: 1)
        try SystemTrashMover().trash(file)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
    }

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("PhotoSorterTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        temporaryDirectories.append(url)
        return url
    }

    @discardableResult
    private func makeFile(_ name: String, in directory: URL, bytes: Int) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data(repeating: 0x41, count: bytes).write(to: url)
        return url
    }

    private func group(_ id: String, _ directory: URL, jpg: Bool, raw: Bool) -> PhotoGroup {
        PhotoGroup(id: id, baseName: id, jpgURL: jpg ? directory.appendingPathComponent("\(id).JPG") : nil,
                   rawURL: raw ? directory.appendingPathComponent("\(id).ARW") : nil,
                   jpgSize: jpg ? 10 : nil, rawSize: raw ? 20 : nil)
    }

    private func barePlan(_ operations: [FileOperation]) -> OrganizationPlan {
        OrganizationPlan(operations: operations, trashGroups: 0, keepJPGGroups: 0, keepRAWGroups: 0,
                         keepAllGroups: 0, unprocessedGroups: 0, trashFiles: 0,
                         keepJPGTrashFiles: 0, keepRAWTrashFiles: 0, bytesToRelease: 0)
    }
}

private final class RecordingTrashMover: TrashMoving {
    var urls: [URL] = []
    func trash(_ url: URL) throws { urls.append(url) }
}
