import Foundation
import ImageIO

@main
struct ValidationMain {
    @MainActor
    static func main() throws {
        var checks = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
            guard condition() else { throw ValidationError.failed(message) }
            checks += 1
            print("✓ \(message)")
        }

        let root = try temporaryDirectory()
        try file("DSC10.JPEG", in: root, bytes: 10)
        try file("dsc10.aRw", in: root, bytes: 20)
        try file("DSC10_edit.TIF", in: root, bytes: 40)
        try file("DSC10_edited_1.JPG", in: root, bytes: 41)
        try file("DSC2.jpg", in: root, bytes: 2)
        try file("ONLYRAW.ARW", in: root, bytes: 30)
        let groups = try PhotoScanner().scan(directory: root)
        try check(groups.map(\.id) == ["dsc2", "dsc10", "onlyraw"], "自然排序与扩展名大小写识别")
        try check(groups[1].jpgURL != nil && groups[1].rawURL != nil, "JPG 与 ARW 正确配对")
        try check(groups[0].rawURL == nil && groups[2].jpgURL == nil, "单独 JPG / ARW 不丢失")
        try check(groups[1].editedURLs.map(\.lastPathComponent) == ["DSC10_edit.TIF", "DSC10_edited_1.JPG"],
                  "后期文件及数字版本后缀归入原照片组")
        try check(groups[1].previewCandidates.map(\.lastPathComponent) == ["DSC10.JPEG", "dsc10.aRw", "DSC10_edit.TIF", "DSC10_edited_1.JPG"],
                  "原片与精修图生成可切换的预览版本")

        let metadata = MetadataLoader.parse(properties: [
            kCGImagePropertyPixelWidth: 6192,
            kCGImagePropertyPixelHeight: 4128,
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFMake: "SONY",
                kCGImagePropertyTIFFModel: "ILCE-6700"
            ],
            kCGImagePropertyExifDictionary: [
                kCGImagePropertyExifLensModel: "E 16-55mm F2.8 G",
                kCGImagePropertyExifFocalLength: 35.0,
                kCGImagePropertyExifFNumber: 2.8,
                kCGImagePropertyExifExposureTime: 0.002,
                kCGImagePropertyExifISOSpeedRatings: [800],
                kCGImagePropertyExifExposureBiasValue: -0.7,
                kCGImagePropertyExifDateTimeOriginal: "2026:09:28 16:30:10"
            ]
        ])
        try check(metadata.camera == "SONY ILCE-6700" && metadata.lens == "E 16-55mm F2.8 G",
                  "相机与镜头元数据解析")
        try check(metadata.focalLength == "35 mm" && metadata.aperture == "ƒ/2.8" && metadata.shutterSpeed == "1/500 s" && metadata.iso == "ISO 800",
                  "专业曝光参数格式化")
        try check(metadata.exposureBias == "-0.7 EV" && metadata.capturedAt == "2026-09-28 16:30:10" && metadata.dimensions == "6192 × 4128  ·  25.6 MP",
                  "曝光补偿、时间与分辨率格式化")

        let planGroups = [
            fakeGroup("trash", root), fakeGroup("all", root), fakeGroup("jpg", root),
            fakeGroup("raw", root), fakeGroup("waiting", root)
        ]
        let classifications: [String: Classification] = [
            "trash": .trash, "all": .keepAll, "jpg": .keepJPG, "raw": .keepRAW
        ]
        let plan = OrganizationPlanner().makePlan(groups: planGroups, classifications: classifications)
        try check(plan.operations.count == 4 && plan.trashGroups == 1 && plan.keepAllGroups == 1 && plan.keepJPGGroups == 1 && plan.keepRAWGroups == 1,
                  "四种分类生成正确操作计划")
        try check(plan.unprocessedGroups == 1 && !plan.operations.contains(where: { $0.groupID == "waiting" }), "未处理照片保持不动")

        var protectedGroup = fakeGroup("protected", root)
        let editedURL = root.appendingPathComponent("protected_edit.psd")
        protectedGroup.editedURLs = [editedURL]
        protectedGroup.editedSize = 99
        let protectedPlan = OrganizationPlanner().makePlan(groups: [protectedGroup], classifications: ["protected": .trash])
        try check(!protectedPlan.operations.contains(where: { $0.url == editedURL }) && protectedPlan.operations.count == 2,
                  "后期文件受保护且不进入废纸篓计划")

        let state = SavedState(currentGroupID: "dsc10", classifications: ["dsc10": .keepRAW],
                               starredGroupIDs: ["dsc10"], notes: ["dsc10": "准备精修人像"],
                               updatedAt: Date(timeIntervalSince1970: 1_700_000_000))
        try StateStore().save(state, to: root)
        guard case .loaded(let loaded) = StateStore().load(from: root) else { throw ValidationError.failed("状态恢复") }
        try check(loaded == state, "分类与星标状态保存、恢复")

        let legacyDirectory = try temporaryDirectory()
        let legacyJSON = """
        {"version":1,"currentGroupID":"old","classifications":{"old":"keepAll"},"updatedAt":"2026-09-28T10:00:00Z"}
        """
        try Data(legacyJSON.utf8).write(to: legacyDirectory.appendingPathComponent(StateStore.fileName))
        guard case .loaded(let legacyState) = StateStore().load(from: legacyDirectory) else {
            throw ValidationError.failed("旧状态兼容")
        }
        try check(legacyState.starredGroupIDs.isEmpty && legacyState.notes.isEmpty && legacyState.classifications["old"] == .keepAll,
                  "兼容不含星标字段的旧状态文件")

        let sessionDirectory = try temporaryDirectory()
        try file("A.JPG", in: sessionDirectory, bytes: 1)
        try file("A.ARW", in: sessionDirectory, bytes: 1)
        try file("A_edit.JPG", in: sessionDirectory, bytes: 1)
        try file("B.JPG", in: sessionDirectory, bytes: 1)
        try file("C.ARW", in: sessionDirectory, bytes: 1)
        let session = PhotoSession()
        session.open(directory: sessionDirectory)
        session.toggleStar()
        session.setNote("降低高光，保留肤色", for: session.groups[0])
        try check(session.starredCount == 1 && session.isStarred(session.groups[0]), "添加星标并更新统计")
        try check(session.notedCount == 1 && session.note(for: session.groups[0]) == "降低高光，保留肤色", "添加备注并实时更新统计")
        _ = session.classify(.trash, autoAdvance: false)
        session.undo()
        try check(session.classifiedCount == 0, "撤销更新状态与统计")
        session.redo()
        try check(session.classifications["a"] == .trash && session.classifiedCount == 1, "重做更新状态与统计")
        let restoredSession = PhotoSession()
        restoredSession.open(directory: sessionDirectory)
        try check(restoredSession.starredCount == 1 && restoredSession.isStarred(restoredSession.groups[0]),
                  "重新打开目录后恢复星标")
        try check(restoredSession.note(for: restoredSession.groups[0]) == "降低高光，保留肤色", "重新打开目录后恢复备注")

        session.setFilter(.unprocessed)
        try check(session.filteredCount == 2 && session.currentGroup?.id == "b", "筛选未处理照片")
        _ = session.classify(.keepAll)
        try check(session.filteredCount == 1 && session.currentGroup?.id == "c", "未处理筛选中分类后自动前进")
        session.setFilter(.starred)
        try check(session.filteredCount == 1 && session.currentGroup?.id == "a", "筛选星标照片")
        session.toggleStar()
        try check(session.filteredCount == 0 && session.currentGroup == nil, "取消星标后移出当前筛选")
        session.setFilter(.trash)
        try check(session.filteredCount == 1 && session.currentGroup?.id == "a", "按分类结果筛选照片")
        session.setFilter(.edited)
        try check(session.filteredCount == 1 && session.currentGroup?.id == "a", "筛选已后期照片")
        session.setFilter(.notEdited)
        try check(session.filteredCount == 2 && session.currentGroup?.id != "a", "筛选未后期照片")
        session.setFilter(.noted)
        try check(session.filteredCount == 1 && session.currentGroup?.id == "a", "筛选有备注照片")
        session.setNote("", for: session.groups[0])
        try check(session.filteredCount == 0 && session.currentGroup == nil, "清空备注后移出有备注筛选")
        session.setFilter(.trash)
        session.setNote("待交付", for: session.groups[0])
        let filterRestoredSession = PhotoSession()
        filterRestoredSession.open(directory: sessionDirectory)
        try check(filterRestoredSession.selectedFilter == .trash && filterRestoredSession.currentGroup?.id == "a",
                  "重新打开目录后恢复筛选条件与位置")
        try check(filterRestoredSession.note(for: filterRestoredSession.groups[0]) == "待交付", "筛选切换后备注仍保持")

        let safeDirectory = try temporaryDirectory()
        let outsideDirectory = try temporaryDirectory()
        let allowed = try file("allowed.JPG", in: safeDirectory, bytes: 1)
        let notScanned = try file("not-scanned.JPG", in: safeDirectory, bytes: 1)
        let outside = try file("outside.JPG", in: outsideDirectory, bytes: 1)
        let snapshot = [PhotoGroup(id: "allowed", baseName: "allowed", jpgURL: allowed, rawURL: nil, jpgSize: 1, rawSize: nil)]
        let unsafePlan = emptyPlan([
            FileOperation(groupID: "allowed", url: allowed),
            FileOperation(groupID: "other", url: notScanned),
            FileOperation(groupID: "outside", url: outside)
        ])
        let recorder = Recorder()
        let result = OrganizationExecutor(trashMover: recorder).execute(plan: unsafePlan, directory: safeDirectory, scannedGroups: snapshot)
        try check(recorder.urls == [allowed] && result.failures.count == 2, "执行器仅处理目录内已扫描文件")

        let trashDirectory = try temporaryDirectory()
        let trashFile = try file("trash-api-\(UUID().uuidString).JPG", in: trashDirectory, bytes: 1)
        try SystemTrashMover().trash(trashFile)
        try check(!FileManager.default.fileExists(atPath: trashFile.path), "临时文件通过 macOS 系统废纸篓 API 移动")
        print("\n全部 \(checks) 项核心验证通过。")
    }

    private static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("PhotoSorterValidation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @discardableResult private static func file(_ name: String, in directory: URL, bytes: Int) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data(repeating: 0x41, count: bytes).write(to: url)
        return url
    }

    private static func fakeGroup(_ id: String, _ directory: URL) -> PhotoGroup {
        PhotoGroup(id: id, baseName: id, jpgURL: directory.appendingPathComponent("\(id).JPG"),
                   rawURL: directory.appendingPathComponent("\(id).ARW"), jpgSize: 10, rawSize: 20)
    }

    private static func emptyPlan(_ operations: [FileOperation]) -> OrganizationPlan {
        OrganizationPlan(operations: operations, trashGroups: 0, keepJPGGroups: 0, keepRAWGroups: 0,
                         keepAllGroups: 0, unprocessedGroups: 0, trashFiles: 0,
                         keepJPGTrashFiles: 0, keepRAWTrashFiles: 0, bytesToRelease: 0)
    }
}

private enum ValidationError: Error { case failed(String) }
private final class Recorder: TrashMoving {
    var urls: [URL] = []
    func trash(_ url: URL) throws { urls.append(url) }
}
