import Foundation

enum OrganizationError: LocalizedError {
    case unsafePath(URL)
    case notScanned(URL)
    case fileChanged(URL)

    var errorDescription: String? {
        switch self {
        case .unsafePath(let url): return "路径不在所选目录内：\(url.path)"
        case .notScanned(let url): return "文件不在扫描清单中：\(url.path)"
        case .fileChanged(let url): return "文件已被移动、删除或无法访问：\(url.path)"
        }
    }
}

struct OrganizationPlanner {
    func makePlan(groups: [PhotoGroup], classifications: [String: Classification]) -> OrganizationPlan {
        var operations: [FileOperation] = []
        var trashGroups = 0, keepJPG = 0, keepRAW = 0, keepAll = 0, unprocessed = 0
        var trashFiles = 0, jpgTrashFiles = 0, rawTrashFiles = 0
        var bytes: Int64 = 0

        for group in groups {
            guard let classification = classifications[group.id] else {
                unprocessed += 1
                continue
            }
            switch classification {
            case .trash:
                trashGroups += 1
                trashFiles += group.originalURLs.count
                for url in group.originalURLs {
                    operations.append(FileOperation(groupID: group.id, url: url))
                }
                bytes += (group.jpgSize ?? 0) + (group.rawSize ?? 0)
            case .keepAll:
                keepAll += 1
            case .keepJPG:
                keepJPG += 1
                if let url = group.rawURL {
                    operations.append(FileOperation(groupID: group.id, url: url))
                    jpgTrashFiles += 1
                    bytes += group.rawSize ?? 0
                }
            case .keepRAW:
                keepRAW += 1
                if let url = group.jpgURL {
                    operations.append(FileOperation(groupID: group.id, url: url))
                    rawTrashFiles += 1
                    bytes += group.jpgSize ?? 0
                }
            }
        }
        return OrganizationPlan(
            operations: operations, trashGroups: trashGroups, keepJPGGroups: keepJPG,
            keepRAWGroups: keepRAW, keepAllGroups: keepAll, unprocessedGroups: unprocessed,
            trashFiles: trashFiles, keepJPGTrashFiles: jpgTrashFiles,
            keepRAWTrashFiles: rawTrashFiles, bytesToRelease: bytes
        )
    }
}

protocol TrashMoving {
    func trash(_ url: URL) throws
}

struct SystemTrashMover: TrashMoving {
    func trash(_ url: URL) throws {
        var resultingURL: NSURL?
        try FileManager.default.trashItem(at: url, resultingItemURL: &resultingURL)
    }
}

struct OrganizationExecutor {
    let trashMover: TrashMoving

    init(trashMover: TrashMoving = SystemTrashMover()) {
        self.trashMover = trashMover
    }

    func execute(plan: OrganizationPlan, directory: URL, scannedGroups: [PhotoGroup]) -> ExecutionResult {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let scanned = Set(scannedGroups.flatMap(\.urls).map { $0.standardizedFileURL.path })
        var succeeded: [FileOperation] = []
        var failures: [OperationFailure] = []

        for operation in plan.operations {
            do {
                let candidate = operation.url.standardizedFileURL
                let parent = candidate.deletingLastPathComponent().resolvingSymlinksInPath()
                guard parent == root else { throw OrganizationError.unsafePath(candidate) }
                guard scanned.contains(candidate.path) else { throw OrganizationError.notScanned(candidate) }
                guard FileManager.default.fileExists(atPath: candidate.path) else { throw OrganizationError.fileChanged(candidate) }
                try trashMover.trash(candidate)
                succeeded.append(operation)
            } catch {
                failures.append(OperationFailure(url: operation.url, reason: error.localizedDescription))
            }
        }
        return ExecutionResult(succeeded: succeeded, failures: failures)
    }
}
