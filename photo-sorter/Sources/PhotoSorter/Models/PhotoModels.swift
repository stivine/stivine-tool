import Foundation

enum Classification: String, Codable, CaseIterable, Identifiable {
    case trash
    case keepAll
    case keepJPG
    case keepRAW

    var id: String { rawValue }

    var title: String {
        switch self {
        case .trash: return "废片"
        case .keepAll: return "全部保留"
        case .keepJPG: return "JPG · 记录"
        case .keepRAW: return "RAW · 精修"
        }
    }

    var key: String {
        switch self {
        case .trash: return "1"
        case .keepAll: return "2"
        case .keepJPG: return "3"
        case .keepRAW: return "4"
        }
    }
}

enum PhotoFilter: String, Codable, CaseIterable, Identifiable {
    case all
    case unprocessed
    case starred
    case trash
    case keepAll
    case keepJPG
    case keepRAW
    case edited
    case notEdited
    case noted

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: return "全部照片"
        case .unprocessed: return "未处理"
        case .starred: return "星标"
        case .trash: return "废片"
        case .keepAll: return "全部保留"
        case .keepJPG: return "JPG · 记录"
        case .keepRAW: return "RAW · 精修"
        case .edited: return "已后期"
        case .notEdited: return "未后期"
        case .noted: return "有备注"
        }
    }

    var systemImage: String {
        switch self {
        case .all: return "photo.stack"
        case .unprocessed: return "circle.dashed"
        case .starred: return "star.fill"
        case .trash: return "trash"
        case .keepAll: return "checkmark.circle"
        case .keepJPG: return "doc.richtext"
        case .keepRAW: return "camera.aperture"
        case .edited: return "wand.and.stars"
        case .notEdited: return "camera.filters"
        case .noted: return "note.text"
        }
    }
}

struct PhotoGroup: Identifiable, Equatable {
    let id: String
    let baseName: String
    let jpgURL: URL?
    let rawURL: URL?
    let jpgSize: Int64?
    let rawSize: Int64?
    var editedURLs: [URL] = []
    var editedSize: Int64 = 0

    var composition: String {
        var parts: [String] = []
        if jpgURL != nil { parts.append("JPG") }
        if rawURL != nil { parts.append("ARW") }
        if !editedURLs.isEmpty { parts.append("后期 ×\(editedURLs.count)") }
        return parts.isEmpty ? "—" : parts.joined(separator: " + ")
    }

    var urls: [URL] { [jpgURL, rawURL].compactMap { $0 } + editedURLs }
    var originalURLs: [URL] { [jpgURL, rawURL].compactMap { $0 } }
    var previewCandidates: [URL] { originalURLs + editedURLs }
    var previewURL: URL? { jpgURL ?? rawURL ?? editedURLs.first }
    var hasOriginal: Bool { jpgURL != nil || rawURL != nil }

    func supports(_ classification: Classification) -> Bool {
        switch classification {
        case .keepRAW: return rawURL != nil
        case .keepJPG: return jpgURL != nil
        case .trash: return hasOriginal
        case .keepAll: return true
        }
    }
}

struct SavedState: Codable, Equatable {
    static let currentVersion = 1
    var version: Int = currentVersion
    var currentGroupID: String?
    var classifications: [String: Classification] = [:]
    var starredGroupIDs: Set<String> = []
    var notes: [String: String] = [:]
    var selectedFilter: PhotoFilter = .all
    var updatedAt: Date = Date()

    init(version: Int = currentVersion, currentGroupID: String?,
         classifications: [String: Classification], starredGroupIDs: Set<String> = [],
         notes: [String: String] = [:], selectedFilter: PhotoFilter = .all, updatedAt: Date) {
        self.version = version
        self.currentGroupID = currentGroupID
        self.classifications = classifications
        self.starredGroupIDs = starredGroupIDs
        self.notes = notes
        self.selectedFilter = selectedFilter
        self.updatedAt = updatedAt
    }

    private enum CodingKeys: String, CodingKey {
        case version, currentGroupID, classifications, starredGroupIDs, notes, selectedFilter, updatedAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decodeIfPresent(Int.self, forKey: .version) ?? Self.currentVersion
        currentGroupID = try container.decodeIfPresent(String.self, forKey: .currentGroupID)
        classifications = try container.decodeIfPresent([String: Classification].self, forKey: .classifications) ?? [:]
        starredGroupIDs = try container.decodeIfPresent(Set<String>.self, forKey: .starredGroupIDs) ?? []
        notes = try container.decodeIfPresent([String: String].self, forKey: .notes) ?? [:]
        selectedFilter = try container.decodeIfPresent(PhotoFilter.self, forKey: .selectedFilter) ?? .all
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
    }
}

struct ClassificationChange: Equatable {
    let groupID: String
    let oldValue: Classification?
    let newValue: Classification
}

struct FileOperation: Equatable {
    let groupID: String
    let url: URL
}

struct OrganizationPlan {
    let operations: [FileOperation]
    let trashGroups: Int
    let keepJPGGroups: Int
    let keepRAWGroups: Int
    let keepAllGroups: Int
    let unprocessedGroups: Int
    let trashFiles: Int
    let keepJPGTrashFiles: Int
    let keepRAWTrashFiles: Int
    let bytesToRelease: Int64
}

struct OperationFailure: Identifiable {
    let id = UUID()
    let url: URL
    let reason: String
}

struct ExecutionResult {
    let succeeded: [FileOperation]
    let failures: [OperationFailure]
}
