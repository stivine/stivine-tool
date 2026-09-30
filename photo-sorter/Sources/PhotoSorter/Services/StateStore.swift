import Foundation

enum StateLoadResult {
    case missing
    case loaded(SavedState)
    case damaged(String)
}

struct StateStore {
    static let fileName = ".photo-sorter.json"

    func load(from directory: URL) -> StateLoadResult {
        let url = directory.appendingPathComponent(Self.fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return .missing }
        do {
            let data = try Data(contentsOf: url)
            return .loaded(try decoder.decode(SavedState.self, from: data))
        } catch {
            return .damaged(error.localizedDescription)
        }
    }

    func save(_ state: SavedState, to directory: URL) throws {
        let data = try encoder.encode(state)
        let target = directory.appendingPathComponent(Self.fileName)
        let temporary = directory.appendingPathComponent(".photo-sorter.\(UUID().uuidString).tmp")
        try data.write(to: temporary, options: .atomic)
        if FileManager.default.fileExists(atPath: target.path) {
            _ = try FileManager.default.replaceItemAt(target, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: target)
        }
    }

    private var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
