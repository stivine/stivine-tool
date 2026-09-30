import AppKit
import Foundation
import SwiftUI

@MainActor
final class PhotoSession: ObservableObject {
    @Published private(set) var directory: URL?
    @Published private(set) var groups: [PhotoGroup] = []
    @Published private(set) var classifications: [String: Classification] = [:]
    @Published private(set) var starredGroupIDs: Set<String> = []
    @Published private(set) var notes: [String: String] = [:]
    @Published private(set) var selectedFilter: PhotoFilter = .all
    @Published var currentIndex = 0
    @Published var message: String?
    @Published var executionResult: ExecutionResult?
    @Published var isBusy = false

    private let scanner = PhotoScanner()
    private let stateStore = StateStore()
    private let planner = OrganizationPlanner()
    private var undoStack: [ClassificationChange] = []
    private var redoStack: [ClassificationChange] = []

    var currentGroup: PhotoGroup? {
        guard groups.indices.contains(currentIndex), matchesFilter(groups[currentIndex]) else { return nil }
        return groups[currentIndex]
    }
    var classifiedCount: Int { groups.reduce(0) { $0 + (classifications[$1.id] == nil ? 0 : 1) } }
    var unprocessedCount: Int { groups.count - classifiedCount }
    var starredCount: Int { starredGroupIDs.count }
    var notedCount: Int { notes.count }
    var filteredIndices: [Int] { groups.indices.filter { matchesFilter(groups[$0]) } }
    var filteredCount: Int { filteredIndices.count }
    var filteredPosition: Int? { filteredIndices.firstIndex(of: currentIndex).map { $0 + 1 } }
    var canGoPrevious: Bool { neighborIndex(before: currentIndex) != nil }
    var canGoNext: Bool { neighborIndex(after: currentIndex) != nil }
    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "选择照片文件夹"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        open(directory: url)
    }

    func open(directory: URL) {
        isBusy = true
        message = nil
        do {
            let scanned = try scanner.scan(directory: directory)
            self.directory = directory.standardizedFileURL
            groups = scanned
            currentIndex = 0
            classifications = [:]
            starredGroupIDs = []
            notes = [:]
            selectedFilter = .all
            undoStack = []
            redoStack = []

            switch stateStore.load(from: directory) {
            case .missing: break
            case .damaged(let reason):
                message = "状态文件损坏，已忽略：\(reason)"
            case .loaded(let state):
                let validIDs = Set(scanned.map(\.id))
                classifications = state.classifications.filter { validIDs.contains($0.key) }
                starredGroupIDs = state.starredGroupIDs.intersection(validIDs)
                notes = state.notes.filter { validIDs.contains($0.key) && !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                selectedFilter = state.selectedFilter
                if let id = state.currentGroupID, let index = scanned.firstIndex(where: { $0.id == id }) {
                    currentIndex = index
                }
            }
            normalizeCurrentSelection()
            if scanned.isEmpty { message = "该文件夹当前层级没有 JPG、JPEG 或 ARW 照片。" }
            persist()
            preloadNeighbors()
        } catch {
            message = "无法读取文件夹：\(error.localizedDescription)"
        }
        isBusy = false
    }

    func rescan() {
        guard let directory else { return }
        let previousID = groups.indices.contains(currentIndex) ? groups[currentIndex].id : nil
        do {
            let scanned = try scanner.scan(directory: directory)
            groups = scanned
            let validIDs = Set(scanned.map(\.id))
            classifications = classifications.filter { validIDs.contains($0.key) }
            starredGroupIDs.formIntersection(validIDs)
            notes = notes.filter { validIDs.contains($0.key) }
            if let previousID, let index = scanned.firstIndex(where: { $0.id == previousID }) {
                currentIndex = index
            } else {
                currentIndex = min(currentIndex, max(scanned.count - 1, 0))
            }
            normalizeCurrentSelection()
            persist()
            preloadNeighbors()
            message = scanned.isEmpty ? "该文件夹当前层级没有可识别的照片。" : nil
        } catch {
            message = "重新扫描失败：\(error.localizedDescription)"
        }
    }

    @discardableResult
    func classify(_ classification: Classification, autoAdvance: Bool = true) -> Bool {
        guard let group = currentGroup else { return false }
        guard group.supports(classification) else {
            message = classification == .keepRAW
                ? "这一组没有 ARW，不能选择“只留 RAW”。"
                : "这一组没有 JPG/JPEG，不能选择“只留 JPG”。"
            return false
        }
        let old = classifications[group.id]
        guard old != classification else {
            if autoAdvance { next() }
            return true
        }
        let classifiedIndex = currentIndex
        classifications[group.id] = classification
        undoStack.append(ClassificationChange(groupID: group.id, oldValue: old, newValue: classification))
        redoStack.removeAll()
        persist()
        if autoAdvance {
            moveAfterItemChanged(from: classifiedIndex)
            persist()
            preloadNeighbors()
        }
        return true
    }

    func next() {
        guard let nextIndex = neighborIndex(after: currentIndex) else { return }
        currentIndex = nextIndex
        persist()
        preloadNeighbors()
    }

    func previous() {
        guard let previousIndex = neighborIndex(before: currentIndex) else { return }
        currentIndex = previousIndex
        persist()
        preloadNeighbors()
    }

    func toggleStar() {
        guard let group = currentGroup else { return }
        let changedIndex = currentIndex
        if starredGroupIDs.contains(group.id) {
            starredGroupIDs.remove(group.id)
        } else {
            starredGroupIDs.insert(group.id)
        }
        if selectedFilter == .starred && !starredGroupIDs.contains(group.id) {
            moveAfterItemChanged(from: changedIndex)
        }
        persist()
        preloadNeighbors()
    }

    func isStarred(_ group: PhotoGroup) -> Bool {
        starredGroupIDs.contains(group.id)
    }

    func note(for group: PhotoGroup) -> String {
        notes[group.id] ?? ""
    }

    func setNote(_ note: String, for group: PhotoGroup) {
        let changedIndex = currentIndex
        if note.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            notes.removeValue(forKey: group.id)
        } else {
            notes[group.id] = note
        }
        if selectedFilter == .noted && notes[group.id] == nil {
            moveAfterItemChanged(from: changedIndex)
        }
        persist()
    }

    func setFilter(_ filter: PhotoFilter) {
        guard selectedFilter != filter else { return }
        selectedFilter = filter
        normalizeCurrentSelection()
        persist()
        preloadNeighbors()
    }

    func count(for filter: PhotoFilter) -> Int {
        groups.reduce(0) { $0 + (matchesFilter($1, filter: filter) ? 1 : 0) }
    }

    func undo() {
        guard let change = undoStack.popLast() else { return }
        if let old = change.oldValue { classifications[change.groupID] = old }
        else { classifications.removeValue(forKey: change.groupID) }
        redoStack.append(change)
        if let index = groups.firstIndex(where: { $0.id == change.groupID }) { currentIndex = index }
        normalizeCurrentSelection(preferredIndex: currentIndex)
        persist()
    }

    func redo() {
        guard let change = redoStack.popLast() else { return }
        classifications[change.groupID] = change.newValue
        undoStack.append(change)
        if let index = groups.firstIndex(where: { $0.id == change.groupID }) { currentIndex = index }
        normalizeCurrentSelection(preferredIndex: currentIndex)
        persist()
    }

    func makePlan() -> OrganizationPlan { planner.makePlan(groups: groups, classifications: classifications) }

    func execute(_ plan: OrganizationPlan) {
        guard let directory else { return }
        isBusy = true
        let snapshot = groups
        Task { [weak self] in
            let result = await Task.detached(priority: .userInitiated) {
                OrganizationExecutor().execute(plan: plan, directory: directory, scannedGroups: snapshot)
            }.value
            guard let self else { return }
            self.executionResult = result
            self.isBusy = false
            self.refreshAfterExecution(result)
        }
    }

    private func refreshAfterExecution(_ result: ExecutionResult) {
        guard let directory else { return }
        do {
            groups = try scanner.scan(directory: directory)
            let remainingIDs = Set(groups.map(\.id))
            // A partially failed group remains classified so the next execution can retry
            // its remaining scanned file. Groups with every file trashed disappear here.
            classifications = classifications.filter { remainingIDs.contains($0.key) }
            starredGroupIDs.formIntersection(remainingIDs)
            notes = notes.filter { remainingIDs.contains($0.key) }
            currentIndex = min(currentIndex, max(groups.count - 1, 0))
            normalizeCurrentSelection()
            persist()
        } catch {
            message = "整理已执行，但刷新目录失败：\(error.localizedDescription)"
        }
    }

    private func persist() {
        guard let directory else { return }
        let state = SavedState(currentGroupID: currentGroup?.id, classifications: classifications,
                               starredGroupIDs: starredGroupIDs, notes: notes, selectedFilter: selectedFilter,
                               updatedAt: Date())
        do { try stateStore.save(state, to: directory) }
        catch { message = "无法保存进度：\(error.localizedDescription)" }
    }

    private func preloadNeighbors() {
        let indices = [neighborIndex(before: currentIndex), neighborIndex(after: currentIndex)].compactMap { $0 }
        PreviewLoader.shared.preload(indices.map { groups[$0] })
    }

    private func matchesFilter(_ group: PhotoGroup, filter: PhotoFilter? = nil) -> Bool {
        switch filter ?? selectedFilter {
        case .all: return true
        case .unprocessed: return classifications[group.id] == nil
        case .starred: return starredGroupIDs.contains(group.id)
        case .trash: return classifications[group.id] == .trash
        case .keepAll: return classifications[group.id] == .keepAll
        case .keepJPG: return classifications[group.id] == .keepJPG
        case .keepRAW: return classifications[group.id] == .keepRAW
        case .edited: return !group.editedURLs.isEmpty
        case .notEdited: return group.editedURLs.isEmpty
        case .noted: return notes[group.id] != nil
        }
    }

    private func neighborIndex(after index: Int) -> Int? {
        filteredIndices.first(where: { $0 > index })
    }

    private func neighborIndex(before index: Int) -> Int? {
        filteredIndices.last(where: { $0 < index })
    }

    private func moveAfterItemChanged(from index: Int) {
        if groups.indices.contains(index), matchesFilter(groups[index]) {
            if let next = neighborIndex(after: index) { currentIndex = next }
            return
        }
        if let next = neighborIndex(after: index) {
            currentIndex = next
        } else if let previous = neighborIndex(before: index) {
            currentIndex = previous
        } else if let first = filteredIndices.first {
            currentIndex = first
        }
    }

    private func normalizeCurrentSelection(preferredIndex: Int? = nil) {
        guard !filteredIndices.isEmpty else { return }
        let preferred = preferredIndex ?? currentIndex
        if groups.indices.contains(preferred), matchesFilter(groups[preferred]) {
            currentIndex = preferred
        } else {
            currentIndex = filteredIndices.first(where: { $0 >= preferred }) ?? filteredIndices[0]
        }
    }
}
