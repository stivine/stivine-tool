import AppKit
import SwiftUI

struct ContentView: View {
    @StateObject private var session = PhotoSession()
    @State private var zoomScale: CGFloat?
    @State private var selectedPreviewPaths: [String: String] = [:]
    @State private var confirmationPlan: OrganizationPlan?

    var body: some View {
        HSplitView {
            VStack(spacing: 12) {
                topBar
                PreviewView(
                    group: session.currentGroup,
                    selectedURL: selectedPreviewURL,
                    emptyMessage: session.directory == nil ? "选择照片文件夹开始分拣" : "当前筛选没有照片",
                    zoomScale: $zoomScale
                )
                bottomBar
            }
            .padding(14)
            .frame(minWidth: 620)

            sidebar.frame(minWidth: 250, idealWidth: 290, maxWidth: 340)
        }
        .frame(minWidth: 920, minHeight: 620)
        .preferredColorScheme(.dark)
        .background(KeyboardCapture(handler: handleKey).frame(width: 0, height: 0))
        .sheet(item: $confirmationPlan) { plan in
            ConfirmationView(plan: plan, confirm: {
                confirmationPlan = nil
                session.execute(plan)
            }, cancel: { confirmationPlan = nil })
        }
        .sheet(item: Binding(get: { session.executionResult.map(ResultBox.init) }, set: { box in
            if box == nil { session.executionResult = nil }
        })) { box in
            ResultView(result: box.result) { session.executionResult = nil }
        }
        .alert("提示", isPresented: Binding(get: { session.message != nil }, set: { if !$0 { session.message = nil } })) {
            Button("知道了") { session.message = nil }
        } message: { Text(session.message ?? "") }
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            Button { session.chooseDirectory() } label: { Label("选择文件夹", systemImage: "folder") }
            Button { session.rescan() } label: { Image(systemName: "arrow.clockwise") }
                .help("重新扫描后期文件 (R)")
                .disabled(session.directory == nil)
            if let directory = session.directory {
                Text(directory.path).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                ForEach(PhotoFilter.allCases) { filter in
                    Button {
                        session.setFilter(filter)
                    } label: {
                        HStack {
                            Label(filter.title, systemImage: filter.systemImage)
                            Text("\(session.count(for: filter))")
                            if session.selectedFilter == filter { Image(systemName: "checkmark") }
                        }
                    }
                }
            } label: {
                Label("\(session.selectedFilter.title) · \(session.filteredCount)", systemImage: "line.3.horizontal.decrease.circle")
            }
            .help("筛选当前浏览范围")
            if let group = session.currentGroup, group.previewCandidates.count > 1 {
                Menu {
                    ForEach(group.previewCandidates, id: \.path) { url in
                        Button {
                            selectPreview(url, for: group)
                        } label: {
                            HStack {
                                Label(previewLabel(url, in: group), systemImage: previewIcon(url, in: group))
                                if selectedPreviewURL == url { Image(systemName: "checkmark") }
                            }
                        }
                    }
                } label: {
                    Label(previewLabel(selectedPreviewURL, in: group), systemImage: "rectangle.2.swap")
                }
                .help("切换原片、RAW 或精修图 (V)")
            }
            Button { adjustZoom(dividingBy: 1.25) } label: { Image(systemName: "minus.magnifyingglass") }
                .help("缩小 (-)")
            Text(zoomScale.map { "\(Int(($0 * 100).rounded()))%" } ?? "适应")
                .font(.system(.caption, design: .monospaced)).frame(minWidth: 42)
            Button { adjustZoom(multiplyingBy: 1.25) } label: { Image(systemName: "plus.magnifyingglass") }
                .help("放大 (+)")
            Button { zoomScale = nil } label: { Image(systemName: "arrow.down.right.and.arrow.up.left") }
                .help("适应窗口")
            Button { zoomScale = 1 } label: { Text("1:1").font(.caption.bold()) }
                .help("100%")
            if let group = session.currentGroup {
                Button { openInPhotoshop(group) } label: { Text("PS").fontWeight(.bold) }
                    .help(group.rawURL == nil ? "用 Photoshop 打开" : "用 Photoshop Camera Raw 打开 ARW")
                Button { session.toggleStar() } label: {
                    Image(systemName: session.isStarred(group) ? "star.fill" : "star")
                        .foregroundStyle(session.isStarred(group) ? .yellow : .primary)
                }
                .help("切换星标 (S)")
            }
            Button { toggleFullScreen() } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                .help("全屏 (F)")
        }
    }

    private var bottomBar: some View {
        HStack {
            Button { session.previous() } label: { Label("上一组", systemImage: "chevron.left") }
                .disabled(!session.canGoPrevious)
            Spacer()
            if let group = session.currentGroup {
                VStack(spacing: 3) {
                    Text(group.baseName).font(.headline)
                    Text(positionText(group))
                        .font(.subheadline).foregroundStyle(.secondary)
                    if let url = selectedPreviewURL {
                        Text("正在预览：\(url.lastPathComponent)")
                            .font(.caption2)
                            .foregroundStyle(group.editedURLs.contains(url) ? Color.purple : Color.secondary)
                    }
                }
            }
            Spacer()
            Button { session.next() } label: { Label("下一组", systemImage: "chevron.right") }
                .disabled(!session.canGoNext)
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 18) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("照片分拣").font(.title2.bold())
                    if let group = session.currentGroup {
                        info(group)
                        Divider()
                        PhotoMetadataView(group: group)
                        Divider()
                        PhotoNoteView(group: group, session: session)
                        Divider()
                        VStack(spacing: 9) {
                            ForEach(Classification.allCases) { classification in
                                ClassificationButton(
                                    classification: classification,
                                    selected: session.classifications[group.id] == classification,
                                    enabled: group.supports(classification)
                                ) { _ = session.classify(classification) }
                            }
                        }
                        Text("当前：\(session.classifications[group.id]?.title ?? "未处理")")
                            .foregroundStyle(session.classifications[group.id]?.color ?? .gray)
                            .fontWeight(.semibold)
                    } else {
                        Text("尚未载入照片").foregroundStyle(.secondary)
                    }
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 7) {
                HStack { Text("已分类"); Spacer(); Text("\(session.classifiedCount)").foregroundStyle(.secondary) }
                HStack { Text("未处理"); Spacer(); Text("\(session.unprocessedCount)").foregroundStyle(.secondary) }
                HStack {
                    Label("星标", systemImage: "star.fill").foregroundStyle(.yellow)
                    Spacer()
                    Text("\(session.starredCount)").foregroundStyle(.secondary)
                }
                HStack {
                    Label("备注", systemImage: "note.text")
                    Spacer()
                    Text("\(session.notedCount)").foregroundStyle(.secondary)
                }
                ProgressView(value: session.groups.isEmpty ? 0 : Double(session.classifiedCount), total: max(Double(session.groups.count), 1))
            }
            HStack {
                Button { session.undo() } label: { Label("撤销", systemImage: "arrow.uturn.backward") }.disabled(!session.canUndo)
                Button { session.redo() } label: { Label("重做", systemImage: "arrow.uturn.forward") }.disabled(!session.canRedo)
            }
            Button(role: .destructive) { confirmationPlan = session.makePlan() } label: {
                Label("执行整理", systemImage: "trash").frame(maxWidth: .infinity)
            }
            .controlSize(.large)
            .disabled(session.groups.isEmpty || session.isBusy)
        }
        .padding(18)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder private func info(_ group: PhotoGroup) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(group.baseName).font(.headline)
            Label(group.composition, systemImage: "photo.stack")
            if group.jpgURL != nil { Text("JPG  \(size(group.jpgSize))").foregroundStyle(.secondary) }
            if group.rawURL != nil { Text("ARW  \(size(group.rawSize))").foregroundStyle(.secondary) }
            if !group.editedURLs.isEmpty {
                Label("已后期 \(group.editedURLs.count) 个  ·  \(size(group.editedSize))", systemImage: "wand.and.stars")
                    .foregroundStyle(.purple)
            }
        }
    }

    private func size(_ bytes: Int64?) -> String {
        bytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "未知大小"
    }

    private func positionText(_ group: PhotoGroup) -> String {
        let position = session.filteredPosition ?? 0
        let scope = session.selectedFilter == .all
            ? "\(position) / \(session.filteredCount)"
            : "\(position) / \(session.filteredCount)（总计 \(session.groups.count)）"
        return "\(scope)  ·  \(group.composition)"
    }

    private func handleKey(_ event: NSEvent) -> Bool {
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if modifiers.contains(.command) {
            guard event.charactersIgnoringModifiers?.lowercased() == "z" else { return false }
            modifiers.contains(.shift) ? session.redo() : session.undo()
            return true
        }
        switch event.keyCode {
        case 123: session.previous(); return true
        case 124: session.next(); return true
        case 49: zoomScale = zoomScale == nil ? 1 : nil; return true
        case 53:
            if zoomScale != nil { zoomScale = nil } else if NSApp.keyWindow?.styleMask.contains(.fullScreen) == true { toggleFullScreen() }
            return true
        default: break
        }
        switch event.charactersIgnoringModifiers?.lowercased() {
        case "1": return session.classify(.trash)
        case "2": return session.classify(.keepAll)
        case "3": return session.classify(.keepJPG)
        case "4": return session.classify(.keepRAW)
        case "a": session.previous(); return true
        case "d": session.next(); return true
        case "z": zoomScale = zoomScale == nil ? 1 : nil; return true
        case "+", "=": adjustZoom(multiplyingBy: 1.25); return true
        case "-", "_": adjustZoom(dividingBy: 1.25); return true
        case "s": session.toggleStar(); return true
        case "r": session.rescan(); return true
        case "v": cyclePreview(); return true
        case "f": toggleFullScreen(); return true
        default: return false
        }
    }

    private func toggleFullScreen() { NSApp.keyWindow?.toggleFullScreen(nil) }

    private func adjustZoom(multiplyingBy amount: CGFloat = 1, dividingBy divisor: CGFloat = 1) {
        zoomScale = min(max((zoomScale ?? 1) * amount / divisor, 0.1), 8)
    }

    private func openInPhotoshop(_ group: PhotoGroup) {
        ExternalEditorService().openInPhotoshop(group: group) { error in
            guard let error else { return }
            DispatchQueue.main.async { session.message = error.localizedDescription }
        }
    }

    private var selectedPreviewURL: URL? {
        guard let group = session.currentGroup else { return nil }
        if let path = selectedPreviewPaths[group.id],
           let selected = group.previewCandidates.first(where: { $0.path == path }) {
            return selected
        }
        return group.previewURL
    }

    private func selectPreview(_ url: URL, for group: PhotoGroup) {
        selectedPreviewPaths[group.id] = url.path
        zoomScale = nil
    }

    private func cyclePreview() {
        guard let group = session.currentGroup, !group.previewCandidates.isEmpty else { return }
        let candidates = group.previewCandidates
        let index = selectedPreviewURL.flatMap { candidates.firstIndex(of: $0) } ?? 0
        selectPreview(candidates[(index + 1) % candidates.count], for: group)
    }

    private func previewLabel(_ url: URL?, in group: PhotoGroup) -> String {
        guard let url else { return "预览" }
        if url == group.jpgURL { return "原片 JPG" }
        if url == group.rawURL { return "原片 RAW" }
        let number = (group.editedURLs.firstIndex(of: url) ?? 0) + 1
        return group.editedURLs.count > 1 ? "精修 \(number)" : "精修图"
    }

    private func previewIcon(_ url: URL, in group: PhotoGroup) -> String {
        group.editedURLs.contains(url) ? "wand.and.stars" : (url == group.rawURL ? "camera.aperture" : "photo")
    }
}

private struct ResultBox: Identifiable {
    let id = UUID()
    let result: ExecutionResult
}

extension OrganizationPlan: Identifiable {
    var id: String { "\(operations.map(\.url.path).joined(separator: "|"))-\(unprocessedGroups)" }
}
