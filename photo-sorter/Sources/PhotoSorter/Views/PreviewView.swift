import AppKit
import SwiftUI

struct PreviewView: View {
    let group: PhotoGroup?
    let selectedURL: URL?
    let emptyMessage: String
    @Binding var zoomScale: CGFloat?
    @State private var image: NSImage?
    @State private var loading = false
    @State private var gestureScale: CGFloat = 1

    var body: some View {
        ZStack {
            Color.black.opacity(0.35)
            if let image {
                if let zoomScale {
                    ScrollView([.horizontal, .vertical]) {
                        Image(nsImage: image)
                            .resizable()
                            .frame(width: image.size.width * zoomScale * gestureScale,
                                   height: image.size.height * zoomScale * gestureScale)
                            .gesture(magnificationGesture)
                    }
                } else {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .padding(12)
                        .gesture(magnificationGesture)
                }
            } else if loading {
                ProgressView("正在载入预览…")
            } else if group != nil {
                VStack(spacing: 12) {
                    Image(systemName: "photo.badge.exclamationmark").font(.system(size: 42))
                    Text("无法生成预览")
                    Text("仍可查看文件信息并进行分类").font(.caption).foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 14) {
                    Image(systemName: "photo.on.rectangle.angled").font(.system(size: 52))
                    Text(emptyMessage).foregroundStyle(.secondary)
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .onTapGesture { NSApp.keyWindow?.makeFirstResponder(nil) }
        .contextMenu { contextMenu }
        .task(id: "\(selectedURL?.path ?? group?.previewURL?.path ?? "none")|\(zoomScale == nil ? "fit" : "full")") {
            image = nil
            guard let group else { return }
            loading = true
            image = await PreviewLoader.shared.image(for: group, selectedURL: selectedURL, fullResolution: zoomScale != nil)
            loading = false
        }
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                if zoomScale == nil { zoomScale = 1 }
                gestureScale = value
            }
            .onEnded { value in
                zoomScale = limited((zoomScale ?? 1) * value)
                gestureScale = 1
            }
    }

    @ViewBuilder private var contextMenu: some View {
        if let group {
            if let image { Button("复制预览图") { copyImage(image) } }
            if let url = selectedURL ?? group.previewURL {
                Button("复制文件") { copyFile(url) }
                Button("复制文件路径") { copyText(url.path) }
            }
            Divider()
            ForEach(group.urls, id: \.path) { url in
                Button("在 Finder 中显示 \(displayName(url, group: group))") {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
            }
        }
    }

    private func limited(_ value: CGFloat) -> CGFloat { min(max(value, 0.1), 8) }

    private func copyImage(_ image: NSImage) {
        guard let data = image.tiffRepresentation else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setData(data, forType: .tiff)
    }

    private func copyFile(_ url: URL) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.writeObjects([url as NSURL])
    }

    private func copyText(_ value: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(value, forType: .string)
    }

    private func displayName(_ url: URL, group: PhotoGroup) -> String {
        if url == group.jpgURL { return "JPG" }
        if url == group.rawURL { return "ARW" }
        return url.lastPathComponent
    }
}
