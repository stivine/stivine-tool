import SwiftUI

struct PhotoMetadataView: View {
    let group: PhotoGroup
    @State private var metadata: PhotoMetadata?
    @State private var loading = true

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("拍摄参数", systemImage: "camera.aperture")
                .font(.headline)
            if loading {
                ProgressView().controlSize(.small)
            } else if let metadata {
                metadataRows(metadata)
            } else {
                Text("未读取到拍摄参数")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .task(id: group.id) {
            loading = true
            metadata = await MetadataLoader.shared.metadata(for: group)
            loading = false
        }
    }

    @ViewBuilder
    private func metadataRows(_ metadata: PhotoMetadata) -> some View {
        if let camera = metadata.camera { row("相机", camera) }
        if let lens = metadata.lens { row("镜头", lens) }
        if let focal = metadata.focalLength { row("焦距", focal) }
        if let aperture = metadata.aperture { row("光圈", aperture) }
        if let shutter = metadata.shutterSpeed { row("快门", shutter) }
        if let iso = metadata.iso { row("感光度", iso) }
        if let bias = metadata.exposureBias { row("曝光补偿", bias) }
        if let date = metadata.capturedAt { row("拍摄时间", date) }
        if let dimensions = metadata.dimensions { row("分辨率", dimensions) }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 58, alignment: .leading)
            Text(value)
                .textSelection(.enabled)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        .font(.caption)
    }
}
