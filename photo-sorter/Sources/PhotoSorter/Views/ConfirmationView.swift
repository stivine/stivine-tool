import SwiftUI

struct ConfirmationView: View {
    let plan: OrganizationPlan
    let confirm: () -> Void
    let cancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("确认执行整理").font(.title2.bold())
            Text("以下文件将移入 macOS 系统废纸篓。此操作现在才会修改文件。")
                .foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 10) {
                row("废片", "\(plan.trashGroups) 组，将移入废纸篓 \(plan.trashFiles) 个文件")
                row("只留 JPG", "\(plan.keepJPGGroups) 组，将移入废纸篓 \(plan.keepJPGTrashFiles) 个 ARW")
                row("只留 RAW", "\(plan.keepRAWGroups) 组，将移入废纸篓 \(plan.keepRAWTrashFiles) 个 JPG")
                row("全部保留", "\(plan.keepAllGroups) 组")
                row("未处理", "\(plan.unprocessedGroups) 组，保持不动")
                row("预计释放", ByteCountFormatter.string(fromByteCount: plan.bytesToRelease, countStyle: .file))
            }
            if plan.unprocessedGroups > 0 {
                Label("尚有 \(plan.unprocessedGroups) 组未分类；它们不会被移动。", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
            }
            HStack {
                Spacer()
                Button("取消", action: cancel).keyboardShortcut(.cancelAction)
                Button("移入废纸篓", role: .destructive, action: confirm)
                    .keyboardShortcut(.defaultAction)
                    .disabled(plan.operations.isEmpty)
            }
        }
        .padding(24)
        .frame(width: 560)
    }

    @ViewBuilder private func row(_ title: String, _ detail: String) -> some View {
        GridRow {
            Text(title).fontWeight(.semibold)
            Text(detail).foregroundStyle(.secondary)
        }
    }
}

struct ResultView: View {
    let result: ExecutionResult
    let dismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(result.failures.isEmpty ? "整理完成" : "整理完成，但有文件失败").font(.title2.bold())
            Text("已移入废纸篓 \(result.succeeded.count) 个文件")
            if !result.failures.isEmpty {
                List(result.failures) { failure in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(failure.url.lastPathComponent).fontWeight(.medium)
                        Text(failure.reason).font(.caption).foregroundStyle(.secondary)
                    }
                }.frame(height: 220)
            }
            HStack { Spacer(); Button("完成", action: dismiss).keyboardShortcut(.defaultAction) }
        }
        .padding(24)
        .frame(width: 600)
    }
}
