import AppKit
import SwiftUI

struct PhotoNoteView: View {
    let group: PhotoGroup
    @ObservedObject var session: PhotoSession

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("备注", systemImage: "note.text").font(.headline)
                Spacer()
                if !session.note(for: group).isEmpty {
                    Button("清除") { session.setNote("", for: group) }
                        .buttonStyle(.link)
                        .font(.caption)
                }
            }
            ZStack(alignment: .topLeading) {
                if session.note(for: group).isEmpty {
                    Text("记录挑选理由、后期思路或交付说明…")
                        .foregroundStyle(.tertiary)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
                TextEditor(text: Binding(
                    get: { session.note(for: group) },
                    set: { session.setNote($0, for: group) }
                ))
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(3)
                .onKeyPress(.escape) {
                    NSApp.keyWindow?.makeFirstResponder(nil)
                    return .handled
                }
            }
            .frame(minHeight: 72, maxHeight: 120)
            .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.secondary.opacity(0.25)))
        }
    }
}
