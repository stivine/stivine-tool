import SwiftUI

struct ClassificationButton: View {
    let classification: Classification
    let selected: Bool
    let enabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Text(classification.key)
                    .font(.system(.body, design: .rounded, weight: .bold))
                    .frame(width: 24, height: 24)
                    .background(.white.opacity(0.13), in: RoundedRectangle(cornerRadius: 5))
                Text(classification.title).fontWeight(.semibold)
                Spacer()
                if selected { Image(systemName: "checkmark.circle.fill") }
            }
            .padding(.horizontal, 12)
            .frame(height: 46)
            .background(selected ? classification.color.opacity(0.42) : classification.color.opacity(0.16))
            .overlay(RoundedRectangle(cornerRadius: 9).stroke(classification.color.opacity(selected ? 0.9 : 0.35)))
            .clipShape(RoundedRectangle(cornerRadius: 9))
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.38)
    }
}

extension Classification {
    var color: Color {
        switch self {
        case .trash: return .red
        case .keepAll: return .green
        case .keepJPG: return .blue
        case .keepRAW: return .orange
        }
    }
}
