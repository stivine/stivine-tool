import AppKit
import SwiftUI

struct KeyboardCapture: NSViewRepresentable {
    let handler: (NSEvent) -> Bool

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> KeyView {
        let view = KeyView()
        view.handler = handler
        context.coordinator.view = view
        context.coordinator.installArrowMonitor()
        DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
        return view
    }

    final class Coordinator {
        weak var view: KeyView?
        private var monitor: Any?

        func installArrowMonitor() {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard event.keyCode == 123 || event.keyCode == 124,
                      !(NSApp.keyWindow?.firstResponder is NSTextView),
                      let view = self?.view else { return event }
                return view.handler?(event) == true ? nil : event
            }
        }

        deinit {
            if let monitor { NSEvent.removeMonitor(monitor) }
        }
    }

    func updateNSView(_ nsView: KeyView, context: Context) {
        nsView.handler = handler
        DispatchQueue.main.async {
            if nsView.window?.firstResponder is NSTextView == false { nsView.window?.makeFirstResponder(nsView) }
        }
    }

    final class KeyView: NSView {
        var handler: ((NSEvent) -> Bool)?
        override var acceptsFirstResponder: Bool { true }
        override func keyDown(with event: NSEvent) {
            if handler?(event) != true { super.keyDown(with: event) }
        }
    }
}
