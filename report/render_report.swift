import AppKit
import Foundation
import PDFKit

guard CommandLine.arguments.count == 3 else {
    fatalError("Usage: swift render_report.swift input.html output.pdf")
}
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let document = try NSAttributedString(
    url: input,
    options: [.documentType: NSAttributedString.DocumentType.html,
              .characterEncoding: String.Encoding.utf8.rawValue],
    documentAttributes: nil)

let app = NSApplication.shared
let info = NSPrintInfo.shared.copy() as! NSPrintInfo
info.paperSize = NSSize(width: 595.28, height: 841.89)
info.leftMargin = 40
info.rightMargin = 40
info.topMargin = 40
info.bottomMargin = 40
info.horizontalPagination = .fit
info.verticalPagination = .automatic
info.jobDisposition = .save
info.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = output

let width = info.paperSize.width - info.leftMargin - info.rightMargin
let view = NSTextView(frame: NSRect(x: 0, y: 0, width: width, height: 100))
view.isVerticallyResizable = true
view.isHorizontallyResizable = false
view.textContainer?.containerSize = NSSize(width: width, height: CGFloat.greatestFiniteMagnitude)
view.textContainer?.heightTracksTextView = false
view.textStorage?.setAttributedString(document)
view.layoutManager?.ensureLayout(for: view.textContainer!)
view.sizeToFit()
let operation = NSPrintOperation(view: view, printInfo: info)
operation.showsPrintPanel = false
operation.showsProgressPanel = false
guard operation.run(), let pdf = PDFDocument(url: output) else {
    fatalError("PDF generation failed")
}
print("PDF pages: \(pdf.pageCount)")
