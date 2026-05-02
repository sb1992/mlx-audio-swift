import Foundation
import MLX

// MARK: - Text Normalizer

enum TadaTextNormalizer {
    private static let substitutions: [Character: String] = [
        "\u{201c}": "\"", "\u{201d}": "\"", "\u{201e}": "\"", "\u{201f}": "\"",
        "\u{2018}": "'", "\u{2019}": "'", "\u{201a}": "'", "\u{201b}": "'",
        "\u{2013}": "-", "\u{2014}": "-", "\u{2015}": "-", "\u{2010}": "-", "\u{2011}": "-",
        "\u{2026}": "...",
        "\u{2039}": "<", "\u{203a}": ">",
        "\u{00ab}": "<<", "\u{00bb}": ">>",
    ]

    static func normalize(_ text: String) -> String {
        var result = text
        for (char, replacement) in substitutions {
            result = result.replacingOccurrences(of: String(char), with: replacement)
        }
        result = result
            .replacingOccurrences(of: "; ", with: ". ")
            .replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: ":", with: ",")
            .replacingOccurrences(of: "(", with: "")
            .replacingOccurrences(of: ")", with: "")
            .replacingOccurrences(of: "--", with: "-")
            .replacingOccurrences(of: "-", with: ", ")
            .replacingOccurrences(of: ",,", with: ",")
            .replacingOccurrences(of: " '", with: " ")
            .replacingOccurrences(of: "' ", with: " ")
            .replacingOccurrences(of: "  ", with: " ")

        result = result.replacingOccurrences(
            of: #"\s+([.,?!])"#, with: "$1",
            options: .regularExpression
        )

        result = result.lowercased()
        let sentencePattern = try! NSRegularExpression(pattern: #"([.!?]\s*)(\w)"#)
        let mutableResult = NSMutableString(string: result)
        sentencePattern.enumerateMatches(
            in: result, range: NSRange(location: 0, length: result.utf16.count)
        ) { match, _, _ in
            guard let match, let wordRange = Range(match.range(at: 2), in: result) else { return }
            let upper = String(result[wordRange]).uppercased()
            mutableResult.replaceCharacters(in: match.range(at: 2), with: upper)
        }
        result = mutableResult as String

        if let first = result.first {
            result = first.uppercased() + result.dropFirst()
        }
        return result
    }
}

// MARK: - Gray Code Duration Decoding

enum TadaGrayCode {
    static func grayCodeToInt(_ gray: MLXArray) -> MLXArray {
        var binary = gray
        var shift = 1
        while shift < 32 {
            binary = binary ^ (binary >> MLXArray(shift))
            shift <<= 1
        }
        return binary
    }

    static func decodeGrayCodeToTime(_ grayBits: MLXArray, numBits: Int) -> MLXArray {
        let bitsBinary = round((grayBits + 1.0) / 2.0).asType(.int32)
        var grayInt = MLXArray.zeros(like: bitsBinary[.ellipsis, 0])
        for i in 0..<numBits {
            let bitSlice = bitsBinary[.ellipsis, numBits - 1 - i]
            grayInt = grayInt + (bitSlice << MLXArray(i))
        }
        return grayCodeToInt(grayInt)
    }
}

// MARK: - Weight Key Sanitization

enum TadaWeightSanitizer {
    static func snakeToCamel(_ snake: String) -> String {
        let parts = snake.split(separator: "_", omittingEmptySubsequences: false)
        guard let first = parts.first else { return snake }
        return String(first) + parts.dropFirst().map { $0.isEmpty ? "_" : $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
    }

    static func sanitizeKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            let components = key.split(separator: ".")
            let camelComponents = components.map { component -> String in
                let s = String(component)
                if Int(s) != nil { return s }
                if s == "weight" || s == "bias" || s == "alpha" || s == "scales" { return s }
                return snakeToCamel(s)
            }
            result[camelComponents.joined(separator: ".")] = value
        }
        return result
    }
}
