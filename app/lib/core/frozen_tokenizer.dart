import 'dart:convert';
import 'package:unorm_dart/unorm_dart.dart' as unorm;

class TokenizedText {
  final List<int> inputIds, attentionMask;
  const TokenizedText(this.inputIds, this.attentionMask);
}

/// Genuine uncased BERT normalization + punctuation splitting + WordPiece.
/// No encoded-text lookup or result cache is used for full-pipeline timing.
class FrozenBertTokenizer {
  final Map<String, int> vocab;
  final int maxWordCharacters;
  final List<String> _special;
  late final RegExp _specialPattern;
  static final _control = RegExp(r'[\p{Cc}\p{Cf}]', unicode: true);
  static final _space = RegExp(r'\p{Zs}', unicode: true);
  static final _mark = RegExp(r'\p{Mn}', unicode: true);
  static final _punctuation = RegExp(r'\p{P}', unicode: true);

  FrozenBertTokenizer._(this.vocab, this.maxWordCharacters, this._special) {
    _special.sort((a, b) => b.length.compareTo(a.length));
    _specialPattern = RegExp(_special.map(RegExp.escape).join('|'));
    for (final t in ['[UNK]', '[CLS]', '[SEP]', '[PAD]']) {
      if (!vocab.containsKey(t)) {
        throw FormatException('Missing frozen token $t');
      }
    }
  }
  factory FrozenBertTokenizer.fromJson(String text) {
    final j = jsonDecode(text) as Map<String, dynamic>;
    final model = j['model'] as Map<String, dynamic>;
    final norm = j['normalizer'] as Map<String, dynamic>;
    if ((model['type'] != null && model['type'] != 'WordPiece') ||
        model['continuing_subword_prefix'] != '##' ||
        norm['type'] != 'BertNormalizer' ||
        norm['lowercase'] != true ||
        norm['clean_text'] != true ||
        norm['handle_chinese_chars'] != true ||
        norm['strip_accents'] == false ||
        (j['pre_tokenizer'] as Map)['type'] != 'BertPreTokenizer') {
      throw const FormatException(
        'Tokenizer differs from the frozen uncased BERT specification',
      );
    }
    final vocab = (model['vocab'] as Map).map(
      (k, v) => MapEntry(k as String, (v as num).toInt()),
    );
    final added = (j['added_tokens'] as List? ?? []).cast<Map>();
    for (final t in added) {
      if (t['single_word'] == true ||
          t['lstrip'] == true ||
          t['rstrip'] == true ||
          t['normalized'] == true) {
        throw const FormatException(
          'Unsupported added-token normalization; no approximation allowed',
        );
      }
    }
    return FrozenBertTokenizer._(
      vocab,
      (model['max_input_chars_per_word'] as num).toInt(),
      added.map((t) => t['content'] as String).toList(),
    );
  }
  bool _isChinese(int c) =>
      (c >= 0x4E00 && c <= 0x9FFF) ||
      (c >= 0x3400 && c <= 0x4DBF) ||
      (c >= 0x20000 && c <= 0x2A6DF) ||
      (c >= 0x2A700 && c <= 0x2B73F) ||
      (c >= 0x2B740 && c <= 0x2B81F) ||
      (c >= 0x2B820 && c <= 0x2CEAF) ||
      (c >= 0xF900 && c <= 0xFAFF) ||
      (c >= 0x2F800 && c <= 0x2FA1F);
  bool _isPunctuation(int c, String ch) =>
      (c >= 33 && c <= 47) ||
      (c >= 58 && c <= 64) ||
      (c >= 91 && c <= 96) ||
      (c >= 123 && c <= 126) ||
      _punctuation.hasMatch(ch);
  List<String> _basic(String text) {
    final cleaned = StringBuffer();
    for (final cp in text.runes) {
      final ch = String.fromCharCode(cp);
      if (cp == 0 || cp == 0xFFFD) continue;
      if (cp == 9 || cp == 10 || cp == 13 || cp == 32 || _space.hasMatch(ch)) {
        cleaned.write(' ');
      } else if (_control.hasMatch(ch)) {
        continue;
      } else if (_isChinese(cp)) {
        cleaned.write(' $ch ');
      } else {
        cleaned.write(ch);
      }
    }
    final normalized = unorm
        .nfd(cleaned.toString().toLowerCase())
        .replaceAll(_mark, '');
    final words = <String>[];
    final current = StringBuffer();
    void flush() {
      if (current.isNotEmpty) {
        words.add(current.toString());
        current.clear();
      }
    }

    for (final cp in normalized.runes) {
      final ch = String.fromCharCode(cp);
      if (ch == ' ') {
        flush();
      } else if (_isPunctuation(cp, ch)) {
        flush();
        words.add(ch);
      } else {
        current.write(ch);
      }
    }
    flush();
    return words;
  }

  List<int> _wordPieces(String word) {
    final chars = word.runes.toList();
    if (chars.length > maxWordCharacters) return [vocab['[UNK]']!];
    final result = <int>[];
    var start = 0;
    while (start < chars.length) {
      int? found;
      var end = chars.length;
      while (end > start) {
        var piece = String.fromCharCodes(chars.sublist(start, end));
        if (start > 0) piece = '##$piece';
        found = vocab[piece];
        if (found != null) break;
        end--;
      }
      if (found == null) return [vocab['[UNK]']!];
      result.add(found);
      start = end;
    }
    return result;
  }

  TokenizedText encode(String text, int maxLength) {
    if (maxLength < 2) throw ArgumentError.value(maxLength, 'maxLength');
    final ids = <int>[];
    void ordinary(String segment) {
      for (final word in _basic(segment)) {
        ids.addAll(_wordPieces(word));
      }
    }

    var cursor = 0;
    if (_special.isNotEmpty) {
      for (final m in _specialPattern.allMatches(text)) {
        ordinary(text.substring(cursor, m.start));
        ids.add(vocab[m.group(0)]!);
        cursor = m.end;
      }
    }
    ordinary(text.substring(cursor));
    final clipped = ids.take(maxLength - 2).toList();
    final input = <int>[vocab['[CLS]']!, ...clipped, vocab['[SEP]']!];
    final mask = List<int>.filled(input.length, 1, growable: true);
    while (input.length < maxLength) {
      input.add(vocab['[PAD]']!);
      mask.add(0);
    }
    return TokenizedText(input, mask);
  }
}
