//! The router between Laya's checkpoints: is this state English Latin text,
//! or something the English checkpoint cannot read? A port of the model's
//! own `laya/lang.py` (Apache-2.0), because which checkpoint answers is part
//! of the model's published behaviour - the English checkpoint collapses to
//! near chance on non-Latin scripts (Hindi 0.100 on 20-option MASSIVE, random
//! 0.050) while reporting high confidence, so this decision is a quality gate,
//! not a heuristic of ours to improve on.
//!
//! Script is the primary signal and exact; the Latin-script language guess is
//! function words, non-English letters and a margin, best effort by design.
//! Every rule, list and threshold below is the reference's, and so is the
//! character model: Python's `str.isalpha` (general category L*), `re`'s
//! `\w` (`isalnum` or `_` - which EXCLUDES combining marks, where the Rust
//! regex `\w` includes them), `str.isspace` (which includes 0x1C-0x1F) and
//! `unicodedata.combining`. Rust's own `is_alphabetic` counts Devanagari vowel
//! signs as letters and Python does not; the script fractions in the reasons
//! would drift, and with them the rare borderline route. The regexes are
//! hand-rolled scanners for the same reason, each checked against its pattern
//! below. Gated against the reference's own routing of its battery.

use std::collections::HashSet;
use std::sync::LazyLock;

use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

use super::pyjson::PyVal;

// ── Python's character model ────────────────────────────────────────────────

fn gc(c: char) -> GeneralCategory {
    c.general_category()
}

/// `str.isalpha`: Lu Ll Lt Lm Lo.
pub(crate) fn py_isalpha(c: char) -> bool {
    use GeneralCategory::*;
    matches!(
        gc(c),
        UppercaseLetter | LowercaseLetter | TitlecaseLetter | ModifierLetter | OtherLetter
    )
}

/// `re`'s `\d`: decimal digits (Nd).
fn py_isdecimal(c: char) -> bool {
    gc(c) == GeneralCategory::DecimalNumber
}

/// `str.isalnum`: a letter, or any number (Nd Nl No).
fn py_isalnum(c: char) -> bool {
    use GeneralCategory::*;
    py_isalpha(c) || matches!(gc(c), DecimalNumber | LetterNumber | OtherNumber)
}

/// `re`'s `\w` on a str pattern.
fn py_word(c: char) -> bool {
    c == '_' || py_isalnum(c)
}

/// `[^\W\d_]`: a word character that is neither a decimal digit nor `_`.
fn py_letter(c: char) -> bool {
    py_word(c) && c != '_' && !py_isdecimal(c)
}

/// `str.isspace` - White_Space plus the four information separators.
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn py_cased_upper(c: char) -> bool {
    c.is_uppercase()
}

fn py_cased_lower(c: char) -> bool {
    c.is_lowercase()
}

fn py_cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase() || gc(c) == GeneralCategory::TitlecaseLetter
}

/// `str.isupper`: at least one cased character, and every cased one upper.
fn py_isupper(s: &str) -> bool {
    let mut any = false;
    for c in s.chars() {
        if py_cased(c) {
            if !py_cased_upper(c) {
                return false;
            }
            any = true;
        }
    }
    any
}

/// `unicodedata.combining(ch) != 0`.
fn py_combining(c: char) -> bool {
    unicode_normalization::char::canonical_combining_class(c) != 0
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// `str.split()` with no argument.
fn py_split_ws(s: &str) -> impl Iterator<Item = &str> {
    s.split(py_isspace).filter(|t| !t.is_empty())
}

/// `round(x)`: half to even.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `round(x, 4)`.
fn py_round4(x: f64) -> f64 {
    (x * 10000.0).round_ties_even() / 10000.0
}

/// `repr(s)` for the reason strings.
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// ── the tables ──────────────────────────────────────────────────────────────

/// Unicode blocks the English checkpoint (ModernBERT-large, 50k English BPE)
/// cannot read, in the reference's order.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function words per language, the reference's lists verbatim (their notes
/// on what is left out and why live there).
const STOP: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
            "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton", "ma",
            "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien", "tout",
            "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit", "merci",
            "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi", "alors",
            "donc",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
            "ich", "wir", "mir", "mich", "dir", "dich", "uns", "mein", "meine", "meinen", "meinem",
            "meiner", "diese", "dieser", "diesen", "dieses", "einen", "einem", "einer", "wie",
            "wo", "wann", "welche", "im", "zum", "zur", "aus", "bei", "nach", "noch", "bitte",
            "heute", "jetzt", "kann", "kannst", "habe", "gibt", "wird", "in", "was",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus", "la", "un",
            "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos", "ni", "dos", "tres", "fue",
            "fueron", "ser", "tiene", "tienen", "tengo", "puede", "pueden", "quiero", "necesito",
            "hemos", "han", "sobre", "entre", "cuando", "donde", "porque", "aunque", "también",
            "ya", "eso", "esto", "esa", "ese", "nada", "algo", "aquí", "hoy", "gracias",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela", "o", "e",
            "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou", "tem", "tenho",
            "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu", "sua", "isso",
            "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais", "já", "ainda",
            "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado", "olá", "você",
            "vocês", "voce", "voces", "vc", "vcs", "nao", "sao", "ja", "até", "tá", "pra",
            "gostaria", "obrigada", "também", "tambem", "estou", "estamos", "meus", "minhas",
            "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite", "depois", "antes",
            "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum", "nenhuma",
            "estava", "ficou", "fiz", "deu",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla", "la", "le", "un", "uno",
            "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi", "ci", "ne", "ho", "hai", "ha",
            "abbiamo", "avete", "hanno", "era", "stato", "stata", "devo", "deve", "devono",
            "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando", "dove", "perche", "molto",
            "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi", "ieri", "grazie", "ciao",
            "scusa", "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo",
            "dagli", "dei", "delle", "dello", "degli", "agli", "alle", "col",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
    (
        "ro",
        &[
            "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
            "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
            "acesta", "aceasta", "mi", "ți", "vă", "nu",
        ],
    ),
    (
        "bn",
        &[
            "ami",
            "amar",
            "amake",
            "amra",
            "amader",
            "apni",
            "apnar",
            "apnake",
            "apnara",
            "tumi",
            "tomar",
            "tomake",
            "tomra",
            "tader",
            "ota",
            "eita",
            "oita",
            "ekta",
            "ei",
            "oi",
            "ki",
            "keno",
            "kivabe",
            "kibhabe",
            "kothay",
            "kokhon",
            "kobe",
            "koto",
            "kintu",
            "jodi",
            "tahole",
            "ar",
            "theke",
            "jonno",
            "sathe",
            "shathe",
            "diye",
            "niye",
            "moddhe",
            "kore",
            "korte",
            "korchi",
            "korsi",
            "korbo",
            "korechi",
            "koreche",
            "korun",
            "koren",
            "korlam",
            "hobe",
            "hoyeche",
            "hoise",
            "hocche",
            "hoyni",
            "chai",
            "chaina",
            "lagbe",
            "parchi",
            "parbo",
            "parchina",
            "peyechi",
            "paini",
            "dite",
            "dilam",
            "diyechi",
            "nai",
            "khub",
            "onek",
            "ekhon",
            "akhon",
            "ekhono",
            "abar",
            "ekbar",
            "duibar",
            "ajke",
            "kalke",
            "taka",
            "bhalo",
            "valo",
            "kharap",
            "shomossa",
            "somossa",
            "dhonnobad",
            "bhai",
            "shob",
            "keu",
            "kichu",
            "bolte",
            "bolun",
            "parben",
            "asbe",
            "jabe",
            "pabo",
            "ferot",
            "dorkar",
            "hoye",
            "geche",
            "gese",
        ],
    ),
    (
        "az",
        &[
            "və", "ve", "bir", "bu", "üçün", "ucun", "ilə", "ile", "olan", "olub", "olmasa", "var",
            "yox", "yoxdur", "mən", "sən", "biz", "siz", "onlar", "daha", "çox", "cox", "hər",
            "nə", "kimi", "görə", "sonra", "əgər", "eger", "deyil", "lakin", "amma", "ancaq",
            "artıq", "artiq", "də", "isə", "həm", "yalnız", "yalniz",
        ],
    ),
];

/// Letters ordinary English does not use (matched after lower-casing).
const NON_EN_DIACRITICS: &str = concat!(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ",
    "ăâîșțşţ",
    "ąćęłńśźż",
    "čďěňřšťůž",
    "őű",
    "ğı",
    "āēģīķļņūž",
    "đ",
    "ə",
);

pub const NON_EN_DIACRITIC_RATE: f64 = 0.02;
pub const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;
pub const NON_LATIN_FRACTION: f64 = 0.2;
pub const NON_LATIN_MIN_FRACTION: f64 = 0.1;
pub const NON_LATIN_MIN_LETTERS: i64 = 10;

struct Tables {
    stop: Vec<(&'static str, HashSet<&'static str>)>,
    shared: HashSet<&'static str>,
    en_only: HashSet<&'static str>,
    diacritics: HashSet<char>,
}

static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    let stop: Vec<(&str, HashSet<&str>)> = STOP
        .iter()
        .map(|(lg, ws)| (*lg, ws.iter().copied().collect()))
        .collect();
    let all: HashSet<&str> = stop.iter().flat_map(|(_, s)| s.iter().copied()).collect();
    let shared: HashSet<&str> = all
        .into_iter()
        .filter(|w| stop.iter().filter(|(_, s)| s.contains(w)).count() > 1)
        .collect();
    let en_only = stop[0].1.difference(&shared).copied().collect();
    Tables {
        stop,
        shared,
        en_only,
        diacritics: NON_EN_DIACRITICS.chars().collect(),
    }
});

fn stop_of(lang: &str) -> Option<&'static HashSet<&'static str>> {
    TABLES
        .stop
        .iter()
        .find(|(lg, _)| *lg == lang)
        .map(|(_, s)| s)
}

// ── the reference's regexes, as scanners ────────────────────────────────────

/// `_WORD.findall(s)`: maximal runs of `[^\W\d_]`.
fn words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if py_letter(c) {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// `_IDENTIFIER.sub(" ", s)`, `(?<![\w-])[\w-]*(?:[.@][\w-]+)+`: a token
/// whose dot or @ joins word characters is an identifier, not prose.
/// `[\w-]` and `[.@]` are disjoint, so a match is a maximal run followed by
/// every `[.@] + run` group that follows it.
fn strip_identifiers(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let wd = |x: char| py_word(x) || x == '-';
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < c.len() {
        if i == 0 || !wd(c[i - 1]) {
            let mut k = i;
            while k < c.len() && wd(c[k]) {
                k += 1;
            }
            let mut groups = 0;
            while k + 1 < c.len() && (c[k] == '.' || c[k] == '@') && wd(c[k + 1]) {
                k += 1;
                while k < c.len() && wd(c[k]) {
                    k += 1;
                }
                groups += 1;
            }
            if groups > 0 {
                out.push(' ');
                i = k;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// `_CODE_LINE.search(s)`: `[=;{}\[\]]|\w\(`.
fn is_code_line(s: &str) -> bool {
    let mut prev_word = false;
    for c in s.chars() {
        if matches!(c, '=' | ';' | '{' | '}' | '[' | ']') || (c == '(' && prev_word) {
            return true;
        }
        prev_word = py_word(c);
    }
    false
}

/// `_JOINED.search(tok)`: `[^\W_][._/\\][^\W_]`.
fn is_joined(tok: &str) -> bool {
    let c: Vec<char> = tok.chars().collect();
    c.windows(3)
        .any(|w| py_isalnum(w[0]) && matches!(w[1], '.' | '_' | '/' | '\\') && py_isalnum(w[2]))
}

/// `_LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), s)`
/// over runs of two or more `[^\W\d_]`.
fn drop_acronyms(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut run = String::new();
    let flush = |run: &mut String, out: &mut String| {
        if run.chars().count() >= 2 && py_isupper(run) {
            out.push(' ');
        } else {
            out.push_str(run);
        }
        run.clear();
    };
    for c in s.chars() {
        if py_letter(c) {
            run.push(c);
        } else {
            flush(&mut run, &mut out);
            out.push(c);
        }
    }
    flush(&mut run, &mut out);
    out
}

// ── the analysis ────────────────────────────────────────────────────────────

/// The string leaves of a state (keys ignored: they are usually English
/// field names), six levels deep.
pub fn iter_text(state: &PyVal) -> Vec<&str> {
    fn walk<'a>(v: &'a PyVal, depth: usize, out: &mut Vec<&'a str>) {
        if depth > 6 {
            return;
        }
        match v {
            PyVal::Str(s) => out.push(s),
            PyVal::Dict(kv) => kv.iter().for_each(|(_, x)| walk(x, depth + 1, out)),
            PyVal::List(items) => items.iter().for_each(|x| walk(x, depth + 1, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(state, 0, &mut out);
    out
}

fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The state flattened for detection, at most `max_chars` characters.
fn state_text(state: &PyVal, max_chars: usize) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let mut budget = max_chars as i64;
    for leaf in iter_text(state) {
        if budget <= 0 {
            break;
        }
        let n = leaf.chars().count() as i64;
        if n > budget {
            parts.push(take_chars(leaf, budget as usize));
            break;
        }
        parts.push(leaf);
        budget -= n + 1;
    }
    take_chars(&parts.join(" "), max_chars).to_owned()
}

/// Letters by script, in the reference's insertion order (named scripts as
/// met, "other", Latin last - so a named script wins a tie against Latin).
fn script_counts(text: &str) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let bump = |name: &'static str, counts: &mut Vec<(&'static str, usize)>| match counts
        .iter_mut()
        .find(|(n, _)| *n == name)
    {
        Some(e) => e.1 += 1,
        None => counts.push((name, 1)),
    };
    let mut latin = 0usize;
    for ch in text.chars() {
        if !py_isalpha(ch) {
            continue;
        }
        let cp = ch as u32;
        if cp < 0x02B0
            || (0x1E00..=0x1EFF).contains(&cp)
            || (0xFF21..=0xFF3A).contains(&cp)
            || (0xFF41..=0xFF5A).contains(&cp)
        {
            latin += 1;
            continue;
        }
        let name = SCRIPT_RANGES
            .iter()
            .find(|(_, r)| r.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)))
            .map_or("other", |(n, _)| n);
        bump(name, &mut counts);
    }
    counts.push(("latin", latin));
    counts
}

fn script_from_counts(counts: &[(&'static str, usize)]) -> &'static str {
    if counts.iter().all(|(_, n)| *n == 0) {
        return "unknown";
    }
    // `max` keeps the first of equal values
    let mut best = counts[0];
    for &c in &counts[1..] {
        if c.1 > best.1 {
            best = c;
        }
    }
    best.0
}

fn profile_from_counts(counts: &[(&'static str, usize)]) -> Vec<(&'static str, f64)> {
    let total: usize = counts.iter().map(|(_, n)| n).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    if let Some((_, n)) = counts.iter().find(|(k, n)| *k == "latin" && *n > 0) {
        out.push(("latin", *n as f64 / total as f64));
    }
    for (k, n) in counts {
        if *k != "latin" && *n > 0 {
            out.push((*k, *n as f64 / total as f64));
        }
    }
    out
}

/// The named non-Latin script of one letter; None for Latin and unclaimed.
fn script_of(ch: char) -> Option<&'static str> {
    let cp = ch as u32;
    if cp < 0x0250
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
    {
        return None;
    }
    SCRIPT_RANGES
        .iter()
        .find(|(_, r)| r.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)))
        .map(|(n, _)| *n)
}

/// Non-Latin runs that read as words, not as a symbol, a capitalised name or
/// a pronunciation inside English prose.
fn non_latin_words(text: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let (mut cur, mut script) = (String::new(), None::<&str>);
    for ch in text.chars() {
        if py_combining(ch) {
            continue;
        }
        let s = script_of(ch);
        if s.is_some() && s == script {
            cur.push(ch);
            continue;
        }
        if !cur.is_empty() {
            runs.push(std::mem::take(&mut cur));
        }
        if s.is_some() {
            cur.push(ch);
            script = s;
        } else {
            script = None;
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs.into_iter()
        .filter(|w| w.chars().count() >= 2 && !w.chars().next().is_some_and(py_cased_upper))
        .collect()
}

fn english_rescued_by_words(words: &[String], diac_rate: f64) -> bool {
    if diac_rate >= ENGLISH_RESCUE_DIACRITIC_RATE {
        return false;
    }
    let t = &*TABLES;
    let set: HashSet<&str> = words.iter().map(String::as_str).collect();
    if set.iter().filter(|w| t.en_only.contains(**w)).count() < 2 {
        return false;
    }
    set.iter()
        .filter(|w| w.chars().any(|c| t.diacritics.contains(&c)))
        .count()
        <= 1
}

struct LatinProfile {
    language: Option<&'static str>,
    diacritic_rate: f64,
    looks_non_english: bool,
}

fn latin_profile(text: &str) -> LatinProfile {
    let t = &*TABLES;
    let ws = words(
        &strip_identifiers(text)
            .replace('\u{130}', "i")
            .to_lowercase(),
    );
    let lowered = text.to_lowercase();
    let n = lowered.chars().count();
    let diac = lowered.chars().filter(|c| t.diacritics.contains(c)).count();
    let diac_rate = diac as f64 / n.max(1) as f64;
    let non_english = diac_rate >= NON_EN_DIACRITIC_RATE;
    if ws.len() < 4 {
        return LatinProfile {
            language: None,
            diacritic_rate: diac_rate,
            looks_non_english: non_english,
        };
    }
    let scores: Vec<(&'static str, usize)> = t
        .stop
        .iter()
        .map(|(lg, s)| (*lg, ws.iter().filter(|w| s.contains(w.as_str())).count()))
        .collect();
    let en = scores[0].1;
    let set: HashSet<&str> = ws.iter().map(String::as_str).collect();
    let mut best: Option<(&'static str, usize)> = None;
    for &(lg, score) in &scores[1..] {
        let s = stop_of(lg).expect("listed");
        let evidenced = set.iter().any(|w| s.contains(w) && !t.shared.contains(w));
        if evidenced && best.is_none_or(|(_, b)| score > b) {
            best = Some((lg, score));
        }
    }
    let lang = match best {
        Some((lg, b)) if b >= 2.max(en + 2) => Some(lg),
        Some((lg, b)) if non_english && b >= 2.max(en) => Some(lg),
        _ if en > 0 && (!non_english || english_rescued_by_words(&ws, diac_rate)) => Some("en"),
        _ => None,
    };
    LatinProfile {
        language: lang,
        diacritic_rate: diac_rate,
        looks_non_english: non_english,
    }
}

fn named_prose_language(segment: &str) -> Option<&'static str> {
    if py_strip(segment).is_empty() || is_code_line(segment) {
        return None;
    }
    let mut prose = py_split_ws(segment)
        .filter(|tok| !is_joined(tok))
        .collect::<Vec<_>>()
        .join(" ");
    if prose.chars().any(py_cased_lower) {
        prose = drop_acronyms(&prose);
    }
    let tokens = words(&prose);
    if tokens.len() < 4 {
        return None;
    }
    let lang = latin_profile(&prose).language?;
    if lang == "en" {
        return None;
    }
    let s = stop_of(lang)?;
    let distinct: HashSet<String> = tokens.iter().map(|w| w.to_lowercase()).collect();
    (distinct.iter().filter(|w| s.contains(w.as_str())).count() >= 2).then_some(lang)
}

fn non_english_segment(state: &PyVal, max_chars: usize) -> Option<(&'static str, String)> {
    let mut seen = 0usize;
    for leaf in iter_text(state) {
        for seg in leaf.split('\n') {
            if seen >= max_chars {
                return None;
            }
            let seg = take_chars(seg, max_chars - seen);
            seen += seg.chars().count();
            if let Some(lang) = named_prose_language(seg) {
                return Some((lang, py_strip(seg).to_owned()));
            }
        }
    }
    None
}

/// What the router saw - the reference's `analyse` result.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub script: &'static str,
    pub script_profile: Vec<(&'static str, f64)>,
    pub language: Option<&'static str>,
    pub is_english: bool,
    pub language_undecided: bool,
    pub diacritic_rate: f64,
    pub non_latin_fraction: f64,
    pub mixed_segment: Option<String>,
}

impl Detection {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "script": self.script,
            "script_profile": self.script_profile.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>(),
            "language": self.language,
            "is_english": self.is_english,
            "language_undecided": self.language_undecided,
            "diacritic_rate": self.diacritic_rate,
            "non_latin_fraction": self.non_latin_fraction,
            "mixed_segment": self.mixed_segment,
        })
    }
}

fn analyse_text(text: &str) -> Detection {
    let counts = script_counts(text);
    let prof = profile_from_counts(&counts);
    let mut script = script_from_counts(&counts);
    let latin_share = prof
        .iter()
        .find(|(k, _)| *k == "latin")
        .map_or(0.0, |(_, v)| *v);
    let non_latin = if prof.is_empty() {
        0.0
    } else {
        py_round4(1.0 - latin_share)
    };
    let alpha = text.chars().filter(|c| py_isalpha(*c)).count();
    let n_non_latin = py_round(non_latin * alpha as f64);
    if script == "latin"
        && !non_latin_words(text).is_empty()
        && (non_latin >= NON_LATIN_FRACTION
            || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS))
    {
        let mut best: Option<(&'static str, f64)> = None;
        for &(k, v) in &prof {
            if k != "latin" && best.is_none_or(|(_, b)| v > b) {
                best = Some((k, v));
            }
        }
        if let Some((k, _)) = best {
            script = k;
        }
    }
    if script == "unknown" {
        return Detection {
            script: "unknown",
            script_profile: prof,
            language: None,
            is_english: true,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: 0.0,
            mixed_segment: None,
        };
    }
    if script != "latin" {
        return Detection {
            script,
            script_profile: prof,
            language: None,
            is_english: false,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: non_latin,
            mixed_segment: None,
        };
    }
    let lp = latin_profile(text);
    let undecided = lp.language.is_none();
    Detection {
        script: "latin",
        script_profile: prof,
        language: lp.language,
        is_english: lp.language == Some("en") || (undecided && !lp.looks_non_english),
        language_undecided: undecided,
        diacritic_rate: py_round4(lp.diacritic_rate),
        non_latin_fraction: non_latin,
        mixed_segment: None,
    }
}

fn leaf_non_english(leaf: &str) -> Option<Detection> {
    let mut best: Option<(usize, Detection)> = None;
    for line in leaf.split('\n') {
        let sample = take_chars(line, 4000);
        if py_strip(sample).is_empty() || is_code_line(sample) {
            continue;
        }
        let det = analyse_text(sample);
        if det.is_english {
            continue;
        }
        let alpha = sample.chars().filter(|c| py_isalpha(*c)).count();
        let keep = if det.language.is_some_and(|l| l != "en") {
            named_prose_language(sample).is_some()
        } else if det.script != "latin" && det.script != "unknown" {
            !non_latin_words(sample).is_empty() && alpha as i64 >= NON_LATIN_MIN_LETTERS
        } else {
            det.language_undecided
                && det.diacritic_rate >= NON_EN_DIACRITIC_RATE
                && words(sample).len() >= 4
        };
        if keep && best.as_ref().is_none_or(|(n, _)| alpha > *n) {
            best = Some((alpha, det));
        }
    }
    best.map(|(_, d)| d)
}

/// The reference's `analyse(state)`.
pub fn analyse(state: &PyVal) -> Detection {
    let mut result = analyse_text(&state_text(state, 4000));
    if result.script == "latin" && result.is_english {
        let leaves = iter_text(state);
        if (leaves.len() > 1 || leaves.iter().any(|l| l.contains('\n')))
            && let Some((lang, mixed)) = non_english_segment(state, 4000)
        {
            result.language = Some(lang);
            result.is_english = false;
            result.language_undecided = false;
            result.mixed_segment = Some(mixed);
        }
    }
    if matches!(state, PyVal::Str(_) | PyVal::Null) || !result.is_english {
        return result;
    }
    let mut best: Option<(usize, Detection)> = None;
    for leaf in iter_text(state) {
        let Some(det) = leaf_non_english(leaf) else {
            continue;
        };
        let alpha = take_chars(leaf, 4000)
            .chars()
            .filter(|c| py_isalpha(*c))
            .count();
        if best.as_ref().is_none_or(|(n, _)| alpha > *n) {
            best = Some((alpha, det));
        }
    }
    if let Some((_, b)) = best {
        result.language = b.language;
        result.is_english = false;
        result.language_undecided = b.language_undecided;
    }
    result
}

// ── the route ───────────────────────────────────────────────────────────────

/// A language code's verdict: Some(true) English, Some(false) not, None when
/// the code names no language (`C`, `und`, empty) - which falls through to
/// detection instead of pinning a checkpoint on no evidence.
pub fn english_from_code(code: &str) -> Option<bool> {
    let code = code.trim().to_lowercase();
    if code.is_empty() {
        return None;
    }
    let code = code.split('.').next().unwrap_or("");
    let primary = code.replace('_', "-");
    let primary = primary.split('-').next().unwrap_or("");
    if primary.is_empty() || ["c", "posix", "und", "zxx", "mul"].contains(&primary) {
        return None;
    }
    Some(["en", "eng", "english"].contains(&primary))
}

/// Which checkpoint reads `state`, and the reference's reason for it. `true`
/// is the multilingual checkpoint.
pub fn route_text(state: &PyVal) -> (bool, String, Detection) {
    let det = analyse(state);
    let pct = |x: f64| format!("{:.0}", 100.0 * x);
    let (multi, reason) = if det.script == "unknown" {
        (
            false,
            "no letters detected in state; using default (english)".to_owned(),
        )
    } else if det.script != "latin" {
        (
            true,
            format!(
                "non-Latin script ({}, {}% of letters); the English checkpoint cannot read it",
                det.script,
                pct(det.non_latin_fraction)
            ),
        )
    } else if !det.is_english {
        let r = if let Some(seg) = &det.mixed_segment {
            format!(
                "Latin script, mostly English, but a line or field reads as {} ({}); the \
                 English checkpoint cannot read it",
                det.language.map_or("None".to_owned(), py_repr),
                py_repr(take_chars(seg, 60))
            )
        } else if let Some(l) = det.language {
            format!(
                "Latin script but language looks like {}, not English",
                py_repr(l)
            )
        } else {
            format!(
                "Latin script, language not identified but {}% non-English letters; not safe \
                 for the English checkpoint",
                pct(det.diacritic_rate)
            )
        };
        (true, r)
    } else if det.language_undecided {
        (
            false,
            "Latin script, language not identified and no non-English letters; using default \
             (english)"
                .to_owned(),
        )
    } else {
        (false, "English Latin text".to_owned())
    };
    (multi, reason, det)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(t: &str) -> PyVal {
        PyVal::Str(t.to_owned())
    }

    #[test]
    fn python_character_classes() {
        // a Devanagari vowel sign is a mark, not a letter, for Python
        assert!(!py_isalpha('\u{093E}') && '\u{093E}'.is_alphabetic());
        // a combining acute is not \w in Python's re
        assert!(!py_word('\u{0301}'));
        assert!(py_letter('é') && !py_letter('5') && !py_letter('_') && py_letter('²'));
        assert!(py_isspace('\u{1f}') && !'\u{1f}'.is_whitespace());
        assert!(py_isupper("COM") && !py_isupper("Com") && !py_isupper("123"));
    }

    #[test]
    fn identifiers_are_not_prose() {
        assert_eq!(strip_identifiers("mail user@example.com now"), "mail   now");
        assert_eq!(strip_identifiers("see github.com/x"), "see  /x");
        // a sentence-final period keeps its word
        assert_eq!(strip_identifiers("non è arrivato."), "non è arrivato.");
        assert_eq!(strip_identifiers("v1.2.3"), " ");
    }

    #[test]
    fn scanners_match_their_patterns() {
        assert!(is_code_line("x = 1") && is_code_line("round(el, 2)"));
        assert!(!is_code_line("Deu erro (500)"));
        assert!(is_joined("Nav/Com") && is_joined("C:\\DOS\\mode") && !is_joined("hello"));
        assert_eq!(drop_acronyms("the MON game at LA"), "the   game at  ");
    }

    #[test]
    fn routes_like_the_reference() {
        let (m, r, _) = route_text(&s("I was charged twice for the same order."));
        assert!(!m, "{r}");
        assert_eq!(r, "English Latin text");
        let (m, r, _) = route_text(&s(
            "Mein Konto wurde zweimal belastet, bitte erstatten Sie mir das Geld.",
        ));
        assert!(m);
        assert_eq!(r, "Latin script but language looks like 'de', not English");
        let (m, r, _) = route_text(&s("मेरे खाते से दो बार पैसे काटे गए हैं"));
        assert!(m);
        assert!(
            r.starts_with("non-Latin script (devanagari, 100% of letters)"),
            "{r}"
        );
        let (m, r, _) = route_text(&s("12345 67890"));
        assert!(!m);
        assert_eq!(r, "no letters detected in state; using default (english)");
    }

    #[test]
    fn language_codes() {
        assert_eq!(english_from_code("en_US.UTF-8"), Some(true));
        assert_eq!(english_from_code("de-DE"), Some(false));
        assert_eq!(english_from_code("C"), None);
        assert_eq!(english_from_code(""), None);
    }
}
