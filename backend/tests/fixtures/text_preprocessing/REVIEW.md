# Text preprocessing incremental review

## Scope

This is one improvement stage, implemented as individually tested increments.
Only `backend/app/text_processing/` and dedicated preprocessing tests/fixtures
were edited. Public `normalize`, `sanitize`, and `process` call signatures remain
compatible; `process` still returns a string. Both switches remain independent.
No frontend, API schema/controller, inference, benchmark, training, deployment,
or general documentation changes were made by this work.

Review type: assistant review of actual text output and deterministic assertions.
This is not a human listening evaluation or a claim that any model sounds better.

## Test-first evidence

- [cases.json](cases.json): 46 input/expected-output cases authored before changes.
- [naturalsoft.txt](naturalsoft.txt): the supplied long-form interview sample.
- [naturalsoft.expected.txt](naturalsoft.expected.txt): intended speech script.
- [baseline-review.json](baseline-review.json): recorded pre-change behavior;
  17/46 cases met the intended contract, while all 29 pre-existing tests passed.
- [final-review.json](final-review.json): actual final outputs; 46/46 pass,
  including repeat-processing equality for every successful case.
- [naturalsoft.baseline.txt](naturalsoft.baseline.txt) and
  [naturalsoft.actual.txt](naturalsoft.actual.txt): directly comparable outputs.

The baseline is historical and is not regenerated from the new implementation.
The email-name expected output was refined during review: address components are
rendered in lowercase, not rewritten using standalone brand aliases. The original
request text is unaffected. This avoids a later pass treating spoken address
components as brand names. This is a speech representation, not a reversible
case-sensitive email serialization.

## Increment ledger

| Increment | Change | Gate passed before proceeding |
| --- | --- | --- |
| 0 | Golden input/output corpus and runnable report | Old suite: 29 passed; baseline failures recorded |
| 1 | Prevent cross-word Markdown underscore consumption | 12/12 cumulative cases; 87 focused tests |
| 2 | Normalize Unicode before semantic rules; preserve token boundaries | 18/18 cases; 93 tests |
| 3 | Convert malformed host/port failures to existing `TextProcessingError` | 24/24 cases; 99 tests |
| 4 | Explicit name aliases, conservative unknown names, address/code contexts | 30/30 cases; 105 tests |
| 5 | Markdown blocks, list boundaries, fenced-code content and paragraphs | 38/38 cases; 113 tests |
| 6 | Explicit relative/Windows/absolute paths and code filenames | 46/46 cases; 121 tests |
| Review | Independent holdouts, toggle matrix, limits and cascading-rule repair | 158 dedicated tests passed |
| Consumer check | Existing API, benchmark and metrics tests, unmodified | 35 passed |

Focused counts include passthrough tests for all corpus inputs. Not-yet-implemented
future output cases were explicitly deselected until their increment, not marked
as successful. Ruff lint/format and `git diff --check` also passed.

## Reviewer findings and decisions

### Identifiers: corrected

Before: `model_registry ... user_name` became `modelregistry ... username`.
Now: each identifier is converted independently without deleting neighboring
content. Actual Markdown emphasis still unwraps correctly, including nested
emphasis around a URL or a currency amount. Code and address spans are recognized
before ordinary-prose rules.

### Unicode and boundaries: corrected for tested cases

Full-width `＄２５` becomes `25 dollars` before sanitization can remove the symbol.
Null characters and zero-width word-break hints do not join adjacent English
tokens. Horizontal whitespace is collapsed; paragraph and line boundaries remain.
The old sanitizer test that required flattening a newline was deliberately revised
to the new structure-preserving contract. It was not merely deleted.

### Malformed addresses: controlled

Invalid ports, missing hosts and malformed brackets raise `TextNormalizationError`
internally, which the processing service translates into the already-supported
`TextProcessingError`. No API module changes were necessary. URL parsing performs
no network access. Normalization-disabled input does not invoke URL parsing.

### Names: intentional rather than accidental

Examples: `SpeechT5` -> `Speech T five`, `FastAPI` -> `Fast A P I`,
`NaturalSoft` -> `Natural Soft`, `OpenVoice Lab` -> `Open Voice Lab`.
The lexicon uses longest exact boundary matches. `McDonald`, `OpenAI`, `MyBrand`,
and larger words such as `NaturalSoftly` remain unchanged. These are candidate
spoken forms; audio pronunciation has not been approved by listening.

### Document structure: corrected for the supported subset

The long sample retains its content order, paragraph boundaries, numbered-list
item content, technical names, and all three explicit code paths. Headings/list
items get sentence boundaries; formatting markers are not narrated. Recognized
flattened `: * item, * item` lists become separate lines. A multiplication
expression such as `A * B * C`, a negative number such as `-5`, and a grammatical
hyphen remain intact.

### Paths: explicit forms corrected; ambiguity retained deliberately

`backend/app/synthesis/service.py` is rendered as
`backend slash app slash synthesis slash service dot P Y`.
Windows drive paths, parent-relative paths and absolute paths are covered.
Code-context paths and filenames are handled independently of prose.

Ambiguous prose such as `and/or`, `loading/caching`, `runner/evaluator`, and
`config/settings` is deliberately preserved unless it is made explicit with code
formatting, a path prefix, a filename extension, or additional path components.
The full sample therefore still contains some literal slashes. We do not claim
every slash should be removed or that all such phrases are speech-optimized.

### Independent review caught an additional bug

The initial improved implementation passed its golden cases, but an all-toggle
review found that `./ --` could become `./—...`, which a second pass interpreted
as a path. The dash rule now avoids that joining behavior. It has a dedicated
regression test in `test_text_preprocessing_safety.py`.

Repeated unmatched brackets also exposed avoidable regex backtracking; the link
pattern no longer treats another opening bracket as ordinary link-label content.
Resource-bound tests exercise long unmatched formatting, the final 5,000-character
ceiling, expansion rejection, and a 50,000-character intermediate normalization
ceiling. Formatting recursion is separately bounded.

## Remaining limitations

- English-focused deterministic rules, not a full CommonMark/HTML/code parser.
- Unrecognized syntax is generally preserved, not guessed or executed.
- Aliases and inserted sentence punctuation still need audio listening checks.
- Lowercase spoken email components do not encode case-sensitive local-part case.
- Only selected operators and currency forms are supported; this does not add
  arbitrary mathematical interpretation or international currency handling.
- Unsupported slash prose and query notation such as `hello+world` can remain raw.
- Passing repeat-processing tests is evidence for covered inputs, not a proof for
  every possible Unicode/Markdown string.
- The frontend was not changed, so its current preview may visually collapse
  newlines even though the returned inference text contains them.
- No new API provenance fields were added, and no historical benchmark/training
  artifact was modified. Those would require permission to change other modules.

## Reproduce

From `D:\OpenVoice-Lab\backend`:

```powershell
.\.venv\Scripts\python.exe tests/test_text_preprocessing.py --through 6
```

The report prints each input, expected output, actual output/error, pass result,
and repeat-processing result. Exit code is nonzero if any case fails.

Run the dedicated suite:

```powershell
$textTests = @(
  'tests/test_text_normalizer.py'
  'tests/test_text_sanitizer.py'
  'tests/test_text_preprocessing.py'
  'tests/test_text_preprocessing_safety.py'
)
.\.venv\Scripts\python.exe -m pytest @textTests -q
```

Run the unchanged consumer regression checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_benchmark.py tests/test_metrics.py -q
```
