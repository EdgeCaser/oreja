"""
Word-error-rate harness for Oreja's transcription pipeline.

Measures the pipeline as production runs it: audio is POSTed to a running
backend's /transcribe endpoint, so the vocabulary prompt, hallucination gates,
VAD and (optionally) accuracy mode are all part of what gets measured.

Reference layout: a directory of audio/reference pairs sharing a base name -
    eval_data/
        meeting1.wav
        meeting1.ref.json     <- Oreja's own JSON export (Save Transcription),
                                 after hand-correcting every segment in the app
        call2.flac
        call2.ref.txt         <- or plain corrected text, if you prefer

A practical way to build references: transcribe a file in the Oreja app, fix
every mistranscribed word in the transcript pane (double-click a segment's text
to edit it), then Save Transcription as JSON and drop it here renamed to
<audio-stem>.ref.json. Speaker labels and timestamps in the export are ignored -
only the words are scored. Privacy mode must be OFF when exporting, or every
segment reads [REDACTED].

Usage (backend must be running):
    python eval_wer.py eval_data/
    python eval_wer.py eval_data/ --language en --accuracy
    python eval_wer.py eval_data/ --json baseline.json

A/B method (matches the house measurement rule): run once before a change and
once after, on the SAME files, and compare the aggregate WER of the two JSON
outputs. Text is normalized (lowercased, punctuation stripped) before scoring,
so only word-level differences count.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
import jiwer

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")
REFERENCE_SUFFIXES = (".ref.json", ".ref.txt")  # first match wins

# Case and punctuation are formatting, not recognition: normalize both sides so
# "Okay," vs "okay" is not counted as an error.
_normalize = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def find_pairs(eval_dir: Path):
    """Yield (audio_path, reference_path) pairs; warn on unpaired files."""
    pairs = []
    for audio in sorted(eval_dir.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        candidates = [audio.with_name(audio.stem + s) for s in REFERENCE_SUFFIXES]
        reference = next((c for c in candidates if c.exists()), None)
        if reference is not None:
            pairs.append((audio, reference))
        else:
            print(f"[skip] {audio.name}: no {candidates[0].name} or {candidates[1].name}",
                  file=sys.stderr)
    return pairs


def load_reference(reference_path: Path) -> str:
    """
    Reference text from either format: Oreja's JSON export (segment texts are
    joined; speakers/timestamps ignored) or a plain text file.
    """
    raw = reference_path.read_text(encoding="utf-8").strip()
    if reference_path.name.lower().endswith(".ref.json"):
        payload = json.loads(raw)
        segments = payload.get("segments") or []
        texts = [(s.get("text") or "").strip() for s in segments]
        if any(t == "[REDACTED]" for t in texts):
            raise ValueError(
                f"{reference_path.name} was exported with privacy mode on - "
                "every segment is [REDACTED]. Re-export with privacy mode off."
            )
        return " ".join(t for t in texts if t).strip()
    return raw


def transcribe(client: httpx.Client, base_url: str, audio_path: Path,
               language: str, accuracy: bool) -> str:
    params = {"source": "file"}
    if language:
        params["language"] = language
    if accuracy:
        params["accuracy"] = "true"
    with open(audio_path, "rb") as handle:
        response = client.post(
            f"{base_url}/transcribe",
            params=params,
            files={"audio": (audio_path.name, handle, "application/octet-stream")},
        )
    response.raise_for_status()
    payload = response.json()
    segments = payload.get("segments") or []
    return " ".join((s.get("text") or "").strip() for s in segments).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("eval_dir", type=Path,
                        help="directory of <name>.<audio-ext> + <name>.ref.txt pairs")
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="backend base URL (default %(default)s)")
    parser.add_argument("--language", default="en",
                        help="language code, 'auto', or '' for the server default "
                             "(default %(default)s)")
    parser.add_argument("--accuracy", action="store_true",
                        help="request accuracy mode (higher beam / OREJA_FILE_MODEL)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write per-file and aggregate results to this file")
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        print(f"Not a directory: {args.eval_dir}", file=sys.stderr)
        return 2

    pairs = find_pairs(args.eval_dir)
    if not pairs:
        print(f"No audio/{REFERENCE_SUFFIX} pairs found in {args.eval_dir}", file=sys.stderr)
        return 2

    references, hypotheses, per_file = [], [], []
    # Long files legitimately transcribe for minutes; never let the client give up first.
    with httpx.Client(timeout=httpx.Timeout(1800.0)) as client:
        for audio, ref_path in pairs:
            try:
                reference = load_reference(ref_path)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[skip] {ref_path.name}: {e}", file=sys.stderr)
                continue
            if not reference:
                print(f"[skip] {ref_path.name}: empty reference", file=sys.stderr)
                continue
            print(f"Transcribing {audio.name}...", flush=True)
            hypothesis = transcribe(client, args.url, audio, args.language, args.accuracy)

            output = jiwer.process_words(
                reference, hypothesis,
                reference_transform=_normalize, hypothesis_transform=_normalize,
            )
            per_file.append({
                "file": audio.name,
                "wer": output.wer,
                "substitutions": output.substitutions,
                "deletions": output.deletions,
                "insertions": output.insertions,
                "hits": output.hits,
                "reference_words": output.hits + output.substitutions + output.deletions,
            })
            references.append(reference)
            hypotheses.append(hypothesis)

    if not references:
        print("Nothing evaluated.", file=sys.stderr)
        return 2

    aggregate = jiwer.process_words(
        references, hypotheses,
        reference_transform=_normalize, hypothesis_transform=_normalize,
    )

    print()
    print(f"{'file':<40} {'WER':>7}  {'sub':>5} {'del':>5} {'ins':>5} {'ref words':>9}")
    for row in per_file:
        print(f"{row['file']:<40} {row['wer']:>6.2%}  {row['substitutions']:>5} "
              f"{row['deletions']:>5} {row['insertions']:>5} {row['reference_words']:>9}")
    print()
    print(f"AGGREGATE WER: {aggregate.wer:.2%}  "
          f"(sub {aggregate.substitutions}, del {aggregate.deletions}, "
          f"ins {aggregate.insertions}, over "
          f"{aggregate.hits + aggregate.substitutions + aggregate.deletions} reference words)")

    if args.json:
        args.json.write_text(json.dumps({
            "url": args.url,
            "language": args.language,
            "accuracy": args.accuracy,
            "aggregate_wer": aggregate.wer,
            "files": per_file,
        }, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
