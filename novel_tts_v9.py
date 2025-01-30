#!/usr/bin/env python3
"""
Advanced Novel TTS (Version 9) with:
 - fastcoref-based coreference,
 - partial compression usage in WAV,
 - ignoring nframes and compname in mismatch checks,
 - skipping zero-length or truly format-incompatible files.

Features:
  1) Full-document coreference with fastcoref (FCoref or LingMessCoref).
  2) Nested quotes detection for multiple quotes in a paragraph.
  3) Speaker info in JSON, narrator assigned, skipping excluded characters.
  4) Avoid splitting words mid-chunk, skipping punctuation-only lines.
  5) Checking WAV parameters but ignoring file length (nframes) and compname.
  6) Optionally compress the final combined WAV with e.g. ULAW if wave supports it.

Usage Example:
  python novel_tts_v9.py story.txt --model_name tts_models/en/vctk/vits --gpu

Dependencies:
  pip install TTS spacy fastcoref
  python -m spacy download en_core_web_sm

Security Note:
  - The wave module built into Python has limited compression types. 
  - For robust compression formats, consider external tools (e.g. pydub + ffmpeg).
"""

import argparse
import os
import sys
import wave
import re
import json
import tempfile
import spacy
from fastcoref import spacy_component
from TTS.api import TTS

##############################################################################
# 0) Utility: Sanitize / ASCII-fy Text
##############################################################################

def sanitize_text(text: str) -> str:
    """
    Replace curly quotes, dashes, ellipses with plain ASCII,
    then remove any leftover non-ASCII chars.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("…", "...").replace("—", "-").replace("–", "-")
    # ASCII-ify
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text

##############################################################################
# 1) Load Character Synonyms (Name Variants) from JSON
##############################################################################

with open("character_synonyms.json", "r", encoding="utf-8") as f:
    CHARACTER_SYNONYMS = json.load(f)

##############################################################################
# 2) Voice Map JSON (Speaker Info with metadata)
##############################################################################

VOICE_MAP_FILE = "character_voices_v9.json"

def load_voice_map() -> dict:
    if os.path.isfile(VOICE_MAP_FILE):
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_voice_map(voice_map: dict):
    with open(VOICE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(voice_map, f, indent=2)

##############################################################################
# 3) WAV Combining With Optional Compression
##############################################################################

def essential_params(p):
    """
    Return the relevant fields for format-checking: (nchannels, sampwidth, framerate, comptype).
    We ignore nframes and compname for mismatch checks.
    """
    return (p.nchannels, p.sampwidth, p.framerate, p.comptype)

def combine_wav_files(wav_files, output_file, compress=False):
    """
    Combine multiple WAV files into a single file, ensuring matching essential params.

    If compress=True, we attempt to set a different compression type (e.g. ULAW) in the final WAV.
    This only works if wave supports it (which is minimal).
    """
    if not wav_files:
        return

    with wave.open(wav_files[0], 'rb') as wf:
        first_params = wf.getparams()
        data_accum = wf.readframes(wf.getnframes())

    base_essentials = essential_params(first_params)

    for wav_f in wav_files[1:]:
        with wave.open(wav_f, 'rb') as wf:
            cur_params = wf.getparams()
            if essential_params(cur_params) != base_essentials:
                raise ValueError("WAV files differ in format (channels/rate/sampwidth/comptype); cannot combine.")
            data_accum += wf.readframes(wf.getnframes())

    # If compress=True, we try e.g. "ULAW" or "ALAW".
    # The wave module isn't guaranteed to support many. 
    # Let's do a check:
    if compress:
        # This is an example; actual wave compression support is minimal in Python.
        compressed_comptype = 'ULAW'  # or 'ALAW'
        # If wave doesn't support that, you might get an error.
        final_params = (first_params.nchannels, first_params.sampwidth, first_params.framerate,
                        compressed_comptype, "Python wave compressed")
    else:
        final_params = first_params

    # Now write
    with wave.open(output_file, 'wb') as wf:
        wf.setparams(final_params)
        wf.writeframes(data_accum)

##############################################################################
# 4) Text Chunking that Avoids Mid-Word Splits
##############################################################################

def chunk_text_for_tts(text: str, max_chars: int = 200) -> list:
    text = text.strip()
    segments = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            space_boundary = text.rfind(" ", start, end)
            if space_boundary != -1 and space_boundary > start:
                end = space_boundary
        segment = text[start:end].strip()
        start = end
        if segment:
            segments.append(segment)

    return segments

##############################################################################
# 5) Nested Quote Detection
##############################################################################

def find_dialogue_spans_nested(text: str):
    OPEN_QUOTES = ["“", "‘", "\"", "'"]
    CLOSE_QUOTES = ["”", "’", "\"", "'"]

    results = []
    stack = []
    start_pos = None
    in_quote = False
    quote_char = None

    for i, ch in enumerate(text):
        if ch in OPEN_QUOTES:
            if not in_quote:
                in_quote = True
                start_pos = i + 1
                quote_char = ch
            else:
                stack.append((quote_char, start_pos))
                start_pos = i + 1
                quote_char = ch
        elif ch in CLOSE_QUOTES and in_quote:
            end_idx = i
            quote_text = text[start_pos:end_idx]
            results.append((quote_text, start_pos, end_idx))

            if stack:
                prev_char, prev_start = stack.pop()
                start_pos = prev_start
                quote_char = prev_char
            else:
                in_quote = False
                quote_char = None
                start_pos = None

    return results

##############################################################################
# 6) FastCoref + spaCy Setup
##############################################################################

def setup_spacy_coref(model_architecture="FCoref", device="cpu"):
    import spacy
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    from fastcoref import spacy_component

    if model_architecture.lower() == "fcoref":
        hf_repo = "biu-nlp/f-coref"
    elif model_architecture.lower() == "lingmesscoref":
        hf_repo = "biu-nlp/lingmess-coref"
    else:
        raise ValueError(f"Unsupported coref model: {model_architecture}")

    nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": model_architecture,
            "model_path": hf_repo,
            "device": device
        }
    )
    return nlp

def resolve_coref_text(nlp, text: str) -> str:
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    if hasattr(doc._, "resolved_text") and doc._.resolved_text:
        return doc._.resolved_text
    return text

##############################################################################
# 7) Speaker Detection, Name Unification, Etc.
##############################################################################

SPEECH_VERBS = ["said", "asked", "replied", "muttered", "shouted", "yelled",
                "whispered", "cried", "called", "snapped", "answered"]

def unify_character_name(raw_name: str) -> str:
    raw_lower = raw_name.strip().title()
    if raw_lower in CHARACTER_SYNONYMS:
        return raw_lower
    for main_name, variants in CHARACTER_SYNONYMS.items():
        if raw_lower == main_name.title():
            return main_name
        for variant in variants:
            if variant.lower() in raw_lower.lower():
                return main_name
    return raw_lower

def guess_speaker_for_quote(quote_text: str, doc_text: str, doc_spacy, q_start: int, q_end: int, known_speakers):
    pos = doc_text.find(quote_text, max(q_start - 5, 0), q_end + 5)
    if pos < 0:
        pos = doc_text.find(quote_text)

    window = 100
    start_ctx = max(0, q_start - window)
    end_ctx = min(len(doc_text), q_end + window)
    context_str = doc_text[start_ctx:end_ctx]

    for speaker in known_speakers:
        for verb in SPEECH_VERBS:
            pattern = rf"\b{speaker}\b\s+{verb}\b"
            if re.search(pattern, context_str, re.IGNORECASE):
                return speaker

    if q_start < 0 or q_end < 0:
        return None

    quote_start_token = None
    quote_end_token = None
    for token in doc_spacy:
        if token.idx <= q_start < (token.idx + len(token.text)):
            quote_start_token = token.i
        if token.idx <= q_end < (token.idx + len(token.text)):
            quote_end_token = token.i

    if quote_start_token is not None and quote_end_token is not None:
        local_range = 10
        t_start = max(0, quote_start_token - local_range)
        t_end = min(len(doc_spacy), quote_end_token + local_range)
        possible_names = set()
        for ent in doc_spacy.ents:
            if ent.label_ == "PERSON":
                if (t_start <= ent.start < t_end) or (t_start <= ent.end < t_end):
                    nm = unify_character_name(ent.text)
                    possible_names.add(nm)
        intersect = possible_names.intersection(set(known_speakers))
        if len(intersect) == 1:
            return intersect.pop()

    return None

##############################################################################
# 8) Main Script: Version 9 (compressed + ignoring nframes + compname)
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Novel TTS v9 with compression, ignoring nframes in mismatch checks, skipping zero-len.")
    parser.add_argument("input_file", help="Path to a .txt file or use --batch.")
    parser.add_argument("--batch", nargs="*", help="Multiple text files.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits", help="Multi-speaker TTS model.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--max_line_chars", type=int, default=200, help="Max chunk size.")
    parser.add_argument("--exclude_characters", nargs="*", default=["Lucy", "Verona", "Avery"],
                        help="Excluded characters are skipped or reassigned.")
    parser.add_argument("--skip_excluded", action="store_true", help="Skip lines from excluded speakers.")
    parser.add_argument("--coref_model", default="FCoref", help="FCoref or LingMessCoref.")
    parser.add_argument("--compress", action="store_true", help="If set, compress final WAV e.g. ULAW.")
    args = parser.parse_args()

    if args.batch:
        input_files = args.batch
    else:
        input_files = [args.input_file]

    for fpath in input_files:
        if not os.path.isfile(fpath):
            print(f"[ERROR] File not found: {fpath}")
            sys.exit(1)
        if not fpath.lower().endswith(".txt"):
            print(f"[ERROR] Only .txt files allowed: {fpath}")
            sys.exit(1)

    device_str = "cuda:0" if args.gpu else "cpu"
    print(f"[INFO] Setting up spaCy + {args.coref_model} on device={device_str}...")
    nlp = setup_spacy_coref(model_architecture=args.coref_model, device=device_str)

    print(f"[INFO] Loading TTS model: {args.model_name}")
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print(f"[ERROR] Failed to load TTS model: {e}")
        sys.exit(1)

    try:
        raw_speakers = tts.speakers
        available_speakers = [s.strip() for s in raw_speakers if s.strip() and s.strip() != "ED"]
        if not available_speakers:
            raise ValueError("No valid speaker IDs found after removing ED.")
    except AttributeError:
        print("[ERROR] TTS model doesn't support multiple speakers.")
        sys.exit(1)

    print(f"[INFO] Available speakers (minus 'ED'): {available_speakers}")

    voice_map = load_voice_map()

    next_speaker_idx = 0
    def get_next_speaker_id_str() -> str:
        nonlocal next_speaker_idx
        sid = available_speakers[next_speaker_idx % len(available_speakers)]
        next_speaker_idx += 1
        return sid

    if "Narrator" not in voice_map:
        nid = get_next_speaker_id_str()
        voice_map["Narrator"] = {
            "speaker_id": nid,
            "description": "Primary narrator voice"
        }
        print(f"[INFO] Assigned Narrator -> {nid}")

    narrator_info = voice_map["Narrator"]
    def get_or_create_speaker_entry(char_name: str) -> dict:
        if char_name in voice_map:
            return voice_map[char_name]
        sid = get_next_speaker_id_str()
        new_entry = {"speaker_id": sid, "description": f"Auto speaker for {char_name}"}
        voice_map[char_name] = new_entry
        return new_entry

    punct_only_re = re.compile(r'^[\W_]+$')

    for file_path in input_files:
        print(f"[INFO] Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Coref
        resolved_text = resolve_coref_text(nlp, raw_text)
        doc_spacy = nlp.make_doc(resolved_text)

        paragraphs = [p.strip() for p in resolved_text.split("\n\n") if p.strip()]

        final_lines = []
        current_offset = 0

        for paragraph in paragraphs:
            spans = find_dialogue_spans_nested(paragraph)
            if not spans:
                final_lines.append((narrator_info, paragraph))
                current_offset += len(paragraph) + 2
                continue

            last_end = 0
            for (quote_text, q_start, q_end) in spans:
                narration_chunk = paragraph[last_end:q_start].strip()
                if narration_chunk:
                    final_lines.append((narrator_info, narration_chunk))

                speaker = guess_speaker_for_quote(
                    quote_text=quote_text,
                    doc_text=paragraph,
                    doc_spacy=doc_spacy,
                    q_start=current_offset + q_start,
                    q_end=current_offset + q_end,
                    known_speakers=list(voice_map.keys())
                )
                if speaker is not None:
                    if speaker in args.exclude_characters:
                        if args.skip_excluded:
                            pass
                        else:
                            final_lines.append((narrator_info, quote_text.strip()))
                else:
                    if quote_text.strip():
                        final_lines.append((narrator_info, quote_text.strip()))
                    continue

                if speaker and speaker not in args.exclude_characters:
                    spk = get_or_create_speaker_entry(speaker)
                    final_lines.append((spk, quote_text.strip()))

                last_end = q_end

            if last_end < len(paragraph):
                trailing = paragraph[last_end:].strip()
                if trailing:
                    final_lines.append((narrator_info, trailing))

            current_offset += len(paragraph) + 2

        if not final_lines:
            print("[INFO] No lines to read for this file.")
            continue

        temp_files = []
        first_params = None

        try:
            for idx, (speaker_dict, line_text) in enumerate(final_lines):
                speaker_str = speaker_dict["speaker_id"]
                chunks = chunk_text_for_tts(line_text, args.max_line_chars)

                for cid, chunk in enumerate(chunks):
                    c_strip = chunk.strip()
                    if not c_strip:
                        print(f"[SKIP] Empty chunk at line {idx+1}, chunk {cid+1}")
                        continue
                    if punct_only_re.match(c_strip):
                        print(f"[SKIP] Punctuation-only chunk at line {idx+1}, chunk {cid+1}: {repr(chunk)}")
                        continue

                    # sanitize
                    c_sanitized = sanitize_text(c_strip)
                    print(f"[DEBUG] line={idx+1}, chunk={cid+1}, speaker={speaker_str}, text={repr(c_sanitized)}")

                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav_name = tmp_wav.name
                    tmp_wav.close()

                    try:
                        tts.tts_to_file(text=c_sanitized, speaker=speaker_str, file_path=tmp_wav_name)
                        with wave.open(tmp_wav_name, "rb") as wf:
                            n_frames = wf.getnframes()
                            current_params = wf.getparams()
                            print("[PARAMS]", current_params)

                        if n_frames == 0:
                            print(f"[SKIP] Zero-length WAV at line {idx+1}, chunk {cid+1}. Removing.")
                            os.remove(tmp_wav_name)
                            continue

                        # Compare ignoring nframes + compname
                        # We'll do essential_params
                        from functools import partial
                        if first_params is None:
                            first_params = essential_params(current_params)
                        else:
                            if essential_params(current_params) != first_params:
                                print(f"[SKIP] Mismatched WAV format (ignoring length) at line {idx+1}, chunk {cid+1}. Removing.")
                                os.remove(tmp_wav_name)
                                continue

                        temp_files.append(tmp_wav_name)

                    except Exception as e:
                        print(f"[ERROR] TTS generation failed on line {idx+1}, chunk {cid+1}: {e}")
                        for tfp in temp_files:
                            if os.path.exists(tfp):
                                os.remove(tfp)
                        sys.exit(1)

            if temp_files:
                base_name = os.path.splitext(file_path)[0]
                output_wav = base_name + "_v9.wav"
                print("[INFO] Combining partial audio segments...")
                # compress=... if you want
                combine_wav_files(temp_files, output_wav, compress=args.compress)
                print(f"[INFO] Audio file generated: {output_wav}")
            else:
                print("[INFO] No valid WAV files to combine. Possibly all skipped?")

        finally:
            for tfp in temp_files:
                if os.path.exists(tfp):
                    os.remove(tfp)

        save_voice_map(voice_map)

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
