#!/usr/bin/env python3
"""
Advanced Novel TTS (Version 7):
 - Uses fastcoref for coreference resolution.
 - Reads speaker synonyms from `character_synonyms.json`.
 - Voice map with dict { "speaker_id": "...", "description": "..." } stored in `character_voices_v7.json`.
 - Avoids splitting words mid-way by chunking on whitespace near max_chars.
 - Skips empty or punctuation-only chunks (e.g., single quotes), preventing WAV param mismatch.
 - Debug prints for each chunk.

Features:
  1) Full-document coreference with fastcoref (FCoref or LingMessCoref).
  2) Nested quotes detection for multiple quotes in a paragraph.
  3) Improved speaker detection (searching for "Name said" patterns, local entities).
  4) Round-robin speaker assignment if text has more unique speakers than the TTS model supports.
  5) Dedicated "Narrator" voice + optional metadata in JSON.
  6) Optionally exclude characters (skip or reassign to narrator).
  7) Chunking long lines for TTS to avoid memory/time issues.
  8) Debugging prints for each chunk + skipping empty lines, to avoid NoneType or invalid inputs to TTS.
  9) Speaker info (speaker_id, description) is stored in JSON, we extract .speaker_id for TTS calls.

Usage Example:
  python novel_tts_v7.py story.txt --model_name tts_models/en/vctk/vits --gpu

For multiple files:
  python novel_tts_v7.py --batch chapter1.txt chapter2.txt ...

Dependencies:
  pip install TTS spacy fastcoref
  python -m spacy download en_core_web_sm

Security Note:
  - Loading Torch models involves pickle; only load from trusted sources.
  - We filter input files to ".txt" to reduce accidental path or file issues.
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
# 1) Load Character Synonyms (Name Variants) from JSON
##############################################################################

with open("character_synonyms.json", "r", encoding="utf-8") as f:
    CHARACTER_SYNONYMS = json.load(f)

##############################################################################
# 2) Voice Map JSON (Speaker Info with metadata)
##############################################################################

VOICE_MAP_FILE = "character_voices_v7.json"

def load_voice_map() -> dict:
    """
    Load or create a global JSON map: { 'CharacterName': { 'speaker_id': '...', 'description': '...' } }
    """
    if os.path.isfile(VOICE_MAP_FILE):
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_voice_map(voice_map: dict):
    """
    Save the updated voice map to JSON: 
      e.g. { "Lucy": {"speaker_id":"p225", "description":"light female voice"}, ...}
    """
    with open(VOICE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(voice_map, f, indent=2)

##############################################################################
# 3) WAV Combining
##############################################################################

def combine_wav_files(wav_files, output_file):
    """
    Combine multiple WAV files into a single file, ensuring matching params.
    """
    if not wav_files:
        return

    with wave.open(wav_files[0], 'rb') as wf:
        params = wf.getparams()
        combined_data = wf.readframes(wf.getnframes())

    for wav_f in wav_files[1:]:
        with wave.open(wav_f, 'rb') as wf:
            if wf.getparams() != params:
                raise ValueError("WAV files differ in format; cannot combine.")
            combined_data += wf.readframes(wf.getnframes())

    with wave.open(output_file, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(combined_data)

##############################################################################
# 4) Text Chunking that Avoids Mid-Word Splits
##############################################################################

def chunk_text_for_tts(text: str, max_chars: int = 200) -> list:
    """
    Break long lines into segments, up to max_chars each, 
    but tries NOT to split in the middle of words.
    """
    text = text.strip()
    segments = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        # Avoid mid-word by searching for last whitespace if not at final segment
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
    """
    Detect multiple nested quotes with a simple stack approach.
    Returns list of (quote_text, start_idx, end_idx).
    """
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
                # pop from stack for nested
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
    """
    If 'LingMessCoref', we use "biu-nlp/lingmess-coref".
    If 'FCoref', we use "biu-nlp/f-coref".
    """
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
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
    """
    Use fastcoref + spacy to produce doc._.resolved_text with 'resolve_text'=True.
    """
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
    """
    Map synonyms to canonical names from `character_synonyms.json`.
    """
    raw_lower = raw_name.strip().title()
    # direct key check
    if raw_lower in CHARACTER_SYNONYMS:
        return raw_lower
    # synonyms check
    for main_name, variants in CHARACTER_SYNONYMS.items():
        if raw_lower == main_name.title():
            return main_name
        for variant in variants:
            if variant.lower() in raw_lower.lower():
                return main_name
    return raw_lower

def guess_speaker_for_quote(quote_text: str, doc_text: str, doc_spacy, q_start: int, q_end: int, known_speakers):
    """
    Attempt to guess speaker by scanning doc near the quote offset.
    Look for "Name said" pattern or PERSON entities in local range.
    """
    pos = doc_text.find(quote_text, max(q_start - 5, 0), q_end + 5)
    if pos < 0:
        pos = doc_text.find(quote_text)

    window = 100
    start_ctx = max(0, q_start - window)
    end_ctx = min(len(doc_text), q_end + window)
    context_str = doc_text[start_ctx:end_ctx]

    # direct "Name said" patterns
    for speaker in known_speakers:
        for verb in SPEECH_VERBS:
            pattern = rf"\b{speaker}\b\s+{verb}\b"
            if re.search(pattern, context_str, re.IGNORECASE):
                return speaker

    # if q_start or q_end invalid
    if q_start < 0 or q_end < 0:
        return None

    # token-level approach for local range
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
# 8) Main Script: Version 7
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Advanced Novel TTS (v7) with fastcoref, dict-based speaker info, improved chunking.")
    parser.add_argument("input_file", help="Path to a .txt file (or multiple with --batch).")
    parser.add_argument("--batch", nargs="*", help="Process multiple text files.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits", help="Coqui multi-speaker TTS model.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--max_line_chars", type=int, default=200, help="Max characters for TTS chunking.")
    parser.add_argument("--exclude_characters", nargs="*", default=["Lucy", "Verona", "Avery"],
                        help="Excluded character names. Lines assigned to them are either skipped or reassigned to narrator.")
    parser.add_argument("--skip_excluded", action="store_true",
                        help="If set, lines from excluded speakers are skipped. Otherwise reassign them to narrator.")
    parser.add_argument("--coref_model", default="FCoref",
                        help="Set to 'LingMessCoref' for bigger model, else 'FCoref'.")
    args = parser.parse_args()

    # Gather input files
    if args.batch:
        input_files = args.batch
    else:
        input_files = [args.input_file]

    for fpath in input_files:
        if not os.path.isfile(fpath):
            print(f"[ERROR] File not found: {fpath}")
            sys.exit(1)
        if not fpath.lower().endswith(".txt"):
            print(f"[ERROR] Only .txt files are allowed: {fpath}")
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

    # Filter out "ED" from TTS speaker list
    try:
        raw_speakers = tts.speakers
        available_speakers = [s.strip() for s in raw_speakers if s.strip() and s.strip() != "ED"]
        if not available_speakers:
            raise ValueError("No valid speaker IDs found after removing 'ED'.")
    except AttributeError:
        print("[ERROR] This TTS model doesn't support multiple speakers.")
        sys.exit(1)

    print(f"[INFO] Available speakers (without 'ED'): {available_speakers}")

    # Load or create voice map
    voice_map = load_voice_map()

    next_speaker_index = 0
    def get_next_speaker_id_str() -> str:
        nonlocal next_speaker_index
        sid = available_speakers[next_speaker_index % len(available_speakers)]
        next_speaker_index += 1
        return sid

    # Ensure we have a narrator entry
    if "Narrator" not in voice_map:
        first_narrator_id = get_next_speaker_id_str()
        voice_map["Narrator"] = {
            "speaker_id": first_narrator_id,
            "description": "Primary narrator voice"
        }
        print(f"[INFO] Assigned Narrator -> {first_narrator_id}")
    narrator_info = voice_map["Narrator"]  # dict with 'speaker_id', 'description'

    def get_or_create_speaker_entry(char_name: str) -> dict:
        if char_name in voice_map:
            return voice_map[char_name]
        else:
            sid = get_next_speaker_id_str()
            new_entry = {
                "speaker_id": sid,
                "description": f"Auto-generated speaker for {char_name}"
            }
            voice_map[char_name] = new_entry
            return new_entry

    # Regex to detect punctuation-only lines
    punct_only_re = re.compile(r'^[\W_]+$')

    for file_path in input_files:
        print(f"[INFO] Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # (A) Full-doc coref
        resolved_text = resolve_coref_text(nlp, raw_text)

        # minimal doc for offset calc
        doc_spacy = nlp.make_doc(resolved_text)
        
        # (B) Split into paragraphs
        paragraphs = [p.strip() for p in resolved_text.split("\n\n") if p.strip()]

        final_lines = []
        current_offset = 0

        for paragraph in paragraphs:
            spans = find_dialogue_spans_nested(paragraph)
            if not spans:
                # entire paragraph => narrator
                final_lines.append((narrator_info, paragraph))
                current_offset += len(paragraph) + 2
                continue

            last_end = 0
            for (quote_text, q_start, q_end) in spans:
                # Narration
                narration_chunk = paragraph[last_end:q_start].strip()
                if narration_chunk:
                    final_lines.append((narrator_info, narration_chunk))

                # Attempt speaker detection
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
                        spk_entry = get_or_create_speaker_entry(speaker)
                        final_lines.append((spk_entry, quote_text.strip()))
                else:
                    # Unknown => narrator
                    final_lines.append((narrator_info, quote_text.strip()))

                last_end = q_end

            # trailing
            if last_end < len(paragraph):
                trailing = paragraph[last_end:].strip()
                if trailing:
                    final_lines.append((narrator_info, trailing))

            current_offset += len(paragraph) + 2

        if not final_lines:
            print("[INFO] No lines to read for this file.")
            continue

        # (C) Generate TTS
        temp_files = []
        try:
            for idx, (speaker_dict, line_text) in enumerate(final_lines):
                speaker_str = speaker_dict["speaker_id"]  # or speaker_dict.get("speaker_id")
                # chunk text
                chunks = chunk_text_for_tts(line_text, args.max_line_chars)

                for cid, chunk in enumerate(chunks):
                    chunk_stripped = chunk.strip()
                    if not chunk_stripped:
                        print(f"[SKIP] Empty chunk at line {idx+1}, chunk {cid+1}")
                        continue

                    # skip punctuation-only
                    if punct_only_re.match(chunk_stripped):
                        print(f"[SKIP] Punctuation-only chunk at line {idx+1}, chunk {cid+1}: {repr(chunk)}")
                        continue

                    # Debug
                    print(f"[DEBUG] line={idx+1}, chunk={cid+1}, speaker={speaker_str}, text={repr(chunk)}")

                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav_name = tmp_wav.name
                    tmp_wav.close()

                    try:
                        tts.tts_to_file(text=chunk, speaker=speaker_str, file_path=tmp_wav_name)
                        
                        # Check if the produced WAV has zero frames
                        with wave.open(tmp_wav_name, "rb") as wf:
                            n_frames = wf.getnframes()
                            if n_frames == 0:
                                print(f"[SKIP] Zero-length WAV at line {idx+1}, chunk {cid+1}. Removing.")
                                wf.close()
                                os.remove(tmp_wav_name)
                                continue  # skip adding to temp_files
                        
                        temp_files.append(tmp_wav_name)
                    except Exception as e:
                        print(f"[ERROR] TTS generation failed on line {idx+1}, chunk {cid+1}: {e}")
                        # cleanup partial
                        for tfp in temp_files:
                            if os.path.exists(tfp):
                                os.remove(tfp)
                        sys.exit(1)

            # Combine partial wavs
            base_name = os.path.splitext(file_path)[0]
            output_wav = base_name + "_v7.wav"
            print("[INFO] Combining partial audio segments...")
            combine_wav_files(temp_files, output_wav)
            print(f"[INFO] Audio file generated: {output_wav}")

        finally:
            # remove tmp files
            for tfp in temp_files:
                if os.path.exists(tfp):
                    os.remove(tfp)

        # Save updated voice map
        save_voice_map(voice_map)

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
