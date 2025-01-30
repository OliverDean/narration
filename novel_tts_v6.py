#!/usr/bin/env python3
"""
Advanced Novel TTS (Version 6) with FastCoref + Enhanced Speaker Attribution
+ Debug Printing of Chunks & Skipping Empty Lines
+ Storing speaker info as a dictionary in the voice map.

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
  python novel_tts_v6_debug.py story.txt --model_name tts_models/en/vctk/vits --gpu

For multiple files:
  python novel_tts_v6_debug.py --batch chapter1.txt chapter2.txt ...

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
# LOAD SYNONYMS FROM JSON
##############################################################################

with open("character_synonyms.json", "r", encoding="utf-8") as f:
    CHARACTER_SYNONYMS = json.load(f)

##############################################################################
# FILE STORAGE FOR VOICE MAP
##############################################################################

# Example structure in JSON (voice_map.json):
# {
#   "Narrator": {
#       "speaker_id": "p234",
#       "description": "Primary narrator voice"
#    },
#   "Lucy": {
#       "speaker_id": "p225",
#       "description": "Light female voice"
#    },
#   ...
# }

VOICE_MAP_FILE = "character_voices_v6.json"

def load_voice_map() -> dict:
    """Load (or create) a global JSON map of character->(dict with speaker_id,description)."""
    if os.path.isfile(VOICE_MAP_FILE):
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_voice_map(voice_map: dict):
    """Save the updated voice map (dict of dict) to JSON."""
    with open(VOICE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(voice_map, f, indent=2)

##############################################################################
# WAV COMBINING
##############################################################################

def combine_wav_files(wav_files, output_file):
    """
    Combine multiple WAV files into a single WAV.
    Ensures they share the same params.
    """
    if not wav_files:
        return

    import wave
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
# TEXT CHUNKING FOR TTS
##############################################################################

def chunk_text_for_tts(text: str, max_chars: int = 200) -> list:
    """
    Break long lines into multiple segments, up to max_chars each.
    Attempts to break at a period if possible.
    """
    text = text.strip()
    segments = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end
    return segments

##############################################################################
# NESTED QUOTE DETECTION
##############################################################################

def find_dialogue_spans_nested(text: str):
    """
    Detect nested quotes in a paragraph using a stack approach.
    Returns a list of (quote_text, start_idx, end_idx).
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
                # Nested quote
                stack.append((quote_char, start_pos))
                start_pos = i + 1
                quote_char = ch
        elif ch in CLOSE_QUOTES and in_quote:
            end_idx = i
            quote_text = text[start_pos:end_idx]
            results.append((quote_text, start_pos, end_idx))

            if stack:
                # Pop from stack for nested scenario
                prev_char, prev_start = stack.pop()
                start_pos = prev_start
                quote_char = prev_char
            else:
                # Quote fully closed
                in_quote = False
                quote_char = None
                start_pos = None
    return results

##############################################################################
# FASTCOREF + SPACY: FULL-DOCUMENT COREFERENCE
##############################################################################

def setup_spacy_coref(model_architecture="FCoref", device="cpu"):
    """
    Initialize a spaCy pipeline with fastcoref. 
    If you want the bigger model: model_architecture="LingMessCoref".
    """
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])

    # handle the correct HF repo name
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
    Use fastcoref + spaCy to produce doc._.resolved_text.
    We enable 'resolve_text' at doc creation time via component_cfg.
    """
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    if hasattr(doc._, "resolved_text") and doc._.resolved_text:
        return doc._.resolved_text
    return text  # fallback if not found

##############################################################################
# ADVANCED SPEAKER DETECTION + NAME UNIFICATION
##############################################################################

SPEECH_VERBS = ["said", "asked", "replied", "muttered", "shouted", "yelled",
                "whispered", "cried", "called", "snapped", "answered"]

def unify_character_name(raw_name: str) -> str:
    raw_lower = raw_name.strip().title()
    # If raw_lower is exactly a key:
    if raw_lower in CHARACTER_SYNONYMS:
        return raw_lower

    # Otherwise, check synonyms
    for main_name, variants in CHARACTER_SYNONYMS.items():
        if raw_lower == main_name.title():
            return main_name
        for variant in variants:
            if variant.lower() in raw_lower.lower():
                return main_name
    return raw_lower

def guess_speaker_for_quote(quote_text: str, doc_text: str, doc_spacy, q_start: int, q_end: int, known_speakers):
    # Attempt to locate snippet
    pos = doc_text.find(quote_text, max(q_start - 5, 0), q_end + 5)
    if pos < 0:
        pos = doc_text.find(quote_text)

    window = 100
    start_ctx = max(0, q_start - window)
    end_ctx = min(len(doc_text), q_end + window)
    context_str = doc_text[start_ctx:end_ctx]

    # 1) direct pattern "Name said"
    for speaker in known_speakers:
        for verb in SPEECH_VERBS:
            pattern = rf"\b{speaker}\b\s+{verb}\b"
            if re.search(pattern, context_str, re.IGNORECASE):
                return speaker

    # 2) spaCy offsets
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
# MAIN SCRIPT (VERSION 6 + DEBUG PRINTS + dictionary-based voice map)
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Novel TTS v6 with fastcoref, advanced speaker attribution, dictionary-based speaker info."
    )
    parser.add_argument("input_file", help="Path to a text file, or use --batch for multiple.")
    parser.add_argument("--batch", nargs="*",
                        help="Process multiple text files in a single run.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits",
                        help="Coqui multi-speaker TTS model name.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available.")
    parser.add_argument("--max_line_chars", type=int, default=200,
                        help="Max characters for TTS chunking.")
    parser.add_argument("--exclude_characters", nargs="*", default=["Lucy", "Verona", "Avery"],
                        help="Characters to exclude. Default: Lucy,Verona,Avery.")
    parser.add_argument("--skip_excluded", action="store_true",
                        help="If set, lines from excluded speakers are skipped entirely. Otherwise reassign to narrator.")
    parser.add_argument("--coref_model", default="FCoref",
                        help="Coref model architecture: 'FCoref' or 'LingMessCoref'.")
    args = parser.parse_args()

    # 1) Gather input files
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

    # 2) Setup spaCy + fastcoref
    device_str = "cuda:0" if args.gpu else "cpu"
    print(f"[INFO] Setting up spaCy + {args.coref_model} on device={device_str}...")
    nlp = setup_spacy_coref(model_architecture=args.coref_model, device=device_str)

    # 3) Load TTS model
    print(f"[INFO] Loading TTS model: {args.model_name}")
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print(f"[ERROR] Failed to load TTS model: {e}")
        print("[SECURITY] Make sure the TTS model is from a trusted source.")
        sys.exit(1)

    # 4) Filter out "ED"
    try:
        raw_speakers = tts.speakers
        available_speakers = [s.strip() for s in raw_speakers if s.strip() and s.strip() != "ED"]
        if not available_speakers:
            raise ValueError("No valid speaker IDs found after removing 'ED'.")
    except AttributeError:
        print("[ERROR] This TTS model does not support multiple speakers.")
        sys.exit(1)

    print(f"[INFO] Available speakers (without 'ED'): {available_speakers}")

    # 5) Load or create voice map
    voice_map = load_voice_map()

    # 6) Round-robin for new speakers
    next_speaker_counter = 0
    def get_next_speaker_id_str():
        """Get the next speaker ID from 'available_speakers' as a string."""
        nonlocal next_speaker_counter
        sid = available_speakers[next_speaker_counter % len(available_speakers)]
        next_speaker_counter += 1
        return sid

    def get_or_create_speaker_entry(char_name: str) -> dict:
        """
        Return a dict like {"speaker_id": "...", "description": "..."}
        If 'char_name' not in voice_map, create new with round-robin speaker_id.
        """
        if char_name in voice_map:
            return voice_map[char_name]  # e.g. {"speaker_id": "...", "description": "..."}
        else:
            sid = get_next_speaker_id_str()
            # You can define a placeholder description or logic to pick something
            new_entry = {
                "speaker_id": sid,
                "description": f"Auto-generated speaker for {char_name}"
            }
            voice_map[char_name] = new_entry
            return new_entry

    # Ensure Narrator is in map
    if "Narrator" not in voice_map:
        # pick next speaker for narrator
        narrator_speaker_id = get_next_speaker_id_str()
        voice_map["Narrator"] = {
            "speaker_id": narrator_speaker_id,
            "description": "Primary narrator voice"
        }
        print(f"[INFO] Assigned Narrator -> {narrator_speaker_id}")
    narrator_info = voice_map["Narrator"]

    for file_path in input_files:
        print(f"[INFO] Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # (A) Full-document coref
        resolved_text = resolve_coref_text(nlp, raw_text)

        # minimal doc for offsets
        doc_spacy = nlp.make_doc(resolved_text)

        # (B) Split into paragraphs
        paragraphs = [p.strip() for p in resolved_text.split("\n\n") if p.strip()]

        final_lines = []
        current_offset = 0

        for paragraph in paragraphs:
            spans = find_dialogue_spans_nested(paragraph)
            if not spans:
                # entire paragraph => narrator
                if paragraph.strip():
                    final_lines.append((narrator_info, paragraph.strip()))
                current_offset += len(paragraph) + 2
                continue

            last_end = 0
            for (quote_text, q_start, q_end) in spans:
                # Narration chunk
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
                    # unify & exclude
                    if speaker in args.exclude_characters:
                        if args.skip_excluded:
                            pass
                        else:
                            if quote_text.strip():
                                final_lines.append((narrator_info, quote_text.strip()))
                    else:
                        speaker_entry = get_or_create_speaker_entry(speaker)
                        if quote_text.strip():
                            final_lines.append((speaker_entry, quote_text.strip()))
                else:
                    # unknown => narrator
                    if quote_text.strip():
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
                # 'speaker_dict' is e.g. {"speaker_id": "p234", "description": "..."}
                # We only want the .speaker_id string for TTS
                speaker_str = speaker_dict["speaker_id"]

                chunks = chunk_text_for_tts(line_text, args.max_line_chars)
                for cid, chunk in enumerate(chunks):
                    if not chunk.strip():
                        print(f"[SKIP] Empty chunk at line {idx+1}, chunk {cid+1}")
                        continue

                    # Debug printing
                    print(f"[DEBUG] line={idx+1}, chunk={cid+1}, speaker={speaker_str}, text={repr(chunk)}")

                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav_name = tmp_wav.name
                    tmp_wav.close()

                    try:
                        tts.tts_to_file(text=chunk, speaker=speaker_str, file_path=tmp_wav_name)
                        temp_files.append(tmp_wav_name)
                    except Exception as e:
                        print(f"[ERROR] TTS generation failed on line {idx+1}, chunk {cid+1}: {e}")
                        for tfp in temp_files:
                            if os.path.exists(tfp):
                                os.remove(tfp)
                        sys.exit(1)

            base_name = os.path.splitext(file_path)[0]
            output_wav = base_name + "_v6.wav"
            print("[INFO] Combining partial audio segments...")
            combine_wav_files(temp_files, output_wav)
            print(f"[INFO] Audio file generated: {output_wav}")

        finally:
            for tfp in temp_files:
                if os.path.exists(tfp):
                    os.remove(tfp)

        # Save updated voice map with any newly assigned speaker IDs
        save_voice_map(voice_map)

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
