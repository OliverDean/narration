#!/usr/bin/env python3
"""
Advanced Novel TTS (Version 10):
 - Reads entire .txt exactly, no rewriting or punctuation removal.
 - Runs fastcoref over full text for improved speaker detection.
 - Uses nested quote detection + offset-based speaker guessing.
 - Assigns lines to speaker from character_voices.json or newly generated.
 - Splits only for TTS chunking (max chars) but otherwise doesn't alter text.
 - Skips zero-length WAVs and checks wave format ignoring nframes & compname.

Usage:
  python novel_tts_v10.py path/to/story.txt --model_name tts_models/en/vctk/vits --gpu
"""

import argparse
import os
import sys
import wave
import re
import json
import tempfile
import spacy
import string
import re
from fastcoref import spacy_component
from TTS.api import TTS


VOICE_MAP_FILE = "character_voices_v10.json"


def load_synonyms_json(path="character_synonyms.json"):
    """Load synonyms for unify_character_name, or return empty if missing."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

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

def load_voice_map():
    if os.path.isfile(VOICE_MAP_FILE):
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_voice_map(voice_map):
    with open(VOICE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(voice_map, f, indent=2)

def essential_params(p):
    """Return (nchannels, sampwidth, framerate, comptype) ignoring nframes/compname."""
    return (p.nchannels, p.sampwidth, p.framerate, p.comptype)

def combine_wav_files(wav_files, output_file):
    if not wav_files:
        return

    with wave.open(wav_files[0], 'rb') as wf:
        base_params = wf.getparams()
        data_accum = wf.readframes(wf.getnframes())

    base_ess = essential_params(base_params)

    for wfile in wav_files[1:]:
        with wave.open(wfile, 'rb') as wf:
            cur = wf.getparams()
            if essential_params(cur) != base_ess:
                raise ValueError("WAV files differ in format (channels/rate/bits/comptype).")
            data_accum += wf.readframes(wf.getnframes())

    with wave.open(output_file, 'wb') as wf:
        wf.setparams(base_params)
        wf.writeframes(data_accum)

def find_dialogue_spans_nested(text: str):
    """
    Finds nested quotes while skipping mid-word apostrophes. That is,
    we only treat "'" as a quote if it is NOT immediately followed by a letter.

    """
    def is_mid_word_apostrophe(txt, idx):
        """
        Return True if this apostrophe is immediately followed by a letter,
        meaning it's likely a contraction, e.g. can't or didn't.
        """
        if idx + 1 < len(txt):
            next_char = txt[idx + 1]
            return next_char.isalpha()  # treat as mid-word if next is letter
        return False

    # We'll keep ' in OPEN/CLOSE quotes for single-quoted dialogue,
    # but skip it if it’s “mid-word”.
    OPEN_QUOTES = ["“", "‘", "\"", "'"] 
    CLOSE_QUOTES = ["”", "’", "\"", "'"]

    results = []
    stack = []
    in_quote = False
    start_pos = None
    quote_char = None

    for i, ch in enumerate(text):
        # Skip mid-word apostrophe
        if ch == "'" and is_mid_word_apostrophe(text, i):
            continue

        if ch in OPEN_QUOTES:
            if not in_quote:
                in_quote = True
                start_pos = i + 1
                quote_char = ch
            else:
                # We might have nested quotes
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

def chunk_text_for_tts(text: str, max_chars: int = 1000) -> list:
    """
    Break text into segments ~ max_chars, avoiding mid-word splits if possible.
    """
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
        seg = text[start:end].strip()
        start = end
        if seg:
            segments.append(seg)
    return segments

def setup_coref(model_architecture="FCoref", device="cpu"):
    nlp = spacy.load("en_core_web_sm", exclude=["parser","lemmatizer","ner","textcat"])

    if model_architecture.lower() == "fcoref":
        repo = "biu-nlp/f-coref"
    elif model_architecture.lower() == "lingmesscoref":
        repo = "biu-nlp/lingmess-coref"
    else:
        raise ValueError("Unknown coref model: " + model_architecture)

    nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": model_architecture,
            "model_path": repo,
            "device": device
        }
    )
    return nlp

def resolve_coref_text(nlp, text: str) -> str:
    """
    Return doc._.resolved_text from fastcoref but do not feed that to TTS. 
    We only use it for speaker detection offset logic.
    """
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    if hasattr(doc._, "resolved_text") and doc._.resolved_text:
        return doc._.resolved_text
    return text

def unify_character_name(raw_name: str, synonyms_map: dict) -> str:
    # synonyms_map is from "character_synonyms.json"
    # e.g. { "Lucy": ["Lucy Ellingson", ...], "Verona": [...], ...}
    candidate = raw_name.strip().title()
    if candidate in synonyms_map:
        return candidate
    for main_name, variants in synonyms_map.items():
        if candidate == main_name.title():
            return main_name
        for v in variants:
            if v.lower() in candidate.lower():
                return main_name
    return candidate

def guess_speaker_for_quote(quote_text: str, doc_text: str, doc_spacy, q_start: int, q_end: int, known_speakers, unify_map: dict):
    """
    Attempt to guess speaker near snippet offset by checking
    doc_spacy ents & "Name said" patterns in ~200 chars window.
    """

    # Find snippet offset
    pos = doc_text.find(quote_text)
    if pos < 0:
        pos = doc_text.find(quote_text[:10])  # fallback partial

    window = 200
    doc_len = len(doc_text)
    start_ctx = max(0, pos - window)
    end_ctx = min(doc_len, pos + len(quote_text) + window)
    context_str = doc_text[start_ctx:end_ctx]

    # simple "Name said" pattern
    speech_verbs = ["said","asked","replied","muttered","shouted","yelled",
                    "whispered","cried","called","snapped","answered"]
    for spk in known_speakers:
        for vb in speech_verbs:
            patt = rf"\b{spk}\b\s+{vb}\b"
            if re.search(patt, context_str, re.IGNORECASE):
                return spk

    # doc offset approach
    if q_start < 0 or q_end < 0:
        return None

    # find tokens in doc_spacy near q_start..q_end
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

        # check PERSON entities
        possible = set()
        for ent in doc_spacy.ents:
            if ent.label_ == "PERSON":
                if (t_start <= ent.start < t_end) or (t_start <= ent.end < t_end):
                    nm = unify_character_name(ent.text, unify_map)
                    possible.add(nm)

        # if exactly 1 known speaker -> pick it
        known_set = set(known_speakers)
        inter = possible.intersection(known_set)
        if len(inter) == 1:
            return inter.pop()

    return None

def main():
    parser = argparse.ArgumentParser(description="Novel TTS v10: advanced pipeline, exact text, quotes->speakers with doc offset.")
    parser.add_argument("input_file", help="Path to .txt")
    parser.add_argument("--batch", nargs="*", help="Multiple .txt files.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits", help="TTS model w/ multi-speaker.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--coref_model", default="FCoref", help="fastcoref: 'FCoref' or 'LingMessCoref'")
    parser.add_argument("--max_line_chars", type=int, default=1000, help="Chunk size for TTS.")
    parser.add_argument("--exclude_characters", nargs="*", default=[], help="Characters to skip or reassign.")
    parser.add_argument("--skip_excluded", action="store_true", help="If set, we skip lines from excluded chars.")
    args = parser.parse_args()

    # gather files
    if args.batch:
        files = args.batch
    else:
        files = [args.input_file]

    # load synonyms
    synonyms_map = load_synonyms_json("character_synonyms.json")

    # load TTS
    print(f"[INFO] Loading TTS model: {args.model_name}")
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print("[ERROR] TTS load failed:", e)
        sys.exit(1)

    # get speaker list ignoring "ED"
    try:
        raw_speakers = tts.speakers
        available_speakers = [s.strip() for s in raw_speakers if s.strip() and s.strip() != "ED"]
        if not available_speakers:
            raise ValueError("No valid speaker IDs found after removing 'ED'.")
    except AttributeError:
        print("[ERROR] Model doesn't support multiple speakers.")
        sys.exit(1)

    print("[INFO] TTS speakers (minus 'ED'):", available_speakers)

    # load or create voice_map
    voice_map = load_voice_map()

    # ensure we have a "Narrator" speaker
    speaker_index = 0
    def get_next_speaker_id():
        nonlocal speaker_index
        sid = available_speakers[speaker_index % len(available_speakers)]
        speaker_index += 1
        return sid

    if "Narrator" not in voice_map:
        new_id = get_next_speaker_id()
        voice_map["Narrator"] = {
            "speaker_id": new_id,
            "description": "Narrator / non-dialogue"
        }
        print(f"[INFO] Created Narrator -> {new_id}")
    narrator_info = voice_map["Narrator"]

    def get_or_create_speaker_entry(char_name: str):
        """Return a dict from voice_map or create new if absent."""
        if char_name in voice_map:
            return voice_map[char_name]
        sid = get_next_speaker_id()
        new_ent = {
            "speaker_id": sid,
            "description": f"Auto speaker for {char_name}"
        }
        voice_map[char_name] = new_ent
        return new_ent

    # we define a re to skip punctuation-only lines
    punct_only_re = re.compile(r'^[\W_]+$')

    # setup spacy + fastcoref
    device_str = "cuda:0" if args.gpu else "cpu"
    print("[INFO] Setting up spaCy + fastcoref model:", args.coref_model)
    nlp = setup_coref(args.coref_model, device=device_str)

    for fpath in files:
        if not os.path.isfile(fpath):
            print("[ERROR] File not found:", fpath)
            continue

        # read entire text
        with open(fpath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 1) full doc coref for speaker detection
        resolved_text = resolve_coref_text(nlp, raw_text)
        doc_spacy = nlp.make_doc(resolved_text)  # minimal doc for offset checks

        # 2) break into paragraphs from the original text (verbatim)
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        # We'll store final lines as (speaker_dict, exact_substring)
        final_lines = []
        current_offset = 0

        for paragraph in paragraphs:
            # find nested quotes in the original text
            spans = find_dialogue_spans_nested(paragraph)
            if not spans:
                # entire paragraph => narrator
                final_lines.append((narrator_info, paragraph))
                current_offset += len(paragraph) + 2
                continue

            last_end = 0
            for (quote_text, q_start, q_end) in spans:
                # narration chunk
                narration_str = paragraph[last_end:q_start]
                if narration_str.strip():
                    final_lines.append((narrator_info, narration_str))

                # guess speaker
                spkr = guess_speaker_for_quote(
                    quote_text=quote_text,
                    doc_text=resolved_text,
                    doc_spacy=doc_spacy,
                    q_start=current_offset + q_start,
                    q_end=current_offset + q_end,
                    known_speakers=list(voice_map.keys()),
                    unify_map=synonyms_map
                )
                if spkr is not None:
                    spkr_unified = spkr.strip().title()
                    if spkr_unified in args.exclude_characters:
                        if args.skip_excluded:
                            # skip
                            pass
                        else:
                            # reassign to narrator
                            final_lines.append((narrator_info, quote_text))
                    else:
                        # create or get from voice_map
                        sp_dict = get_or_create_speaker_entry(spkr_unified)
                        final_lines.append((sp_dict, quote_text))
                else:
                    # unknown => narrator
                    final_lines.append((narrator_info, quote_text))

                last_end = q_end

            # trailing
            if last_end < len(paragraph):
                trailing_str = paragraph[last_end:]
                if trailing_str.strip():
                    final_lines.append((narrator_info, trailing_str))

            current_offset += len(paragraph) + 2

        if not final_lines:
            print("[INFO] No lines to read in file:", fpath)
            continue

        # TTS generation
        temp_files = []
        first_params = None

        try:
            line_count = 0
            for (speaker_dict, text_chunk) in final_lines:
                line_count += 1
                # chunk for TTS
                segments = chunk_text_for_tts(text_chunk, args.max_line_chars)
                for cid, seg in enumerate(segments, start=1):
                    s_stripped = seg.strip()
                    if not s_stripped:
                        print(f"[SKIP] Empty chunk at line {line_count}, chunk {cid}")
                        continue
                    if punct_only_re.match(s_stripped):
                        print(f"[SKIP] Punctuation-only chunk at line {line_count}, chunk {cid}: {repr(seg)}")
                        continue

                    # no sanitizing => EXACT text
                    speaker_id = speaker_dict["speaker_id"]
                    print(f"[DEBUG] line={line_count}, chunk={cid}, speaker={speaker_id}, text={repr(s_stripped)}")

                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav_name = tmp_wav.name
                    tmp_wav.close()

                    try:
                        tts.tts_to_file(text=s_stripped, speaker=speaker_id, file_path=tmp_wav_name)
                        with wave.open(tmp_wav_name, "rb") as wf:
                            n_frames = wf.getnframes()
                            cparams = wf.getparams()
                            print("[PARAMS]", cparams)
                        if n_frames == 0:
                            print(f"[SKIP] Zero-length chunk at line={line_count}, chunk={cid}")
                            os.remove(tmp_wav_name)
                            continue
                        # compare ignoring nframes, compname
                        param_tuple = (cparams.nchannels, cparams.sampwidth, cparams.framerate, cparams.comptype)
                        if first_params is None:
                            first_params = param_tuple
                        else:
                            if param_tuple != first_params:
                                print(f"[SKIP] Mismatch format ignoring length at line={line_count}, chunk={cid}")
                                os.remove(tmp_wav_name)
                                continue

                        temp_files.append(tmp_wav_name)

                    except Exception as e:
                        print(f"[ERROR] TTS generation failed at line={line_count}, chunk={cid}: {e}")
                        for tfp in temp_files:
                            if os.path.exists(tfp):
                                os.remove(tfp)
                        sys.exit(1)

            if temp_files:
                base_name = os.path.splitext(fpath)[0]
                output_wav = base_name + "_v10.wav"
                print("[INFO] Combining partial audio segments...")
                combine_wav_files(temp_files, output_wav)
                print(f"[INFO] Audio file generated:", output_wav)
            else:
                print("[INFO] No valid WAV chunks. Possibly all skipped?")

        finally:
            for tfp in temp_files:
                if os.path.exists(tfp):
                    os.remove(tfp)

        # save voice map with any new speakers
        save_voice_map(voice_map)

    print("[INFO] All done. Version 10 complete.")

if __name__ == "__main__":
    main()
