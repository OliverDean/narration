#!/usr/bin/env python3
"""
Advanced Novel TTS (Version 10) + MP3 Compression:
 - Reads entire .txt exactly, ignoring mid-word apostrophes in quotes.
 - Uses fastcoref for improved speaker detection.
 - Splits text for TTS, generates PCM WAV partials, then combines them.
 - Converts final combined WAV to MP3 (via pydub + ffmpeg).
 - Saves updated speakers in character_voices_v10.json if any new are created.

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
from pydub import AudioSegment  # for mp3 compression
from pydub.utils import mediainfo, which
from fastcoref import spacy_component
from TTS.api import TTS

VOICE_MAP_FILE = "character_voices_v10.json"

AudioSegment.converter = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"

AudioSegment.ffmpeg = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"

##############################################################################
# 1) Helper: Load synonyms, voice map
##############################################################################

def load_synonyms_json(path="character_synonyms.json"):
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

##############################################################################
# 2) Quoting, chunking
##############################################################################

def is_mid_word_apostrophe(txt, idx):
    """
    Return True if the apostrophe at txt[idx] is followed by a letter,
    e.g. "can't", "didn't", so we skip it as a quote delimiter.
    """
    if idx+1 < len(txt):
        return txt[idx+1].isalpha()
    return False

def find_dialogue_spans_nested(text: str):
    """
    Finds nested quotes while skipping mid-word apostrophes.
    We only treat "'" as a quote if NOT mid-word.
    """
    OPEN_QUOTES = ["“", "‘", "\"", "'"]
    CLOSE_QUOTES = ["”", "’", "\"", "'"]

    results = []
    stack = []
    in_quote = False
    start_pos = None
    quote_char = None

    for i, ch in enumerate(text):
        # Skip mid-word ' as quote delimiter
        if ch == "'" and is_mid_word_apostrophe(text, i):
            continue

        if ch in OPEN_QUOTES:
            if not in_quote:
                in_quote = True
                start_pos = i+1
                quote_char = ch
            else:
                stack.append((quote_char, start_pos))
                start_pos = i+1
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

def chunk_text_for_tts(text: str, max_chars: int = 1000):
    """
    Break text into ~max_chars segments, avoiding mid-word splits if possible.
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

##############################################################################
# 3) fastcoref + spaCy
##############################################################################

def setup_coref(model_architecture="FCoref", device="cpu"):
    import spacy
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

##############################################################################
# 4) Speaker detection
##############################################################################

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

##############################################################################
# 5) Combining Partial WAV + Converting to MP3
##############################################################################

def essential_params(p):
    return (p.nchannels, p.sampwidth, p.framerate, p.comptype)

def combine_wav_files(input_wavs, output_wav):
    """
    Combine multiple PCM WAV files with matching format into one WAV.
    """
    import wave
    if not input_wavs:
        return
    with wave.open(input_wavs[0], 'rb') as wf:
        params = wf.getparams()
        combined_data = wf.readframes(wf.getnframes())

    base_ess = essential_params(params)

    for wfile in input_wavs[1:]:
        with wave.open(wfile, 'rb') as wf:
            cur = wf.getparams()
            if essential_params(cur)!=base_ess:
                raise ValueError("Mismatch wave params.")
            combined_data += wf.readframes(wf.getnframes())

    with wave.open(output_wav, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(combined_data)

def compress_to_mp3(input_wav, output_mp3, bitrate="192k"):
    """
    Use pydub + ffmpeg to compress the WAV into MP3 with the given bitrate.
    """
    sound = AudioSegment.from_wav(input_wav)
    sound.export(output_mp3, format="mp3", bitrate=bitrate)

##############################################################################
# 6) Main
##############################################################################

def main():
    import tempfile
    parser = argparse.ArgumentParser(description="Version 10 + pydub mp3 compression.")
    parser.add_argument("input_file", help="Path to .txt")
    parser.add_argument("--batch", nargs="*", help="Multiple .txt files.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits", help="TTS model w/ multi-speaker.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--coref_model", default="FCoref", help="fastcoref: 'FCoref' or 'LingMessCoref'")
    parser.add_argument("--max_line_chars", type=int, default=1000, help="Chunk size for TTS.")
    parser.add_argument("--exclude_characters", nargs="*", default=[], help="Characters to skip or reassign.")
    parser.add_argument("--skip_excluded", action="store_true", help="If set, we skip lines from excluded chars.")
    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (default 192k)")
    args = parser.parse_args()

    if args.batch:
        files = args.batch
    else:
        files = [args.input_file]

    synonyms_map = load_synonyms_json("character_synonyms.json")
    
    voice_map = load_voice_map()

    print("[INFO] Loading TTS model:", args.model_name)
    from TTS.api import TTS
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print("[ERROR] TTS load failed:", e)
        sys.exit(1)

    # filter out "ED"
    try:
        raw_speakers = tts.speakers
        available_speakers = [s.strip() for s in raw_speakers if s.strip() and s.strip()!="ED"]
        if not available_speakers:
            raise ValueError("No valid speaker IDs found after removing ED.")
    except AttributeError:
        print("[ERROR] Model doesn't support multi-speakers.")
        sys.exit(1)

    print("[INFO] TTS speakers:", available_speakers)

    # ensure narrator
    spk_index = 0
    def get_next_speaker_id():
        nonlocal spk_index
        sid = available_speakers[spk_index % len(available_speakers)]
        spk_index += 1
        return sid

    if "Narrator" not in voice_map:
        nid = get_next_speaker_id()
        voice_map["Narrator"] = {"speaker_id": nid, "description":"Narrator / non-dialogue"}
        print("[INFO] Created Narrator ->", nid)
    narrator_info = voice_map["Narrator"]

    def get_or_create_speaker_entry(char_name: str):
        if char_name in voice_map:
            return voice_map[char_name]
        new_id = get_next_speaker_id()
        new_ent = {
            "speaker_id": new_id,
            "description": f"Auto speaker for {char_name}"
        }
        voice_map[char_name] = new_ent
        return new_ent

    # skip punctuation-only lines
    punct_only_re = re.compile(r'^[\W_]+$')

    print("[INFO] Setting up spaCy + fastcoref:", args.coref_model)
    nlp = setup_coref(args.coref_model, device="cuda:0" if args.gpu else "cpu")

    for fpath in files:
        if not os.path.isfile(fpath):
            print("[ERROR] File not found:", fpath)
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # use resolved text only for offset, not for TTS
        resolved_text = resolve_coref_text(nlp, raw_text)
        doc_spacy = nlp.make_doc(resolved_text)

        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        final_lines = []
        current_offset = 0

        for paragraph in paragraphs:
            spans = find_dialogue_spans_nested(paragraph)
            if not spans:
                final_lines.append((narrator_info, paragraph))
                current_offset += len(paragraph)+2
                continue

            last_end = 0
            for (quote_text, q_start, q_end) in spans:
                narration_str = paragraph[last_end:q_start]
                if narration_str.strip():
                    final_lines.append((narrator_info, narration_str))

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
                            pass
                        else:
                            final_lines.append((narrator_info, quote_text))
                    else:
                        sp_dict = get_or_create_speaker_entry(spkr_unified)
                        final_lines.append((sp_dict, quote_text))
                else:
                    final_lines.append((narrator_info, quote_text))
                last_end = q_end

            if last_end<len(paragraph):
                trailing_str = paragraph[last_end:]
                if trailing_str.strip():
                    final_lines.append((narrator_info, trailing_str))
            current_offset += len(paragraph)+2

        if not final_lines:
            print("[INFO] No lines to read in file:", fpath)
            continue

        temp_files = []
        first_params = None

        try:
            line_count=0
            for (speaker_dict, text_chunk) in final_lines:
                line_count+=1
                segments = chunk_text_for_tts(text_chunk, args.max_line_chars)
                for cid, seg in enumerate(segments, start=1):
                    seg_strip = seg.strip()
                    if not seg_strip:
                        print(f"[SKIP] Empty chunk at line {line_count}, chunk {cid}")
                        continue
                    if punct_only_re.match(seg_strip):
                        print(f"[SKIP] Punctuation-only chunk at line {line_count}, chunk {cid}: {repr(seg)}")
                        continue

                    spkid = speaker_dict["speaker_id"]
                    print(f"[DEBUG] line={line_count}, chunk={cid}, speaker={spkid}, text={repr(seg_strip)}")

                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav_name = tmp_wav.name
                    tmp_wav.close()

                    try:
                        tts.tts_to_file(text=seg_strip, speaker=spkid, file_path=tmp_wav_name)
                        with wave.open(tmp_wav_name, "rb") as wf:
                            n_frames = wf.getnframes()
                            cparams = wf.getparams()
                            print("[PARAMS]", cparams)
                        if n_frames==0:
                            print(f"[SKIP] Zero-length at line={line_count}, chunk={cid}")
                            os.remove(tmp_wav_name)
                            continue
                        param_tuple = (cparams.nchannels, cparams.sampwidth, cparams.framerate, cparams.comptype)
                        if first_params is None:
                            first_params = param_tuple
                        else:
                            if param_tuple!=first_params:
                                print(f"[SKIP] Mismatch format ignoring length line={line_count}, chunk={cid}")
                                os.remove(tmp_wav_name)
                                continue
                        temp_files.append(tmp_wav_name)

                    except Exception as e:
                        print(f"[ERROR] TTS gen fail line={line_count}, chunk={cid} -> {e}")
                        for tf in temp_files:
                            if os.path.exists(tf):
                                os.remove(tf)
                        sys.exit(1)

            # Combine partial WAVs -> final .wav, then compress -> mp3
            if temp_files:
                base_name = os.path.splitext(fpath)[0]
                final_wav = base_name + "_v10.wav"
                print("[INFO] Combining partial segments ->", final_wav)
                combine_wav_files(temp_files, final_wav)

                # Now compress to MP3
                mp3_output = base_name + "_v10.mp3"
                print("[INFO] Compressing to mp3 ->", mp3_output)
                compress_to_mp3(final_wav, mp3_output, bitrate=args.bitrate)

                # Remove big PCM WAV
                os.remove(final_wav)
                print("[INFO] MP3 file generated:", mp3_output)
            else:
                print("[INFO] No valid WAV chunks. Possibly all skipped?")

        finally:
            # remove partial chunk files
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)

        # Save voice map with new speakers
        save_voice_map(voice_map)

    print("[INFO] All done. Generated MP3 versions of your TTS output.")

if __name__=="__main__":
    main()
