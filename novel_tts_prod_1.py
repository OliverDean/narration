#!/usr/bin/env python3
"""
Main script to read a text, detect person entities, assign voices, generate TTS audio,
and produce a final MP3. Uses spaCy for NER & coreference, Coqui TTS for audio,
and pydub for MP3.
"""

import argparse
import os
import sys
import math
import json
import re
import wave
import tempfile
import spacy
from typing import Dict, List, Tuple
from fastcoref import spacy_component
from rapidfuzz.fuzz import ratio
from pydub import AudioSegment
from TTS.api import TTS

MALE_SPEAKERS = [
    "p226", "p228", "p229", "p230", "p231", "p232", "p233", "p234",
    "p234", "p236", "p238", "p239", "p241", "p251", "p253", "p254",
    "p255", "p256", "p257", "p258", "p262", "p264", "p265", "p266",
    "p267", "p269", "p272", "p279", "p281", "p282", "p285", "p286",
    "p287", "p299", "p301", "p302", "p307", "p312", "p313", "p317",
    "p318", "p326", "p330", "p340", "p351", "p376",
]
FEMALE_SPEAKERS = [
    "p225", "p227", "p237", "p240", "p243", "p244", "p245", "p246",
    "p247", "p248", "p249", "p250", "p252", "p259", "p260", "p261",
    "p263", "p268", "p270", "p271", "p273", "p274", "p275", "p276",
    "p277", "p278", "p280", "p283", "p284", "p292", "p293", "p294",
    "p295", "p297", "p300", "p303", "p304", "p305", "p308", "p310",
    "p311", "p314", "p316", "p323", "p329", "p333", "p334", "p335",
    "p336", "p339", "p341", "p343", "p345", "p347", "p360", "p361",
    "p362", "p363", "p364", "p374",
]
DEFAULT_STOPNAMES = {
    "mom", "dad", "mother", "father", "grandma",
    "grandfather", "sir", "ma'am", "mr", "lady", "gentleman"
}

# Set ffmpeg paths for AudioSegment
AudioSegment.converter = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"

def load_json_file(file_path: str):
    """
    Load JSON data from the specified file path.
    Returns a Python object or an empty dictionary if file not found.
    """
    return json.load(open(file_path, "r", encoding="utf-8")) if os.path.isfile(file_path) else {}

def save_json_file(data, file_path: str):
    """
    Save the given data as JSON to the specified file path.
    """
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)

def remove_blacklisted_names(voices_map: Dict, synonyms_map: Dict, user_blacklist_file: str = None):
    """
    Remove blacklisted names from the given voices map and synonyms map,
    merging with default stop names if a user blacklist file is provided.
    """
    blacklist_set = {x.lower() for x in DEFAULT_STOPNAMES}
    if user_blacklist_file and os.path.isfile(user_blacklist_file):
        with open(user_blacklist_file, "r", encoding="utf-8") as file_handle:
            for stop_name in json.load(file_handle):
                blacklist_set.add(stop_name.lower())

    remove_keys = [k for k in voices_map if k.lower() in blacklist_set]
    for k in remove_keys:
        voices_map.pop(k, None)
        synonyms_map.pop(k, None)

    for key, values in synonyms_map.items():
        synonyms_map[key] = [val for val in values if val.lower() not in blacklist_set]

def clean_variant_text(text: str) -> str:
    """
    Remove trailing punctuation and possessives from the given text, returning the cleaned variant.
    """
    text = re.sub(r"[.,!?;:\"]+$", "", text.strip())
    text = re.sub(r"[’']s\b", "", text, flags=re.IGNORECASE)
    return text.strip()

def unify_synonyms_in_place(synonyms_map: Dict, voices_map: Dict, threshold: int = 80):
    """
    Unify synonyms in place, merging duplicates for entries that 
    exceed a certain fuzzy match threshold (default=80).
    """
    for canonical, variants in synonyms_map.items():
        temp_list = []
        for candidate in variants:
            cleaned_candidate = clean_variant_text(candidate)
            merged = False
            for i, existing in enumerate(temp_list):
                if ratio(cleaned_candidate.lower(), existing.lower()) >= threshold:
                    # Keep the shorter string
                    if len(cleaned_candidate) < len(existing):
                        temp_list[i] = cleaned_candidate
                    merged = True
                    break
            if not merged:
                temp_list.append(cleaned_candidate)
        synonyms_map[canonical] = temp_list

    canonical_list = list(synonyms_map.keys())
    changed = True
    while changed:
        changed = False
        length = len(canonical_list)
        i = 0
        while i < length - 1:
            j = i + 1
            merged_any = False
            while j < length:
                a = canonical_list[i]
                b = canonical_list[j]
                if ratio(a.lower(), b.lower()) >= threshold:
                    for val_b in synonyms_map[b]:
                        if val_b not in synonyms_map[a]:
                            synonyms_map[a].append(val_b)
                    synonyms_map.pop(b, None)
                    voices_map.pop(b, None)
                    canonical_list.remove(b)
                    length = len(canonical_list)
                    changed = True
                    merged_any = True
                    break
                else:
                    j += 1
            if not merged_any:
                i += 1

def get_gender_from_coreference(entity_span, doc) -> str:
    """
    Determine likely gender ("male"/"female") from coreference data 
    in the entity's sentence context. Returns None if unknown.
    """
    male_count, female_count = 0, 0
    male_set = {"he", "him", "his"}
    female_set = {"she", "her", "hers"}

    for token in entity_span.sent:
        lower_word = token.lower_
        if lower_word in male_set:
            male_count += 1
        elif lower_word in female_set:
            female_count += 1

    if male_count > female_count and male_count > 0:
        return "male"
    elif female_count > male_count and female_count > 0:
        return "female"
    return None

def get_next_speaker_id(gender: str, male_index: List[int], female_index: List[int]) -> str:
    """
    Get the next speaker ID, rotating through the male or female 
    speaker lists based on detected gender. Defaults to male if unknown.
    """
    if gender == "male":
        speaker = MALE_SPEAKERS[male_index[0] % len(MALE_SPEAKERS)]
        male_index[0] += 1
        return speaker
    elif gender == "female":
        speaker = FEMALE_SPEAKERS[female_index[0] % len(FEMALE_SPEAKERS)]
        female_index[0] += 1
        return speaker
    else:
        speaker = MALE_SPEAKERS[male_index[0] % len(MALE_SPEAKERS)]
        male_index[0] += 1
        return speaker

def process_text_entities(
    text: str,
    doc_ner,
    doc_coref,
    voices_map: Dict,
    synonyms_map: Dict,
    male_index: List[int],
    female_index: List[int],
    threshold: int
):
    """
    Scan the text for PERSON entities, assign new voices if needed, 
    update synonyms map, and track how many new entries were created/updated.
    Returns (new_characters_count, new_synonyms_count).
    """
    characters_set = set(voices_map.keys())
    new_characters_count = 0
    new_synonyms_count = 0

    for ent in doc_ner.ents:
        if ent.label_ == "PERSON":
            name = clean_variant_text(ent.text)
            if len(name) < 2:
                continue
            if name not in voices_map:
                gender = get_gender_from_coreference(ent, doc_ner)
                voices_map[name] = {
                    "speaker_id": get_next_speaker_id(gender, male_index, female_index),
                    "gender": gender if gender else "unknown",
                    "desc": "auto"
                }
                synonyms_map.setdefault(name, [])
                characters_set.add(name)
                new_characters_count += 1

            best_score = 0
            best_candidate = None
            for candidate in characters_set:
                score = ratio(name.lower(), candidate.lower())
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_score >= threshold and best_candidate:
                if name not in synonyms_map[best_candidate]:
                    synonyms_map[best_candidate].append(name)
                    new_synonyms_count += 1

    return new_characters_count, new_synonyms_count

def get_wave_essential_params(wave_handle: wave.Wave_read):
    """
    Get essential wave parameters such as 
    channels, sample width, frame rate, and compression type.
    """
    return (
        wave_handle.getnchannels(),
        wave_handle.getsampwidth(),
        wave_handle.getframerate(),
        wave_handle.getcomptype()
    )

def combine_wav_files(wav_paths: List[str], output_file: str):
    """
    Combine multiple WAV files into a single WAV file, 
    ensuring their parameters match.
    """
    if not wav_paths:
        return

    combined_data = b""
    with wave.open(wav_paths[0], "rb") as w1:
        params = w1.getparams()
        combined_data += w1.readframes(w1.getnframes())

    base_params = (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype
    )

    for wav_file in wav_paths[1:]:
        with wave.open(wav_file, "rb") as w2:
            cp = w2.getparams()
            current_params = (
                cp.nchannels,
                cp.sampwidth,
                cp.framerate,
                cp.comptype
            )
            if base_params != current_params:
                raise ValueError("WAV file parameter mismatch.")
            combined_data += w2.readframes(w2.getnframes())

    with wave.open(output_file, "wb") as wout:
        wout.setparams(params)
        wout.writeframes(combined_data)

def compress_wav_to_mp3(input_file: str, output_file: str, bitrate: str = "192k"):
    """
    Convert a WAV file to an MP3 file at a specified bitrate using pydub.
    """
    sound = AudioSegment.from_wav(input_file)
    sound.export(output_file, format="mp3", bitrate=bitrate)

def chunk_text_for_tts(text: str, max_chars: int = 1000) -> List[str]:
    """
    Split the text into chunks of up to 'max_chars', 
    preferably splitting on spaces to avoid truncating words.
    """
    text = text.strip()
    segments = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            space_break = text.rfind(" ", start, end)
            if space_break > start:
                end = space_break
        chunk = text[start:end].strip()
        start = end
        if chunk:
            segments.append(chunk)
    return segments

def segment_text_by_line(text: str, voices_map: Dict) -> List[Tuple[str, str]]:
    """
    Split the input text into segments by line (or paragraph), preserving the text unchanged.
    For each non-empty line, attempt to detect a speaker attribution using a regex.
    If a speaker is found (e.g. a line like:
        “Yes?” Verona asked Avery.
    the regex captures "Verona"), that speaker is used.
    Otherwise, the segment is assigned to "Narrator".
    Returns a list of (speaker, segment_text) tuples.
    """
    segments = []
    lines = text.splitlines()
    # This regex looks for a dialogue in quotes followed by a speaker attribution.
    speaker_pattern = re.compile(r'“.*?”\s+([A-Z][a-z]+)\s+(said|asked|answered|conceded)')
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        match = speaker_pattern.search(stripped_line)
        if match:
            speaker = match.group(1)
            if speaker not in voices_map:
                speaker = "Narrator"
        else:
            speaker = "Narrator"
        segments.append((speaker, line))
    return segments

def run_pipeline(
    text_file_path: str,
    voice_map_path: str,
    synonyms_map_path: str,
    blacklist_file_path: str,
    coreference_model: str = "FCoref",
    fuzzy_threshold: int = 80,
    chunk_size: int = 0,
    tts_model_name: str = "tts_models/en/vctk/vits",
    use_gpu: bool = False,
    max_line_chars: int = 1000,
    bitrate: str = "192k"
):
    """
    Main pipeline to read text, detect person entities, assign voices, generate TTS,
    and produce a final MP3 file.
    """
    if not os.path.isfile(text_file_path):
        print("[ERR] text not found:", text_file_path)
        return

    voices_map = load_json_file(voice_map_path)
    synonyms_map = load_json_file(synonyms_map_path)
    remove_blacklisted_names(voices_map, synonyms_map, blacklist_file_path)

    # Ensure every existing voice entry has a synonyms list
    for character_name in list(voices_map.keys()):
        if character_name not in synonyms_map:
            synonyms_map[character_name] = []

    with open(text_file_path, "r", encoding="utf-8") as file_handle:
        full_text = file_handle.read()

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("fastcoref", config={"model_architecture": coreference_model, "device": "cpu"})
    nlp_ner = spacy.load("en_core_web_sm")

    try:
        tts_engine = TTS(model_name=tts_model_name, gpu=use_gpu)
    except Exception as exc:
        print("[ERROR] TTS load fail:", exc)
        sys.exit(1)

    available_speakers = [x.strip() for x in tts_engine.speakers if x.strip() and x.strip() != "ED"]
    if not available_speakers:
        print("[ERR] No valid speaker IDs after removing ED.")
        return

    # Simple speaker rotation for narrator if needed
    speaker_index = [0]
    def get_next_narrator_speaker():
        speaker_id = available_speakers[speaker_index[0] % len(available_speakers)]
        speaker_index[0] += 1
        return speaker_id

    # Ensure a narrator entry exists
    if "Narrator" not in voices_map:
        narrator_speaker = get_next_narrator_speaker()
        voices_map["Narrator"] = {"speaker_id": narrator_speaker, "desc": "Narrator"}

    steps = 1
    if chunk_size > 0:
        steps = math.ceil(len(full_text) / chunk_size)

    male_index = [0]
    female_index = [0]
    total_new_characters = 0
    total_new_synonyms = 0

    # Process text in chunks for entity detection
    for i in range(steps):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(full_text)) if chunk_size > 0 else len(full_text)
        partial_text = full_text[start_index:end_index]

        doc_core = nlp(partial_text)
        doc_ner_data = nlp_ner(partial_text)
        new_chars, new_syns = process_text_entities(
            partial_text,
            doc_ner_data,
            doc_core,
            voices_map,
            synonyms_map,
            male_index,
            female_index,
            fuzzy_threshold
        )
        total_new_characters += new_chars
        total_new_synonyms += new_syns

    # Finalize synonyms and remove any newly blacklisted names
    unify_synonyms_in_place(synonyms_map, voices_map, fuzzy_threshold)
    remove_blacklisted_names(voices_map, synonyms_map, blacklist_file_path)

    # Save updated voices & synonyms
    save_json_file(voices_map, voice_map_path)
    save_json_file(synonyms_map, synonyms_map_path)

    # Split the text by lines and use each line unchanged.
    segments = segment_text_by_line(full_text, voices_map)
    temporary_files = []
    first_wave_params = None

    # For each segment (line), generate TTS using the detected speaker.
    for speaker, segment_text in segments:
        chunks = chunk_text_for_tts(segment_text, max_line_chars)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Get speaker ID from voices_map; fallback to Narrator if not found.
            speaker_entry = voices_map.get(speaker, voices_map["Narrator"])
            speaker_id = speaker_entry["speaker_id"]

            temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_path = temp_wav_file.name
            temp_wav_file.close()

            try:
                tts_engine.tts_to_file(
                    text=chunk,
                    speaker=speaker_id,
                    file_path=temp_wav_path
                )

                with wave.open(temp_wav_path, "rb") as wav_file_handle:
                    num_frames = wav_file_handle.getnframes()
                    wave_params = wav_file_handle.getparams()

                if num_frames == 0:
                    os.remove(temp_wav_path)
                    continue

                current_wave_params = (
                    wave_params.nchannels,
                    wave_params.sampwidth,
                    wave_params.framerate,
                    wave_params.comptype
                )

                if first_wave_params is None:
                    first_wave_params = current_wave_params
                else:
                    if current_wave_params != first_wave_params:
                        os.remove(temp_wav_path)
                        continue

                temporary_files.append(temp_wav_path)

            except Exception as exc:
                # Cleanup and exit on error
                for tmp_f in temporary_files:
                    if os.path.exists(tmp_f):
                        os.remove(tmp_f)
                print("[ERR] TTS fail:", exc)
                sys.exit(1)

    # Combine all WAV segments and export to MP3
    if temporary_files:
        base_name = os.path.splitext(text_file_path)[0]
        final_wav_file = base_name + "_final.wav"
        combine_wav_files(temporary_files, final_wav_file)

        final_mp3_file = base_name + "_final.mp3"
        compress_wav_to_mp3(final_wav_file, final_mp3_file, bitrate=bitrate)
        os.remove(final_wav_file)

        print("[INFO] Created mp3:", final_mp3_file)
    else:
        print("[WARN] No audio segments were generated.")

    # Cleanup temporary WAV files
    for temp_path in temporary_files:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"[INFO] Done. Found {total_new_characters} new characters, {total_new_synonyms} new synonyms.")

def main():
    """
    Parse command-line arguments and run the main pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file")
    parser.add_argument("--voice_map", default="character_voices.json")
    parser.add_argument("--synonyms", default="character_synonyms.json")
    parser.add_argument("--blacklist", default="black_list.json")
    parser.add_argument("--coref_model", default="FCoref")
    parser.add_argument("--fuzzy_thresh", type=int, default=80)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max_line_chars", type=int, default=1000)
    parser.add_argument("--bitrate", default="192k")
    args = parser.parse_args()

    run_pipeline(
        text_file_path=args.text_file,
        voice_map_path=args.voice_map,
        synonyms_map_path=args.synonyms,
        blacklist_file_path=args.blacklist,
        coreference_model=args.coref_model,
        fuzzy_threshold=args.fuzzy_thresh,
        chunk_size=args.chunk_size,
        tts_model_name=args.model_name,
        use_gpu=args.gpu,
        max_line_chars=args.max_line_chars,
        bitrate=args.bitrate
    )

if __name__ == "__main__":
    main()
