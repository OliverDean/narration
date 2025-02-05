#!/usr/bin/env python3
"""
Audiobook Narration Application

This script reads one or more text files (or a directory of text files), detects person entities,
assigns voices, generates TTS audio (using Coqui TTS), and produces a final MP3 for each input text.
It leverages advanced logging, error handling, resource management, and caching.
If the required JSON files (voice map, synonyms, blacklist) do not exist, the script will create them.
A new command-line option (--skip_coref) lets you disable coreference-based gender detection,
but by default an optimized version is used.
"""

import argparse
import os
import sys
import math
import json
import re
import wave
import tempfile
import logging
import shutil
import hashlib
from typing import Dict, List, Tuple, Optional
import spacy
from fastcoref import spacy_component
from rapidfuzz.fuzz import ratio
from pydub import AudioSegment
from TTS.api import TTS

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants for pronouns
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}

# Set default ffmpeg path from environment variable (if provided)
DEFAULT_FFMPEG_PATH = os.environ.get("FFMPEG_PATH", r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe")
AudioSegment.converter = DEFAULT_FFMPEG_PATH
AudioSegment.ffmpeg = DEFAULT_FFMPEG_PATH

# Global speaker lists
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

def load_json_file(file_path: str) -> Dict:
    """Load JSON data from the specified file."""
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return {}
    return {}

def save_json_file(data: Dict, file_path: str) -> None:
    """Save the given data as JSON to the specified file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")

def ensure_file_exists(file_path: str, default_content) -> None:
    """If file_path does not exist, create it with the given default content."""
    if not os.path.exists(file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(default_content, f, indent=2)
            logger.info(f"Created default file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")

def remove_blacklisted_names(voices_map: Dict, synonyms_map: Dict, user_blacklist_file: Optional[str] = None) -> None:
    """
    Remove names that are in the default or user-provided blacklist from the voices and synonyms maps.
    This ensures that any character found in the blacklist will not be updated or added.
    """
    blacklist_set = {x.lower() for x in DEFAULT_STOPNAMES}
    if user_blacklist_file and os.path.isfile(user_blacklist_file):
        try:
            with open(user_blacklist_file, "r", encoding="utf-8") as f:
                user_blacklist = json.load(f)
                for stop_name in user_blacklist:
                    blacklist_set.add(stop_name.lower())
        except Exception as e:
            logger.error(f"Error reading blacklist file {user_blacklist_file}: {e}")
    # Remove any keys (character names) that appear in the blacklist.
    remove_keys = [k for k in voices_map if k.lower() in blacklist_set]
    for k in remove_keys:
        voices_map.pop(k, None)
        synonyms_map.pop(k, None)
    # Also, remove any synonyms that match a blacklisted name.
    for key, values in synonyms_map.items():
        synonyms_map[key] = [val for val in values if val.lower() not in blacklist_set]

def clean_variant_text(text: str) -> str:
    """Remove trailing punctuation and possessives from the text."""
    text = re.sub(r"[.,!?;:\"]+$", "", text.strip())
    text = re.sub(r"[’']s\b", "", text, flags=re.IGNORECASE)
    return text.strip()

def clean_tts_input(text: str) -> str:
    """Clean the TTS input by removing extraneous leading/trailing non-word characters."""
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', text)
    return cleaned.strip()

def unify_synonyms_in_place(synonyms_map: Dict, voices_map: Dict, threshold: int = 80) -> None:
    """Unify synonyms in place by merging duplicates based on fuzzy matching."""
    for canonical, variants in synonyms_map.items():
        temp_list = []
        for candidate in variants:
            cleaned_candidate = clean_variant_text(candidate)
            merged = False
            for i, existing in enumerate(temp_list):
                if ratio(cleaned_candidate.lower(), existing.lower()) >= threshold:
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

# --- New Helper: Build Coreference Mention Map ---

def build_coref_mention_map(doc) -> Dict[Tuple[int, int], object]:
    """
    Build and return a mapping from (start, end) boundaries to their coreference cluster.
    This is built once per document for fast lookup.
    """
    mapping: Dict[Tuple[int, int], object] = {}
    if hasattr(doc._, "coref_clusters") and doc._.coref_clusters is not None:
        for cluster in doc._.coref_clusters:
            mentions = getattr(cluster, "mentions", cluster)
            for mention in mentions:
                if hasattr(mention, "start") and hasattr(mention, "end"):
                    mapping[(mention.start, mention.end)] = cluster
                elif isinstance(mention, tuple) and len(mention) == 2:
                    mapping[(mention[0], mention[1])] = cluster
    return mapping

def get_gender_from_coreference(entity_span, doc, coref_mention_map: Optional[Dict[Tuple[int,int], object]] = None) -> Optional[str]:
    """
    Determine the likely gender ("male" or "female") for an entity using coreference data.
    This version uses a pre-built mapping for efficiency. If the entity's span is found in the map,
    it counts pronouns in the entire cluster; otherwise it falls back to the sentence context.
    """
    if coref_mention_map is None:
        coref_mention_map = build_coref_mention_map(doc)
    male_count = 0
    female_count = 0
    cluster = coref_mention_map.get((entity_span.start, entity_span.end))
    if cluster is not None:
        mentions = getattr(cluster, "mentions", cluster)
        for mention in mentions:
            if hasattr(mention, "start") and hasattr(mention, "end"):
                m_span = mention
            elif isinstance(mention, tuple) and len(mention) == 2:
                m_span = doc[mention[0]:mention[1]]
            else:
                continue
            for token in m_span:
                token_lower = token.text.lower()
                if token_lower in MALE_PRONOUNS:
                    male_count += 1
                elif token_lower in FEMALE_PRONOUNS:
                    female_count += 1
        if male_count or female_count:
            return "male" if male_count > female_count else "female"
    # Fallback: check pronouns in the sentence containing the entity
    for token in entity_span.sent:
        token_lower = token.text.lower()
        if token_lower in MALE_PRONOUNS:
            male_count += 1
        elif token_lower in FEMALE_PRONOUNS:
            female_count += 1
    if male_count or female_count:
        return "male" if male_count > female_count else "female"
    return None

def get_next_speaker_id(gender: Optional[str], male_index: List[int], female_index: List[int]) -> str:
    """Rotate through speaker lists based on gender, defaulting to male if unknown."""
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

def is_valid_character_name(name: str) -> bool:
    """Check if the name is a valid character name based on common patterns."""
    name = name.strip()
    if len(name) < 3:
        return False
    if not any(ch.isupper() for ch in name):
        return False
    if not any(ch.isalpha() for ch in name):
        return False
    common_non_character_words = {"ahh", "um", "uh", "hey", "hmm", "mmh", "wall", "door"}
    if name.lower() in common_non_character_words:
        return False
    tokens = name.split()
    if any(token and not token[0].isupper() for token in tokens):
        return False
    return True

def process_text_entities(
    text: str,
    doc_ner,
    doc_coref,
    voices_map: Dict,
    synonyms_map: Dict,
    male_index: List[int],
    female_index: List[int],
    threshold: int,
    skip_coref: bool = False
) -> Tuple[int, int]:
    """Process named entities in the text, updating character and synonyms maps."""
    new_characters_count = 0
    new_synonyms_count = 0

    # For efficiency, if coreference is not skipped, build the mention map once.
    coref_map = None
    if not skip_coref:
        coref_map = build_coref_mention_map(doc_coref)

    # Iterate over all PERSON entities.
    for ent in doc_ner.ents:
        if ent.label_ != "PERSON":
            continue

        name = clean_variant_text(ent.text)
        if not is_valid_character_name(name):
            continue

        # Only update if the candidate name is not already present in synonyms_map.
        if name in synonyms_map:
            continue

        # Get gender using coreference information (if not skipped)
        gender = None
        if not skip_coref:
            gender = get_gender_from_coreference(ent, doc_coref, coref_mention_map=coref_map)
        voices_map[name] = {
            "speaker_id": get_next_speaker_id(gender, male_index, female_index),
            "gender": gender if gender else "unknown",
            "desc": "auto"
        }
        # Add a new entry for synonyms.
        synonyms_map[name] = []
        new_characters_count += 1

        # Now check against all existing characters for fuzzy matches.
        # If the candidate is similar enough to an existing one, add it as a synonym.
        for candidate in voices_map.keys():
            if candidate == name:
                continue
            if not is_valid_character_name(candidate):
                continue
            score = ratio(name.lower(), candidate.lower())
            if score >= threshold:
                if name not in synonyms_map[candidate]:
                    synonyms_map[candidate].append(name)
                    new_synonyms_count += 1

    return new_characters_count, new_synonyms_count

def combine_audio_segments(wav_paths: List[str], output_file: str) -> None:
    """Combine multiple WAV files into one using pydub."""
    if not wav_paths:
        return
    try:
        combined = AudioSegment.empty()
        for path in wav_paths:
            segment = AudioSegment.from_wav(path)
            combined += segment
        combined.export(output_file, format="wav")
    except Exception as e:
        logger.error(f"Error combining WAV files: {e}")
        raise

def compress_wav_to_mp3(input_file: str, output_file: str, bitrate: str = "192k") -> None:
    """Convert a WAV file to MP3 format using pydub."""
    try:
        sound = AudioSegment.from_wav(input_file)
        sound.export(output_file, format="mp3", bitrate=bitrate)
    except Exception as e:
        logger.error(f"Error compressing WAV to MP3: {e}")
        raise

def generate_tts_cache_key(speaker_id: str, text: str) -> str:
    """Generate a unique cache key for a TTS segment using speaker_id and text."""
    hash_input = f"{speaker_id}:{text}".encode("utf-8")
    return hashlib.md5(hash_input).hexdigest()

def chunk_text_for_tts(text: str, max_chars: int = 1000) -> List[str]:
    """
    Split the text into chunks up to max_chars while ensuring that chunks do not break
    inside a quoted segment and the natural flow is preserved.
    """
    text = text.strip()
    segments = []
    length = len(text)

    # Pre-calculate quote spans (straight or smart quotes).
    quote_pattern = re.compile(r'(["“”])(.+?)(\1)')
    quote_spans: List[Tuple[int, int]] = []
    for match in quote_pattern.finditer(text):
        quote_spans.append((match.start(), match.end()))

    def is_inside_quote(index: int) -> Optional[Tuple[int, int]]:
        for (qstart, qend) in quote_spans:
            if qstart < index < qend:
                return (qstart, qend)
        return None

    start = 0
    while start < length:
        desired_end = min(start + max_chars, length)
        space_break = text.rfind(" ", start, desired_end)
        candidate_end = space_break if (space_break > start) else desired_end

        quote_span = is_inside_quote(candidate_end)
        if quote_span:
            qstart, qend = quote_span
            if qend - start <= max_chars * 1.5:
                candidate_end = qend
            else:
                candidate_end = qstart if qstart > start else candidate_end

        if candidate_end == desired_end:
            quote_span = is_inside_quote(candidate_end)
            if quote_span:
                qstart, qend = quote_span
                if qend - start <= max_chars * 1.5:
                    candidate_end = qend
                else:
                    candidate_end = qstart if qstart > start else candidate_end

        candidate_end = min(candidate_end, length)
        chunk = text[start:candidate_end].strip()
        if chunk:
            segments.append(chunk)
        if candidate_end <= start:
            candidate_end = start + 1
        start = candidate_end

    return segments

def segment_text_by_dialogue_advanced(text: str, voices_map: Dict) -> List[Tuple[str, str]]:
    """
    Segment the text into dialogue and narrative segments with speaker assignments.
    Dialogue (text in quotes) is assigned to a speaker (via attribution if detected),
    while narrative text is assigned to "Narrator".
    """
    segments = []
    last_dialogue_speaker = None
    conversation_speakers = []  # Up to two speakers.
    current = 0

    dialogue_pattern = re.compile(r'([“"])(.+?)([”"])', re.DOTALL)
    attribution_pattern = re.compile(
        r'^[\s,;:\-\"]*(?P<speaker>[A-Z][a-z]+)\s+(said|asked|answered|conceded)',
        re.IGNORECASE
    )

    for match in dialogue_pattern.finditer(text):
        if match.start() > current:
            narrative = text[current:match.start()]
            segments.append(("Narrator", narrative))
        
        dialogue_full = match.group(0)
        dialogue_end = match.end()
        post_text = text[dialogue_end: dialogue_end + 100]
        attr_match = attribution_pattern.search(post_text)
        
        if attr_match:
            explicit_speaker = attr_match.group("speaker")
            if explicit_speaker not in voices_map:
                explicit_speaker = "Narrator"
            dialogue_speaker = explicit_speaker
            last_dialogue_speaker = dialogue_speaker

            if dialogue_speaker not in conversation_speakers:
                conversation_speakers.append(dialogue_speaker)
                if len(conversation_speakers) > 2:
                    conversation_speakers = conversation_speakers[-2:]
            
            segments.append((dialogue_speaker, dialogue_full))
            attribution_text = post_text[:attr_match.end()].strip()
            segments.append(("Narrator", attribution_text))
            current = dialogue_end + attr_match.end()
        else:
            if last_dialogue_speaker is None:
                dialogue_speaker = "Narrator"
            else:
                if len(conversation_speakers) == 2:
                    dialogue_speaker = (
                        conversation_speakers[1]
                        if last_dialogue_speaker == conversation_speakers[0]
                        else conversation_speakers[0]
                    )
                else:
                    dialogue_speaker = last_dialogue_speaker
            segments.append((dialogue_speaker, dialogue_full))
            current = dialogue_end

    if current < len(text):
        segments.append(("Narrator", text[current:]))
    
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
    bitrate: str = "192k",
    update_characters: bool = True,
    ffmpeg_path: Optional[str] = None,
    skip_coref: bool = False
) -> None:
    """Main pipeline: read text, update character/synonym maps, generate TTS audio, and produce the final MP3 file."""
    if not os.path.isfile(text_file_path):
        logger.error(f"Text file not found: {text_file_path}")
        return

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path

    voices_map = load_json_file(voice_map_path)
    synonyms_map = load_json_file(synonyms_map_path)

    if update_characters:
        remove_blacklisted_names(voices_map, synonyms_map, blacklist_file_path)
    else:
        logger.info("Using existing character and synonyms lists without update.")

    for character in list(voices_map.keys()):
        if character not in synonyms_map:
            synonyms_map[character] = {}

    if "Narrator" not in voices_map or not voices_map["Narrator"].get("speaker_id"):
        voices_map["Narrator"] = {"speaker_id": None, "desc": "Narrator"}

    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        logger.error(f"Error reading text file {text_file_path}: {e}")
        return

    try:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("fastcoref", config={"model_architecture": coreference_model, "device": "cpu"})
        nlp_ner = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Error loading spaCy models: {e}")
        return

    try:
        tts_engine = TTS(model_name=tts_model_name, gpu=use_gpu)
    except Exception as e:
        logger.error(f"TTS engine load failed: {e}")
        sys.exit(1)

    available_speakers = [s.strip() for s in tts_engine.speakers if s.strip() and s.strip() != "ED"]
    if not available_speakers:
        logger.error("No valid speaker IDs available after filtering.")
        return

    if voices_map["Narrator"]["speaker_id"] is None:
        voices_map["Narrator"]["speaker_id"] = available_speakers[0]

    steps = 1 if chunk_size <= 0 else math.ceil(len(full_text) / chunk_size)
    male_index = [0]
    female_index = [0]
    total_new_characters = 0
    total_new_synonyms = 0

    if update_characters:
        logger.info("Updating character and synonyms lists...")
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
                fuzzy_threshold,
                skip_coref=skip_coref
            )
            total_new_characters += new_chars
            total_new_synonyms += new_syns
        unify_synonyms_in_place(synonyms_map, voices_map, fuzzy_threshold)
        remove_blacklisted_names(voices_map, synonyms_map, blacklist_file_path)
        save_json_file(voices_map, voice_map_path)
        save_json_file(synonyms_map, synonyms_map_path)
        logger.info(f"Updated character list: {total_new_characters} new characters, {total_new_synonyms} new synonyms.")
    else:
        logger.info("Skipping character list update.")

    segments = segment_text_by_dialogue_advanced(full_text, voices_map)
    temporary_files: List[str] = []
    tts_cache: Dict[str, str] = {}

    segment_iter = tqdm(segments, desc="Generating TTS Audio") if tqdm else segments
    with tempfile.TemporaryDirectory() as temp_dir:
        for speaker, segment_text in segment_iter:
            chunks = chunk_text_for_tts(segment_text, max_line_chars)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk or not any(ch.isalpha() for ch in chunk):
                    continue
                tts_input = clean_tts_input(chunk) or chunk
                speaker_entry = voices_map.get(speaker, voices_map["Narrator"])
                speaker_id = speaker_entry.get("speaker_id", available_speakers[0])
                cache_key = generate_tts_cache_key(speaker_id, tts_input)
                if cache_key in tts_cache:
                    temp_wav_path = tts_cache[cache_key]
                else:
                    temp_wav_path = os.path.join(temp_dir, f"tts_{cache_key}.wav")
                    try:
                        tts_engine.tts_to_file(
                            text=tts_input,
                            speaker=speaker_id,
                            file_path=temp_wav_path
                        )
                        with wave.open(temp_wav_path, "rb") as wav_file:
                            if wav_file.getnframes() == 0:
                                logger.warning(f"TTS produced empty audio for chunk: {tts_input}")
                                if os.path.exists(temp_wav_path):
                                    os.remove(temp_wav_path)
                                continue
                        tts_cache[cache_key] = temp_wav_path
                    except Exception as exc:
                        logger.warning(f"TTS failed for chunk: {tts_input}. Error: {exc}")
                        if os.path.exists(temp_wav_path):
                            os.remove(temp_wav_path)
                        continue
                temporary_files.append(temp_wav_path)

        if temporary_files:
            base_name = os.path.splitext(text_file_path)[0]
            final_wav_file = base_name + "_final.wav"
            try:
                combine_audio_segments(temporary_files, final_wav_file)
            except Exception as e:
                logger.error(f"Failed to combine audio segments: {e}")
                return
            final_mp3_file = base_name + "_final.mp3"
            try:
                compress_wav_to_mp3(final_wav_file, final_mp3_file, bitrate=bitrate)
                os.remove(final_wav_file)
                logger.info(f"Created MP3: {final_mp3_file}")
            except Exception as e:
                logger.error(f"Failed to compress WAV to MP3: {e}")
        else:
            logger.warning("No audio segments were generated.")

        logger.info(f"Done. Found {total_new_characters} new characters, {total_new_synonyms} new synonyms.")

def main() -> None:
    """
    Parse command-line arguments and run the main pipeline.
    Supports input as a single text file or a directory of text files.
    If the required JSON files (voice map, synonyms, blacklist) do not exist, they will be created.
    """
    parser = argparse.ArgumentParser(
        description="TTS Narration Pipeline: process text files, assign voices, generate TTS audio, and produce final MP3s."
    )
    parser.add_argument("text_file", help="Path to the input text file or directory containing text files.")
    parser.add_argument("--voice_map", default="character_voices.json",
                        help="JSON file for character voices mapping (default: character_voices.json).")
    parser.add_argument("--synonyms", default="character_synonyms.json",
                        help="JSON file for character synonyms mapping (default: character_synonyms.json).")
    parser.add_argument("--blacklist", default="black_list.json",
                        help="JSON file with names to ignore (default: black_list.json).")
    parser.add_argument("--coref_model", default="FCoref",
                        help="Coreference resolution model to use (default: FCoref).")
    parser.add_argument("--fuzzy_thresh", type=int, default=80,
                        help="Fuzzy matching threshold (default: 80).")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Size of text chunks to process at a time (default: 0 for full text).")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits",
                        help="TTS model name to use (default: tts_models/en/vctk/vits).")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for TTS if available.")
    parser.add_argument("--max_line_chars", type=int, default=1000,
                        help="Maximum number of characters per TTS chunk (default: 1000).")
    parser.add_argument("--bitrate", default="192k",
                        help="Bitrate for final MP3 (default: 192k).")
    parser.add_argument("--no_update_characters", action="store_true",
                        help="Do not update character and synonyms lists; use existing ones.")
    parser.add_argument("--ffmpeg", default=None,
                        help="Path to the ffmpeg binary (overrides FFMPEG_PATH env variable if provided).")
    parser.add_argument("--skip_coref", action="store_true",
                        help="Skip coreference-based gender detection (for testing efficiency).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Ensure required JSON files exist; if not, create them with default empty content.
    ensure_file_exists(args.voice_map, {})
    ensure_file_exists(args.synonyms, {})
    ensure_file_exists(args.blacklist, [])

    # If input is a directory, process all .txt files within.
    if os.path.isdir(args.text_file):
        for root, dirs, files in os.walk(args.text_file):
            for file in files:
                if file.lower().endswith(".txt"):
                    file_path = os.path.join(root, file)
                    logger.info(f"Processing file: {file_path}")
                    run_pipeline(
                        text_file_path=file_path,
                        voice_map_path=args.voice_map,
                        synonyms_map_path=args.synonyms,
                        blacklist_file_path=args.blacklist,
                        coreference_model=args.coref_model,
                        fuzzy_threshold=args.fuzzy_thresh,
                        chunk_size=args.chunk_size,
                        tts_model_name=args.model_name,
                        use_gpu=args.gpu,
                        max_line_chars=args.max_line_chars,
                        bitrate=args.bitrate,
                        update_characters=not args.no_update_characters,
                        ffmpeg_path=args.ffmpeg,
                        skip_coref=args.skip_coref
                    )
    else:
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
            bitrate=args.bitrate,
            update_characters=not args.no_update_characters,
            ffmpeg_path=args.ffmpeg,
            skip_coref=args.skip_coref
        )

if __name__ == "__main__":
    main()
