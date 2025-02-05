#!/usr/bin/env python3
import os
import re
import sys

def rename_chapters(directory):
    """
    Renames files of the form:
       <words>_<arc>_<chapter>.txt
    where:
       - arc is digits (e.g. '11')
       - chapter is EITHER all digits (e.g. '4') OR a single letter (e.g. 'z').
    to:
       <arc>_<chapter>_<words>.txt
    
    Skips files that do not match the above pattern.
    """
    for fname in os.listdir(directory):
        # Only operate on .txt files
        if not fname.lower().endswith(".txt"):
            continue
        
        old_path = os.path.join(directory, fname)
        
        base, ext = os.path.splitext(fname)  # e.g. ("dash_to_pieces_11_4", ".txt")
        parts = base.split("_")              # e.g. ["dash", "to", "pieces", "11", "4"]
        
        # Need at least 3 parts: <some_name>_<arc>_<chapter>
        if len(parts) < 3:
            continue
        
        arc = parts[-2]         # e.g. "11"
        chapter = parts[-1]     # e.g. "4" or "z"
        name_parts = parts[:-2] # e.g. ["dash", "to", "pieces"]
        
        # 1) arc must be purely digits
        if not arc.isdigit():
            continue
        
        # 2) chapter must be either all digits OR a single letter
        #    - all digits: "4", "10", etc.
        #    - single letter: "z", "a", ...
        #    We'll allow single letter (a-z or A-Z), or all digits
        if not (chapter.isdigit() or re.match(r'^[A-Za-z]$', chapter)):
            continue
        
        # If the pattern matches, rename
        chapter_name_str = "_".join(name_parts)  # e.g. "dash_to_pieces"
        new_filename = f"{arc}_{chapter}_{chapter_name_str}{ext}"  # e.g. "11_4_dash_to_pieces.txt"
        new_path = os.path.join(directory, new_filename)
        
        print(f"Renaming '{fname}' -> '{new_filename}'")
        os.rename(old_path, new_path)

def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."

    if not os.path.isdir(directory):
        print("[ERROR] Not a directory:", directory)
        sys.exit(1)

    rename_chapters(directory)
    print("Done.")

if __name__ == "__main__":
    main()
