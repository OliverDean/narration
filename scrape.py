import os
import time
import re
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode

def to_snake_case(text):
    """
    Convert a string like "Cutting Class – 6.6" to "cutting_class_6_6"
    by lowercasing, replacing non-alphanumeric characters with underscores,
    and stripping extra underscores.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text

def clean_text(text):
    """
    Clean the text so that accidental newline breaks that isolate punctuation
    (or split words) are removed. In particular, we:
      1. Replace any single newline (that is not part of a double newline)
         with a space. This joins lines that were accidentally broken.
      2. Remove extra spaces before punctuation.
    """
    # Replace any newline that is not part of a paragraph break (i.e. not doubled)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Remove any extra whitespace before punctuation characters.
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text

def extract_chapter_text(soup):
    """
    Extract the story text between <hr /> tags, then convert to plain ASCII
    using unidecode, and finally clean up any unwanted newline breaks.
    """
    hr_tags = soup.find_all('hr')
    if len(hr_tags) < 2:
        return ""
    
    start_hr = hr_tags[0]
    end_hr = hr_tags[1]

    # Collect everything between the first and second <hr>
    content_elements = []
    current = start_hr.next_sibling
    while current and current != end_hr:
        content_elements.append(current)
        current = current.next_sibling

    chunk_html = "".join(str(elem) for elem in content_elements)
    chunk_soup = BeautifulSoup(chunk_html, "html.parser")
    text = chunk_soup.get_text(separator="\n").strip()

    # Convert to plain ASCII (e.g. “ becomes " and – becomes -)
    text_ascii = unidecode(text)
    # Clean the text so that lines aren’t broken in the middle of sentences.
    text_clean = clean_text(text_ascii)

    return text_clean

def scrape_chapter(url):
    """
    Given a chapter URL, return:
      (chapter_title, chapter_text, next_chapter_url)
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url} (status code: {response.status_code})")

    soup = BeautifulSoup(response.text, "html.parser")

    # 1. Get the chapter title from <h1 class="entry-title">
    h1_tag = soup.find("h1", class_="entry-title")
    if not h1_tag:
        raise Exception(f"No <h1 class='entry-title'> found on page {url}")
    chapter_title = h1_tag.get_text(strip=True)

    # 2. Extract the chapter text (converted to plain ASCII and cleaned)
    chapter_text = extract_chapter_text(soup)

    # 3. Find the "Next" link
    nav_next = soup.find("span", class_="nav-next")
    next_chapter_url = None
    if nav_next and nav_next.find("a"):
        next_chapter_url = nav_next.find("a")["href"]

    return chapter_title, chapter_text, next_chapter_url

def parse_chapter_title(title, overall_count):
    """
    Parse the chapter title to extract:
      - arc_name
      - arc_number (from the chapter numbering)
      - chapter-of-arc
      - a flag indicating if it is an interlude

    Expected formats:
      • "Arc Name – X.Y" for a standard chapter (e.g. "Cutting Class – 6.6")
      • If the title contains "interlude" (case-insensitive) it is marked as an interlude.
    
    For chapters that don't match the numbering pattern (like "Brochure Experience Kennet"),
    assume they belong to the "spoilers" arc with arc number "0" and chapter "spoilers".
    """
    title = title.strip()
    interlude = False

    # Handle interlude chapters first
    if "interlude" in title.lower():
        parts = re.split(r'[-–]', title)
        if len(parts) >= 2:
            arc_name = parts[0].strip()
        else:
            arc_name = title
        arc_number = "interlude"
        arc_chapter = "interlude"
        return arc_name, arc_number, arc_chapter, True

    # Look for the pattern: "Arc Name – X.Y"
    m = re.search(r'^(.*?)\s*[-–]\s*(\d+(?:[.]\d+)?)', title)
    if m:
        arc_name = m.group(1).strip()
        chap_num = m.group(2).strip()
        if '.' in chap_num:
            arc_number, arc_chapter = chap_num.split('.', 1)
        else:
            arc_number = chap_num
            arc_chapter = chap_num
        return arc_name, arc_number, arc_chapter, False
    else:
        # Fallback: assign arc number 0 and chapter 'spoilers'
        arc_name = title
        arc_number = "0"
        arc_chapter = "spoilers"
        return arc_name, arc_number, arc_chapter, False

def generate_header(overall_count, arc_name, arc_number, arc_chapter, interlude):
    """
    Generate a header string to prepend to each chapter's text.
    This header announces the overall chapter number, the arc info,
    and whether this is an interlude.
    """
    if interlude:
        header = f"Chapter {overall_count}: Interlude - Arc {arc_number}: {arc_name}\n\n"
    else:
        header = f"Chapter {overall_count}: Arc {arc_number} - Chapter {arc_chapter} of {arc_name}\n\n"
    return header

def main():
    # Create the output directory (using an underscore-friendly name)
    out_dir = "pale_witch"
    os.makedirs(out_dir, exist_ok=True)

    # Starting URL for the first chapter
    start_url = "https://palewebserial.wordpress.com/2020/05/05/blood-run-cold-0-0/"
    current_url = start_url

    # We stop after about 360 chapters
    max_chapters = 360
    chapter_count = 0

    while current_url and chapter_count < max_chapters:
        chapter_count += 1
        print(f"Scraping chapter #{chapter_count}: {current_url}")

        try:
            chapter_title, chapter_text, next_url = scrape_chapter(current_url)
        except Exception as e:
            print(e)
            break

        # Parse the chapter title to get arc information
        arc_name, arc_number, arc_chapter, interlude = parse_chapter_title(chapter_title, chapter_count)

        # Generate a header for TTS purposes
        header = generate_header(chapter_count, arc_name, arc_number, arc_chapter, interlude)

        # Prepend the header to the chapter text
        full_text = header + chapter_text

        # Create the filename: count_arcnumber_chapternumber_nameofarc.txt
        filename = f"{chapter_count}_{arc_number}_{arc_chapter}_{to_snake_case(arc_name)}.txt"
        file_path = os.path.join(out_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Saved chapter \"{chapter_title}\" to '{file_path}'.\n")

        # Move on to the next chapter
        current_url = next_url
        time.sleep(0.2)  # a brief pause to be polite

    print("Done. Scraped chapters:", chapter_count)

if __name__ == "__main__":
    main()
