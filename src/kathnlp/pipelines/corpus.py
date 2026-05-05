from __future__ import annotations



import re

from dataclasses import dataclass

from pathlib import Path



import pandas as pd





SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[\.\!\?;])\s+")

TOKEN_SPLIT_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

CLAUSE_SPLIT_REGEX = re.compile(r"(?<=[,·:])\s+")





@dataclass

class SentenceRecord:

    sentence_id: str

    text: str

    source_sheet: str

    source_row: int





def simple_sentence_split(text: str) -> list[str]:

    text = (text or "").replace("\r", "\n")

    initial = [part.strip() for part in SENTENCE_SPLIT_REGEX.split(text) if part.strip()]

    chunks: list[str] = []

    for part in initial:

        for line_part in [x.strip() for x in part.split("\n") if x.strip()]:

            if len(line_part.split()) > 80:

                chunks.extend(

                    [c.strip() for c in CLAUSE_SPLIT_REGEX.split(line_part) if c.strip()]

                )

            else:

                chunks.append(line_part)

    final_chunks: list[str] = []

    for chunk in chunks:

        words = chunk.split()

        if len(words) <= 80:

            final_chunks.append(chunk)

            continue

        # Hard window split for very long fragments.

        for i in range(0, len(words), 60):

            piece = " ".join(words[i : i + 60]).strip()

            if piece:

                final_chunks.append(piece)

    return [chunk for chunk in final_chunks if len(chunk) >= 8]





def simple_tokenize(text: str) -> list[str]:

    return TOKEN_SPLIT_REGEX.findall(text)





def extract_answer_files_sentences(

    csv_paths: list[Path],

    answer_column_name: str = "Answer Files",

) -> list[SentenceRecord]:

    records: list[SentenceRecord] = []

    for csv_path in csv_paths:

        frame = pd.read_csv(csv_path)

        if answer_column_name not in frame.columns:

            continue

        for idx, raw_text in frame[answer_column_name].fillna("").items():

            if not raw_text.strip():

                continue

            for sentence_index, sentence in enumerate(simple_sentence_split(raw_text), start=1):

                sentence_id = f"{csv_path.stem}-{idx+1}-{sentence_index}"

                records.append(

                    SentenceRecord(

                        sentence_id=sentence_id,

                        text=sentence,

                        source_sheet=csv_path.stem,

                        source_row=idx + 1,

                    )

                )

    return records

