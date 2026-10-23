#!/usr/bin/env python3
"""Generate the synthetic stand-in corpus that ships with the repository.

The real personal corpus is never committed, so without this nobody -- a
recruiter, a reviewer, or me on a fresh machine -- could run the pipeline end
to end.  This script fabricates two authors with deliberately different habits:

* ``sam``     -- lowercase, hedges, ellipses, emoji, short clauses
* ``control`` -- capitalised, formal connectives, semicolons, longer sentences

The style gap is what gives the evaluation harness something to detect, so the
stylometry and authorship tests have a real signal to measure rather than
asserting against noise.  It is a fixture, not data: no claim is made that it
resembles anybody.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.data.document import Document, make_id, write_jsonl  # noqa: E402

TOPICS = [
    "the gradient descent assignment", "that stats lecture", "the group project",
    "my thesis draft", "the dataset we scraped", "the tutorial questions",
    "the kaggle comp", "the linear algebra revision", "the internship application",
    "the deployment pipeline", "the transformer paper", "the eval harness",
]
FEELINGS = ["cooked", "fine honestly", "a bit rough", "actually alright", "painful", "so much better"]
SAM_OPENERS = ["ngl", "tbh", "honestly", "ok so", "wait", "yeah nah", "lowkey", "bruh"]
SAM_CLOSERS = ["...", " lol", " 😅", " :((", " haha", "", " fr", " 👀"]
CTRL_CONNECT = ["Moreover", "Therefore", "In addition", "Consequently", "However", "Furthermore"]

ESSAY_SAM = (
    "i spent most of this week on {topic} and it was {feeling}. the thing that kept "
    "tripping me up was the setup - not the maths, just getting the data into a shape "
    "that made sense. once i stopped trying to be clever about it and wrote the dumb "
    "version first, it got a lot easier. next time i want to start with the dumb version "
    "on purpose{closer}\n\nthe part i still don't fully get is why the results move around "
    "so much between runs. i think it's the seed but i haven't actually checked properly, "
    "which is probably the real lesson here"
)
ESSAY_CTRL = (
    "This report summarises the work undertaken on {topic}. {connect}, the principal "
    "difficulty was not analytical but procedural; considerable effort was required to "
    "transform the raw data into a tractable representation. {connect2}, an initial "
    "simplified implementation proved more productive than a premature optimisation.\n\n"
    "A remaining source of uncertainty is the variance observed across repeated runs. "
    "{connect3}, this variance has not yet been attributed to a specific cause, and a "
    "controlled investigation is warranted before firm conclusions are drawn."
)

CODE_SAM = '''import pandas as pd


def load_{name}(path):
    # NOTE: this is slow on the big file, cache it later
    df = pd.read_csv(path)
    df = df.dropna(subset=["{col}"])
    return df


def summarise_{name}(df):
    out = df.groupby("{col}").size()
    print(f"got {{len(out)}} groups")
    return out
'''
CODE_CTRL = '''from __future__ import annotations

import pandas as pd


def load_{name}(path: str) -> pd.DataFrame:
    """Load the {name} dataset and drop incomplete records.

    Args:
        path: Location of the CSV file.

    Returns:
        A dataframe containing only complete records.
    """
    frame = pd.read_csv(path)
    return frame.dropna(subset=["{col}"])


def summarise_{name}(frame: pd.DataFrame) -> pd.Series:
    """Return the number of records per {col}."""
    return frame.groupby("{col}").size()
'''

NAMES = ["taxi", "housing", "weather", "sensor", "survey", "transit", "energy", "retail"]
COLS = ["region", "category", "timestamp", "segment", "label", "zone"]


def sam_message(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    body = rng.choice([
        f"{rng.choice(SAM_OPENERS)} {topic} was {rng.choice(FEELINGS)}",
        f"{rng.choice(SAM_OPENERS)} i think i broke {topic} again",
        f"did u finish {topic}",
        f"{rng.choice(SAM_OPENERS)} im just gonna submit what i have",
        f"kinda stuck on {topic} rn",
        f"{rng.choice(SAM_OPENERS)} that took me like 3 hours",
        f"same, {topic} is {rng.choice(FEELINGS)}",
    ])
    return body + rng.choice(SAM_CLOSERS)


def control_message(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    return rng.choice([
        f"I have finished {topic}; it was more demanding than expected.",
        f"Could you confirm whether {topic} is due on Friday?",
        f"{rng.choice(CTRL_CONNECT)}, I would suggest we revisit {topic} tomorrow.",
        f"I am still working through {topic}, and will share my notes shortly.",
        f"Thank you for sending {topic}. I will review it this evening.",
    ])


def build(author: str, n_chat: int, n_essay: int, n_code: int, rng: random.Random) -> list[Document]:
    is_sam = author == "sam"
    docs: list[Document] = []

    # Chat: several consecutive messages grouped into one utterance block, the
    # same way the Meta export parser groups a burst of messages.
    for i in range(n_chat):
        burst = [sam_message(rng) if is_sam else control_message(rng) for _ in range(rng.randint(1, 4))]
        text = "\n".join(burst)
        docs.append(Document(id=make_id(author, "chat", i), source="chat", text=text, author=author))

    for i in range(n_essay):
        if is_sam:
            text = ESSAY_SAM.format(
                topic=rng.choice(TOPICS), feeling=rng.choice(FEELINGS), closer=rng.choice(SAM_CLOSERS)
            )
        else:
            text = ESSAY_CTRL.format(
                topic=rng.choice(TOPICS), connect=rng.choice(CTRL_CONNECT),
                connect2=rng.choice(CTRL_CONNECT), connect3=rng.choice(CTRL_CONNECT),
            )
        docs.append(Document(id=make_id(author, "longform", i), source="longform", text=text, author=author))

    for i in range(n_code):
        template = CODE_SAM if is_sam else CODE_CTRL
        text = template.format(name=rng.choice(NAMES), col=rng.choice(COLS))
        docs.append(Document(id=make_id(author, "code", i), source="code", text=text, author=author))

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample_synthetic", type=Path)
    parser.add_argument("--chat", type=int, default=1200)
    parser.add_argument("--essays", type=int, default=200)
    parser.add_argument("--code", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    for author in ("sam", "control"):
        rng = random.Random(args.seed + (0 if author == "sam" else 1))
        docs = build(author, args.chat, args.essays, args.code, rng)
        path = args.out / f"{author}.jsonl"
        write_jsonl(path, docs)
        chars = sum(d.n_chars for d in docs)
        print(f"{author:8s} {len(docs):5d} docs  {chars:8,d} chars  -> {path}")


if __name__ == "__main__":
    main()
