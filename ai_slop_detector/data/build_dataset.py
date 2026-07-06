"""Builds the small, hand-authored, labeled AI-vs-human text dataset.

IMPORTANT / HONEST SCOPE NOTE: this sandbox has no network access to
download a real scraped corpus (e.g. HC3, GPT-wiki-intro) or to call a live
LLM API. So this dataset is *synthetically constructed* from templates: one
generator writes deliberately informal, personal, "human blog/forum post"
style paragraphs; the other writes deliberately formal, hedge-heavy,
buzzword-heavy paragraphs in the style default LLM output is well known for
("Furthermore...", "In today's fast-paced world...", "leverage", "delve",
"seamless", etc.). Both generators pull from banks of phrases/sentence
templates and randomize topic, order, and specific details (with a fixed
seed for reproducibility), so texts are non-identical but the *label is
true by construction* -- unlike a scraped dataset there is no labeling
noise, but there is also no guarantee this generalizes to real-world LLM
output as well as a real corpus would. See ai_slop_detector/README.md
"Limitations" for the honest version of this tradeoff.

Run as a script to (re)write data/samples.csv:
    python -m ai_slop_detector.data.build_dataset
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 20260706
OUTPUT_PATH = Path(__file__).parent / "samples.csv"

TOPICS = [
    "a weekend camping trip",
    "trying a new pasta recipe",
    "the movie I watched last night",
    "my daily commute",
    "a new pair of running shoes",
    "starting a home workout routine",
    "the book I just finished",
    "adopting a rescue dog",
    "working from home",
    "a coffee shop near my apartment",
    "a rainy Tuesday",
    "learning to play the guitar",
    "a job interview I had",
    "moving to a new apartment",
    "a budgeting app I tried",
    "a road trip down the coast",
    "my neighbor's garden",
    "switching to a new phone",
    "a pottery class",
    "cleaning out the garage",
]

# ---------------------------------------------------------------------------
# Human-style generator: informal, contractions, first person, uneven
# sentence lengths, slang, mild tangents.
# ---------------------------------------------------------------------------

HUMAN_OPENERS = [
    "So {topic} happened this week and honestly I have thoughts.",
    "Okay so {topic} -- gonna be honest, it wasn't what I expected.",
    "I keep thinking about {topic}, not gonna lie.",
    "Quick post about {topic} because a couple people asked.",
    "Not sure why but {topic} has been on my mind all day.",
    "Someone asked me about {topic} today and I realized I never wrote it up.",
]

HUMAN_MIDDLES = [
    "It's kind of a mess honestly, but in a good way?",
    "I didn't plan for it to take this long, but here we are.",
    "My roommate thinks I'm overreacting, maybe I am, idk.",
    "There were like three moments where I almost gave up.",
    "It's not perfect and I'm not gonna pretend it is.",
    "I laughed way harder than I should've at one point.",
    "Took me forever to get the hang of it, tbh.",
    "Half the time I had no clue what I was doing.",
    "It's one of those things that's better than it sounds on paper.",
    "I've done worse, but I've definitely done better too.",
]

HUMAN_CLOSERS = [
    "Anyway, we'll see how it goes.",
    "Would I do it again? Probably, yeah.",
    "Not a five-star experience but I don't regret it either.",
    "That's it, that's the post lol.",
    "Might update this later if anything changes.",
    "10/10 would recommend to literally no one and also everyone.",
    "So yeah. That's where things stand right now.",
    "Still figuring it out as I go, honestly.",
]

HUMAN_ASIDES = [
    "(also my phone died halfway through which didn't help)",
    "-- side note, I definitely spent more money than planned --",
    "which, fun fact, is exactly what my mom warned me about",
    "and don't even get me started on the parking situation",
    "which is a whole separate story I'll save for another day",
]


def _gen_human(rng: random.Random, topic: str) -> str:
    opener = rng.choice(HUMAN_OPENERS).format(topic=topic)
    n_middles = rng.choice([1, 2, 2, 3])
    middles = rng.sample(HUMAN_MIDDLES, k=n_middles)
    sentences = [opener]
    for i, m in enumerate(middles):
        if rng.random() < 0.35:
            m = f"{m} {rng.choice(HUMAN_ASIDES)}."
        sentences.append(m)
    if rng.random() < 0.5:
        sentences.append(rng.choice(HUMAN_CLOSERS))
    # Occasionally tack on a short punchy fragment for rhythm variety.
    if rng.random() < 0.3:
        sentences.append(rng.choice(["Wild.", "Anyway.", "No regrets.", "We'll see."]))
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# AI-style generator: formal, hedge phrases, buzzwords, uniform sentence
# length, generic summarizing close.
# ---------------------------------------------------------------------------

AI_OPENERS = [
    "When it comes to {topic}, there are several important factors to consider.",
    "In today's fast-paced world, {topic} has become increasingly significant.",
    "{topic_cap} is a topic that many people find worth exploring in more depth.",
    "It is worth taking a closer look at {topic}, as it offers valuable insights.",
    "Navigating the complexities of {topic} can be both rewarding and challenging.",
]

AI_MIDDLES = [
    "Furthermore, it is important to note that consistency plays a crucial role in achieving positive outcomes.",
    "Moreover, this approach can help foster a more holistic and well-rounded experience overall.",
    "Additionally, taking a comprehensive and structured approach can help unlock meaningful, long-term benefits.",
    "It is also worth noting that a seamless, well-planned strategy tends to yield more robust results.",
    "Notably, this underscores the importance of careful planning and thoughtful execution.",
    "In many respects, this reflects a broader trend toward more mindful and intentional decision-making.",
    "This, in turn, can help elevate the overall experience and foster a genuine sense of accomplishment.",
    "Ultimately, embarking on this journey requires patience, dedication, and a willingness to adapt.",
]

AI_CLOSERS = [
    "In conclusion, {topic} is a rewarding endeavor that offers something valuable for everyone.",
    "Overall, {topic} serves as a testament to the power of consistency and thoughtful planning.",
    "To sum up, embracing {topic} can lead to a more fulfilling and well-rounded experience.",
    "In summary, {topic} underscores the importance of balance, patience, and a holistic mindset.",
]


def _gen_ai(rng: random.Random, topic: str) -> str:
    opener = rng.choice(AI_OPENERS).format(topic=topic, topic_cap=topic[0].upper() + topic[1:])
    n_middles = rng.choice([2, 3, 3])
    middles = rng.sample(AI_MIDDLES, k=n_middles)
    closer = rng.choice(AI_CLOSERS).format(topic=topic)
    return " ".join([opener, *middles, closer])


def build_samples(seed: int = SEED) -> list[tuple[str, str]]:
    """Returns a list of (text, label) pairs, label in {"human", "ai"}."""
    rng = random.Random(seed)
    samples: list[tuple[str, str]] = []
    variants_per_topic = 5
    for topic in TOPICS:
        for _ in range(variants_per_topic):
            samples.append((_gen_human(rng, topic), "human"))
        for _ in range(variants_per_topic):
            samples.append((_gen_ai(rng, topic), "ai"))
    rng.shuffle(samples)
    return samples


def write_csv(path: Path = OUTPUT_PATH, seed: int = SEED) -> int:
    samples = build_samples(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in samples:
            writer.writerow([text, label])
    return len(samples)


if __name__ == "__main__":
    n = write_csv()
    print(f"Wrote {n} labeled samples to {OUTPUT_PATH}")
