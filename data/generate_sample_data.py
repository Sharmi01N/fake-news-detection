"""
generate_sample_data.py
------------------------
Creates a small SYNTHETIC demo dataset (sample_data.csv) so the pipeline
(preprocessing -> training -> app) can be tested end-to-end immediately,
without waiting to download the real datasets.

IMPORTANT: This is NOT a substitute for the real datasets mentioned in the
proposal (LIAR, ISOT, Kaggle Fake News). Replace data/sample_data.csv with
the real, combined dataset before training the models you actually submit.

Real dataset links to download manually and merge (title,text,label -> 0/1):
  - ISOT Fake News Dataset: https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/
  - Kaggle Fake and Real News: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  - LIAR Dataset: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
"""
import csv
import random

random.seed(42)

real_templates = [
    "The central bank announced a {rate}% interest rate {direction} on {day} to control inflation.",
    "Researchers at {uni} published a peer-reviewed study on {topic} in a leading scientific journal.",
    "The government has approved a new budget of {amount} million taka for the {sector} sector this fiscal year.",
    "The national cricket team won the match against {country} by {n} runs, officials confirmed.",
    "Local authorities completed the construction of a new bridge in {city}, improving connectivity for residents.",
    "The health ministry reported {n} new cases as part of its routine weekly surveillance update.",
    "A university committee reviewed the admission policy and released the notice on the official website.",
    "Stock markets closed {direction} today following quarterly earnings reports from major companies.",
    "The city corporation announced a schedule for waste collection in {city} starting next month.",
    "Election commission officials confirmed the voter list will be updated ahead of the {day} deadline.",
    "Scientists confirmed that the new vaccine trial results were reviewed by an independent ethics board.",
    "The ministry of education released the exam results for {n} thousand students today.",
]

fake_templates = [
    "SHOCKING: Doctors HATE this one trick that cures {topic} overnight, government trying to hide it!",
    "BREAKING: Secret documents PROVE the {sector} sector was destroyed by aliens, media refuses to report!",
    "You won't believe what {country} is hiding about the {topic} — share before this gets DELETED!",
    "Miracle cure for {topic} discovered by a farmer, big pharma companies are furious and suing him!",
    "Government secretly plans to ban {topic} completely, leaked audio reveals shocking truth!",
    "Celebrity secretly confesses {topic} is a hoax created to control the population, wake up!",
    "URGENT: Scientists CONFIRM {topic} chemicals are being added to water supply in {city}!",
    "This one weird trick will make you rich overnight, banks don't want you to know about {topic}!",
    "Anonymous insider reveals the {sector} sector is collapsing tomorrow, withdraw your money NOW!",
    "Forbidden footage leaked showing {country} officials admitting the {topic} was staged all along!",
    "Click here to see the terrifying truth about {topic} that they don't want you to see!",
    "Ancient prophecy predicted this exact event about {topic}, and it's happening right now!",
]

fill = {
    "rate": ["0.25", "0.5", "0.75", "1"],
    "direction": ["hike", "cut", "up", "down"],
    "day": ["Monday", "Friday", "the 15th", "next week"],
    "uni": ["RUET", "BUET", "Dhaka University", "MIT", "Oxford"],
    "topic": ["cancer", "diabetes", "climate change", "5G networks", "vaccines", "inflation", "elections"],
    "amount": ["500", "1200", "3400", "780"],
    "sector": ["health", "education", "agriculture", "banking", "energy"],
    "country": ["India", "the USA", "China", "the UK", "Pakistan"],
    "n": ["12", "45", "128", "300", "7"],
    "city": ["Rajshahi", "Dhaka", "Chittagong", "Khulna", "Sylhet"],
}

def fill_template(t):
    for key, options in fill.items():
        if "{" + key + "}" in t:
            t = t.replace("{" + key + "}", random.choice(options))
    return t

rows = []
for _ in range(150):
    t = random.choice(real_templates)
    rows.append((fill_template(t), 0))  # 0 = real
for _ in range(150):
    t = random.choice(fake_templates)
    rows.append((fill_template(t), 1))  # 1 = fake

random.shuffle(rows)

with open("/home/claude/fake-news-detection/data/sample_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Generated {len(rows)} synthetic rows -> data/sample_data.csv")
