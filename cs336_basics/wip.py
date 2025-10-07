import json

vocab_filepath = "../tokenizers/owt_tokenizer_32k_2/vocab.json"
merges_filepath = "../tokenizers/owt_tokenizer_32k/merges.txt"
text = """
Topic: The Deep Sea

The deep sea is a realm of crushing pressure, perpetual darkness, and alien beauty. Below a thousand meters, sunlight fades completely, and what remains is an otherworldly world where life has evolved in ways that defy our surface logic. Bioluminescent creatures flicker like drifting constellations, using light not for vision but for communication, deception, and predation. This is a place where jellyfish glow like lanterns and fish wield their own headlamps—a galaxy beneath the waves.

Pressure here is not a metaphor but a law—more than a thousand times what we feel on land. Yet life thrives. Tubeworms near hydrothermal vents feast on chemical energy rather than sunlight, turning sulfur into sustenance through chemosynthesis. These vents, spewing mineral-rich water from the Earth’s crust, are reminders that life’s story might not need the Sun at all. Some scientists even suspect that such environments resemble the cradle of life itself—hot, dark, and full of possibility.

Despite its distance, the deep sea mirrors our anxieties about exploration. It is both the planet’s last wilderness and a growing target for mining and resource extraction. To study it is to confront the limits of our technology and imagination. Every descent into that abyss is a conversation with the unknown, a reminder that our world still holds mysteries vast enough to humble us.
"""

# with open(vocab_filepath, "r", encoding="utf-8") as f:
#     hash = json.loads(f.read())

# for i in range(30):
#     max_key_value = max(hash.items(), key=lambda val: len(val[1]))
#     print(max_key_value[1])
#     del hash[max_key_value[0]]

with open(merges_filepath, "r", encoding="utf-8") as f:
    merges = f.readlines()

print(merges[-1].split())
