import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
model = AutoModel.from_pretrained("hyp1231/blair-roberta-base")

language_context = 'I need a product that can scoop, measure, and rinse grains without the need for multiple utensils and dishes. It would be great if the product has measurements inside and the ability to rinse and drain all in one. I just have to be careful not to pour too much accidentally.'
item_metadata = [
  'Talisman Designs 2-in-1 Measure Rinse & Strain | Holds up to 2 Cups | Food Strainer | Fruit Washing Basket | Strainer & Colander for Kitchen Sink | Dishwasher Safe - Dark Blue. The Measure Rinse & Strain by Talisman Designs is a 2-in-1 kitchen colander and strainer that will measure and rinse up to two cups. Great for any type of food from rice, grains, beans, fruit, vegetables, pasta and more. After measuring, fill with water and swirl to clean. Strain then pour into your pot, pan, or dish. The convenient size is easy to hold with one hand and is compact to fit into a kitchen cabinet or pantry. Dishwasher safe and food safe.',
  'FREETOO Airsoft Gloves Men Tactical Gloves for Hiking Cycling Climbing Outdoor Camping Sports (Not Support Screen Touch).'
]
texts = [language_context] + item_metadata

inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs, return_dict=True).last_hidden_state[:, 0]
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

print(embeddings[0] @ embeddings[1])    # tensor(0.8564)
print(embeddings[0] @ embeddings[2])    # tensor(0.5741)
