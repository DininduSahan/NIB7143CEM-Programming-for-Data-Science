from pathlib import Path
import json

data_folder = Path("Data/Raw")

json_files = list(data_folder.glob("*.json"))

print("Total JSON files:", len(json_files))

if json_files:
    file_path = json_files[0]

    with open(file_path, "r", encoding="utf-8") as file:
        match_data = json.load(file)

    print("First match file:", file_path.name)
    print("Match data loaded successfully!")
    print(match_data.keys())
else:
    print("No JSON files found.")

# Explore match information
print("\nMatch Information:")

print(json.dumps(match_data["info"], indent=4))    

from pathlib import Path
data_folder = Path("Data/Raw")
json_files = list(data_folder.glob("*.json"))
print("Total JSON files:", len(json_files))

innings = match_data["innings"]
print("Number of innings:", len(innings))
first_innings = innings[0]
print("Batting team:", first_innings["team"])
first_over = first_innings["overs"][0]
first_delivery = first_over["deliveries"][0]
print("First delivery:")
print(first_delivery)
print("Delivery fields:")
print(first_delivery.keys())



