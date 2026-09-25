import os

# osunlp/SMolInstruct: https://huggingface.co/datasets/osunlp/SMolInstruct
SMOL_ALL_TASKS = [
    "forward_synthesis",
    "retrosynthesis",
    "molecule_captioning",
    "molecule_generation",
    "name_conversion-i2f",
    "name_conversion-i2s",
    "name_conversion-s2f",
    "name_conversion-s2i",
    "property_prediction-esol",
    "property_prediction-lipo",
    "property_prediction-bbbp",
    "property_prediction-clintox",
    "property_prediction-hiv",
    "property_prediction-sider",
]
SMOL_DIR = os.environ.get("SMOL_DIR", "data/SMolInstruct")
