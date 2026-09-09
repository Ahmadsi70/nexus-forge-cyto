from datasets import load_dataset
ds = load_dataset("MedOtter/Pan-Cancer-Nuclei-Seg", split="train")
print(ds.column_names)
