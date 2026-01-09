import sys
import pandas as pd
from tqdm import tqdm
from aa2dna import aa2dna  
import os

# Get chunk index from command line argument
chunk_idx = int(sys.argv[1])

# Read in the correct chunk
lib = pd.read_csv(f"/projects/p30802/Cydney/aa2dna/meta/meta_100K_chunk_{chunk_idx}.csv") #replace with path to csv with sequences

# Maximum length of protein sequences
#maxLen = max([len(seq) for seq in lib['protein_sequence']]i)
maxLen = 80
# Process the chunk (with tqdm for progress tracking)
lib["dna_no_adapt"] = [aa2dna(seq, True, maxLen) for seq in tqdm(lib['protein_sequence'])]

# Save the processed chunk to a new CSV
lib.to_csv(f"/projects/p30802/Cydney/aa2dna/meta/meta_100K_chunk_{chunk_idx}_out.csv", index=False) #save the DNA sequences

