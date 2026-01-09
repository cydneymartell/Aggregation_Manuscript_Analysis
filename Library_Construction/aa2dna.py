"""
    Lines in the DNAWorks input file
        solutions defines the number of solutions from the optimization that you want (we want 1)
        repeat defines the length of a repeat that will be tolerated, long repeats can result in mis-priming, 8 is the default value
        LOGFILE string: the string defines the name of the logfile
        patterns to screen the genes for (some of these are restriction enzymes, I'm guessing we dont want these in our sequences so they arent cleaved)
        codon frequency for ecoli2 will be used
        the protein section contains the protein sequences
"""

lines = """solutions 1
repeat 8
LOGFILE temp.log
pattern
  BamHI GGATCC
  NdeI CATATG
  XhoI CTCGAG
  NheI GCTAGC
  BsaI GGTCTC
  BsaI GAGACC
  PolyA AAAAAAAA
  PolyG GGGGG
  PolyT TTTTTTTT
  PolyC CCCCCCCC
  Aarl  CACCTGC
//

codon ecoli2
protein
%s
//

""".split('\n')


import os
import subprocess
import shutil
def aa2dna(seq, pad, maxLen):
    """
    This function takes in a protein sequence and uses dnaworks to convert it to a DNA sequence 

    Parameters:
        seq (str): A string of the amino acid sequence to be converted to DNA sequence
        pad (bool): A boolean where true means you want padding added on and false means no
        maxLen (int): An integer of the maximum length of a sequence in the library

    Returns:
        dna: the dna sequence

"""

    #We want all of the DNA sequences to be the same length because we dont want to bias certain 
    #sequences in the library during PCR. 
    #Pads the protein sequence with GGS to make the length of the protein sequence the same as the max length
    if pad:
        padding = "GGS"
        protLen = len(seq)
        if len(seq) < maxLen:
            diff = maxLen - len(seq)
            numTrip = int(diff/3) 
            numRem = diff % 3 #Accounts for if the amount of padding to be added is not a multiple of 3  
            seq = seq +  numTrip*padding + padding[:numRem] #Adds padding to the c terminus of the protein
    


    os.makedirs('d_%s' % seq, exist_ok=True)
    os.chdir('d_%s' % seq) 

    #Writes an input file for DNAWorks
    with open('temp.job','w') as file:
        for line in lines:
            file.write(line.replace('%s',seq) + '\n')

    cmd = '/projects/b1107/jane/software/dnaworks/DNAWorks/dnaworks temp.job'

    # Run the command and capture both stdout and stderr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Communicate to get the output and error
    stdout, stderr = process.communicate()

    # Decode both stdout and stderr
    stdout_decoded = stdout.decode('utf-8')
    stderr_decoded = stderr.decode('utf-8')

    # Print both the output and the error (if any)
    print("STDOUT:")
    print(stdout_decoded)

    if stderr_decoded:
        print("STDERR:")
        print(stderr_decoded)

    with open('temp.log') as file:
        outlines = file.readlines()
    dna = ''
    
    for j in range(len(outlines)):
        if ' The DNA sequence #   1 is:' in outlines[j]:
            j += 2
            while '-----' not in outlines[j]:
                if len(outlines[j].split()) == 2: dna += outlines[j].split()[-1]
                j += 1
            break
    os.chdir('..') 
    
    shutil.rmtree('d_%s' % seq) 
    
    #Adds stop codons on to the end of the protein sequence before the padding, using TAA
    if pad:
        numCodon = protLen*3
        dna = dna[:numCodon]+'TAA'+dna[numCodon:]
    else:
        dna += 'TAA'

    
    return dna
