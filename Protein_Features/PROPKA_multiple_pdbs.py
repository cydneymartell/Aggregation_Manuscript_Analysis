
import argparse
import subprocess
import glob
import os
import re
import pandas as pd
from tqdm import tqdm


def run_propka(pdb_dir, propka_exec, propka_out, ph):
    """
    Run PROPKA on all PDB files in pdb_dir.
    pdb files can be AlphaFold predicted structures if no solved structure exists
    """
    os.makedirs(propka_out, exist_ok=True)

    pdbs = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    if len(pdbs) == 0:
        raise ValueError(f"No PDB files found in {pdb_dir}")

    for pdb in tqdm(pdbs, desc="Running PROPKA"):
        cmd = [
            propka_exec,
            "-o", str(ph),
            pdb
        ]
        subprocess.run(
            cmd,
            cwd=propka_out,
            check=True
        )


def parse_propka_outputs(propka_out):
   
    """ Parse the output .pka files to extract the calculated pI and charge for each protein. 
        Returns a DataFrame with each row containing the name of the protein, propka predicted pI and charge for folded and unfolded
    """

    pka_files = glob.glob(os.path.join(propka_out, "*.pka"))
    if len(pka_files) == 0:
        raise ValueError(f"No .pka files found in {propka_out}")

    records = []

    for file_path in tqdm(pka_files, desc="Parsing PROPKA output"):
        prot = os.path.basename(file_path).replace(".pka", "")

        with open(file_path, "r") as f:
            lines = f.read().splitlines()

        # Find start of charge table
        start_index = None
        for i, line in enumerate(lines):
            if "pH  unfolded  folded" in line:
                start_index = i + 1
                break

        if start_index is None:
            continue

        text = "\n".join(lines[start_index:])

        #Extract pI values
        pI_match = re.search(
            r"The pI is\s+([\d\.]+)\s+\(folded\)\s+and\s+([\d\.]+)\s+\(unfolded\)",
            text
        )

        if pI_match:
            pI_folded = float(pI_match.group(1))
            pI_unfolded = float(pI_match.group(2))
        else:
            pI_folded = None
            pI_unfolded = None

        #Extract charge calculations
        charge_matches = re.findall(
            r"(-?\d+\.\d{2})\s+(-?\d+\.\d{2})\s+(-?\d+\.\d{2})",
            text
        )
        
        #Saves the charge at pH 4 and pH 7 for analysis done in the paper but this can be changed
        charge_4_u = charge_4_f = None
        charge_7_u = charge_7_f = None

        for ph, unfolded, folded in charge_matches:
            ph = float(ph)
            if ph == 4.00:
                charge_4_u = float(unfolded)
                charge_4_f = float(folded)
            elif ph == 7.20:
                charge_7_u = float(unfolded)
                charge_7_f = float(folded)

        records.append([
            prot,
            pI_unfolded,
            pI_folded,
            charge_4_u,
            charge_4_f,
            charge_7_u,
            charge_7_f
        ])

    cols = [
        "name",
        "propka_pI_unfolded",
        "propka_pI_folded",
        "propka_charge_4_unfolded",
        "propka_charge_4_folded",
        "propka_charge_7_2_unfolded",
        "propka_charge_7_2_folded",
    ]

    return pd.DataFrame(records, columns=cols)


def main():
    parser = argparse.ArgumentParser(
        description="Run PROPKA on multiple PDBs and return summarized pI and charge data"
    )

    parser.add_argument("--pdb_dir", required=True,
                        help="Directory containing PDB files")
    parser.add_argument("--propka_exec", required=True,
                        help="Path to propka3 executable")
    parser.add_argument("--propka_out", required=True,
                        help="Directory to store PROPKA output files")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV file")
    parser.add_argument("--ph", type=float, default=7.2,
                        help="pH value passed to PROPKA (default: 7.2) for stability calculations")

    args = parser.parse_args()

    run_propka(
        pdb_dir=args.pdb_dir,
        propka_exec=args.propka_exec,
        propka_out=args.propka_out,
        ph=args.ph
    )

    df = parse_propka_outputs(args.propka_out)
    df.to_csv(args.out_csv, index=False)

    print(f"Saved results to {args.out_csv}")


if __name__ == "__main__":
    main()
