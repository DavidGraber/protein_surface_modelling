from Bio.PDB.PDBParser import PDBParser
import numpy as np

def parse_pdb(parser, protein_id, filename):
    
    protein_id = parser.get_structure(protein_id, filename)
    models = list(protein_id.get_models())
    chains = list(models[0].get_chains())
    residues = list(chains[0].get_residues())
    
    protein = {p+1:{} for p in range(len(residues))}
    protein["residues"] = residues
    
    prot_coords = np.zeros((1,3))
    for i, residue in enumerate(residues):
        atoms = list(residue.get_atoms())
        protein[i+1]["atoms"]=atoms
        coords = []
        for atom in atoms: 
            coord = list(atom.get_vector())
            coords.append(coord)
        protein[i+1]["coords"] = np.asarray(coords)
        prot_coords = np.concatenate((prot_coords, np.asarray(coords)))
    
    protein["coords"] = prot_coords
        
    return protein