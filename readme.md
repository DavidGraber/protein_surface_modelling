# Protein Molecular Surface Fingerprinting on Mutants of GB1 protein

Input Data: Sequence of 23'400 mutants of GB1 protein (protein G subunit B1), whose structures were predicted with AlphaFold. Around each mutant, a discretized molecular surface with geometric and chemical features has been computed (see https://github.com/FreyrS/dMaSIF). The WT GB1 protein binds to the constant region of human IgG with high affinity. The GB1 variants are randomly mutated at four selected sites. Data of the mutant's binding affinity to IgG is taken from laboratory measurements (see https://elifesciences.org/articles/16965)

This repository contains code for molecular surface fingerprinting and prediction of GB1-IgG interaction. 
- The interaction of the WT GB1 protein with IgG is examined closely and a radial patch of radius 12A (containing the most important AA-residues that are involved in the interaction) is extracted from the discretized surface. 
- The geodesic distances between the surface points are extracted and used to flatten the patches into the plane with MDS.
- The flattened patches are mapped into a polar grid with 72 angular and 10 radial bins, generating a "fingerprint" of the extracted surface patch. 
- These fingerprints are generated for various mutants of GB1 and are used to train a convolutional neural network that aims to reduce the dimensionality of the fingerprints and predict the binding affinity of the mutants. 
