'''
Generate conformers from SMILES or InChI strings.
'''
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors









def conformalize(molecular_identifier, input_type = 'SMILES',
                 n_conf_heuristic = True, n_jobs = 1, seed = -1,
                 maxIters = 2000, useRandomCoords = False, enforceChirality = True):
    '''
    Obtain an approximation for the most stable conformer from a 2D molecular identifier
    based on the MMFF94s force field.


    Parameters
    ----------
    molecular_identifier : str
        It can be either an InChI or a SMILES string.
    input_type : {'SMILES', 'InChI'}, default = 'SMILES'
    n_conf_heuristic : bool, default = True
        Whether to use an heuristic based on the number of rotatable bonds
        to select the number of conformers generated. Otherwise, 300 will be obtained.
    n_jobs : int, default = 1
        Number of parallel jobs. Set to 0 for the maximum supported by the system.
    seed : int, default = -1
        Random seed for generating conformers. If -1 (default), the RNG will not be seeded.
    maxIters : int, default = 2000
        Maximum number of iterations when performing force field energy optimization.
        Set higher in case the returned error is 5.
    useRandomCoords : bool, default = False
        Whether to use random coordinates for the initial geometry or rather from the distance matrix.
        It might be useful to turn on in case the returned error is 3 or 4 (see below).
    enforceChirality : bool, default = True
        Whether to enforce the conformer embedder to respect the input stereochemistry.
        It might be useful to turn on in case the returned error is 3 or 4
        (see below, recommended to try useRandomCoords = True first).
    
        
    Returns
    -------
    conformer : rdkit.Chem.rdchem Mol object or None
        Explicit hydrogens will automatically be added.
    error : int
        0 no error
        1 for input_type ValueError
        2 for molecular identifier reading fail
        3 for conformer enumeration error
        4 for force field optimization
        5 for convergence fail of the force field optimization
    '''
    # --- 1. Read-in the molecule
    if input_type == 'SMILES':
        molecule = Chem.MolFromSmiles(molecular_identifier) # generate molecule with sanitizer on
    elif input_type == 'InChI':
        molecule = Chem.MolFromInchi(molecular_identifier) # explicit hydrogens are removed for better computational efficiency
    else:
        return None, 1
    '''
    When reading molecules, it automatically does sanitation (SanitizeMol() is different from the others: 
    it does a small amount of normalization - fixing groups like nitro which are commonly drawn in a
    hypervalent state but which can be represented in a charge-separated form without needing weird valences - and some validation - 
    rejecting molecules with atoms that have non-physical valences, rejecting molecules that cannot be kekulized - and a bunch of 
    chemistry perception - ring finding, calculating valences, finding aromatic systems, etc.)
    (https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg10668.html)
    '''
    if molecule is None:
        return None, 2
    molecule = Chem.AddHs(molecule)



    # --- 2. Select number of conformers to generate (https://pubs.acs.org/doi/10.1021/ci2004658)
    n_rots = rdMolDescriptors.CalcNumRotatableBonds(molecule)
    if n_rots == 0:
        num_conformers = 1
    else:
        if n_conf_heuristic:
            if n_rots <= 7:
                num_conformers = 50
            elif n_rots <= 12:
                num_conformers = 200
            else:
                num_conformers = 300
        else:
            num_conformers = 300



    # --- 3. KDG embedding
    '''
    the ET terms bias the sampling of conformer space to favor torsion values that have been observed in crystal structures. 
    These are quite useful if you want conformers for condensed phases or for protein-ligand docking, 
    but can miss regions of conformer space that you may want to sample if you are working with compounds in the gas phase or solution. 
    For those cases you are probably best served by sticking with KDG, not ETKDGv3.
    (https://github.com/rdkit/rdkit/discussions/8226)
    '''
    try:
        params = AllChem.KDG()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.5 # from https://pubs.acs.org/doi/10.1021/ci2004658
        params.numThreads = n_jobs
        # maxAttempts=0 by default decide the maximum number of attempts automatically
        params.useRandomCoords = useRandomCoords
        params.enforceChirality = enforceChirality
        conformers = list(AllChem.EmbedMultipleConfs(
            molecule,
            numConfs=num_conformers,
            params=params
        ))

    except Exception:
        return None, 3



    # --- 4. MMFF94s force field optimization
    '''
    MMFF94 is particularly good with organic compounds.
    MMFF94 and MMFF94s use the same functional form to calculate the potential energy. 
    They only differ in the Torsion and Out-Of-Plane bending parameters used. 
    The s in MMFF94s stands for static and this set of parameters is more suited for tasks where the output is static.
    (https://avogadro.cc/docs/optimizing-geometry/molecular-mechanics/)
    '''
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(
            molecule,
            numThreads = n_jobs,
            maxIters = maxIters, # RDKit default 200
            mmffVariant='MMFF94s'
        )
    except Exception:
        return None, 4



    # --- 5. select only best conformer based on FF energy optimization
    ranking = [
        (conf_id, not_converged, energy)
        for conf_id, (not_converged, energy) in zip(conformers, results)
    ]
    ranking.sort(key=lambda x: (x[1], x[2])) # converged first, then lowest energy

    best_conf_id, not_converged, _ = ranking[0]
    if not_converged == 1:
        return None, 5
    


    best_molecule = Chem.Mol(molecule) # better than copy.deepcopy(mol) https://sourceforge.net/p/rdkit/mailman/message/33652439/
    best_molecule.RemoveAllConformers()
    best_molecule.AddConformer(molecule.GetConformer(best_conf_id), assignId=True)
    return best_molecule, 0