# Utility script for feature generation
# Md Mahmudulla Hassan
# The University of Texas at El Paso
# Last Modified: 12/19/2018

import os
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import shutil

MAYACHEMTOOLS_DIR = "~/sfs0/mayachemtools"

class FeatureGenerator:
	""" 
	This is a class for mathematical operations on SMILE to generate molecular fingerprints
      
    Attributes: 
        SMILE (sring): SMILE string
	""" 

    def __init__(self, smiles):
    	""" 
        The constructor for FeatureGenerator class. 
  
        Parameters: 
           SMILE (sring): SMILE string 
           """ 
        self.smiles = smiles
        self.temp_dir = tempfile.mkdtemp()


    def toString(self):
        return self.smiles


    def toSDF(self):
    	""" 
    	The function to convert SMILE to SDF format. 
          
        Returns: 
            SDF: SDF file from supplied SMILE is returned and saved a temp dir 
    	""" 
        # Try to get the rdkit mol
        mol = Chem.MolFromSmiles(self.smiles)
        #if mol == None: raise("Error in mol object")
        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)
        mol.SetProp("smiles", self.smiles)
        #self.sdf_filepath = os.path.join(self.temp_dir, "temp.sdf")
        w = Chem.SDWriter(os.path.join(self.temp_dir, "temp.sdf"))
        w.write(mol)
        w.flush()


    def toTPATF(self):
    	""" 
    	The function to calculate the Topological Pharmacophore Atom Triplets Fingerprints
          
        Returns: 
            SDF: Topological Pharmacophore Atom Triplets Fingerprints from supplied SDF file.
    	""" 
        features = []
        script_path = os.path.join(MAYACHEMTOOLS_DIR,
                "bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl")
        self.toSDF()
        # Now generate the TPATF features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")):
            print("Error: sdf file could not be created.")
            return None
        command = ("perl " + script_path + " -r " +
                   os.path.join(self.temp_dir, "temp") +
                   " --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o "
                   + os.path.join(self.temp_dir, "temp.sdf"))
        os.system(command)
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]
        # Clean up the temporary files
        self._cleanup()
        return features


    def toTPAPF(self):
    	""" 
    	The function to calculate the Topological Pharmacophore Atom Pairs Fingerprints 
          
        Returns: 
            SDF: Topological Pharmacophore Atom Pairs Fingerprints from supplied SDF file.
    	""" 
        features = []
        script_path = os.path.join(MAYACHEMTOOLS_DIR,
                "bin/TopologicalPharmacophoreAtomPairsFingerprints.pl")
        # Generate the sdf file
        self.toSDF()
        # Now generate the TPATF features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")):
            print("Error: sdf file could not be created.")
            return None
        command = ("perl " + script_path + " -r " +
                   os.path.join(self.temp_dir, "temp") +
                   " --AtomPairsSetSizeToUse FixedSize -v ValuesString -o " +
                   os.path.join(self.temp_dir, "temp.sdf"))
        os.system(command)
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]
        # Clean up the temporary files
        self._cleanup()
        return features


    def _cleanup(self):
        shutil.rmtree(self.temp_dir)

# Example: Extracting TPATF features
#  from utility import FeatureGenerator
#  ft = FeatureGenerator("O=C(Cc1ccccc1)Nc2ncc(s2)C3CCC3")
#  features = ft.toTPATF()