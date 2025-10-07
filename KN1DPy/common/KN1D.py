# Contains Class definitions for shared variables primarily used in KN1D

#   From the KN1D_internal common block
class KN1D_Internal:

    def __init__(self):

        self.fH_s = None
        self.fH2_s = None
        self.nH2_s = None
        self.SpH2_s = None
        self.nHP_s = None
        self.THP_s = None

    #Setup string conversion for printing
    def __str__(self):
        string = "KN1D_Internal:\n"
        string += "    fH_s: " + str(self.fH_s) + "\n"
        string += "    fH2_s: " + str(self.fH2_s) + "\n"
        string += "    nH2_s: " + str(self.nH2_s) + "\n"
        string += "    SpH2_s: " + str(self.SpH2_s) + "\n"
        string += "    nHP_s: " + str(self.nHP_s) + "\n"
        string += "    THP_s: " + str(self.THP_s) + "\n"

        return string