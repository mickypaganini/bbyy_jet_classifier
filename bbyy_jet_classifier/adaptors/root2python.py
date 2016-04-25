from collections import OrderedDict
import ROOT

RTYPE_2_CHAR = { "Int_t":"I", "Double_t":"D", "Float_t":"F" }
CHAR_2_TYPE = { "I":"i4", "D":"f8", "F":"f4" }

def get_tree_variables( input_tree, excluded_variables=[] ) :
  """
  Definition:
  -----------
    Retrieve all branches and types from a ROOT tree

  Args:
  -----
    input_tree = a ROOT tree whose branches we are interested in
    excluded_variables = a list of branch names to be excluded from consideration

  Returns:
  --------
    variable_dict = a dictionary which has the branch names to be used for training as keys
  """
  variable_dict = OrderedDict()
  for leaf in sorted(input_tree.GetListOfLeaves()) :
    variable_name = leaf.GetName()
    if variable_name not in excluded_variables :
      print "Ignoring variable {}!".format(variable_name)
    else :
      variable_dict[variable_name] = RTYPE_2_CHAR[leaf.GetTypeName()]
  return variable_dict



def get_branch_info( input_filename, input_treename, excluded_variables=[] ) :
  """
  Definition:
  -----------
    Retrieve all branches and types from a ROOT tree, given the file name and the tree name

  Args:
  -----
    input_tree = a ROOT tree whose branches we are interested in
    excluded_variables = a list of branch names to be excluded from consideration

  Returns:
  --------
    variable_dict = a dictionary which has the branch names to be used for training as keys
  """
  f_input = ROOT.TFile( input_filename, "READ" )
  variable_dict = get_tree_variables( f_input.Get(input_treename), excluded_variables )
  f_input.Close()
  return variable_dict
