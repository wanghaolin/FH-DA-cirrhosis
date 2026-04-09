"""
The :mod:`deslib.des` provides a set of key dynamic ensemble selection
algorithms (DES).
"""

from .base import BaseDES
from .des_clustering import DESClustering
from .des_knn import DESKNN
from .des_mi import DESMI
from .des_p import DESP
from .knop import KNOP
from .knora_e import KNORAE
from .knora_u import KNORAU

from .des_fh import DESFH
from .des_FHMW_JFB_vector import DESFHMW_JFB_vector
from .des_FHMW_prior_vector import DESFHMW_prior_vector
from .des_FHMW_AllBoxes_vector import DESFHMW_allboxes_vector
from .fh_des_AllBoxes_vector import FHDES_Allboxes_vector
from .fh_des_JFB_vector import FHDES_JFB_vector
from .fh_des_prior_vector import FHDES_prior_vector


from .meta_des import METADES
from my_deslib.des.probabilistic.base import BaseProbabilistic
from my_deslib.des.probabilistic.minimum_difference import MinimumDifference
from my_deslib.des.probabilistic.deskl import DESKL
from my_deslib.des.probabilistic.rrc import RRC
from my_deslib.des.probabilistic.exponential import Exponential
from my_deslib.des.probabilistic.logarithmic import Logarithmic

__all__ = ['BaseDES',

           'DESFH',
           'DESFHMW_JFB_vector',
           'DESFHMW_allboxes_vector',
           'DESFHMW_prior_vector',
           'FHDES_Allboxes_vector',
           'FHDES_JFB_vector',
           'FHDES_JFB_DE',
           'FHDES_prior_vector',

           'METADES',
           'KNORAE',
           'KNORAU',
           'KNOP',
           'DESP',
           'DESKNN',
           'DESClustering',
           'DESMI',
           'BaseProbabilistic',
           'RRC',
           'DESKL',
           'MinimumDifference',
           'Exponential',
           'Logarithmic']
