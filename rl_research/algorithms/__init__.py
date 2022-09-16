from .reinforce import Reinforce
from .reinforce_baseline import ReinforceBaseline
from .reinforce_count_states import ReinforceCountState
from .reinforce_count_states_actions import ReinforceCountStateAction
from .reinforce_count_seq import ReinforceCountSeq
from .reinforce_count_seq_edited import ReinforceCountSeqEdited
from .reinforce_seq_comp import ReinforceSeqComp
from .temporal_difference import td_learning, q_learning
from .training import train

agents = {
     #"reinforce": ReinforceBaseline,
    #"reinforce_st": ReinforceCountState,
    "reinforce_st_at":ReinforceCountStateAction,
    #"reinforce_seq": ReinforceCountSeq,
    "reinforce_seq_edited": ReinforceCountSeqEdited,
    #"reinforce_seq_comp": ReinforceSeqComp
}
