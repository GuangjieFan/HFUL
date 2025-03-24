Mutual Unlearning Across Different Identities in Heterogeneous Federated Learning
Abstract
The Heterogeneous Federated Learning (HFL) framework is an emerging distributed learning paradigm that enables participants to train their own models independently, without aggregating them into a unified global model as in traditional federated learning. Consequently, traditional federated unlearning methods, which are based on the conventional federated learning framework, struggle to effectively handle the unlearning needs of participants in HFL tasks.To address this issue, we introduce a novel heterogeneous federated unlearning (HFUL) algorithm. The algorithm uses Low-Rank Adaptation (LoRA) to fine-tune the local models of participants requesting unlearning, removing out-domain knowledge. It also incorporates masking operations on the cross-correlation matrix to block the acquisition of out-domain knowledge during unlearning, and dynamically adjusts distillation ratios across models to facilitate mutual unlearning between the requesting participant and others. Our extensive evaluations show that HFUL effectively reduces the influence of unlearned data by an average of 7.91%, with a corresponding decrease of 3.58% in local model performance.
Heterogeneous Federated Learning
@inproceedings{FCCL_CVPR22,
    title={Learn from others and be yourself in heterogeneous federated learning},
    author={Huang, Wenke and Ye, Mang and Du, Bo},
    booktitle={CVPR},
    year={2022}
}
