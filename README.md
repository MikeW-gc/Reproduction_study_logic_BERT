# Reproduction_study_logic_BERT

### Original Study (Wang \& Pan) Repo at https://github.com/happywwy/RuleFusionForIE

Dependencies of this project in requirements.txt (Any general ML env with torch(with CUDA Nvidia) should be able to run all the scripts.

Re-used code/data: main_trec.py & utils.py and everything in data came from Wang \& Pan

To run the original end-to-end entity & relation extraction pipeline, run 'python ./main_trec.py'

To run the original end-to-end entity & relation extraction pipeline without logic loss, run 'python ./main_trec_withoutlogic.py'

To run the BERT end-to-end entity & relation extraction pipeline without logic loss, run 'python ./bert.py'

To run the logically infused BERT end-to-end entity & relation extraction pipeline without logic loss, run 'python ./bert_with_logic.py'
