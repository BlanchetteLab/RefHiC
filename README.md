# RefHiC
A reference panel guided topological structure annotation of Hi-C data

## Installation
RefHiC relies on several libraries including pytorch, scipy, etc. We suggest users using conda to create a virtual environment for it. You can run the command snippets below to install RefHiC:
<pre>
git clone https://github.com/BlanchetteLab/RefHiC.git
cd RefHiC
conda create --name refhic  --file requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda activate refhic
pip install torchmetrics
pip install -U git+https://github.com/fadel/pytorch_ema
pip install tensorboard
pip install torchlars
pip install -U scikit-learn
pip install --editable .
</pre>
