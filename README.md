```bash
export MY_ENV_NAME=allenact-foundation
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"
conda env create --file ./environment-gpu.yml --name $MY_ENV_NAME

conda activate $MY_ENV_NAME
pip install -e .
pip install -r requirements.txt
pip install imageio==2.4.1  # ai2thor-colab
cd t5x; pip install -e .; cd ..
pip install --editable=git+https://github.com/openai/CLIP.git@e184f608c5d5e58165682f7c332c3a8b4c1545f2#egg=clip

```
