#!/bin/bash

if [[ "$(basename -- "$0")" == "setup_venv.sh" ]]; then
    echo "Don't run $0, source it." >&2
    echo "Script execution terminated."
    exit 1
fi

dpkg -s "python3-venv" &> /dev/null

if [[ ! $? -eq 0 ]]; then
    echo "Ensure python3-venv is installed." >&2
    return
fi

VENV_NAME="engage_venv"

mkdir "$PWD/$VENV_NAME"
python3.6 -m venv "$$VENV_NAME"
source "$PWD/$VENV_NAME/bin/activate"

pip3 --no-cache-dir install numpy
pip3 --no-cache-dir install pandas
pip3 --no-cache-dir install matplotlib
pip3 --no-cache-dir install seaborn
pip3 --no-cache-dir install gensim
pip3 --no-cache-dir install sklearn
pip3 --no-cache-dir install joblib
pip3 --no-cache-dir install future
pip3 --no-cache-dir install pyhsmm
pip3 uninstall scipy
pip3 --no-cache-dir install scipy==0.19.1
#pip3 uninstall scipy
#pip3 --no-cache-dir install scipy==1.1.0
#pip3 install git+git://github.com/NickHoernle/pyhsmm.git@155fb49065f4296800c479760dd5196cdb691cde
