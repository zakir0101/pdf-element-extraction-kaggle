#_____________________________________________________________
# 0. Setup System requirements


# if [ ! -f ~/mineru.json	 ];then 
# 	cp -T config.json ~/mineru.json	 
# 	echo $'copied the conf.json to \$HOME/mineru.json ..\n'
# else
# 	echo $'~/mineru.json already exist !\n'
# fi

#_____________________________________________________________

# 1. Install python Modules

if [  -d "$1" ]; then
	echo $'venv already exist !\n';
else
	echo $'creating python virtual enviroment (./$1)\n';
	python3 -m venv $1;
fi

echo "*********************************************************************"
echo "*********************** START PIP INSTALL ***************************"
echo $'*********************************************************************\n'

source $1/bin/activate
pip install uv
uv pip install -r requirements-local.txt

echo $'\n***************************  [DONE]  *******************************\n'

echo $'finished installing dependencies\n' 

#_____________________________________________________________

# %% [markdown]
#  3. downloading models 


if [ ! -d ../cache ];then 
	echo $'setting cache dir\n';
	mkdir ../cache;
	export HF_HUB_CACHE=$(realpath ../cache);
else
	echo $'cache dir already exist !\n';
fi

echo $'setting model-source == huggingface\n'
export MINERU_MODEL_SOURCE=huggingface

echo $'start downloading models ...\n '

echo "*********************************************************************"
echo "************************* START DOWNLOAD ****************************"
echo $'*********************************************************************\n'

mineru-models-download -m all -s huggingface

echo $'\n***************************  [DONE]  *******************************\n'

echo $'setting model-source == local\n'
export MINERU_MODEL_SOURCE=local
