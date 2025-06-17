
# 1. Setup System requirements
# NOTE: not nessesary !!

# apt-get update &&  apt-get install -y   software-properties-common &&  add-apt-repository ppa:deadsnakes/ppa -y 
# apt-get update 

# apt-get install -y    python3.10    python3.10-venv    python3.10-distutils    python3-pip    wget    git    libgl1    libreoffice    fonts-noto-cjk    fonts-wqy-zenhei    fonts-wqy-microhei    ttf-mscorefonts-installer    fontconfig    libglib2.0-0    libxrender1    libsm6    libxext6    poppler-utils \

# update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


#____ 

# 2. Install python Modules
#
# TODO: : first check if venv already exist do not recreate it and skip this part 
#
python3 -m venv venv
source venv/bin/activate
pip install -r path/to/repo/requirements-server.txt


# ## 3. download Models


# %% [markdown]
# ## 4. configure GPUs

wget https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json &&  cp magic-pdf.template.json /root/magic-pdf.json
sed -i 's|cpu|cuda|g' /root/magic-pdf.json
cat /root/magic-pdf.json

# %% [markdown]
# ## 5. Code [Flask server]

# rm $(ls ./ ) -r
