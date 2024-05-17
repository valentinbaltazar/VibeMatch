# VibeMatch Web App 

## Setup & Installation

It is good practive to make a new environment and install the required packages (here I call it "vbmatch", you can call it anything):

```
conda create -n vbmatch python=3.9
```

Activate the environment:

```
conda activate vbmatch
```

Go to any directory and clone the repo (Here I am in the Desktop folder called "apps"):

```
cd /Desktop/apps/
```

```bash
git clone https://github.com/valentinbaltazar/VibeMatch.git
```

Install the packages needed (may take a while):
```bash
pip install -r requirements.txt
```

## Running The App

Now run the main.py file and you are set!
```bash
python main.py
```

## Viewing The App
This app has no server (yet) so it will run locally at:

Follow link `http://127.0.0.1:5000`