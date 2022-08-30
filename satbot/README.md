### SAT chatbot web app

#### Notes: 

1) Before running the code in this folder, please obtain the classifiers by running the notebooks in the language_models folder or download the trained models in Google Drive (links are in the language_models folder)

2) You may need to change the file paths in 'classifiers.py' and 'rule_based_model.py' to your local paths when running locally

3) This chatbot uses the react-chatbot-kit library: https://fredrikoseberg.github.io/react-chatbot-kit-docs/


#### To run the code in this folder locally, after cloning open a terminal window and do:

$ conda create -n sat3 python=3.8

$ conda activate sat3

$ cd ./model

$ python3 -m pip install -r requirements.txt

$ set FLASK_APP=flask_backend_with_aws

$ flask db init

$ flask db migrate -m "testDB table"

$ flask db upgrade

$ nano .env   ---->  add DATABASE_URL="sqlite:////YOUR LOCAL PATH TO THE app.db FILE" to the .env file, save and exit (relative path works as well)

$ flask run


#### To launch the front end, open another terminal tab and do:

$ cd ./SATbot/view

$ npm install

$ npm run start
