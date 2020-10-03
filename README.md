Avant de compiler le projet, si vous n'avez pas encore téléchargé les paquets du paquet nltk, merci d'enlever les commentaires à la ligne 14 du projet_final.py

Deux variables de configuration possibles :
	*	config_preload : True pour charger les données lémmatisées préchargées, pour gagner du temps
	*	config_tuning : False pour enlever l'étape paramètre-tuning (les hyper-param par défault sont déjà optimisées)

Lien vers les données :
https://drive.google.com/file/d/1Lwu2AdZ9bm2AQudE5ngiCCS3KZPXUQtv/view?usp=sharing

Une fois que vous avez téléchargé l'archive de données. Extrayez-la au dossier contenant projet_final.py. La bonne structure du projet serait :
.
├── execution.txt
├── fonctions.py
├── liste_des_emojis_HTML.txt
├── mbti-type
│   ├── mbti_1.csv
│   ├── testdata_X.csv
│   ├── testdata_y.csv
│   ├── traindata_X.csv
│   └── traindata_y.csv
├── pipeline.py
├── projet_final.py
├── rapport.pdf
└── README.txt


