# Projet7

Projet 7 du parcours data scientist d'Openclassrooms.

Le but du projet est de créer un modèle de scoring concernant les clients d'une banque, pour savoir si celle-ci accepte d'accorder le prêt demandé selon le score du client.

Pour chaque client, le modèle de prédiction donne une réponse positive ou négative, ainsi qu'un indice de confiance A/B/C/D (A = confiance élevée, D = confiance basse) basé sur les probabilités de sortie du modèle, ce qui permet de voir plus précisément où se situe le client sur l'échelle de confiance que la banque est prête à lui accorder.

Le dashboard interactif permet au chargé clientèle (et au client) de voir rapidement si le prêt est accordé ou non, l'indice de confiance du client, et de comprendre à travers divers KPIs les raisons de l'acceptation ou du refus de la banque.

Pour lancer l'application (sous Python), l'utilisateur doit lancer la commande suivante dans son invite de commande :

  streamlit run dashboard.py
  
Un nouvel onglet du navigateur par défaut s'ouvre alors et l'application se lance automatiquement une première fois.

Les instructions d'utilisation se trouvent directement sur la page de l'application.

Les packages nécessaires à l'utilisation de l'application se trouvent dans le fichier "requirements.txt".
