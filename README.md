# 🚀 Projet : Quantization Flux 4-bits pour Petits GPU

Bienvenue dans ce projet expérimental ! L'objectif ici est de vous permettre de **tester et d'exécuter des modèles basés sur l'architecture Flux** en utilisant une **quantization 4-bits ultra-légère**, spécifiquement conçue pour les **GPU avec des ressources VRAM limitées** (y compris les puces Apple Silicon comme M1, M2, M3).

Dites adieu aux erreurs de mémoire lors de l'exécution de modèles gourmands sur votre matériel modeste !

## ✨ Fonctionnalités Clés

- **Quantization 4-bits Ultra-Légère :** Réduction drastique de l'empreinte mémoire des modèles pour tenir sur de petits GPU.
- **Optimisé Flux :** Se concentre sur l'expérimentation avec les modèles basés sur l'architecture Flux.
- **Démarrage Simplifié :** Configurez et lancez le projet rapidement grâce à `make`.
- **Support Apple Silicon (M1/M2/M3) :** Instructions incluses pour utiliser le backend Metal.

## 📋 Prérequis

Avant de commencer, assurez-vous que les outils suivants sont installés sur votre système :

- [**Docker**](https://docs.docker.com/get-docker/) : Pour construire et exécuter le conteneur de l'application.
- [**Make**](https://www.gnu.org/software/make/) : Pour automatiser les étapes d'initialisation et de démarrage.
- **Clé API Hugging Face :** Nécessaire pour télécharger les modèles. Si vous n'en avez pas, créez-en une ici : [Vos Tokens d'Accès Hugging Face](https://huggingface.co/settings/tokens).

## 🚀 Démarrage Rapide

Suivez ces étapes simples pour mettre en place et lancer l'environnement :

1.  **Configuration de l'environnement :**
    Créez le fichier de configuration de votre environnement local en copiant l'exemple fourni :

    ```bash
    cp .env.example .env
    ```

2.  **Ajout de votre Clé Hugging Face :**
    Ouvrez le fichier `.env` que vous venez de créer et ajoutez votre clé API Hugging Face à la ligne `HF_API_KEY`.

    ```dotenv
    # Exemple de contenu de votre fichier .env
    HF_API_KEY=votre_super_cle_huggingface_ici
    ```

    N'oubliez pas que vous pouvez trouver ou gérer votre clé ici : [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3.  **Initialisation du Projet :**
    Lancez la commande `make init`. Cette étape va construire l'image Docker nécessaire pour le projet.

    ```bash
    make init
    ```

    _Cette commande peut prendre un certain temps lors de la première exécution car elle télécharge les dépendances et construit l'image Docker._

4.  **Lancement de l'Application :**
    Une fois l'initialisation terminée, vous pouvez lancer l'application principale avec la commande `make start`.
    ```bash
    make start
    ```
    Cette commande démarre le conteneur Docker et exécute le script Python principal (`flux.py`) qui génère l'image.

## 🔧 Amélioration de la Résolution et de la Qualité

Le script de génération d'image (`flux.py`) utilise plusieurs paramètres qui influencent la taille et la qualité de l'image finale. Vous pouvez les modifier directement dans ce fichier Python pour expérimenter.

Voici les paramètres clés que vous pourriez vouloir ajuster dans `flux.py` :

1.  **Modèle d'Upscale (`model_path`) :**

    - Dans la fonction `upscale_image`, la ligne `model_path = os.path.join("models", "modelx4.ort")` spécifie le modèle ONNX utilisé pour l'upscaling.
    - Par défaut, il utilise `modelx4.ort` pour un facteur d'agrandissement de x4.
    - **Pour utiliser un modèle x2**, changez cette ligne en :
      ```python
      model_path = os.path.join("models", "modelx2.ort")
      ```
    - _Note : Assurez-vous d'avoir le fichier `modelx2.ort` dans le dossier `models/` de votre projet._

2.  **Taille de l'Image de Base :**

    - Dans la fonction `generate_image`, les lignes `base_image_width = int(image_with // 4)` et `base_image_height = int(image_height // 4)` définissent les dimensions de l'image générée par le pipeline Flux _avant_ l'upscaling.
    - Elles divisent les dimensions finales souhaitées (`image_with`, `image_height`) par 4.
    - **Pour générer une image de base plus grande** (ce qui, combiné à un upscale x2, peut donner une meilleure qualité), changez la division par 4 en division par 2 :
      ```python
      base_image_width = int(image_with // 2)
      base_image_height = int(image_height // 2)
      ```
    - _Attention : Augmenter la taille de l'image de base consomme plus de VRAM sur votre GPU._

3.  **Nombre d'Étapes d'Inférence (`num_inference_steps`) :**
    - La variable `num_inference_steps` (définie en début de script et utilisée dans `pipe_args`) contrôle le nombre d'étapes que le pipeline Flux utilise pour générer l'image de base.
    - Un nombre d'étapes plus élevé peut potentiellement améliorer la qualité et les détails de l'image, mais cela augmente aussi le temps de génération.
    - **Pour augmenter le nombre d'étapes**, modifiez la valeur de cette variable. Par exemple, pour passer de 28 à 40 étapes :
      ```python
      num_inference_steps=40
      ```

**En combinant ces modifications (utiliser `modelx2.ort`, diviser la taille de base par 2, et augmenter `num_inference_steps`), vous générerez une image de base deux fois plus grande qui sera ensuite upscalée par un facteur de 2, résultant en une image finale de la même dimension cible (`image_with` x `image_height`), mais potentiellement avec une meilleure qualité perçue grâce à l'image de base plus grande et aux étapes d'inférence supplémentaires.**

N'hésitez pas à expérimenter avec ces valeurs pour trouver le meilleur équilibre entre qualité, vitesse et consommation de mémoire pour votre configuration matérielle.

## 🍎 Note pour les utilisateurs de Mac (Apple Silicon - M1/M2/M3)

Ce projet est conçu pour fonctionner sur des GPU avec des ressources limitées, y compris les puces Apple Silicon.

Pour vous assurer que PyTorch utilise correctement le backend Metal (qui remplace CUDA sur Mac) :

1.  **Vérifiez votre installation de PyTorch :** Assurez-vous d'avoir installé la version de PyTorch compatible avec 'mps' (Metal Performance Shaders). L'installation via `pip install torch torchvision torchaudio` dans un environnement compatible (comme dans votre Dockerfile si vous adaptez l'image de base ou si vous l'installez localement) devrait gérer cela.

---

Bonne experimentation avec la quantization 4-bits ! ✨
