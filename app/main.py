
from loguru import logger
import torch
import traceback
import torch
import dotenv
from tools.flux import generate_image

dotenv.load_dotenv()

def run_agent():
    logger.debug("\n" + "="*60)
    logger.debug("🌟 GÉNÉRATEUR D'IMAGES OPTIMISE 4bits")
    logger.debug("="*60)
    
    user_theme = input("\n🧑 Description de l'image : ")
    
    try:
       path = generate_image(user_theme)
       logger.debug(f"Image enregistrée : {path}")
    except KeyboardInterrupt:
        logger.debug("\n\n⚠️ Interruption manuelle détectée. Nettoyage des ressources...")
        torch.cuda.empty_cache()
        logger.debug("Programme arrêté par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de l'exécution du workflow: {e}")
        logger.error(traceback.format_exc())
    
    run_agent()

if __name__ == "__main__":
    logger.info("=== Démarrage du générateur d'images manga par signe astrologique ===")
    run_agent()