import os
import shutil
import random

def create_test_set(source_dir, test_dir, test_ratio=0.1):
    # Certifique-se de que o diretório de teste existe
    os.makedirs(test_dir, exist_ok=True)

    for category in ['normal', 'parkinson']:
        # Crie os diretórios de categoria no diretório de teste
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # Lista todas as imagens na categoria
        source_category_dir = os.path.join(source_dir, category)
        images = os.listdir(source_category_dir)

        # Calcule o número de imagens para mover
        num_test = int(len(images) * test_ratio)

        # Selecione aleatoriamente as imagens para o conjunto de teste
        test_images = random.sample(images, num_test)

        # Mova as imagens selecionadas para o diretório de teste
        for image in test_images:
            source_path = os.path.join(source_category_dir, image)
            dest_path = os.path.join(test_dir, category, image)
            shutil.move(source_path, dest_path)

        print(f"Movidas {num_test} imagens de {category} para o conjunto de teste.")

if __name__ == "__main__":
    source_dir = "imagens/parkinsons_dataset"
    test_dir = os.path.join(source_dir, "test")
    
    create_test_set(source_dir, test_dir)

    print("Separação do conjunto de teste concluída.")