# import os
# import glob
# import re
# from PIL import Image

# def natural_key(string_):
#     return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

# def processar_imagem(caminho, tamanho_maximo):
#     """
#     Abre a imagem, converte para RGB e reduz a resolução.
#     tamanho_maximo: tupla (largura, altura) - ex: (800, 800)
#     """
#     img = Image.open(caminho)
    
#     # Reduz a resolução mantendo a proporção (thumbnail NUNCA aumenta a imagem)
#     img.thumbnail(tamanho_maximo, Image.Resampling.LANCZOS)  # LANCZOS dá a melhor qualidade
    
#     # Converte para RGB (necessário para GIF)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
    
#     return img

# def gerar_imagens(arquivos, tamanho_maximo):
#     """Gerador que processa uma imagem por vez, economizando memória."""
#     for arquivo in arquivos:
#         yield processar_imagem(arquivo, tamanho_maximo)

# def criar_gif(prefixo, step=5, tamanho_maximo=(800, 800)):
#     """
#     Cria um GIF pulando frames e com resolução reduzida.
    
#     Parâmetros:
#     - prefixo: nome base dos arquivos
#     - step: pega 1 a cada 'step' frames (ex: 5)
#     - tamanho_maximo: tupla (largura, altura) para redimensionar
#     """
#     padrao = os.path.join("plots", f"{prefixo}_*.png")
#     arquivos = sorted(glob.glob(padrao), key=natural_key)

#     if not arquivos:
#         print(f"Nenhuma imagem encontrada em: {padrao}")
#         return

#     # Pula frames para reduzir a quantidade
#     arquivos_filtrados = arquivos[::step]

#     if not arquivos_filtrados:
#         print("Nenhum frame restante após o filtro.")
#         return

#     gif_path = f"animacao_{prefixo}.gif"

#     # Processa a PRIMEIRA imagem (fora do gerador, pois ela é a base do save)
#     primeira = processar_imagem(arquivos_filtrados[0], tamanho_maximo)

#     # Ajusta a duração para manter a velocidade original
#     duracao_ajustada = 50 * step

#     # Salva usando o gerador para as demais imagens
#     primeira.save(
#         gif_path,
#         save_all=True,
#         append_images=gerar_imagens(arquivos_filtrados[1:], tamanho_maximo),
#         duration=duracao_ajustada,
#         loop=0,
#         optimize=True  # <-- BÔNUS: ativa a otimização interna do GIF (remove pixels duplicados)
#     )
    
#     primeira.close()
    
#     # Pega o tamanho do arquivo em MB para mostrar
#     tamanho_mb = os.path.getsize(gif_path) / (1024 * 1024)
#     print(f"GIF criado: {gif_path}")
#     print(f"  -> {len(arquivos_filtrados)} frames")
#     print(f"  -> Duração: {duracao_ajustada}ms por frame")
#     print(f"  -> Resolução máxima: {tamanho_maximo[0]}x{tamanho_maximo[1]}")
#     print(f"  -> Tamanho do arquivo: {tamanho_mb:.2f} MB")

# # Execução (ajuste o tamanho_maximo conforme sua necessidade)
# # - (1200, 1200) para qualidade alta porém menor
# # - (800, 800)  para qualidade média (recomendado)
# # - (600, 600)  para qualidade baixa e arquivo bem leve
# criar_gif("velocity", step=5, tamanho_maximo=(600, 600))
# criar_gif("pressure", step=5, tamanho_maximo=(600, 600))
# criar_gif("vorticity", step=5, tamanho_maximo=(600, 600))




# import os
# import glob
# import re
# from PIL import Image

# def natural_key(string_):
#     return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

# def gerar_imagens(arquivos):
#     """Gerador que abre uma imagem por vez, economizando memória."""
#     for arquivo in arquivos:
#         img = Image.open(arquivo)
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         yield img

# def criar_gif(prefixo, step=5):
#     """
#     Cria um GIF pulando frames para reduzir o tamanho do arquivo.
#     step: pega 1 a cada 'step' frames (ex: step=5 reduz de 1500 para 300 frames).
#     """
#     padrao = os.path.join("plots", f"{prefixo}_*.png")
#     arquivos = sorted(glob.glob(padrao), key=natural_key)

#     if not arquivos:
#         print(f"Nenhuma imagem encontrada em: {padrao}")
#         return

#     # --- PULA FRAMES AQUI (ex: pega 1 a cada 5) ---
#     arquivos_filtrados = arquivos[::step]  
#     # ----------------------------------------------

#     if not arquivos_filtrados:
#         print("Nenhum frame restante após o filtro.")
#         return

#     gif_path = f"animacao_{prefixo}.gif"

#     # Abre a primeira imagem
#     primeira = Image.open(arquivos_filtrados[0])
#     if primeira.mode != 'RGB':
#         primeira = primeira.convert('RGB')

#     # Ajusta a duração para manter a velocidade original
#     duracao_ajustada = 50 * step  # Se step=5, vira 250ms por frame

#     # Salva usando o gerador para as demais imagens
#     primeira.save(
#         gif_path,
#         save_all=True,
#         append_images=gerar_imagens(arquivos_filtrados[1:]),
#         duration=duracao_ajustada,
#         loop=0
#     )
    
#     primeira.close()
#     print(f"GIF criado: {gif_path} ({len(arquivos_filtrados)} frames, duração {duracao_ajustada}ms)")

# # Execução (agora com step=5 para ficar leve)
# criar_gif("velocity", step=5)
# criar_gif("pressure", step=5)
# criar_gif("vorticity", step=5)






import os
import glob
import re
from PIL import Image

def natural_key(string_):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

def gerar_imagens(arquivos):
    """Gerador que abre uma imagem por vez, evitando carregar todas na memória."""
    for arquivo in arquivos:
        img = Image.open(arquivo)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        yield img  # Retorna a imagem e 'pausa' o loop até a próxima iteração

def criar_gif(prefixo):
    # Busca os arquivos e aplica a ordenação natural (corrigido: removida a linha duplicada)
    padrao = os.path.join("plots", f"{prefixo}_*.png")
    arquivos = sorted(glob.glob(padrao), key=natural_key)

    if not arquivos:
        print(f"Nenhuma imagem encontrada em: {padrao}")
        return

    gif_path = f"animacao_{prefixo}.gif"

    # Abre a PRIMEIRA imagem separadamente (ela vai ficar na memória como base)
    primeira = Image.open(arquivos[0])
    if primeira.mode != 'RGB':
        primeira = primeira.convert('RGB')

    # Salva o GIF passando o GERADOR para as imagens restantes.
    # O Pillow vai consumir o gerador frame a frame, escrevendo no disco
    # e descartando cada imagem da memória assim que usada.
    primeira.save(
        gif_path,
        save_all=True,
        append_images=gerar_imagens(arquivos[1:]),  # <--- AQUI ESTÁ A MÁGICA
        duration=25,
        loop=0
    )
    
    primeira.close()
    print(f"GIF criado com sucesso: {gif_path} ({len(arquivos)} frames)")

# Execução
criar_gif("velocity")
criar_gif("pressure")
criar_gif("vorticity")



