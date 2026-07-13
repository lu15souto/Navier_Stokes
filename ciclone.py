
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 00:59:43 2026

@author: luisguilhermesoutomiranda

Projeto:

Descrição:


Fluxo logico do codigo:

INÍCIO
│
├─ 1. DEFINIR parâmetros físicos e numéricos
│    (U_inf, D, Re, nu, Cs, dimensões do canal, dt, T_end, AMR)
│
├─ 2. GERAR MALHA INICIAL (Gmsh)
│    ├─ Criar retângulo (canal)
│    ├─ Criar disco (cilindro)
│    ├─ Subtrair disco do retângulo (operação booleana)
│    ├─ Aplicar campo de tamanho para refino local (atualmente linear)
│    ├─ Gerar malha triangular 2D
│    ├─ Converter .msh → .xdmf (via Meshio)
│    └─ Carregar malha no FEniCS
│
├─ 3. CONFIGURAR ESPAÇOS FUNCIONAIS
│    ├─ V = VectorFunctionSpace(P2)  → Velocidade
│    ├─ Q = FunctionSpace(P1)        → Pressão
│    └─ W = MixedFunctionSpace(V × Q) → Sistema acoplado
│
├─ 4. DEFINIR CONDIÇÕES DE CONTORNO
│    ├─ Entrada (Inlet): u_x = U_inf, u_y = 0
│    ├─ Cilindro: u_x = 0, u_y = 0 (no-slip)
│    ├─ Paredes superior/inferior: u_y = 0 (slip)
│    └─ Saída: condição natural (Neumann implícita)
│
├─ 5. DEFINIR MODELO LES E FORMA FRACA
│    ├─ Função nu_t(u): calcula viscosidade turbulenta (Smagorinsky)
│    ├─ Função b_form(u, v, w): termo convectivo skew-symmetric
│    └─ Função variational_form(w, w_n, w_n1):
│         ├─ u_star = 2*u_n - u_{n-1}  (Adams-Bashforth)
│         ├─ u_mid = (u + u_n)/2       (Crank-Nicolson)
│         ├─ nu_eff = nu + nu_t(u_n)
│         └─ Retorna F = 0 (equação residual)
│
├─ 6. LOOP PRINCIPAL (AMR)
│   PARA cada ciclo de refinamento (até max_refinements):
│   │
│   ├─ 6.1. INICIALIZAR variáveis de tempo (t=0, step=0)
│   │       Zerar w, w_n, w_n1
│   │
│   ├─ 6.2. LOOP TEMPORAL (enquanto t < T_end):
│   │   │
│   │   ├─ Calcular dt adaptativo via CFL:
│   │   │    dt = min(dt_max, CFL * h_min / u_max)
│   │   │
│   │   ├─ Montar sistema não-linear (F(w) = 0)
│   │   │
│   │   ├─ TENTAR resolver com Newton-GMRES
│   │   │   ├─ SE convergir: atualizar w_n1 ← w_n, w_n ← w
│   │   │   ├─ SENÃO: reduzir CFL e REPETIR passo (continue)
│   │   │   └─ FIM SE
│   │   │
│   │   ├─ Atualizar tempo: t += dt, step += 1
│   │   │
│   │   └─ SE (step % plot_steps == 0):
│   │        ├─ Extrair u_n, p_n de w
│   │        ├─ Calcular vorticidade e magnitude
│   │        ├─ Gerar gráficos PNG (Velocidade, Pressão, Vorticidade)
│   │        └─ Exportar solução para VTK/PVD
│   │   └─ FIM LOOP TEMPORAL
│   │
│   ├─ 6.3. REFINAMENTO ADAPTATIVO (AMR)
│   │   ├─ Calcular erro = gradiente da vorticidade (∇×u)
│   │   ├─ Selecionar fração (30%) das células com maior erro
│   │   ├─ Refinar malha localmente (refine())
│   │   ├─ Reconstruir espaços funcionais V, Q, W na nova malha
│   │   ├─ Projetar (interpolar) a solução antiga para a nova malha
│   │   └─ Atualizar w, w_n, w_n1 com a solução projetada
│   │
│   └─ FIM CICLO AMR
│
└─ FIM (FIM DA SIMULAÇÃO)


"""
#=====================================
# Bibliotecas
#=====================================
from dolfin import *            # FEniCS (dolfin) para simulação por Elementos Finitos
import numpy as np              # NumPy para manipulação de matrizes e operações matemáticas vetoriais
import matplotlib.pyplot as plt # Matplotlib para geração de gráficos e visualizações 2D
import os                       # Para manipulação de caminhos e criação de diretórios
import gmsh                     # API do Gmsh para geração de malhas geométricas complexas
import meshio                   # Meshio para converter formatos de arquivos de malhas entre diferentes softwares
import time
import matplotlib.tri as tri


#=====================================
# Parametros
#=====================================
U_inf = 10.0             # Velocidade de escoamento livre do fluido na entrada do canal (1.0 m/s)
D = 1.0                 # Diâmetro do cilindro que servirá como obstáculo (1.0 metro)
Re = 1000               # Número de Reynolds da simulação
nu = (U_inf * D) / Re   # Viscosidade cinemática do fluido
rho = 1.0               # Densidade do fluido

Cs = 0.15    # Constante de Smagorinsky, usada em modelos LES para turbulência

Lx_up = -5.0 * D    # Posição da parede de entrada do canal
Lx_down = 15.0 * D  # Posição da parede de saída do canal
Ly = 8.0 * D        # Altura total do canal
n = 10              # Número de divisões base para determinar o tamanho dos elementos da malha

lc = (Lx_down - Lx_up) / n      # Calcula o tamanho padrão dos elementos da malha
lc_min = 0.1  # Tamanho bem refinado colado no cilindro
lc_max = 1.0  # Tamanho máximo controlado para as paredes e longe do centro

dt = 0.001          # Passo de tempo discreto para o avanço da simulação
T_end = 20.0        # Tempo físico total em que a simulação será encerrada

max_refinements = 5     # Quantidade máxima de vezes que a malha pode sofrer refinamento adaptativo automático
refine_fraction = 0.5   # Fração das células com maior erro que serão escolhidas para serem refinadas

plot_interval_time = 0.05                    # Tempo físico no qual os resultados serão salvos/plotados
plot_steps = int(plot_interval_time / 0.002)   # Quantos passos de tempo ocorrem dentro do intervalo de plotagem


#=====================================
# Pastas
#=====================================
os.makedirs("plots", exist_ok=True)     # Cria a pasta para salvar os plots
os.makedirs("vtk", exist_ok=True)       # Cria a pasta para salvar os arquivos de visualização


#=====================================
# Malha
#=====================================
# Constrói a geometria do domínio e gera a malha usando o motor do Gmsh
def gerar_malha_gmsh(Lx_up, Lx_down, Ly, D, n, lc, lc_min, lc_max):

    print("Iniciando geração de malha com Gmsh...")
    gmsh.initialize()                   # Inicializa a API interna do Gmsh
    gmsh.model.add("canal_cilindro")    # Cria e nomeia um novo modelo de malha

    #-------------------------------------------
    # 1. Criar a geometria do canal (retângulo)
    #-------------------------------------------
    # Adiciona um retângulo via modelo OpenCASCADE (OCC) definindo o ponto inicial X, ponto inicial Y, Z=0, largura e altura
    canal = gmsh.model.occ.addRectangle(Lx_up, -Ly/2, 0, Lx_down - Lx_up, Ly)

    #-------------------------------------------
    # 2. Criar a geometria do cilindro (disco)
    #-------------------------------------------
    # Adiciona um disco circular (2D) centrado na origem (0.0, 0.0), Z=0, com raio no eixo X e raio no eixo Y igual a D/2
    cilindro = gmsh.model.occ.addDisk(0.0, 0.0, 0, D/2, D/2)

    #-------------------------------------------
    # 3. Booleana: Recortar o cilindro do canal
    #-------------------------------------------
    # Aplica uma operação booleana de corte: subtrai a área do cilindro da área do canal
    domain, _ = gmsh.model.occ.cut([(2, canal)], [(2, cilindro)])

    # Sincroniza os comandos geométricos do OpenCASCADE com o modelo principal do Gmsh para atualizar a árvore geométrica
    gmsh.model.occ.synchronize()

    #-------------------------------------------
    # 4. Refinamento local ao redor do cilindro
    #-------------------------------------------
    # Adiciona um campo matemático do tipo "Distance" (índice 1) que calcula a distância de qualquer ponto até entidades geométricas
    gmsh.model.mesh.field.add("Distance", 1)
    # Configura o campo de distância para não focar em pontos específicos isolados (lista vazia)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [])
    # Adiciona um campo do tipo "MathEval" (índice 2) que permite escrever fórmulas matemáticas para definir o tamanho da malha
    gmsh.model.mesh.field.add("MathEval", 2)
    # Define a fórmula matemática: o tamanho do elemento cresce linearmente conforme se afasta da origem (onde está o cilindro)
    gmsh.model.mesh.field.setString(2, "F", f"Min({lc_max}, {lc_min} + 0.1 * sqrt(x^2 + y^2), {lc} * (1 + (sqrt(x^2 + y^2) / ({D}/2))^2))")
    # Aplica essa fórmula matemática (campo 2) como a malha de fundo que dita a densidade de elementos no domínio
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    #-------------------------------------------
    # 5. Gerar a malha 2D
    #-------------------------------------------
    # Executa o algoritmo do Gmsh para quebrar a geometria contínua em elementos discretos bidimensionais (triângulos)
    gmsh.model.mesh.generate(2)

    # Define uma string com o nome do arquivo no formato padrão do Gmsh (.msh)
    msh_filename = "vtk/malha_inicial.msh"
    # Salva a malha gerada com o nome definido em msh_filename
    gmsh.write(msh_filename)
    print(f"Malha do Gmsh (.msh) salva em: vtk/malha_inicial.msh")
    # Encerra e limpa a memória utilizada pela API do Gmsh
    gmsh.finalize()

    print("Convertendo malha do Gmsh para o formato FEniCS via Meshio...")

    #-------------------------------------------
    # 6. Ler o arquivo do GMSH
    #-------------------------------------------
    # Usa a biblioteca Meshio para ler o arquivo ".msh" temporário gerado e interpretar seus dados brutos
    msh = meshio.read(msh_filename)

    # Cria uma lista vazia que armazenará exclusivamente os índices dos nós que formam os triângulos 2D
    triangle_cells = []
    # Percorre todos os blocos de células identificados pelo Meshio dentro da malha lida
    for cell in msh.cells:
        # Verifica se o tipo estrutural do bloco atual de células é composto por triângulos (elemento 2D)
        if cell.type == "triangle":
            # Se for o primeiro bloco de triângulos encontrado, armazena os dados diretamente na variável
            if len(triangle_cells) == 0:
                triangle_cells = cell.data
            # Se já existirem triângulos salvos, empilha verticalmente (combina) os novos dados aos anteriores
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])

    # # Criar um objeto meshio purificado de elementos espúrios ou mistos
    # # Instancia um novo objeto de malha limpo contendo apenas os dados estritamente necessários para o ambiente 2D do FEniCS
    # meshio_mesh = meshio.Mesh(
    #     # Filtra e repassa apenas as coordenadas X e Y de cada ponto (remove a coordenada Z, já que o FEniCS 2D gera erro se contiver Z)
    #     points=msh.points[:, :2],
    #     # Associa as células triangulares filtradas anteriormente ao novo objeto limpo
    #     cells=[("triangle", triangle_cells)]
    # )

    # # Define o caminho de texto completo onde a malha convertida pronta para o FEniCS será salva temporariamente
    # #xml_path = os.path.join("vtk", "malha_inicial.xdmf")
    xdmf_path = "vtk/malha_inicial.xdmf"

    # # Grava o objeto de malha limpo no formato estruturado XDMF utilizando codificação interna em XML
    # meshio.write(xml_path, meshio_mesh, data_format="dolfin-xml")

    meshio.write_points_cells(xdmf_path, msh.points[:, :2], [("triangle", triangle_cells)])

    print(f"Malha salva em: {xdmf_path}")

    # Instancia um objeto de malha vazio nativo da classe do FEniCS (Mesh)
    mesh = Mesh()
    # Abre o arquivo estruturado XDMF que acabou de ser gravado para realizar a leitura de dados
    with XDMFFile(xdmf_path) as xdmf:
        # Alimenta o objeto vazio nativo do FEniCS lendo a estrutura geométrica contida no arquivo XDMF
        xdmf.read(mesh)

    print(f"Malha gerada com Gmsh e carregada com sucesso: {mesh.num_cells()} células.")
    # Retorna o objeto de malha estruturado do FEniCS pronto para receber equações diferenciais e condições de contorno
    return mesh

# Chama a função definida acima passando as variáveis globais para gerar a malha física da simulação
mesh = gerar_malha_gmsh(Lx_up, Lx_down, Ly, D, n, lc, lc_min, lc_max)

print("Salvando arquivo final da malha...")

# Para capturar possíveis erros de gravação em tempo de execução
try:
    # Tenta instanciar e abrir um arquivo de gravação de alto desempenho no formato XDMF dentro do diretório 'vtk'
    with XDMFFile(os.path.join("vtk", "malha_inicial.xdmf")) as xdmf:
        # Grava os dados da malha gerada para o arquivo XDMF aberto
        xdmf.write(mesh)
    print("Sucesso! Malha salva em: vtk/malha_inicial.xdmf")
# Caso ocorra qualquer falha ou erro
except Exception as e:
    # Mostra na tela uma mensagem detalhando o erro encontrado e avisa que tentará o formato legado PVD
    print(f"Erro ao salvar em XDMF ({e}). Tentando formato alternativo PVD...")
    # Cria uma instância de gravação de arquivos clássica no formato PVD (Paraview Data) apontando para a pasta 'vtk'
    pvd_file = File(os.path.join("vtk", "malha_inicial.pvd"))
    # Utiliza o operador de fluxo (`<<`) nativo do FEniCS para descarregar todos os dados estruturais da malha no arquivo PVD
    pvd_file << mesh
    print("Sucesso! Malha salva em: vtk/malha_inicial.pvd")


#=====================================
# Espaco de elementos finitos
#=====================================
#-------------------------------------------
# 1. Espaços independentes (NÃO subespaços)
#-------------------------------------------
# Cria um espaço funcional vetorial (VectorFunctionSpace) para a velocidade usando elementos de Lagrange Contínuos ('CG') de grau 2 ($P_2$)
V = VectorFunctionSpace(mesh, 'CG', 2)  # Velocidade
# Cria um espaço funcional escalar (FunctionSpace) para a pressão usando elementos de Lagrange Contínuos ('CG') de grau 1 ($P_1$)
Q = FunctionSpace(mesh, 'CG', 1)       # Pressão

#-------------------------------------------
# 2. spaço misto usando MixedElement
#-------------------------------------------
# Define o elemento geométrico vetorial de grau 2 ($P_2$) baseado no formato de célula da malha para compor o espaço misto
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
# Define o elemento geométrico escalar de grau 1 ($P_1$) baseado no formato de célula da malha para compor o espaço misto
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
# Combina os dois elementos estruturais ($P_2$ e $P_1$) em um único elemento composto (Misto), metodo de Taylor-Hood
W_element = MixedElement([P2, P1])
# Cria o espaço funcional misto unificado (W) na malha
W = FunctionSpace(mesh, W_element)

#-------------------------------------------
# 3. Funções
#-------------------------------------------
# Cria o vetor de incógnitas principal 'w' associado ao espaço misto W, que conterá os coeficientes da velocidade e pressão do passo atual
w = Function(W)
# Cria o vetor de funções 'w_n' para armazenar os campos de velocidade e pressão calculados no passo de tempo imediatamente anterior ($t_n$)
w_n = Function(W)
# Cria o vetor de funções 'w_n1' para armazenar os campos de velocidade e pressão do passo de tempo retrasado ($t_{n-1}$)
w_n1 = Function(W)

#-------------------------------------------
# 4. Extrair subfunções
#-------------------------------------------
# Separa a função mista atual 'w' em seus componentes físicos: o vetor velocidade 'u' e o escalar pressão 'p'
u, p = split(w)
# Separa a função mista do passo anterior 'w_n' em velocidade anterior 'u_n' e pressão anterior 'p_n'
u_n, p_n = split(w_n)
# Separa a função mista do passo retrasado 'w_n1' em velocidade retrasada 'u_n1' e pressão retrasada 'p_n1'
u_n1, p_n1 = split(w_n1)


#=====================================
# CONDIÇÕES DE CONTORNO
#=====================================
# Cria uma classe customizada herdando de 'SubDomain' para mapear geometricamente a parede do cilindro
class CylinderBoundary(SubDomain):
    # Define o método interno que avalia se uma coordenada espacial 'x' pertence a esta fronteira específica
    def inside(self, x, on_boundary):
        # Retorna Verdadeiro se a distância radial do ponto $(x^2 + y^2)$ for menor que o raio do cilindro ao quadrado (com tolerância de 1e-6)
        return (x[0]**2 + x[1]**2) < (D/2)**2 + 1e-6

# Cria uma classe customizada herdando de 'SubDomain' para mapear a região de entrada de fluido (Inlet)
class Inlet(SubDomain):
    # Define o método interno que avalia se o ponto está na entrada do canal
    def inside(self, x, on_boundary):
        # Retorna Verdadeiro se o ponto estiver fisicamente na borda externa global e sua coordenada X estiver muito próxima de Lx_up
        return on_boundary and near(x[0], Lx_up, 1e-3)

#Cria uma classe customizada herdando de 'SubDomain' para mapear a região de saída de fluido (Outlet)
class Outlet(SubDomain):
    # Define o método interno que avalia se o ponto está na saida
    def inside(self, x, on_boundary):
        # Retorna Verdadeiro se o ponto estiver na borda externa e sua coordenada X estiver no fim do canal (Lx_down)
        return on_boundary and near(x[0], Lx_down, 1e-3)

# Cria uma classe customizada herdando de 'SubDomain' para mapear as paredes superior e inferior do canal
class TopBottom(SubDomain):
    # Define o método interno que avalia se o ponto pertence ao teto ou ao piso do domínio
    def inside(self, x, on_boundary):
        # Retorna Verdadeiro se o ponto estiver na borda externa E o valor absoluto de Y estiver muito próximo de metade da altura total (Ly/2)
        return on_boundary and (near(abs(x[1]), Ly/2, 1e-3))

# Cria uma classe customizada herdando de 'SubDomain' para colocar pressão em um ponto
class PressurePoint(SubDomain):
    # Define o método interno que avalia se é o ponto
    def inside(self, x, on_boundary):
        return (abs(x[0] - Lx_down) < 1e-3 and abs(x[1] - 0.0) < 1e-3)

bc_pressure_point = DirichletBC(W.sub(1), Constant(0.0), PressurePoint())

# Aplica a condição de Dirichlet no cilindro: define velocidade zero (componentes X e Y = 0.0) no subespaço 0 (velocidade) da malha mista W
bc_cylinder = DirichletBC(W.sub(0), Constant((0.0, 0.0)), CylinderBoundary())
# Aplica a condição de Dirichlet na entrada: força o fluido a entrar com velocidade horizontal estável U_inf e vertical nula
bc_inflow = DirichletBC(W.sub(0), Constant((U_inf, 0.0)), Inlet())
# Aplica p = 0 na saída: força a pressão no subespaço 1 (pressão) a ser nula na fronteira Outlet
# bc_outlet_pressure = DirichletBC(W.sub(1), Constant(0.0), Outlet())
# Aplica a condição de escorregamento (slip) nas paredes: extrai o componente vertical da velocidade (sub(0).sub(1)) e zera apenas ele (bloqueia o fluxo através da parede)
bc_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), TopBottom())
# Reúne todas as restrições físicas criadas em uma lista unificada (bcs) que será repassada diretamente ao solucionador do FEniCS
bcs = [bc_inflow, bc_cylinder, bc_slip, bc_pressure_point]


#=====================================
# Modelo LES
#=====================================
# Função que calcula a viscosidade turbulenta ($\nu_t$) com base no campo de velocidade atual
def compute_nu_t(u_func):
    # Calcula o tensor taxa de deformação simétrico $S = \frac{1}{2}(\nabla u + (\nabla u)^T)$
    S = sym(grad(u_func))
    # Calcula a magnitude (norma de Frobenius) do tensor de deformação: $|S| = \sqrt{2 \sum_{ij} S_{ij}S_{ij}}$
    S_mag = sqrt(2.0 * inner(S, S))
    # Obtém uma expressão simbólica do FEniCS correspondente a area de cada célula da malha
    cell_area = CellVolume(mesh)
    # Define a escala do filtro do modelo LES ($\Delta$) como a raiz quadrada da área do elemento triangular
    delta = cell_area**0.5
    # Aplica a fórmula algébrica de Smagorinsky para calcular a viscosidade turbulenta: $\nu_t = (C_s \cdot \Delta)^2 \cdot |S|$
    nu_t = (Cs * delta)**2 * S_mag
    # Retorna a expressão da viscosidade turbulenta que varia ponto a ponto no domínio
    return nu_t

# Declara o termo convectivo modificado para garantir estabilidade numérica em malhas mais grossas
def b_form(u, v, w):
    # Calcula a forma antissimétrica (skew-symmetric) do termo convectivo: $\frac{1}{2}(u \cdot \nabla v, w) - \frac{1}{2}(u \cdot \nabla w, v)$
    return 0.5 * inner(dot(u, nabla_grad(v)), w) * dx - 0.5 * inner(dot(u, nabla_grad(w)), v) * dx

# Declara a função que constrói o resíduo da equação variacional unificada (Forma Fraca)
def variational_form(w, w_old, w_old2):
    # Separa os componentes de velocidade 'u' e pressão 'p' da incógnita atual do sistema misto 'w'
    u, p = split(w)
    # Separa a velocidade 'u_old' do passo de tempo anterior ($t_n$), descartando a pressão antiga com o caractere '_'
    u_old, _ = split(w_old)
    # Separa a velocidade 'u_old2' do passo de tempo retrasado ($t_{n-1}$), descartando também sua respectiva pressão
    u_old2, _ = split(w_old2)

    #-------------------------------------------
    # Extrapolação Adams-Bashforth
    #-------------------------------------------
    # Realiza uma extrapolação linear explícita de segunda ordem para estimar a velocidade de transporte no tempo intermediário: $u^* = 2u^n - u^{n-1}$
    u_star = 2.0 * u_old - u_old2
    # # Projeta o campo algébrico 'u_old' no espaço vetorial contínuo V para transformá-lo em uma função manipulável pelo modelo LES
    # u_old_func = project(u_old, V)  # project já retorna Function em V
    # Chama a função definida anteriormente para calcular a viscosidade de redemoinhos ($\nu_t$) gerada pelo escoamento
    nu_t = compute_nu_t(u_old)
    # Soma a viscosidade molecular natural ($\nu$) com a turbulenta ($\nu_t$) para obter a viscosidade efetiva total do fluido
    nu_eff = nu + nu_t
    # Define a velocidade no ponto médio temporal ($t_{n+1/2}$) para aplicar o esquema de Crank-Nicolson (precisão de 2ª ordem no tempo)
    u_mid = 0.5 * (u + u_old)
    # Instancia as funções de teste virtuais 'v' (vetorial para quantidade de movimento) e 'q' (escalar para continuidade) do espaço misto W
    v, q = TestFunctions(W)

    # Monta a equação de resíduo final $F = 0$, integrando todos os termos em todo o domínio de elementos finitos ('dx')
    F = (1.0/dt) * inner(u - u_old, v) * dx \
        + inner(nu_eff * nabla_grad(u_mid), nabla_grad(v)) * dx \
        + b_form(u_star, u_mid, v) \
        - p * div(v) * dx \
        + div(u) * q * dx
    # Retorna a forma variacional completa não-linear que o solucionador de Newton tentará zerar a cada passo de tempo
    return F


#=====================================
# SOLVER
#=====================================
# Define um dicionário Python contendo as configurações numéricas para o solucionador do FEniCS
solver_parametros = {
    # Solucionador principal para o problema não-linear será Newton-Raphson
    "nonlinear_solver": "newton",
    # Abre um sub-dicionário de configurações específicas para ajustar o comportamento interno do solucionador de Newton
    "newton_solver": {
        # Define a tolerância de erro relativa
        "relative_tolerance": 1e-4,
        # Define a tolerância de erro absoluta
        "absolute_tolerance": 1e-4,
        # Limita o número máximo de correções de Newton, antes de forçar a parada com erro por não-convergência
        "maximum_iterations": 50,
        # Selecionar o solver linearGMRES (Generalized Minimal Residual), um método iterativo muito eficiente para sistemas lineares grandes e não-simétricos
        "linear_solver": "lu",  #"gmres",
        # Aplica o pré-condicionador algébrico multigrelha (Hypre Algebraic Multigrid), que acelera drasticamente a convergência do GMRES
        "preconditioner": "none"    #"hypre_amg"
    }
}

#=====================================
# LOOP PRINCIPAL
#=====================================
def main():
    # Informa ao Python que a função alterará variáveis estruturais globais criadas fora deste escopo
    global V, Q, W, w, w_n, w_n1, mesh, u, p, u_n, p_n, dt

    # PARAMETRIZAÇÃO DO CFL
    dt_maximo = dt     # O maior dt que você aceita que a simulação use
    cfl_alvo = 0.5     # Coeficiente de segurança clássico para Crank-Nicolson
    cfl_atual = cfl_alvo

    # Inicia o loop de Refinamento Adaptativo de Malha (AMR) que rodará até atingir o limite estipulado
    for refine_step in range(max_refinements):
        # Imprime no terminal o ciclo atual de refinamento em que a simulação se encontra
        print(f"\nCiclo AMR {refine_step+1} / {max_refinements}")
        # Imprime no terminal a quantidade atual de triângulos (elementos) contidos na malha ativa
        print(f"Elementos: {mesh.num_cells()}")

        # Inicialização com campo uniforme
        u_init = interpolate(Constant((U_inf, 0.0)), V)
        p_init = interpolate(Constant(0.0), Q)
        w_init = Function(W)
        assign(w_init.sub(0), u_init)
        assign(w_init.sub(1), p_init)

        w.assign(w_init)
        w_n.assign(w_init)
        w_n1.assign(w_init)

        # Inicializa o contador de tempo físico da simulação em zero segundos
        t = 0.0
        # Inicializa o contador discreto de passos de tempo em zero
        step = 0
        # Inicializa o contador de arquivos salvos para nomear sequencialmente os gráficos exportados
        plot_counter = 0
        # Contar tempo de processo
        inicio_cpu = time.time()

        # Cria (ou abre) o arquivo XDMF na pasta 'vtk/' para salvar as soluções temporais de forma eficiente
        xdmf_file = XDMFFile(f"vtk/simulation_results_cycle_{refine_step+1}.xdmf")

        # Configura o arquivo para permitir a adição consecutiva de passos de tempo e atualizar a malha se necessário
        xdmf_file.parameters["flush_output"] = True

        # Executa o loop temporal enquanto o tempo físico acumulado for menor que o tempo final
        while t < T_end:
            #---------------------------------------------------------
            # Criterio CFL
            #---------------------------------------------------------
            u_anterior, _ = w_n.split()
            u_mag = project(sqrt(dot(u_anterior, u_anterior)), FunctionSpace(mesh, 'CG', 1))
            max_u = u_mag.vector().max()
            h_min = mesh.hmin()

            # Evita divisão por zero impondo uma velocidade mínima
            safe_u = max(max_u, 1e-3)
            dt_cfl = (cfl_atual * h_min) / safe_u

            # O dt será o menor entre o limite físico do CFL e o teto máximo definido
            dt = min(dt_maximo, dt_cfl)

            # Chama a função para construir a forma fraca residual 'F' baseada nos estados atual, anterior e retrasado
            F = variational_form(w, w_n, w_n1)
            print("passei F")
            # Calcula simbolicamente o Jacobiano 'J' (Matriz Tangente) através da derivada direcional de F em relação a w
            J = derivative(F, w)
            print("passei J")

            # Instancia a estrutura matemática do problema não-linear contendo o resíduo F, a incógnita w, os contornos e o Jacobiano J
            problem = NonlinearVariationalProblem(F, w, bcs, J)
            print("passei problema")
            # Instancia o resolvedor não-linear de alto desempenho associado ao problema configurado
            solver = NonlinearVariationalSolver(problem)
            # Carrega o dicionário global de parâmetros numéricos para dentro deste solver ativo
            solver.parameters.update(solver_parametros)
            print("passei por aqui")

            # Abre um bloco de monitoramento para capturar falhas matemáticas e divergências durante a solução
            try:
                # Tenta executar as iterações de Newton-Raphson para encontrar os valores de velocidade e pressão do passo atual
                print("entrei try")
                solver.solve()
                print("sai try")
            # Caso o solver falhe (não convirja devido a instabilidades ou passos de tempo muito grandes)
            except RuntimeError as e:
                # Imprime um aviso no terminal indicando exatamente em qual tempo físico ocorreu o erro
                print(f"Erro no solver em t={t}: {e}")
                # Avisa ao usuário que uma estratégia de recuperação por redução de passo será executada
                print("Reduzindo o coeficiente de segurança do CFL...")
                cfl_atual = max(cfl_atual * 0.5, 1e-4)
                continue

            # Se o passo passou sem erros, tentamos retornar o CFL devagar ao valor alvo original
            cfl_atual = min(cfl_alvo, cfl_atual * 1.1)

            # Transfere os dados do passo anterior ($t_n$) para o espaço de armazenamento do passo retrasado ($t_{n-1}$)
            w_n1.assign(w_n)
            # Transfere a solução recém-calculada ($w$) para o espaço de armazenamento do passo anterior ($t_n$)
            w_n.assign(w)

            # Avança o relógio físico da simulação somando o incremento do passo de tempo ajustado
            t += dt
            # Incrementa em uma unidade o contador de passos temporais executados
            step += 1

            #---------------------------------------------------------
            # Plots
            #---------------------------------------------------------
            # Verifica se o passo atual atingiu o intervalo configurado para salvar os dados
            if step % plot_steps == 0:
                # Incrementa o número de identificação dos arquivos de saída
                plot_counter += 1
                # Imprime no terminal uma mensagem indicando o progresso do salvamento dos arquivos
                print(f"Salvando t = {t:.3f}s")
                # Separa os subcampos locais de velocidade e pressão de 'w' para fins de processamento visual
                u_n, p_n = w.split()

                # Calcula o rotacional da velocidade (Vorticidade, $\omega = \nabla \times u$) e projeta em um espaço escalar contínuo CG1
                omega = project(curl(u_n), FunctionSpace(mesh, 'CG', 1))
                # Calcula a magnitude escalar da velocidade ($\sqrt{u_x^2 + u_y^2}$) e projeta no mesmo espaço CG1
                mag = project(sqrt(dot(u_n, u_n)), FunctionSpace(mesh, 'CG', 1))

                # Inicializa uma janela de figura em branco na Matplotlib com tamanho proporcional de 10x6 polegadas
                plt.figure(figsize=(12,6))
                # Plota o mapa de cores de fundo representando a magnitude da velocidade com a paleta térmica 'coolwarm'
                # plot(mag, title=f"Velocidade t={t:.2f}s", cmap='coolwarm', interactive=False)
                fig_vel = plot(mag, title=f"Magnitude da Velocidade (m/s) t={t:.2f}s", cmap='viridis')
                # Adicionamos uma barra de cores
                plt.colorbar(fig_vel)
                # Sobrepõe vetores/setas indicando a direção do fluxo de velocidade na escala de tamanho 0.3
                plot(u_n, mode="velocity", scale=0.2, color='white', interactive=False)

                plt.xlim(-5, 15)
                plt.ylim(-4, 4)
                plt.gca().set_aspect('equal', adjustable='box')
                # Salva a imagem gerada na pasta 'plots/' formatando o nome com zeros à esquerda (ex: velocity_0001.png)
                plt.savefig(f"plots/velocity_{plot_counter:04d}.png", dpi=200, bbox_inches='tight')
                # Fecha a janela de visualização atual para liberar a memória RAM do computador
                plt.close()

                # Inicializa uma nova janela de figura para a renderização do campo de pressão
                plt.figure(figsize=(12,6))
                # Plota o mapa de cores da pressão utilizando a paleta divergente 'RdBu' (Red-Blue)
                # plot(p_n, title=f"Pressão t={t:.2f}s", cmap='RdBu', interactive=False)
                fig_pres = plot(p_n, title=f"Campo de Pressão (Pa) t={t:.2f}s", cmap='coolwarm')
                # Adicionamos uma barra de cores
                plt.colorbar(fig_pres)

                plt.xlim(-5, 15)
                plt.ylim(-4, 4)
                plt.gca().set_aspect('equal', adjustable='box')
                # Salva a figura de pressão na pasta de destino correspondente
                plt.savefig(f"plots/pressure_{plot_counter:04d}.png", dpi=200, bbox_inches='tight')
                # Fecha a janela gráfica da pressão
                plt.close()

                # Inicializa uma nova janela de figura para a renderização do campo de vorticidade (vórtices)
                plt.figure(figsize=(12,6))


                levels = np.linspace(-10, 10, 50)

                mesh_coordinates = mesh.coordinates()
                triangles = mesh.cells()
                triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
                omega_array = omega.compute_vertex_values(mesh)
                
                plt.tricontourf(triangulation, omega_array, levels=levels, cmap='RdBu_r', extend='both')
                plt.title(f"Campo de Vorticidade (1/s) t={t:.2f}s")
                
                # Barra de cores com limites fixos
                cbar = plt.colorbar(ticks=[-10, -5, 0, 5, 10])
                cbar.set_label('Vorticidade $\omega_z$')
                
                # Eixos e proporção
                plt.xlim(-5, 15)
                plt.ylim(-4, 4)
                plt.gca().set_aspect('equal', adjustable='box')
                
                # Adicionamos o contorno do cilindro manualmente para clareza
                circle = plt.Circle((0, 0), 0.5, color='black', fill=False, linewidth=2)
                plt.gca().add_patch(circle)

                plt.savefig(f"plots/vorticity_{plot_counter:04d}.png", dpi=200, bbox_inches='tight')
                plt.close()

                # # Plota a vorticidade com a paleta 'seismic', excelente para destacar giros horários (azul) e anti-horários (vermelho)
                # plot(omega, title=f"Vorticidade t={t:.2f}s", cmap='seismic', interactive=False)
                # # Salva a imagem da estrutura de esteira de vórtices em formato PNG
                # plt.savefig(f"plots/vorticity_{plot_counter:04d}.png", dpi=150)
                # # Fecha a janela gráfica da vorticidade
                # plt.close()

                # # Cria e descarrega a solução mista completa num arquivo PVD compatível com pós-processadores externos como o ParaView
                # File(f"vtk/solution_{plot_counter:04d}.pvd") << w

                # Escreve a função mista 'w' atrelando-a explicitamente ao tempo físico atual 't' da simulação
                xdmf_file.write(w, t)

                print(f"t = {t:.2f}s, CPU acum = {time.time() - inicio_cpu:.2f}s")

        #---------------------------------------------------------
        # AMR
        #---------------------------------------------------------
        # Extrai os componentes de velocidade e pressão da última solução obtida para guiar a lógica de refinamento
        u_n, p_n = w.split()
        # Calcula novamente o campo de vorticidade baseado no fluxo de velocidade estabilizado ao final do tempo total
        omega = project(curl(u_n), FunctionSpace(mesh, 'CG', 1))
        # Calcula o gradiente da vorticidade ($\nabla \omega$) e projeta em um espaço descontínuo de ordem zero (DG0, constante por célula)
        grad_omega = project(grad(omega), FunctionSpace(mesh, 'DG', 0))
        # Define o gradiente de vorticidade como o nosso indicador matemático de erro (onde ele for alto, a malha deve ser refinada)
        error_indicator = grad_omega

        # Cria uma lista booleana atrelada às células (triângulos) da topologia atual da malha para marcá-las
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        # Extrai os valores numéricos locais do indicador de erro de todas as células e converte em um vetor NumPy plano
        errors = error_indicator.vector().get_local()
        # Se a lista de erros contiver dados válidos (comprimento maior que zero)
        if len(errors) > 0:
            # Calcula o valor de corte (percentil) para selecionar apenas os 30% (`refine_fraction`) de triângulos com maiores erros
            threshold = np.percentile(errors, 100*(1 - refine_fraction))
            # Inicia uma varredura passando por cada uma das células geométricas individuais da malha
            for cell in cells(mesh):
                # Se o indicador de erro avaliado no centro (midpoint) da célula atual for maior que o valor de corte estabelecido
                if error_indicator(cell.midpoint()) > threshold:
                    # Marca esta célula específica como Verdadeira (True), agendando-a para ser dividida em triângulos menores
                    cell_markers[cell] = True
                # Caso o erro na célula seja baixo e aceitável
                else:
                    # Marca a célula como Falsa (False), indicando que seu tamanho geométrico deve ser mantido intacto
                    cell_markers[cell] = False
        # Caso o vetor de erros esteja corrompido ou vazio (situação limite de malha já ultra saturada)
        else:
            # Exibe um alerta no console informando que o processo adaptativo foi encerrado preventivamente
            print("Malha muito fina, parando AMR.")
            # Interrompe o loop principal de ciclos AMR imediatamente
            break

        # Executa a subdivisão geométrica da malha original dividindo ao meio os triângulos que foram marcados como True
        mesh = refine(mesh, cell_markers)

        # Reconstrói o espaço funcional de velocidade em cima da nova malha modificada e refinada
        V_new = VectorFunctionSpace(mesh, 'CG', 2)
        # Reconstrói o espaço funcional escalar de pressão em cima da nova malha gerada
        Q_new = FunctionSpace(mesh, 'CG', 1)

        # Recria os elementos geométricos mistos $P_2$ e $P_1$ adaptados para a nova topologia celular
        P2_new = VectorElement("CG", mesh.ufl_cell(), 2)
        P1_new = FiniteElement("CG", mesh.ufl_cell(), 1)
        # Combina novamente os novos elementos na estrutura mista unificada
        W_element_new = MixedElement([P2_new, P1_new])
        # Instancia o novo espaço funcional misto global W_new acoplado à nova malha refinada
        W_new = FunctionSpace(mesh, W_element_new)

        # Instancia uma nova função mista vazia acoplada a este novo espaço gerado
        w_new = Function(W_new)
        # Projeta a velocidade antiga para ajustar-se perfeitamente aos novos nós inseridos pelo refinamento
        u_proj = project(u_n, V_new)
        # Projeta o mapa de pressões antigo para os novos graus de liberdade da malha expandida
        p_proj = project(p_n, Q_new)
        # Atribui o vetor de velocidade interpolado ao subespaço correspondente (0) da nova função mista
        assign(w_new.sub(0), u_proj)
        # Atribui o escalar de pressão interpolado ao subespaço correspondente (1) da nova função mista
        assign(w_new.sub(1), p_proj)

        # Substitui os espaços funcionais antigos pelos novos espaços atualizados nas variáveis globais
        V, Q, W = V_new, Q_new, W_new

    # Imprime no terminal uma mensagem limpa sinalizando que toda a rotina de simulação e loops foi finalizada com sucesso
    print("\n--- FIM ---")

if __name__ == "__main__":
    main()