# Indoor-AutonomousNavigationNetwork

Esse projeto consiste na parte operacional do meu Trabalho de Conclusão de curso (TCC) que aborda temas baseados em Visão Computacional como Redes Neurais Convolucionais, Reconhecimento de padrões e Aprendizagem Profunda de Máquina para controle do Drone Dji Tello.

Testado com Python 3.6.8, mas pode ser compatível com outras versões 3.6.x


## Pré-requesitos

Tanto usuários de Windows quanto usuários de Linux podem utilizar o programa. Dito isso, é necessário baixar e instalar o Python 3.6.

### Windows
1. Visite a página do Python para Windows através desse [link](https://www.python.org/downloads/windows)

2. Procure na lista "_Stable Release_" por "**Python 3.6.8**" e depois escolha o executável referente a arquitetura do seu computador.

3. Clique no arquivo baixado e siga as instruções de instalação da interface. Obs.: não esqueça de adicionar o caminho do arquivo "python.exe" a variável de ambiente do sistema "**PYTHONPATH**". 

   - Reinicie o computador se necessário.
   - O importante é que após a utilização do comando "_python_" no **Prompt de comando(cmd)** apareça a versão "**Python 3.6.8**" como ativa.

### Linux 
1. Abra o Terminal e utilize os seguintes comandos em ordem:

> $ sudo apt-get update && sudo apt-get upgrade
> $ sudo apt-get install python

2. Adicionar o caminho do python a variável de ambiente do sistema "**PYTHONPATH**" através do comando 'export PYTHONPATH="${PYTHONPATH}:path/to/your/python'

   - Reinicie o comnputador se necessário

## Preparação e instalação de dependências
Abra o terminal do seu sistema, vá até a pasta em que o projeto ocupará (escolhido por você) e depois use os comandos abaixo para realizar a preparação do ambiente: 
1. > git clone https://github.com/LeonFerreirax/Indoor-AutonomousNavigation.git
2. > cd Indoor-AutonomousNavigation/
3. > pip install -r requirements.txt
4. É necessário baixar os _pesos_ da Rede Neural Alexnet treinada disponível em [alexnet_epoch178]( https://drive.google.com/file/d/1VbWI-2oCERg5gDiOtl4TmyMMYlkY0zkI/view?usp=sharing)
5. Adicione od arquivo baixadod e descompactadod no diretório **tf_alexnet** deste repositótio
6. Abra o código "final.py" e verifique se os caminhos do metagraph e do checkpoint estão enderaçados corretamente.   

## Executando o código
-**Passo 1:** Ligue o Drone Tello e conecte seu dispositivo ao Drone via wifi.

-**Passo 2:** Abra o diretório em um terminal:

> python main.py

-**Passo 3:** Apertar botão escolhido para iniciar a decolagem (t)

O comportamento/navegação do Drone é baseado na probabilidade de certeza que a rede tem após o frame capturado da câmera embutida de qual comando, dentre 6(seis) disponíveis, executar: frente, esquerda, direita, rotacionar a esquerda, rotacionar a direita, parar.

Dessa forma, o Drone terá um valor inteiro baseado nessa procentagem de certeza multiplicado por uma constante de velocidade com um peso maior para a classe escolhida e assim terá uma velocidade resultante para uma direção também resultante.
