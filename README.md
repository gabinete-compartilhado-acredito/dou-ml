# Modelo de machine learning para seleção de matérias mais relevantes do DOU

Este é um repositório de notebooks de Python utilizados para criar e testar modelos de machine learning que
fazem uma pré-seleção das matérias mais relevantes do [Diário Oficial da União](https://www.in.gov.br/leiturajornal)
(DOU) de acordo com um conjunto de exemplos de matérias mais ou menos relevantes (classificados manualmente).
Os modelos treinados e testados são exportados para arquivos `.joblib` que são postos em produção: eles são carregados
pela [AWS Lambda](https://aws.amazon.com/pt/lambda/) e utilizados dentro de um sistema maior de captura e filtragem
de matérias do DOU para a produção do Boletim DOU Acredito (grupo de notícias no whastapp).

Veja também o [DOUTOR](https://github.com/gabinete-compartilhado-acredito/DOUTOR), nosso código de captura e filtragem por palavra-chave do Diário Oficial da União.

#### Disclaimer

Embora disponibilizados aqui, estes notebooks não foram construídos especificamente para serem distribuídos nem testados em outros ambientes
além do utilizado pelo desenvolvedor. Por isso, suas execuções podem depender de ajustes no seu conteúdo.

## O conteúdo deste repositório

### Pacotes necessários

A execução dos notebooks depende de alguns pacotes de Python. Os principais pacotes necessários e que não são tipicamente
inclusos na instalação do Python estão listados no arquivo `requirements.txt`. Alguns pacotes são utilizados apenas em trechos
antigos e dispensáveis para construção dos modelos (e.g. `spacy`) e, por isso, foram deixados de fora do `requirements.txt`.

### Dados utilizados

Os dados utilizados no treinamento dos modelos são arquivos CSV contendo matérias do DOU com sua relevância classificada
manualmente numa escala de 1 (menos relevante) a 5 (mais relevante). Por ocuparem bastante espaço, eles não estão inclusos
neste repositório e devem ser baixados deste
[link](https://storage.googleapis.com/gab-compartilhado-publico/monitor-dou/materias_dou_classificadas_manualmente.tar.gz)
e salvos diretamente na pasta `dados/`, e.g. `dados/*.csv`.

### Os notebooks

Existem dois notebooks principais neste repositório, um que cria modelos para a seção 1 do DOU e outro que cria
modelos para a seção 2. Ambos os modelos apresentam basicamente a mesma estrutura:

1. os dados são carregados;
2. os dados são divididos entre amostras de treino, validação e teste;
3. uma pipeline contendo pré-processamento e o modelo em si é construída da maneira desejada (modificações podem ser feitas no pré-processamento e no modelo);
4. o ajuste de hiperparâmetros é feito para a pipeline escolhida, utilizando uma amostra fixa de validação;
5. o modelo selecionado é testado novamente na amostra de validação, sob diversos aspectos;
6. o modelo final é testado na amostra de teste;
7. o modelo final é salvo para ser colocado em produção, junto com seus dados de treinamento e validação e com suas métricas;
8. os modelos antigos podem ser carregados e testados nos dados novos (seção utilizada para fazer o acompanhamento da deterioração do modelo);
9. algumas seções antigas e obsoletas ou utilizadas para tarefas que foram descontinuadas aparecem no final, mais como forma de registro.

### Modelos produzidos

Os modelos postos em produção utilizam a vetorização de textos "bag of words binário" (outras vetorizações, tais como com pesos tf-idf, foram testadas mas
apresentaram desempenho pior), com palavras e bigramas. Eles são modelos de regressão, sendo que o que melhor funcionou no caso da seção 1 foi uma regressão
linear com regularização (Ridge) e o que melhor funcionou na seção 2 foi um ensemble de Random Forest com Regressão Ridge. Os modelos já postos em produção
podem ser obtidos em formato joblib deste
[link](https://storage.googleapis.com/gab-compartilhado-publico/monitor-dou/modelo_dou_treinados.tar.gz).
Modelos do tipo transformers e RNN, pré-treinados, ainda não foram testados. 

Caso haja interesse em carregar os modelos já treinados, é importante ter instalados os pacotes necessários nas versões utilizadas. As versões necessárias
aos modelos estão listadas no arquivo `modelos/requirements.txt`. Os modelos precisam ser baixados do link acima e salvos na pasta `modelos/`.

## Contato

Quaisquer dúvidas a respeito deste trabalho podem ser encaminhadas a [Henrique Xavier](http://henriquexavier.net) (<https://github.com/hsxavier>).
