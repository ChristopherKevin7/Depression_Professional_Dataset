# Saúde Mental e Estilo de Vida: Modelagem Preditiva com Machine Learning

## Visão Geral do Dataset

Este dataset explora a relação entre saúde mental e diversos fatores demográficos, de estilo de vida e relacionados ao trabalho. Ele inclui informações como:
- Gênero, idade e pressão no trabalho.
- Satisfação no emprego, duração do sono e hábitos alimentares.
- Estresse financeiro, horas de trabalho e indicadores de saúde mental, como depressão, pensamentos suicidas e histórico familiar de doenças mentais.

O objetivo do dataset é fornecer insights sobre como condições de vida e trabalho influenciam o bem-estar mental. Ele é adequado para análise exploratória de dados, modelagem preditiva e pesquisas estatísticas. As aplicações potenciais incluem:
- Identificação de fatores de risco para problemas de saúde mental.
- Entendimento do impacto do equilíbrio entre vida pessoal e profissional.
- Predição de desfechos de saúde mental com base em padrões de estilo de vida.

---

## Pré-Processamento dos Dados

### Codificação de Variáveis Categóricas
- **Variáveis Binárias**: Variáveis como `gender`, `depression`, `suicidal thoughts` e `family history` foram codificadas utilizando One-Hot Encoding.
- **Variáveis Ordenadas**: Colunas como `Sleep Duration` e `Dietary Habits` foram ordenadas hierarquicamente com valores numéricos.

---

## Análise de Correlação

### Principais Achados
- **Depressão x Idade**: Apresentou a maior correlação com a variável alvo.
- **Depressão x Pensamentos Suicidas**: Correlação forte.
- **Depressão x Pressão no Trabalho**: Correlação significativa.
- **Depressão x Gênero**: Apresentou a menor correlação com a variável alvo.

---

## Modelos de Machine Learning

### K-Nearest Neighbors (KNN)
#### Relatório de Classificação
```plaintext
              precisão    recall  f1-score   suporte

           0       0.94      0.99      0.96       368
           1       0.86      0.44      0.58        43

    acurácia                          0.93       411
   média macro       0.90      0.72      0.77       411
média ponderada       0.93      0.93      0.92       411
```

O modelo KNN apresentou um bom desempenho geral, mas enfrentou dificuldades com a classe minoritária (casos positivos de depressão), refletidas no baixo valor de recall para a classe 1.

#### Considerações Importantes
- Oversampling/Undersampling: A geração de dados sintéticos pode comprometer a confiabilidade do modelo, enquanto a redução de amostras saudáveis diminuiria significativamente o tamanho do dataset (~5x menor), comprometendo a representatividade.
- Para lidar com o desbalanceamento sem alterar os dados, modelos como Regressão Logística e Random Forest, que permitem configuração de pesos entre as classes, foram explorados.

###  Regressão Logística
#### Relatório de Classificação
```plaintext

              precisão    recall  f1-score   suporte

           0       1.00      1.00      1.00       368
           1       0.98      1.00      0.99        43

    acurácia                          1.00       411
   média macro       0.99      1.00      0.99       411
média ponderada       1.00      1.00      1.00       411
```

#### AUC-ROC
Conjunto de Treinamento: 1.00
Conjunto de Teste: 1.00
#### Validação Cruzada
```plaintext
Pontuações de validação cruzada: [0.9854, 0.9708, 0.9757, 0.9806, 0.9805, 0.9756, 0.9902, 0.9756, 0.9902, 0.9902]
Acurácia média: 0.98
```
#### Análise de Overfitting
- Acurácia Treino x Teste: Não houve queda significativa; ambos os valores são altos e consistentes (Treinamento: 0.98, Teste: 1.00).
- Validação Cruzada: A alta acurácia com baixa variação entre os folds indica boa generalização.
- AUC-ROC: Valores idênticos para treino e teste reforçam a capacidade do modelo de distinguir entre as classes.
#### Conclusão
O modelo de Regressão Logística apresentou excelente desempenho sem fortes indicios de overfitting.

### Random Forest
#### Relatório de Classificação (Threshold = 0.25)
```plaintext
              precisão    recall  f1-score   suporte

           0       0.98      0.96      0.97       368
           1       0.73      0.86      0.79        43

    acurácia                          0.95       411
   média macro       0.85      0.91      0.88       411
média ponderada       0.96      0.95      0.95       411
```
#### AUC-ROC
Score Geral: 0.98
#### Validação Cruzada
```plaintext
Pontuações de validação cruzada: [0.9515, 0.9417, 0.9515, 0.9563, 0.9415, 0.9463, 0.9610, 0.9512, 0.9512, 0.9415]
Acurácia média: 0.95
```
#### Ajuste de Threshold
Threshold = 0.25: Bom equilíbrio entre precisão e recall para a classe minoritária.
Threshold = 0.2: Recall muito alto (0.98) para a classe minoritária, mas redução da precisão (0.66).

#### Conclusão
O Random Forest oferece flexibilidade para priorizar recall ou precisão ajustando o threshold. O threshold 0.25 equilibra bem as métricas, enquanto 0.2 pode ser útil em cenários que priorizem a minimização de falsos negativos.

### Conclusão e Recomendações
- Regressão Logística: Melhor desempenho geral, com métricas consistentes entre treino, teste e validação cruzada. É confiável e interpretável, ideal para produção.
- Random Forest: Métricas robustas e flexibilidade no ajuste de thresholds, adequado para cenários específicos que priorizem recall ou precisão.
- KNN: Desempenho limitado para datasets desbalanceados, mais adequado para conjuntos de dados balanceados.
### Próximos Passos
- Explorar novas features ou fontes externas de dados para melhorar o poder preditivo.
- Testar métodos de ensemble combinando Regressão Logística e Random Forest.
- Avaliar os modelos em cenários reais para confirmar sua aplicabilidade.

## Como Executar

Clone o repositório:
  ```bash
    git clone https://github.com/ChristopherKevin7/Depression_Professional_Dataset.git
  ```

Execute o notebook DepressionProfessional.ipynb para reproduzir os resultados.

### Links do Dataset Utilizado

- [Depression Professional Dataset](https://www.kaggle.com/datasets/ikynahidwin/depression-professional-dataset/data)

## Contribuições

Feedbacks e sugestões são bem-vindos! Sinta-se à vontade para abrir uma issue ou enviar um pull request.
