## Resumo: ReLU, Sigmoid, Transformers e Self-Attention

Esse documento apresenta uma explicação simplificada sobre funções de ativação, Transformers e o mecanismo de Self-Attention, com exemplos práticos para facilitar o entendimento dos meus estudos sobre o assunto.

---

### 1. Funções de Ativação: ReLU e Sigmoid
As funções de ativação introduzem não-linearidade em redes neurais, permitindo que elas aprendam padrões complexos.

#### **Sigmoid Activation:**
- Produz valores entre 0 e 1, útil para classificação binária.
- Exemplo:
  - Imagine um classificador de emails que prevê se um email é spam ou não.
  - A saída do modelo pode ser um valor entre 0 e 1, onde valores próximos de 1 indicam spam e valores próximos de 0 indicam um email legítimo.

#### **ReLU (Rectified Linear Unit):**
- Fórmula: \( ReLU(x) = \max(0, x) \)
- Exemplo:
  - Se um modelo está aprendendo a prever preços de imóveis, valores negativos não fazem sentido.
  - O ReLU ajuda ignorando números negativos e mantendo apenas os positivos.

---

### 2. O que são Transformers?
Os **Transformers** são uma arquitetura de deep learning projetada para processar sequências de dados (como texto) de maneira eficiente. Eles substituem redes recorrentes (RNNs) e convolucionais (CNNs) ao utilizar o mecanismo de **Self-Attention**.

#### **Exemplo:**
- Considere a frase: "O cachorro corre no parque."
- Uma RNN processaria essa frase palavra por palavra, enquanto um Transformer pode analisar todas as palavras simultaneamente, compreendendo que "cachorro" está relacionado a "corre".

---

### 3. Feed-Forward e Positional Encoding

#### **Feed-Forward:**
- Cada camada do Transformer contém uma **Feed-Forward Network (FFN)**, que transforma os vetores processados pelo Self-Attention.

#### **Positional Encoding:**
- Como os Transformers processam palavras em paralelo, precisam de um mecanismo para identificar a ordem das palavras.
- Isso é feito com **Positional Encoding**, que usa funções senoidais e cosenoidais para adicionar informações sobre a posição de cada palavra.

#### **Exemplo:**
- A frase "Eu gosto de café" e "Café eu gosto de" possuem as mesmas palavras, mas significados diferentes.
- O Positional Encoding ajuda o modelo a entender essa diferença.

---

### 4. Self-Attention Deep Dive
O **Self-Attention** permite que o modelo determine quais palavras são mais importantes para entender o contexto.

#### **Três matrizes principais:**
- **Query (Q):** Representa a palavra que busca informações.
- **Key (K):** Representa palavras que podem ser relevantes.
- **Value (V):** Representa a informação associada às palavras.

#### **Exemplo:**
- Na frase: "O gato subiu no telhado porque viu um pássaro."
- O modelo precisa entender que "viu" está relacionado a "gato", e não a "telhado".
- O Self-Attention dá mais peso à relação entre "viu" e "gato".

#### **Masked Self-Attention:**
- No **Decoder**, o modelo só pode olhar palavras passadas, impedindo que ele "trapaceie" prevendo palavras futuras.

#### **Multi-Head Attention:**
- Utiliza múltiplas "cabeças" para capturar diferentes relações entre palavras.

---

### 5. Forward Pass e Desvio Padrão

#### **Forward Pass:**
- O **Forward Pass** é o processo de propagação dos dados pela rede neural, desde a entrada até a saída.
- Cada camada aplica transformações nos dados antes de passar para a próxima.

#### **Desvio Padrão (Standard Deviation):**
- Mede a dispersão dos valores em um conjunto de dados.
- Em redes neurais, é útil para normalizar os dados e melhorar a estabilidade do treinamento.

---

### 6. Transformer Blocks

Os **Transformers Blocks** são compostos por múltiplas camadas de **Self-Attention** e **Feed-Forward Networks**, permitindo que o modelo capture padrões complexos.

#### **Componentes principais:**
1. **Multi-Head Attention**
2. **Feed-Forward Networks**
3. **Positional Encoding**
4. **Camadas de Normalização**

Cada Transformer Block processa a entrada e passa o resultado para o próximo bloco.

---

### 7. Diferença entre Pre-Training e Fine-Tuning

#### **Pre-Training:**
- O modelo é treinado em grandes quantidades de dados para aprender padrões gerais.
- Exemplo: O GPT é treinado com bilhões de palavras da internet para aprender a estrutura da linguagem.

#### **Fine-Tuning:**
- O modelo é ajustado em dados específicos para uma tarefa particular.
- Exemplo: Ajustar o GPT para responder perguntas médicas usando um conjunto de dados especializado.

---

### 8. Diferença entre Encoder e Decoder
O **Encoder** e o **Decoder** são as duas partes principais do Transformer.

#### **Encoder:**
- Processa a entrada completa de uma vez.
- Usa **Self-Attention** para capturar relações entre todas as palavras.
- Exemplo: Ao traduzir "O céu é azul" para inglês, o Encoder processa a frase inteira antes de gerar a tradução.

#### **Decoder:**
- Gera a resposta palavra por palavra.
- Utiliza **Masked Self-Attention** para evitar olhar palavras futuras.
- Exemplo: Ao gerar "The sky is blue", o Decoder prevê cada palavra com base nas anteriores.

---

### 9. Arquitetura do GPT
O **GPT (Generative Pre-trained Transformer)** é baseado apenas no **Decoder**, projetado para geração de texto.

- Utiliza **Masked Self-Attention** para prever a próxima palavra sem olhar adiante.
- Treinado em grandes quantidades de texto para aprender padrões linguísticos.

#### **Exemplo:**
- Dado um início de frase como "Hoje o tempo está...", o modelo pode prever "ensolarado" ou "chuvoso" com base no contexto aprendido.

---

### Conclusão
- **ReLU e Sigmoid** são funções de ativação fundamentais.
- **Transformers** superam RNNs/LSTMs ao usar **Self-Attention** para processar textos de forma eficiente.
- **Self-Attention** é o coração dos Transformers, permitindo que o modelo aprenda relações entre palavras.
- **Feed-Forward Networks e Positional Encoding** são componentes essenciais do Transformer.
- **Transformer Blocks** organizam camadas para capturar padrões complexos.
- **Pre-Training e Fine-Tuning** são estratégias para adaptar modelos de IA.
- **GPT** usa apenas o Decoder do Transformer para gerar texto.



