# Projeto 1: Algoritmo de Traçado de Raio
**Aluno:** Lucca Vieira Rocha
**Disciplina:** INF2608 - Fundamentos da Computação Gráfica

## 1. Abordagem e Arquitetura: Data-Oriented Design (DOD)
Para a implementação deste projeto, decidi afastar a solução do paradigma tradicional Orientado a Objetos (onde lida-se com um pixel e um raio de cada vez recursivamente) e implementei uma arquitetura vetorizada em **Data-Oriented Design** utilizando `NumPy`.

Em vez de processar `Ray` e `Hit` individualmente, a aplicação achata toda a geometria da cena em grandes tensores (*Structure of Arrays* ao invés de *Array of Structures*). Todos os raios da câmera são gerados simultaneamente em uma matriz `(N, 3)` e as interseções contra esferas e AABB (Boxes) são resolvidas via *broadcasting* matricial.

O gargalo comum do Python para laços de repetição profundos foi eliminado, garantindo uma performance e um tempo de renderização ordens de grandeza melhores do que uma implementação em Python clássica, aproximando a matemática da pipeline utilizada nativamente em GPUs e algoritmos de Machine Learning.

## 2. Requisitos Básicos Implementados (7.0 pts)
* **Geometria:** Interseção matemática matricial contra N Esferas simultâneas (Cálculo do discriminante de Bhaskara via vetores) e N Caixas AABB (Método dos Slabs adaptado para tensores paralelos).
* **Iluminação Direta e Sombras:** O cálculo de Phong ocorre em uma única passagem multiplicando o array de Normais pelo array do Vetor da Luz. O lançamento de sombras foi otimizado aplicando uma máscara booleana (`in_shadow`) que zera a cor dos pixels cujo `t_min` do raio secundário é menor que a distância para a luz.
* **Múltiplas Amostras (Antialiasing):** A renderização principal ocorre dentro de um loop de `SAMPLES_PER_PIXEL`. Para cada iteração, as matrizes de origem `X` e `Y` da câmera recebem um desvio de *jittering* gerado por ruído uniforme. No final, as matrizes são somadas e divididas.

## 3. Extensões Implementadas (3.0 pts)
Para fechar os 10.0 pontos através das extensões, implementei a lógica voltada ao recálculo dos raios:
* **Instanciação de objetos reflexivos (1.0 pt):** Um loop principal gerencia até `MAX_BOUNCES`. Em vez de estourar a pilha com recursão, uma máscara booleana `active_mask` é atualizada. Para os pontos atingidos que possuem material reflexivo, um novo vetor `R = V - 2(V.N)N` substitui a matriz de direções e a atenuação da luz é aplicada, repetindo o traçado matricial para as reflexões na cena perfeitamente (demonstrado pela esfera espelhada central no render).

## 4. Conclusão e Resultados
O arquivo em anexo `render_data_oriented.png` ilustra uma cena similar a Cornell Box com todos os requisitos integrados (paredes coloridas utilizando AABB boxes e esferas difusas e reflexivas). O código Python provido (`numpy_raytracer.py`) roda do começo ao fim sem dependências obscuras, validando que a adoção do Data-Oriented Design não apenas enxuga a escrita do código, como entrega um Motor de Renderização extremamente robusto e escalável.