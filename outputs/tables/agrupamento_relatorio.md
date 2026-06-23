# Analise de Agrupamento - ENEM 2024

## Objetivo

O objetivo desta analise foi identificar perfis de escolas a partir do desempenho medio dos participantes no ENEM 2024. A tecnica escolhida foi a Analise de Agrupamento, conforme a aula sobre formacao de grupos homogeneos.

A unidade de analise foi a escola. Essa escolha permite interpretar os resultados como perfis escolares de desempenho, em vez de agrupar participantes individualmente.

## Base e variaveis

Foi utilizada a base `RESULTADOS_2024.csv`, a mesma base do trabalho da disciplina. Foram considerados apenas registros com:

- codigo de escola informado;
- presenca confirmada nas quatro provas objetivas;
- notas completas em Ciencias da Natureza, Ciencias Humanas, Linguagens, Matematica e Redacao.

As variaveis usadas no agrupamento foram:

- media da escola em Ciencias da Natureza;
- media da escola em Ciencias Humanas;
- media da escola em Linguagens;
- media da escola em Matematica;
- media da escola em Redacao;
- media geral da escola.

Tambem foi aplicado um filtro minimo de 30 participantes por escola, para reduzir instabilidade em medias baseadas em poucos alunos.

## Metodo

As variaveis foram padronizadas antes do agrupamento, pois as notas possuem escalas e variabilidades diferentes. A medida de dissimilaridade adotada foi baseada na distancia euclidiana. O metodo utilizado foi K-medias, uma tecnica de particao que busca formar grupos com alta semelhanca interna e maior diferenca entre grupos.

O numero de grupos foi avaliado de `k=2` a `k=8`, usando duas medidas:

- inercia, associada ao metodo do cotovelo;
- indice de silhueta, usado como regra objetiva principal.

## Resultados

Foram lidos 4.332.944 registros da base. Apos os filtros de escola, presenca e notas completas, restaram 1.193.432 registros validos. Com o filtro de pelo menos 30 participantes, foram analisadas 12.817 escolas.

A maior silhueta ocorreu com `k=2`, com valor de 0,5914. Assim, a solucao final separou as escolas em dois perfis:

| Perfil | Escolas | Participantes medios | Media geral | Interpretacao |
|---:|---:|---:|---:|---|
| 1 | 9.388 | 75,46 | 500,63 | Desempenho mais baixo |
| 2 | 3.429 | 78,10 | 613,57 | Desempenho intermediario/alto |

A diferenca entre as medias gerais dos dois grupos foi de 112,94 pontos.

## Interpretacao

Os grupos se diferenciam principalmente pelo nivel medio de desempenho nas cinco areas avaliadas. O segundo perfil apresentou medias superiores em todas as notas, com destaque para Matematica e Redacao, que contribuem para ampliar a distancia entre os grupos.

Como as variaveis foram padronizadas, nenhuma prova dominou o agrupamento apenas por estar em uma escala mais dispersa. A analise deve ser interpretada como descritiva: ela identifica perfis de escolas segundo semelhancas nas notas, mas nao permite concluir causalidade sobre os fatores que explicam o desempenho.

## Arquivos gerados

- `outputs/tables/agrupamento_validacao_k.csv`: avaliacao de `k` por inercia e silhueta.
- `outputs/tables/agrupamento_perfis_clusters.csv`: resumo dos perfis encontrados.
- `outputs/tables/agrupamento_escolas_clusters.csv`: classificacao de cada escola.
- `outputs/figures/agrupamento_cotovelo_silhueta.svg`: grafico de validacao.
- `outputs/figures/agrupamento_clusters_pca.svg`: visualizacao dos grupos.
- `outputs/figures/agrupamento_perfis_notas.svg`: comparacao das notas medias por perfil.
