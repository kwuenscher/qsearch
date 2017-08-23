<h1 align="center">
  <img
    alt="compete_graph_animation"
    src="https://user-images.githubusercontent.com/9468947/29244436-bc4e66ee-7faf-11e7-8d3a-e1e419d60ca5.gif"
  />
  <br />
  qsearch
</h1>

<h4 align="center">
  Spatial quantum search (Grover's search) package based on the continuous time quantum walk.
</h4>


### Dependencies

1. numpy
2. networkx
3. scipy

### Usage

Quantum walks

```
G = nx.complete_graph(5)
qwalker = Qwalker(G)
N = 1/len(G)
t = 10
localisations = walk(qwalker, 1/N, t, 0.1)

```

Quantum Search

```
G = nx.complete_graph(10)
qwalker = Qwalker(G)
N = len(G)
t = 10
oracle = createMarkedVertex(N, 0)
success_probability = evolve(qwalker, 1/N, oracle, t, 0.1)

```
