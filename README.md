# Clustering through trade in an information economy
## Algorithm
Clustering of the 150 sample Iris vector dataset with three types of flowers is implemented as thus:
The economy consists of three players.
Each player is given 50 randomly selected samples from the Iris vector dataset.
For each round, two players are randomly selected, and a vector is randomly selected from each player.
The two players calculate their current entropy, and then swap vectors.
The two players calculate their new entropy, and if both players gain in entropy then the trade is successful.
If both players don't gain in entropy then the vector swap is reversed.
The previous steps are repeated a large number of times resulting in the clustering of the vectors.
## Entropy calculation
An adjacency matrix is calculated by multiplying all input vectors (50 vectors from a particular player) by all input vectors.
The adjacency matrix is fed into page rank.
The entropy is calculated over the page rank output probabilities.
## Future work
- The adjacency matrix is symmetric which may result in a loss of clustering accuracy. The adjacency matrix can be made asymmetric through the dropout algorithm, adding in noise, or multiplying the Iris vectors by random transforms.
- It might be possible to randomly select vectors in a non-uniform manner by using the page ranks (the algorithm now requiries fewer iterations).
- A distributed implementation might work. In a simple implementation trades might be one sided. If Ethereum contracts are used, trades could be fair (with high computational overhead).
